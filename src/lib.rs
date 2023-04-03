#![forbid(unsafe_code)]
//! # QOA - Quite OK Audio Format
//!
//! A library for decoding qoa files.
//!
use std::collections::VecDeque;
use std::fmt::Display;
use std::time::Duration;

const QOA_MIN_FILESIZE: usize = 16;

const QOA_SLICE_LEN: usize = 20;
const QOA_HEADER_SIZE: usize = 8;
const QOA_LMS_LEN: usize = 4;
const QOA_MAGIC: u32 = 0x716f6166; // "qoaf"

#[derive(Clone)]
pub struct QoaDecoder {
    channels: u32,
    sample_rate: u32,
    samples: u32,
    lms: Vec<QoaLms>,

    // TODO: This shouldn't be here...
    decoded_samples: Option<VecDeque<i16>>,
}

impl Iterator for QoaDecoder {
    type Item = i16;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: This is totally a hack, but it works for now.
        self.decoded_samples
            .as_mut()
            .and_then(|samples| samples.pop_front())
    }
}

#[derive(Clone)]
struct QoaLms {
    history: [i32; QOA_LMS_LEN],
    weights: [i32; QOA_LMS_LEN],
}

impl QoaDecoder {
    pub fn new(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut qoa = QoaDecoder::decode_header(bytes)?;
        let samples = qoa.decode_frames(bytes);
        qoa.decoded_samples = Some(samples);

        Ok(qoa)
    }

    pub fn samples(&self) -> u32 {
        self.samples
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.samples as f64 / self.sample_rate as f64)
    }

    pub fn decoded_samples(&self) -> Option<VecDeque<i16>> {
        self.decoded_samples.clone()
    }

    pub fn decode_header(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < QOA_MIN_FILESIZE {
            return Err(DecodeError::LessThanMinimumFileSize);
        }

        let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        if magic != QOA_MAGIC {
            return Err(DecodeError::NotQoaFile);
        }

        let samples = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        if samples == 0 {
            return Err(DecodeError::NoSamples);
        }

        let frame_header = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let channels = ((frame_header >> 56) & 0x0000ff) as u32;
        let sample_rate = ((frame_header >> 32) & 0xffffff) as u32;

        if channels == 0 || sample_rate == 0 {
            return Err(DecodeError::InvalidHeader);
        }

        let lms = vec![
            QoaLms {
                history: [0; QOA_LMS_LEN],
                weights: [0; QOA_LMS_LEN],
            };
            channels as usize
        ];

        Ok(Self {
            channels,
            sample_rate,
            samples,
            lms,
            decoded_samples: None,
        })
    }

    pub fn decode_frames(&mut self, bytes: &[u8]) -> VecDeque<i16> {
        let mut already_processed_bytes = QOA_HEADER_SIZE;
        let total_samples = (self.samples * self.channels) as usize;
        let mut sample_data = vec![0i16; total_samples];

        let mut sample_index = 0;
        while sample_index < self.samples {
            let (frame_len, frame_size) = self.decode_frame(
                &bytes[already_processed_bytes..],
                &mut sample_data[(sample_index * self.channels) as usize..],
            );

            if frame_size == 0 {
                break;
            }

            already_processed_bytes += frame_size;
            sample_index += frame_len as u32;
        }

        sample_data.into()
    }

    fn decode_frame(&mut self, bytes: &[u8], samples_data: &mut [i16]) -> (usize, usize) {
        if bytes.len() < 8 + QOA_LMS_LEN * 4 * self.channels as usize {
            return (0, 0);
        }

        let frame_header = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let channels = ((frame_header >> 56) & 0x0000ff) as u32;
        let sample_rate = ((frame_header >> 32) & 0xffffff) as u32;
        let total_samples = ((frame_header >> 16) & 0x00ffff) as usize;
        let frame_size = (frame_header & 0x00ffff) as usize;

        let data_size = frame_size - 8 - QOA_LMS_LEN * 4 * self.channels as usize;
        let num_slices = data_size / 8;
        let max_total_samples = num_slices * QOA_SLICE_LEN;

        if channels != self.channels
            || sample_rate != self.sample_rate
            || frame_size > bytes.len()
            || total_samples * self.channels as usize > max_total_samples
        {
            return (0, 0);
        }

        // Read the LMS state: 4 x 2 bytes history, 4 x 2 bytes weights per channel
        let mut p = 8;
        for c in 0..self.channels as usize {
            let mut history = u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap());
            let mut weights = u64::from_be_bytes(bytes[p + 8..p + 16].try_into().unwrap());
            p += 16;

            for i in 0..QOA_LMS_LEN {
                self.lms[c].history[i] = ((history >> 48) as i16) as i32;
                history <<= 16;
                self.lms[c].weights[i] = ((weights >> 48) as i16) as i32;
                weights <<= 16;
            }
        }

        let (frame_len, decoded_bytes) = self.decode_slices(bytes, p, total_samples, samples_data);
        (frame_len, decoded_bytes + p)
    }

    fn decode_slices(
        &mut self,
        bytes: &[u8],
        start: usize,
        total_samples: usize,
        samples_data: &mut [i16],
    ) -> (usize, usize) {
        let mut p = start;
        let mut sample_index = 0;

        while sample_index < total_samples && p + 8 < bytes.len() {
            for c in 0..self.channels as usize {
                let mut slice = u64::from_be_bytes(bytes[p..p + 8].try_into().unwrap());

                let scale_factor = ((slice >> 60) & 0xf) as i32;
                let slice_end =
                    (sample_index + QOA_SLICE_LEN).min(total_samples) * self.channels as usize + c;

                let slice_start = sample_index * self.channels as usize + c;
                for si in (slice_start..slice_end).step_by(self.channels as usize) {
                    let prediction = self.lms[c].predict();
                    let quantized = ((slice >> 57) & 0x7) as usize;
                    let dequantized = QOA_DEQUANT_TAB[scale_factor as usize][quantized];
                    let reconstructed = (prediction + dequantized).clamp(-32768, 32767) as i16;

                    samples_data[si] = reconstructed;
                    slice <<= 3;

                    self.lms[c].update(reconstructed, dequantized);
                }

                p += 8;
            }

            sample_index += QOA_SLICE_LEN;
        }

        (sample_index, p - start)
    }
}

impl QoaLms {
    fn predict(&self) -> i32 {
        let mut prediction = 0;
        for i in 0..QOA_LMS_LEN {
            prediction += self.weights[i] * self.history[i];
        }
        prediction >> 13
    }

    fn update(&mut self, sample: i16, residual: i32) {
        let delta = residual >> 4;
        for i in 0..QOA_LMS_LEN {
            self.weights[i] += if self.history[i] < 0 { -delta } else { delta };
        }

        for i in 0..QOA_LMS_LEN - 1 {
            self.history[i] = self.history[i + 1];
        }
        self.history[QOA_LMS_LEN - 1] = sample as i32;
    }
}

const QOA_DEQUANT_TAB: [[i32; 8]; 16] = [
    [1, -1, 3, -3, 5, -5, 7, -7],
    [5, -5, 18, -18, 32, -32, 49, -49],
    [16, -16, 53, -53, 95, -95, 147, -147],
    [34, -34, 113, -113, 203, -203, 315, -315],
    [63, -63, 210, -210, 378, -378, 588, -588],
    [104, -104, 345, -345, 621, -621, 966, -966],
    [158, -158, 528, -528, 950, -950, 1477, -1477],
    [228, -228, 760, -760, 1368, -1368, 2128, -2128],
    [316, -316, 1053, -1053, 1895, -1895, 2947, -2947],
    [422, -422, 1405, -1405, 2529, -2529, 3934, -3934],
    [548, -548, 1828, -1828, 3290, -3290, 5117, -5117],
    [696, -696, 2320, -2320, 4176, -4176, 6496, -6496],
    [868, -868, 2893, -2893, 5207, -5207, 8099, -8099],
    [1064, -1064, 3548, -3548, 6386, -6386, 9933, -9933],
    [1286, -1286, 4288, -4288, 7718, -7718, 12005, -12005],
    [1536, -1536, 5120, -5120, 9216, -9216, 14336, -14336],
];

#[derive(Debug, Clone)]
pub enum DecodeError {
    LessThanMinimumFileSize,
    NotQoaFile,
    NoSamples,
    InvalidHeader,
}

impl std::error::Error for DecodeError {}

impl Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DecodeError::LessThanMinimumFileSize => {
                write!(f, "File is less than minimum size of 16 bytes")
            }
            DecodeError::NotQoaFile => write!(f, "File is not a qoa file"),
            DecodeError::NoSamples => write!(f, "File has no samples"),
            DecodeError::InvalidHeader => write!(f, "File has invalid header"),
        }
    }
}

#[cfg(feature = "rodio")]
mod rodio_integration {
    use super::*;

    impl rodio::Source for QoaDecoder {
        fn current_frame_len(&self) -> Option<usize> {
            None
        }

        fn channels(&self) -> u16 {
            self.channels() as u16
        }

        fn sample_rate(&self) -> u32 {
            self.sample_rate()
        }

        fn total_duration(&self) -> Option<std::time::Duration> {
            Some(self.duration())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let qoa_bytes = include_bytes!("../fixtures/julien_baker_sprained_ankle.qoa");

        let qoa = QoaDecoder::new(qoa_bytes).unwrap();
        assert_eq!(qoa.channels(), 2);
        assert_eq!(qoa.sample_rate(), 44100);
        assert_eq!(qoa.samples(), 2394122);
    }
}
