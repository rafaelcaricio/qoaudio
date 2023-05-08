#![forbid(unsafe_code)]
//! # QOA - Quite OK Audio Format
//!
//! A library for decoding qoa files.
//!
use std::collections::VecDeque;
use std::fmt::Display;
use std::time::Duration;

pub const QOA_MIN_FILESIZE: usize = 16;
pub const QOA_SLICE_LEN: usize = 20;
pub const QOA_LMS_LEN: usize = 4;
pub const QOA_HEADER_SIZE: usize = 8;
pub const QOA_MAGIC: u32 = u32::from_be_bytes(*b"qoaf");

#[derive(Debug, Clone)]
pub enum ProcessingMode {
    FixedSamples {
        channels: u32,
        sample_rate: u32,
        samples: u32,
    },
    Streaming,
}

impl ProcessingMode {
    pub fn duration(&self) -> Option<Duration> {
        match self {
            ProcessingMode::FixedSamples {
                channels: _channels,
                sample_rate,
                samples,
            } => Some(Duration::from_secs_f64(
                (*samples as f64) / (*sample_rate as f64),
            )),
            ProcessingMode::Streaming => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QoaDecoder {
    mode: ProcessingMode,
    lms: Vec<QoaLms>,
}

#[derive(Debug, Clone)]
struct QoaLms {
    history: [i32; QOA_LMS_LEN],
    weights: [i32; QOA_LMS_LEN],
}

impl QoaDecoder {
    pub fn streaming() -> Self {
        Self {
            mode: ProcessingMode::Streaming,
            lms: Vec::new(),
        }
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

        if samples == 0 {
            // Indicates we are in streaming mode.
            Ok(Self::streaming())
        } else {
            Ok(Self {
                mode: ProcessingMode::FixedSamples {
                    channels,
                    sample_rate,
                    samples,
                },
                lms,
            })
        }
    }

    pub fn mode(&self) -> &ProcessingMode {
        &self.mode
    }

    pub fn decode_frames(&mut self, bytes: &[u8]) -> Result<Vec<Frame>, DecodeError> {
        // Check if we begin with the header, if so skip it. Otherwise, assume
        // we are already at the start of the first frame.
        let mut already_processed_bytes = if bytes.len() >= QOA_MIN_FILESIZE {
            let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
            if magic == QOA_MAGIC {
                QOA_HEADER_SIZE
            } else {
                0
            }
        } else {
            0
        };

        let mut decoded_frames = Vec::new();
        while already_processed_bytes < bytes.len() {
            let (sample_data, frame_size) = self.decode_frame(&bytes[already_processed_bytes..])?;

            already_processed_bytes += frame_size;
            decoded_frames.push(sample_data);
        }

        Ok(decoded_frames)
    }

    fn decode_frame(&mut self, bytes: &[u8]) -> Result<(Frame, usize), DecodeError> {
        if bytes.len() < QOA_HEADER_SIZE {
            // Error with not enough bytes to read the frame header
            return Err(DecodeError::NotEnoughBytes);
        }

        let frame_header = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let channels = ((frame_header >> 56) & 0x0000ff) as u32;
        let sample_rate = ((frame_header >> 32) & 0xffffff) as u32;
        let total_samples = ((frame_header >> 16) & 0x00ffff) as usize;
        let frame_size = (frame_header & 0x00ffff) as usize;

        if bytes.len() < 8 + QOA_LMS_LEN * 4 * channels as usize {
            // Error with not enough bytes to decode the frame
            return Err(DecodeError::NotEnoughBytes);
        }

        let data_size = frame_size - 8 - QOA_LMS_LEN * 4 * channels as usize;
        let num_slices = data_size / 8;
        let max_total_samples = num_slices * QOA_SLICE_LEN;

        if let ProcessingMode::FixedSamples {
            channels: decoded_channels,
            sample_rate: decoded_sample_rate,
            ..
        } = self.mode
        {
            if channels != decoded_channels || sample_rate != decoded_sample_rate {
                // Error with invalid frame header, incompatible with the decoder metadata
                return Err(DecodeError::IncompatibleFrame);
            }
        }

        if frame_size > bytes.len() || total_samples * channels as usize > max_total_samples {
            return Err(DecodeError::InvalidHeader);
        }

        let mut frame = Frame {
            channels,
            sample_rate,
            samples: vec![0i16; total_samples * channels as usize],
        };

        // Initialize the LMS state if needed, when streaming the number of channels might change
        if self.lms.len() != channels as usize && matches!(self.mode, ProcessingMode::Streaming) {
            self.lms = vec![
                QoaLms {
                    history: [0; QOA_LMS_LEN],
                    weights: [0; QOA_LMS_LEN],
                };
                channels as usize
            ];
        } else if self.lms.len() != channels as usize {
            // Error with invalid number of channels in non-streaming mode
            return Err(DecodeError::InvalidHeader);
        }

        // Read the LMS state: 4 x 2 bytes history, 4 x 2 bytes weights per channel
        let mut processed_bytes = 8;
        for c in 0..channels as usize {
            let mut history = u64::from_be_bytes(
                bytes[processed_bytes..processed_bytes + 8]
                    .try_into()
                    .unwrap(),
            );
            let mut weights = u64::from_be_bytes(
                bytes[processed_bytes + 8..processed_bytes + 16]
                    .try_into()
                    .unwrap(),
            );
            processed_bytes += 16;

            for i in 0..QOA_LMS_LEN {
                self.lms[c].history[i] = ((history >> 48) as i16) as i32;
                history <<= 16;
                self.lms[c].weights[i] = ((weights >> 48) as i16) as i32;
                weights <<= 16;
            }
        }

        let decoded_bytes = self.decode_slices(bytes, processed_bytes, &mut frame);
        Ok((frame, decoded_bytes + processed_bytes))
    }

    fn decode_slices(&mut self, bytes: &[u8], start: usize, frame: &mut Frame) -> usize {
        // The bytes is the full frame, so we need to skip the header which the start parameter
        // indicated the start of the slices part of the data.
        let mut processed_bytes = start;
        let mut sample_index = 0;

        let channels = frame.channels as usize;
        let data = &mut frame.samples;
        let total_samples = data.len() / channels;

        while sample_index < total_samples && processed_bytes + 8 < bytes.len() {
            for c in 0..channels {
                let mut slice = u64::from_be_bytes(
                    bytes[processed_bytes..processed_bytes + 8]
                        .try_into()
                        .unwrap(),
                );

                let scale_factor = ((slice >> 60) & 0xf) as i32;
                let slice_end = (sample_index + QOA_SLICE_LEN).min(total_samples) * channels + c;

                let slice_start = sample_index * channels + c;
                for si in (slice_start..slice_end).step_by(channels) {
                    let prediction = self.lms[c].predict();
                    let quantized = ((slice >> 57) & 0x7) as usize;
                    let dequantized = QOA_DEQUANT_TAB[scale_factor as usize][quantized];
                    let reconstructed = (prediction + dequantized).clamp(-32768, 32767) as i16;

                    data[si] = reconstructed;
                    slice <<= 3;

                    self.lms[c].update(reconstructed, dequantized);
                }

                processed_bytes += 8;
            }

            sample_index += QOA_SLICE_LEN;
        }

        // Returns the number of samples processed and the total amount of bytes processed
        // since we started at the start index, we subtract it to get the total.
        processed_bytes - start
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
pub struct Frame {
    samples: Vec<i16>,
    channels: u32,
    sample_rate: u32,
}

impl Frame {
    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(
            self.samples.len() as f64 / self.channels as f64 / self.sample_rate as f64,
        )
    }
}

#[derive(Debug, Clone)]
pub enum DecodeError {
    LessThanMinimumFileSize,
    NotQoaFile,
    NoSamples,
    InvalidHeader,
    NotEnoughBytes,
    IncompatibleFrame,
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
            DecodeError::NotEnoughBytes => write!(f, "Not enough bytes to decode"),
            DecodeError::IncompatibleFrame => write!(f, "Incompatible frame header"),
        }
    }
}

pub struct DecodedAudio {
    channels: u32,
    sample_rate: u32,
    samples: VecDeque<i16>,
}

impl Iterator for DecodedAudio {
    type Item = i16;

    fn next(&mut self) -> Option<Self::Item> {
        self.samples.pop_front()
    }
}

impl TryFrom<Vec<Frame>> for DecodedAudio {
    type Error = DecodeError;

    fn try_from(frames: Vec<Frame>) -> Result<Self, Self::Error> {
        if frames.is_empty() {
            return Err(DecodeError::NoSamples);
        }

        let channels = frames[0].channels();
        let sample_rate = frames[0].sample_rate();

        let mut samples = Vec::new();
        for frame in frames {
            if frame.channels() != channels {
                return Err(DecodeError::InvalidHeader);
            }

            if frame.sample_rate() != sample_rate {
                return Err(DecodeError::InvalidHeader);
            }

            samples.extend_from_slice(frame.samples());
        }

        Ok(DecodedAudio {
            channels,
            sample_rate,
            samples: samples.into(),
        })
    }
}

impl DecodedAudio {
    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(
            self.samples.len() as f64 / self.channels as f64 / self.sample_rate as f64,
        )
    }
}

#[cfg(feature = "rodio")]
mod rodio_integration {
    use super::*;

    impl rodio::Source for DecodedAudio {
        fn current_frame_len(&self) -> Option<usize> {
            None
        }

        fn channels(&self) -> u16 {
            self.channels() as u16
        }

        fn sample_rate(&self) -> u32 {
            self.sample_rate()
        }

        fn total_duration(&self) -> Option<Duration> {
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

        let mut qoa = QoaDecoder::decode_header(qoa_bytes).unwrap();
        assert!(matches!(
            qoa.mode,
            ProcessingMode::FixedSamples {
                channels: 2,
                sample_rate: 44100,
                samples: 2394122,
                ..
            }
        ));

        let audio = qoa.decode_frames(qoa_bytes).unwrap();
        assert!(!audio.is_empty());
        assert_eq!(audio[0].channels(), 2);
        assert_eq!(audio[0].sample_rate(), 44100);

        let audio = DecodedAudio::try_from(audio).unwrap();
        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.duration(), Duration::from_secs_f64(54.288480726));
    }
}
