#![forbid(unsafe_code)]
//! # QOA - Quite OK Audio Format
//!
//! A library for decoding qoa files.
use std::fmt::Display;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::time::Duration;
use std::{fmt, io};

pub const QOA_SLICE_LEN: usize = 20;
pub const QOA_LMS_LEN: usize = 4;
pub const QOA_HEADER_SIZE: usize = 8;
pub const QOA_MAGIC: u32 = u32::from_be_bytes(*b"qoaf");
pub const MAX_SLICES_PER_CHANNEL_PER_FRAME: usize = 256;

/// The decoding mode of the QOA file.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingMode {
    /// Total number of samples is known and sample rate and number of channels
    /// is fixed for the entire file.
    FixedSamples {
        /// Number of channels.
        channels: u8,
        /// The sample rate for all channels in HZ (e.g. 44100)
        sample_rate: u32,
        /// Number of samples per channel in the entire file.
        samples: u32,
    },
    /// Total number of samples is not known and the sample rate and number of
    /// channels can change each frame.
    Streaming,
}

/// Decoder of QOA files.
///
/// Decoded samples are obtained by calling [`QoaDecoder::next`]. This is a
/// streaming decoder. It reads bytes in small chunks as needed. Memory
/// consumption per channel is less than 100 bytes.
///
/// ## Details
///
/// While the spec requires a valid QOA file to have all but the last frame to
/// have all 256 slices, this decoder will accept partial interior frames
/// without error.
#[derive(Debug)]
pub struct QoaDecoder<R> {
    mode: ProcessingMode,
    lms: Vec<QoaLms>,
    reader: R,
    current_frame: CurrentFrame,
    pending_samples: Box<[i16]>,
    next_pending_sample_idx: usize,
    returned_first_frame_header: bool,
}

#[derive(Debug, Clone, Default)]
pub struct QoaLms {
    pub history: [i32; QOA_LMS_LEN],
    pub weights: [i32; QOA_LMS_LEN],
}

#[derive(Debug)]
pub enum EncodeError {
    IoError(io::Error),
    InvalidInput(&'static str),
    UnsupportedSampleRate,
    UnsupportedChannelCount,
}

pub struct QoaEncoder<W: io::Write> {
    writer: W,
    lms: Vec<QoaLms>,
    num_channels: u8,
    sample_rate: u32,
    total_samples_written_per_channel: u32,
    expected_total_samples_per_channel: u32,
}

impl<W: io::Write> QoaEncoder<W> {
    pub fn new(
        mut writer: W,
        num_channels: u8,
        sample_rate: u32,
        total_samples_per_channel_in_file: u32,
    ) -> Result<Self, EncodeError> {
        if num_channels == 0 { // num_channels as usize > u8::MAX is implicitly handled by u8 type
            return Err(EncodeError::UnsupportedChannelCount);
        }
        if sample_rate == 0 || sample_rate > 0xFFFFFF {
            return Err(EncodeError::UnsupportedSampleRate);
        }

        writer.write_all(&QOA_MAGIC.to_be_bytes())?;
        writer.write_all(&total_samples_per_channel_in_file.to_be_bytes())?;

        let lms_states = vec![QoaLms::default(); num_channels as usize];

        Ok(Self {
            writer,
            lms: lms_states,
            num_channels,
            sample_rate,
            total_samples_written_per_channel: 0,
            expected_total_samples_per_channel: total_samples_per_channel_in_file,
        })
    }

    pub fn write_frame(&mut self, samples_for_frame_per_channel: &[Vec<i16>]) -> Result<(), EncodeError> {
        if samples_for_frame_per_channel.len() != self.num_channels as usize {
            return Err(EncodeError::InvalidInput("Channel count mismatch for frame"));
        }

        // encode_frame performs other necessary validations
        let (frame_header_info, frame_bytes) =
            encode_frame(samples_for_frame_per_channel, &mut self.lms, self.sample_rate)?;

        self.writer.write_all(&frame_bytes)?;
        self.total_samples_written_per_channel += frame_header_info.num_samples_per_channel as u32;

        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), EncodeError> {
        if self.expected_total_samples_per_channel != 0
            && self.total_samples_written_per_channel != self.expected_total_samples_per_channel
        {
            return Err(EncodeError::InvalidInput(
                "Total samples written does not match expected total in file header",
            ));
        }
        self.writer.flush()?;
        Ok(())
    }
}

impl From<io::Error> for EncodeError {
    fn from(inner: io::Error) -> Self {
        EncodeError::IoError(inner)
    }
}

impl<R> QoaDecoder<R>
where
    R: io::Read,
{
    /// Read the file header and first frame header of a QOA file read from
    /// `reader`.
    ///
    /// QoaDecoder makes many small reads so wrapping a `File` with a
    /// `BufReader` is recommended. This is done automatically when using
    /// [`QoaDecoder::open`].
    pub fn new(mut reader: R) -> Result<Self, DecodeError> {
        let magic = read_u32_be(&mut reader)?;
        if magic != QOA_MAGIC {
            return Err(DecodeError::NotQoaFile);
        }

        let samples = read_u32_be(&mut reader)?;
        let mode = if samples == 0 {
            // Indicates we are in streaming mode.
            ProcessingMode::Streaming
        } else {
            ProcessingMode::FixedSamples {
                // replaced on first call to decode_frame_header_and_lms below
                channels: 0,
                sample_rate: 0,
                samples,
            }
        };

        // replaced on first call to decode_frame_header_and_lms below
        let current_frame: CurrentFrame = Default::default();

        let mut to_return = Self {
            mode,
            lms: Vec::new(),
            reader,
            current_frame,
            pending_samples: Box::new([]),
            next_pending_sample_idx: 0,
            returned_first_frame_header: false,
        };

        // If we are in streaming mode, then there is no frame header to read.
        if to_return.mode != ProcessingMode::Streaming {
            let found_frame = to_return.decode_frame_header_and_lms(true)?;
            if !found_frame {
                return Err(DecodeError::NoSamples);
            }
        }

        Ok(to_return)
    }

    /// Consume this `QoaDecoder` and return the inner reader.
    pub fn into_inner(self) -> R {
        self.reader
    }

    /// Returns if the decoder is streaming (i.e. total number of samples are
    /// not known and sample rate and number of channels can change each frame)
    /// or fixed (i.e. total number of samples are known and sample rate and
    /// number of channels does not change).
    pub fn mode(&self) -> &ProcessingMode {
        &self.mode
    }

    /// The header of the frame currently being processed (i.e. most recently
    /// returned from next).
    pub fn current_frame_header(&self) -> &FrameHeader {
        &self.current_frame.header
    }

    pub fn total_duration(&self) -> Option<Duration> {
        match &self.mode {
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

    /// Returns Ok(true) if a frame was read. Returns Ok(false) if EOF was
    /// encountered before any bytes were read.
    fn decode_frame_header_and_lms(&mut self, first: bool) -> Result<bool, DecodeError> {
        let frame_header = match read_u64_be(&mut self.reader) {
            Ok(h) => h,
            Err(e) => {
                return if e.kind() == io::ErrorKind::UnexpectedEof {
                    // This is expected.
                    Ok(false)
                } else {
                    Err(e.into())
                };
            }
        };
        let num_channels = ((frame_header >> 56) & 0x0000ff) as u8;
        let sample_rate = ((frame_header >> 32) & 0xffffff) as u32;
        let num_samples_per_channel = ((frame_header >> 16) & 0x00ffff) as u16;
        let frame_size = (frame_header & 0x00ffff) as u16;
        let frame_header = FrameHeader {
            num_channels,
            sample_rate,
            num_samples_per_channel,
        };

        if num_channels == 0 || sample_rate == 0 {
            return Err(DecodeError::InvalidFrameHeader);
        }

        const LMS_SIZE: usize = 4;
        let non_sample_data_size = QOA_HEADER_SIZE + QOA_LMS_LEN * LMS_SIZE * num_channels as usize;
        if frame_size as usize <= non_sample_data_size {
            return Err(DecodeError::InvalidFrameHeader);
        }
        let data_size = frame_size as usize - non_sample_data_size;
        let num_slices = data_size / 8;

        if num_slices % num_channels as usize != 0 {
            return Err(DecodeError::InvalidFrameHeader);
        }
        if num_slices / num_channels as usize > MAX_SLICES_PER_CHANNEL_PER_FRAME {
            return Err(DecodeError::InvalidFrameHeader);
        }

        if let ProcessingMode::FixedSamples {
            channels: decoded_channels,
            sample_rate: decoded_sample_rate,
            ..
        } = &mut self.mode
        {
            if first {
                *decoded_channels = num_channels;
                *decoded_sample_rate = sample_rate;
            } else if num_channels != *decoded_channels || sample_rate != *decoded_sample_rate {
                // Error with invalid frame header, incompatible with the decoder metadata
                return Err(DecodeError::IncompatibleFrame);
            }
        }

        // Initialize the LMS state if needed, when streaming the number of channels
        // might change
        if self.lms.len() != num_channels as usize {
            // Already checked number of channels did not change in Fixed mode above.
            assert!(matches!(self.mode, ProcessingMode::Streaming) || first);
            self.lms
                .resize_with(num_channels as usize, Default::default);
        }

        // Read the LMS state: 4 x 2 bytes history, 4 x 2 bytes weights per channel
        for c in 0..num_channels as usize {
            let mut history = read_u64_be(&mut self.reader)?;
            let mut weights = read_u64_be(&mut self.reader)?;

            for i in 0..QOA_LMS_LEN {
                self.lms[c].history[i] = ((history >> 48) as i16) as i32;
                history <<= 16;
                self.lms[c].weights[i] = ((weights >> 48) as i16) as i32;
                weights <<= 16;
            }
        }

        self.current_frame = CurrentFrame {
            header: frame_header,
            num_samples_per_channel_remaining: num_samples_per_channel,
        };

        Ok(true)
    }

    fn decode_one_slice_per_channel(&mut self) -> Result<(), DecodeError> {
        assert!(self.next_pending_sample_idx >= self.pending_samples.len());
        let channels = self.current_frame.header.num_channels as usize;
        let full_slices_num_samples = QOA_SLICE_LEN * channels;
        if self.pending_samples.len() != full_slices_num_samples {
            self.pending_samples = vec![0_i16; full_slices_num_samples].into_boxed_slice();
        }
        self.next_pending_sample_idx = 0;

        for channel_idx in 0..channels {
            let mut slice = read_u64_be(&mut self.reader)?;

            let scale_factor = ((slice >> 60) & 0xf) as usize;
            for sample_in_channel_slice_idx in 0..QOA_SLICE_LEN {
                let prediction = self.lms[channel_idx].predict();
                let quantized = ((slice >> 57) & 0x7) as usize;
                let dequantized = QOA_DEQUANT_TAB[scale_factor][quantized];
                let reconstructed = (prediction + dequantized).clamp(-32768, 32767) as i16;

                let data_idx = sample_in_channel_slice_idx * channels + channel_idx;

                self.pending_samples[data_idx] = reconstructed;
                slice <<= 3;

                self.lms[channel_idx].update(reconstructed, dequantized);
            }
        }
        let num_samples_per_channel = self.current_frame.num_samples_per_channel_remaining;
        if (num_samples_per_channel as usize) < QOA_SLICE_LEN {
            let total_num_samples = num_samples_per_channel as usize * channels;
            // If this is the last slice of the file, it might not have all 20
            // samples and be zero filled. Cut off the excess 0 samples. This
            // is done after the loop so the loop can have a constant size for
            // better compiler optimizations.
            self.pending_samples = self.pending_samples[0..total_num_samples]
                .to_vec()
                .into_boxed_slice();
            self.current_frame.num_samples_per_channel_remaining -= num_samples_per_channel;
        } else {
            self.current_frame.num_samples_per_channel_remaining -= QOA_SLICE_LEN as u16;
        }
        Ok(())
    }
}

impl QoaDecoder<io::BufReader<File>> {
    /// Open a file, wrap it with BufReader and create a new decoder.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<QoaDecoder<io::BufReader<File>>, DecodeError> {
        let file = File::open(path)?;
        QoaDecoder::new(io::BufReader::new(file))
    }
}

impl QoaDecoder<Cursor<Vec<u8>>> {
    /// Create a new decoder for use in streaming mode.
    ///
    /// This allows for decoding a single frame at a time. This is useful for
    /// decoding QOA content that is being streamed over the network.
    pub fn new_streaming() -> Result<Self, DecodeError> {
        let streaming_header: Vec<u8> = [QOA_MAGIC, 0]
            .iter()
            .flat_map(|&x| x.to_be_bytes())
            .collect();
        QoaDecoder::new(Cursor::new(streaming_header))
    }

    /// Decode a single frame in streaming mode.
    pub fn decode_frame(&mut self, frame_data: &[u8]) -> Result<Vec<i16>, DecodeError> {
        self.reader.get_mut().extend_from_slice(frame_data);
        let mut to_return = Vec::new();
        for item in self {
            match item? {
                QoaItem::Sample(s) => to_return.push(s),
                QoaItem::FrameHeader(_) => (),
            }
        }
        Ok(to_return)
    }
}

/// Return type of [`QoaDecoder::next`].
#[derive(Debug)]
pub enum QoaItem {
    Sample(i16),
    FrameHeader(FrameHeader),
}

impl<R: io::Read> Iterator for QoaDecoder<R> {
    type Item = Result<QoaItem, DecodeError>;

    /// Get the next sample or the frame header if a new frame is starting.
    ///
    /// If an error is returned, iteration should be considered finished and
    /// next should not be called again.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(sample) = self.pending_samples.get(self.next_pending_sample_idx) {
            self.next_pending_sample_idx += 1;
            return Some(Ok(QoaItem::Sample(*sample)));
        }
        if !self.returned_first_frame_header {
            // For consistency return the header read in the `new` function.
            self.returned_first_frame_header = true;
            return Some(Ok(QoaItem::FrameHeader(self.current_frame.header)));
        }
        if self.current_frame.num_samples_per_channel_remaining > 0 {
            if let Err(e) = self.decode_one_slice_per_channel() {
                return Some(Err(e));
            }
        } else {
            return match self.decode_frame_header_and_lms(false) {
                Ok(true) => Some(Ok(QoaItem::FrameHeader(self.current_frame.header))),
                Ok(false) => None,
                Err(e) => Some(Err(e)),
            };
        }
        debug_assert!(!self.pending_samples.is_empty());
        self.next()
    }
}

/// A fully decoded QOA file.
pub struct DecodedQoa {
    /// Number of channels in `samples`
    pub num_channels: u8,
    /// Sample rate in HZ of `samples`
    pub sample_rate: u32,
    /// Interleaved samples of all channels (e.g. L-R-L-R-L-R... if there are
    /// two channels)
    pub samples: Vec<i16>,
}

/// Decode all samples of a QOA file read from `reader`.
///
/// If not all frames in the file have the same number of channels and the same
/// sample rate DecodeError::IncompatibleFrame is returned.
/// This is a convenience function that uses QoaDecoder internally. QoaDecoder
/// is able to decode samples even if frames have different number of channels
/// or sample rate.
pub fn decode_all<R: io::Read>(reader: R) -> Result<DecodedQoa, DecodeError> {
    let mut decoder = QoaDecoder::new(reader)?;
    let mut samples = Vec::new();
    if let &ProcessingMode::FixedSamples {
        samples: samples_per_channel,
        channels,
        ..
    } = decoder.mode()
    {
        samples.reserve_exact(samples_per_channel as usize * channels as usize);
    }
    let QoaItem::FrameHeader(FrameHeader {
        num_channels,
        sample_rate,
        ..
    }) = decoder.next().unwrap()?
    else {
        unreachable!();
    };
    for item in decoder {
        match item? {
            QoaItem::Sample(s) => samples.push(s),
            QoaItem::FrameHeader(header) => {
                if num_channels != header.num_channels || sample_rate != header.sample_rate {
                    return Err(DecodeError::IncompatibleFrame);
                }
            }
        }
    }
    Ok(DecodedQoa {
        num_channels,
        sample_rate,
        samples,
    })
}

/// Same as [`decode_all`] but open the file and wrap in a BufReader first.
pub fn open_and_decode_all<P: AsRef<Path>>(path: P) -> Result<DecodedQoa, DecodeError> {
    let file = File::open(path.as_ref())?;
    let reader = io::BufReader::new(file);
    decode_all(reader)
}

#[derive(Debug, Default)]
struct CurrentFrame {
    header: FrameHeader,
    /// Number of samples to be read per channel before this frame is done
    num_samples_per_channel_remaining: u16,
}

/// The metadata at the beginning of each frame of slices.
#[derive(Debug, Copy, Clone, Default)]
pub struct FrameHeader {
    /// Number of channels in this frame
    pub num_channels: u8,
    /// Sample rate in HZ for this frame
    pub sample_rate: u32,
    /// Samples per channel in this frame
    pub num_samples_per_channel: u16,
}

fn read_u32_be<R: io::Read>(mut reader: R) -> io::Result<u32> {
    Ok(u32::from_be_bytes(read_array(&mut reader)?))
}

fn read_u64_be<R: io::Read>(mut reader: R) -> io::Result<u64> {
    Ok(u64::from_be_bytes(read_array(&mut reader)?))
}

fn read_array<R: io::Read, const LEN: usize>(mut reader: R) -> io::Result<[u8; LEN]> {
    let mut bytes = [0_u8; LEN];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

impl QoaLms {
    #[inline(always)]
    pub fn predict(&self) -> i32 {
        let mut prediction: i32 = 0;
        for i in 0..QOA_LMS_LEN {
            // The spec does not specify a precision for these operations or
            // how to handle overflow. The reference C implementation does
            // not prevent signed integer overflow (undefined behavior). Since
            // overflow should not occur in a properly encoded file, we take the
            // lower overhead option of just allowing wrapping.
            prediction = prediction.wrapping_add(self.weights[i].wrapping_mul(self.history[i]));
        }
        prediction >> 13
    }

    #[inline(always)]
    pub fn update(&mut self, sample: i16, residual: i32) {
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

pub const QOA_DEQUANT_TAB: [[i32; 8]; 16] = [
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

#[derive(Debug)]
pub enum DecodeError {
    NotQoaFile,
    NoSamples,
    InvalidFrameHeader,
    IncompatibleFrame,
    IoError(io::Error),
}

impl std::error::Error for DecodeError {}

pub fn find_closest_dequantized_value_in_row(residual: i32, scale_factor_row: &[i32; 8]) -> (usize, i32) {
    let mut best_quantized_idx = 0;
    let mut actual_dequantized_value = scale_factor_row[0];
    let mut min_diff = (residual - actual_dequantized_value).abs();

    for idx in 1..8 {
        let current_val = scale_factor_row[idx];
        let diff = (residual - current_val).abs();

        if diff < min_diff {
            min_diff = diff;
            best_quantized_idx = idx;
            actual_dequantized_value = current_val;
        } else if diff == min_diff {
            if current_val < actual_dequantized_value { // Tie-breaking for same diff
                best_quantized_idx = idx;
                actual_dequantized_value = current_val;
            }
        }
    }
    (best_quantized_idx, actual_dequantized_value)
}

pub fn quantize_residual(residual: i32, scale_factor: usize) -> usize {
    let (best_quantized_idx, _) = find_closest_dequantized_value_in_row(residual, &QOA_DEQUANT_TAB[scale_factor]);
    best_quantized_idx
}

pub fn find_best_scale_factor_quantized_and_dequantized(residual: i32) -> (usize, usize, i32) {
    let mut overall_best_sf = 0;
    let (mut overall_best_quantized_idx, mut overall_actual_dequantized_value) =
        find_closest_dequantized_value_in_row(residual, &QOA_DEQUANT_TAB[0]);
    let mut overall_min_diff = (residual - overall_actual_dequantized_value).abs();

    for sf in 1..16 { // Iterate scale_factors from 1 to 15
        let (current_quantized_idx, current_actual_dequantized_value) =
            find_closest_dequantized_value_in_row(residual, &QOA_DEQUANT_TAB[sf]);
        let current_diff = (residual - current_actual_dequantized_value).abs();

        if current_diff < overall_min_diff {
            overall_min_diff = current_diff;
            overall_best_sf = sf;
            overall_best_quantized_idx = current_quantized_idx;
            overall_actual_dequantized_value = current_actual_dequantized_value;
        }
        // Tie-breaking for scale_factor (prefer smaller sf if errors are equal)
        // is implicitly handled by iterating sf from low to high and only updating
        // overall_best_sf if a *strictly* smaller diff is found.
    }
    (overall_best_sf, overall_best_quantized_idx, overall_actual_dequantized_value)
}

pub fn encode_slice(input_samples: &[i16], lms: &mut QoaLms) -> Result<(u64, Vec<i16>), EncodeError> {
    if input_samples.len() != QOA_SLICE_LEN {
        return Err(EncodeError::InvalidInput("Slice must contain QOA_SLICE_LEN samples"));
    }

    // 1. Determine slice_scale_factor
    let mut max_ideal_sf = 0;
    let mut temp_lms_for_sf_finding = lms.clone(); // Clone LMS to avoid altering it during sf search

    for &sample_val_i16 in input_samples.iter().take(QOA_SLICE_LEN) {
        let predicted = temp_lms_for_sf_finding.predict();
        let residual = sample_val_i16 as i32 - predicted;
        let (ideal_sf, _, dequantized_for_temp_lms) = find_best_scale_factor_quantized_and_dequantized(residual);
        if ideal_sf > max_ideal_sf {
            max_ideal_sf = ideal_sf;
        }
        // Update temp_lms for next ideal_sf calculation. Reconstruct and update.
        let reconstructed_for_temp_lms = (predicted + dequantized_for_temp_lms).clamp(-32768, 32767) as i16;
        temp_lms_for_sf_finding.update(reconstructed_for_temp_lms, dequantized_for_temp_lms);
    }
    let slice_scale_factor = max_ideal_sf;

    let mut reconstructed_samples = Vec::with_capacity(QOA_SLICE_LEN);
    let mut slice_data: u64 = 0;

    // Add scale factor to slice_data
    slice_data |= (slice_scale_factor as u64) << 60;

    for i in 0..QOA_SLICE_LEN {
        let current_input_sample = input_samples[i];
        let predicted_sample = lms.predict();
        let residual = current_input_sample as i32 - predicted_sample;
        
        let quantized_value_3bit = quantize_residual(residual, slice_scale_factor);
        let dequantized_residual = QOA_DEQUANT_TAB[slice_scale_factor][quantized_value_3bit];
        
        let reconstructed_sample_i32 = predicted_sample + dequantized_residual;
        let reconstructed_sample_i16 = reconstructed_sample_i32.clamp(-32768, 32767) as i16;
        
        reconstructed_samples.push(reconstructed_sample_i16);
        lms.update(reconstructed_sample_i16, dequantized_residual);
        
        slice_data |= (quantized_value_3bit as u64) << (57 - (i * 3));
    }

    Ok((slice_data, reconstructed_samples))
}

fn write_frame_header_to_buf(buf: &mut Vec<u8>, num_channels: u8, sample_rate: u32, samples_in_frame_per_channel: u16, frame_size: u16) {
    buf.push(num_channels);
    // sample_rate is u24, so we take the last 3 bytes of its u32 BE representation
    let sr_bytes = sample_rate.to_be_bytes();
    buf.extend_from_slice(&sr_bytes[1..4]); 
    buf.extend_from_slice(&samples_in_frame_per_channel.to_be_bytes());
    buf.extend_from_slice(&frame_size.to_be_bytes());
}

pub fn encode_frame(
    input_samples_per_channel: &[Vec<i16>],
    lms_states: &mut [QoaLms],
    sample_rate: u32,
) -> Result<(FrameHeader, Vec<u8>), EncodeError> {
    // 1. Input Validation
    if input_samples_per_channel.is_empty() || lms_states.is_empty() {
        return Err(EncodeError::InvalidInput("Input samples or LMS states cannot be empty."));
    }

    let num_channels = input_samples_per_channel.len();
    if num_channels != lms_states.len() {
        return Err(EncodeError::InvalidInput("Number of channels in samples and LMS states must match."));
    }
    if num_channels > u8::MAX as usize {
        return Err(EncodeError::UnsupportedChannelCount); // Or InvalidInput
    }

    if sample_rate == 0 || sample_rate > 0xFFFFFF {
        return Err(EncodeError::UnsupportedSampleRate); // Or InvalidInput
    }

    let num_samples_this_frame_per_channel = input_samples_per_channel[0].len();
    if num_samples_this_frame_per_channel == 0 {
        return Err(EncodeError::InvalidInput("Number of samples per channel cannot be zero."));
    }

    for c in 1..num_channels {
        if input_samples_per_channel[c].len() != num_samples_this_frame_per_channel {
            return Err(EncodeError::InvalidInput("All channels must have the same number of samples."));
        }
    }

    if num_samples_this_frame_per_channel % QOA_SLICE_LEN != 0 {
        return Err(EncodeError::InvalidInput("Number of samples per channel must be a multiple of QOA_SLICE_LEN."));
    }
    
    let num_slices_per_channel = num_samples_this_frame_per_channel / QOA_SLICE_LEN;
    if num_slices_per_channel > MAX_SLICES_PER_CHANNEL_PER_FRAME {
        return Err(EncodeError::InvalidInput("Number of slices per channel exceeds MAX_SLICES_PER_CHANNEL_PER_FRAME."));
    }
    if num_samples_this_frame_per_channel > u16::MAX as usize {
         return Err(EncodeError::InvalidInput("Number of samples per channel exceeds u16::MAX."));
    }


    // 2. Calculate Frame Parameters & Validate Frame Size
    // LMS state: 4 history (i16) + 4 weights (i16) = 8 i16s = 16 bytes per channel
    let lms_data_size: usize = num_channels * 16; 
    let num_total_slices: usize = num_slices_per_channel * num_channels;
    let slices_data_size: usize = num_total_slices * 8; // Each slice is 8 bytes (u64)

    let calculated_frame_size_usize: usize = QOA_HEADER_SIZE + lms_data_size + slices_data_size;
    if calculated_frame_size_usize > u16::MAX as usize {
        return Err(EncodeError::InvalidInput("Calculated frame size exceeds u16::MAX. Consider fewer channels or samples per frame."));
    }
    let frame_size_val = calculated_frame_size_usize as u16;

    // 3. Initialize Output Buffer
    let mut frame_bytes: Vec<u8> = Vec::with_capacity(frame_size_val as usize);

    // 4. Assemble and Write Frame Header (8 bytes)
    let frame_header_struct = FrameHeader {
        num_channels: num_channels as u8,
        sample_rate,
        num_samples_per_channel: num_samples_this_frame_per_channel as u16,
    };
    
    // Write header using helper or manually
    frame_bytes.push(frame_header_struct.num_channels);
    let sr_bytes = frame_header_struct.sample_rate.to_be_bytes();
    frame_bytes.extend_from_slice(&sr_bytes[1..4]); // u24 for sample_rate
    frame_bytes.extend_from_slice(&frame_header_struct.num_samples_per_channel.to_be_bytes());
    frame_bytes.extend_from_slice(&frame_size_val.to_be_bytes());


    // 5. Write LMS States (num_channels * 16 bytes)
    for c in 0..num_channels {
        for i in 0..QOA_LMS_LEN {
            frame_bytes.extend_from_slice(&(lms_states[c].history[i] as i16).to_be_bytes());
        }
        for i in 0..QOA_LMS_LEN {
            frame_bytes.extend_from_slice(&(lms_states[c].weights[i] as i16).to_be_bytes());
        }
    }

    // 6. Encode and Collect Slice Data
    // let mut all_encoded_slices_data: Vec<u8> = Vec::with_capacity(slices_data_size); // Already accounted for in frame_bytes capacity
    for slice_idx in 0..num_slices_per_channel {
        for c in 0..num_channels {
            let start = slice_idx * QOA_SLICE_LEN;
            let end = start + QOA_SLICE_LEN;
            let current_slice_samples = &input_samples_per_channel[c][start..end];
            
            let (encoded_slice_u64, _reconstructed_samples) = encode_slice(current_slice_samples, &mut lms_states[c])?;
            frame_bytes.extend_from_slice(&encoded_slice_u64.to_be_bytes());
        }
    }
    // frame_bytes.extend_from_slice(&all_encoded_slices_data); // Appended directly

    // 7. Return
    Ok((frame_header_struct, frame_bytes))
}


impl Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DecodeError::NotQoaFile => write!(f, "File is not a qoa file"),
            DecodeError::NoSamples => write!(f, "File has no samples"),
            DecodeError::InvalidFrameHeader => write!(f, "File has invalid frame header"),
            DecodeError::IncompatibleFrame => write!(f, "Incompatible frame header"),
            DecodeError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl From<io::Error> for DecodeError {
    fn from(inner: io::Error) -> Self {
        DecodeError::IoError(inner)
    }
}

#[cfg(feature = "rodio")]
pub struct QoaRodioSource<R: io::Read> {
    decoder: QoaDecoder<R>,
}

#[cfg(feature = "rodio")]
mod rodio_integration {
    use super::*;

    impl<R: io::Read> QoaRodioSource<R> {
        /// Wrap a decoder as a Rodio Source for playback.
        pub fn new(decoder: QoaDecoder<R>) -> QoaRodioSource<R> {
            Self { decoder }
        }
    }

    impl<R: io::Read> Iterator for QoaRodioSource<R> {
        type Item = i16;

        /// Return samples of i16 for Rodio.
        ///
        /// Errors stop iteration. Also ensure that `channels` and `sample_rate`
        /// reflect the next sample to be returned.
        fn next(&mut self) -> Option<Self::Item> {
            loop {
                return match self.decoder.next() {
                    Some(Ok(QoaItem::Sample(s))) => {
                        if self.decoder.next_pending_sample_idx
                            >= self.decoder.pending_samples.len()
                            && self.decoder.current_frame.num_samples_per_channel_remaining == 0
                        {
                            // This frame is done. We need to process the next frame header now so
                            // the current channels and sample rate will
                            // be returned correctly to Rodio.
                            match self.decoder.next() {
                                Some(Ok(QoaItem::FrameHeader(_))) => (),
                                Some(Ok(QoaItem::Sample(_))) => unreachable!(),
                                Some(Err(_)) => return None,
                                None => (), // We will return None again on the next call.
                            }
                        }
                        Some(s)
                    }
                    Some(Ok(QoaItem::FrameHeader(_))) => continue,
                    Some(Err(_)) => None,
                    None => None,
                };
            }
        }
    }

    impl<R: io::Read> rodio::Source for QoaRodioSource<R> {
        fn current_frame_len(&self) -> Option<usize> {
            if matches!(self.decoder.mode, ProcessingMode::Streaming) {
                let num_samples = self.decoder.current_frame.num_samples_per_channel_remaining
                    as usize
                    * self.decoder.current_frame.header.num_channels as usize;
                Some(num_samples)
            } else {
                None
            }
        }

        fn channels(&self) -> u16 {
            self.decoder.current_frame.header.num_channels.into()
        }

        fn sample_rate(&self) -> u32 {
            self.decoder.current_frame.header.sample_rate
        }

        fn total_duration(&self) -> Option<Duration> {
            self.decoder.total_duration()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    static QOA_BYTES: &[u8] = include_bytes!("../fixtures/julien_baker_sprained_ankle.qoa");

    #[test]
    fn test_iterating_through_whole_file() {
        let qoa = QoaDecoder::new(Cursor::new(QOA_BYTES)).unwrap();
        assert!(matches!(
            qoa.mode(),
            ProcessingMode::FixedSamples {
                channels: 2,
                sample_rate: 44100,
                samples: 2394122,
                ..
            }
        ));

        let mut frame_headers_seen = 0;
        let mut samples_seen = 0;

        for item in qoa {
            let item = item.expect("No io errors should happen");
            match item {
                QoaItem::Sample(_) => samples_seen += 1,
                QoaItem::FrameHeader(header) => {
                    assert_eq!(header.num_channels, 2);
                    assert_eq!(header.sample_rate, 44100);
                    frame_headers_seen += 1;
                    if frame_headers_seen < 468 {
                        assert_eq!(header.num_samples_per_channel, 5120);
                    } else {
                        assert_eq!(header.num_samples_per_channel, 3082);
                    }
                }
            }
        }
        assert_eq!(frame_headers_seen, 468);
        assert_eq!(samples_seen, 2394122 * 2);
    }

    #[test]
    fn test_decode_streaming_frames() {
        let mut qoa = QoaDecoder::new_streaming().unwrap();
        assert!(matches!(qoa.mode(), ProcessingMode::Streaming));

        // Read first frame from sample file.
        // We skip the standard file header and read the frame size present in the header. We
        // use the frame size to capture the entire frame.
        let frame_header =
            read_u64_be(Cursor::new(QOA_BYTES[QOA_HEADER_SIZE..16].to_vec())).unwrap();
        let frame_size = (frame_header & 0x00ffff) as u16;
        let first_frame_end = 8 + frame_size as usize;

        let samples = qoa
            .decode_frame(&QOA_BYTES[QOA_HEADER_SIZE..first_frame_end])
            .unwrap();
        // We know the first frame has 5120 samples per channel.
        assert_eq!(samples.len(), 5120 * 2);

        // Read second frame from sample file.
        let frame_header = read_u64_be(Cursor::new(
            QOA_BYTES[first_frame_end..first_frame_end + QOA_HEADER_SIZE].to_vec(),
        ))
        .unwrap();
        let frame_size = (frame_header & 0x00ffff) as u16;
        let second_frame_end = first_frame_end + frame_size as usize;

        let samples = qoa
            .decode_frame(&QOA_BYTES[first_frame_end..second_frame_end])
            .unwrap();
        // We know the first frame has 5120 samples per channel.
        assert_eq!(samples.len(), 5120 * 2);
    }

    #[test]
    fn test_decode_all() {
        let decoded = decode_all(Cursor::new(QOA_BYTES)).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.num_channels, 2);
        assert_eq!(decoded.samples.len(), 2394122 * 2);
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_quantization_logic() {
        // find_closest_dequantized_value_in_row tests
        let (idx, val) = find_closest_dequantized_value_in_row(0, &QOA_DEQUANT_TAB[0]);
        assert_eq!((idx, val), (1, -1), "Residual 0, SF 0");

        let (idx, val) = find_closest_dequantized_value_in_row(6, &QOA_DEQUANT_TAB[0]);
        assert_eq!((idx, val), (4, 5), "Residual 6, SF 0");

        // find_best_scale_factor_quantized_and_dequantized tests
        let (sf, q_idx, dequant_val) = find_best_scale_factor_quantized_and_dequantized(0);
        assert_eq!((sf, q_idx, dequant_val), (0, 1, -1), "Residual 0");
        
        let (sf, q_idx, dequant_val) = find_best_scale_factor_quantized_and_dequantized(8);
        assert_eq!((sf, q_idx, dequant_val), (0, 6, 7), "Residual 8");

        let (sf, q_idx, dequant_val) = find_best_scale_factor_quantized_and_dequantized(50);
        assert_eq!((sf, q_idx, dequant_val), (0, 6, 49), "Residual 50");
    }

    #[test]
    fn test_encode_decode_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let num_channels: usize = 2;
        let sample_rate: u32 = 44100;
        let samples_per_frame_per_channel: usize = QOA_SLICE_LEN * 10; // 200 samples
        let num_frames: usize = 3;
        let total_samples_per_channel: usize = samples_per_frame_per_channel * num_frames; // 600 samples

        // Generate Original Audio Data: original_audio_data[frame_idx][channel_idx][sample_idx_in_frame]
        let mut original_audio_data: Vec<Vec<Vec<i16>>> =
            vec![vec![vec![0i16; samples_per_frame_per_channel]; num_channels]; num_frames];
        
        let mut global_sample_idx_per_channel = vec![0usize; num_channels];

        for frame_idx in 0..num_frames {
            for chan_idx in 0..num_channels {
                for sample_idx_in_frame in 0..samples_per_frame_per_channel {
                    let s = global_sample_idx_per_channel[chan_idx];
                    original_audio_data[frame_idx][chan_idx][sample_idx_in_frame] =
                        (((s * (chan_idx + 1)) % 2000) - 1000) as i16;
                    global_sample_idx_per_channel[chan_idx] += 1;
                }
            }
        }
        
        // Encode
        let mut writer = Cursor::new(Vec::new());
        let mut encoder = QoaEncoder::new(
            &mut writer,
            num_channels as u8,
            sample_rate,
            total_samples_per_channel as u32,
        )?;

        for frame_idx in 0..num_frames {
            let current_frame_samples: Vec<Vec<i16>> = original_audio_data[frame_idx].clone();
            encoder.write_frame(&current_frame_samples)?;
        }
        encoder.finish()?;
        let encoded_bytes = writer.into_inner();
        assert!(!encoded_bytes.is_empty(), "Encoded bytes should not be empty");

        // Decode
        let reader = Cursor::new(encoded_bytes); // reader needs to be mutable for decode_all
        let decoded_qoa = decode_all(reader)?;

        assert_eq!(decoded_qoa.num_channels, num_channels as u8, "Channel count mismatch");
        assert_eq!(decoded_qoa.sample_rate, sample_rate, "Sample rate mismatch");
        assert_eq!(
            decoded_qoa.samples.len(),
            total_samples_per_channel * num_channels,
            "Total decoded sample count mismatch"
        );

        // Compare Samples
        let error_threshold = 250; // Generous threshold
        let mut decoded_sample_idx = 0;
        for frame_idx in 0..num_frames {
            for sample_idx_in_frame in 0..samples_per_frame_per_channel {
                for chan_idx in 0..num_channels {
                    let orig_s = original_audio_data[frame_idx][chan_idx][sample_idx_in_frame];
                    let deco_s = decoded_qoa.samples[decoded_sample_idx];
                    decoded_sample_idx += 1;

                    let diff = (orig_s as i32 - deco_s as i32).abs();
                    assert!(
                        diff < error_threshold,
                        "Large difference at frame {}, chan {}, sample_in_frame {}: orig = {}, deco = {}, diff = {}",
                        frame_idx, chan_idx, sample_idx_in_frame, orig_s, deco_s, diff
                    );
                }
            }
        }
        Ok(())
    }
}
