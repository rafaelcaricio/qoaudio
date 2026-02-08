#![forbid(unsafe_code)]
//! # QOA - Quite OK Audio Format
//!
//! A library for encoding and decoding qoa files.
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
pub const QOA_SLICES_PER_FRAME: usize = 256;
pub const QOA_FRAME_LEN: usize = QOA_SLICES_PER_FRAME * QOA_SLICE_LEN;
pub const QOA_MAX_CHANNELS: usize = 8;

/// QOA encoder quantization table
const QOA_QUANT_TAB: [i32; 17] = [7, 7, 7, 5, 5, 3, 3, 1, 0, 0, 2, 2, 4, 4, 6, 6, 6];

/// QOA encoder reciprocal table
const QOA_RECIPROCAL_TAB: [i32; 16] = [
    65536, 9363, 3121, 1457, 781, 475, 311, 216, 156, 117, 90, 71, 57, 47, 39, 32,
];

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
    pending_samples_end: usize,
    next_pending_sample_idx: usize,
    returned_first_frame_header: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct QoaLms {
    history: [i32; QOA_LMS_LEN],
    weights: [i32; QOA_LMS_LEN],
}

/// Encoder for QOA files.
///
/// Encodes 16-bit PCM audio into QOA format using lossy compression.
/// The encoder uses LMS prediction and quantization to achieve compression.
///
/// Supports both one-shot encoding via [`encode`](QoaEncoder::encode) and
/// frame-at-a-time streaming via [`write_header`](QoaEncoder::write_header)
/// followed by repeated [`encode_frame`](QoaEncoder::encode_frame) calls.
#[derive(Debug)]
pub struct QoaEncoder {
    channels: u8,
    sample_rate: u32,
    samples: u32,
    lms: [QoaLms; QOA_MAX_CHANNELS],
    prev_scalefactor: [usize; QOA_MAX_CHANNELS],
}

/// Description of QOA file properties for encoding
#[derive(Debug, Clone)]
pub struct QoaDesc {
    /// Number of audio channels (1-8)
    pub channels: u8,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Total number of samples per channel
    pub samples: u32,
}

/// Errors that can occur during QOA encoding
#[derive(Debug)]
pub enum EncodeError {
    /// Invalid number of channels (must be 1-8)
    InvalidChannels,
    /// Invalid sample rate (must be > 0)
    InvalidSampleRate,
    /// Invalid number of samples (must be > 0)
    InvalidSamples,
    /// I/O error
    IoError(io::Error),
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
            pending_samples_end: 0,
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

        if !num_slices.is_multiple_of(num_channels as usize) {
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
        assert!(self.next_pending_sample_idx >= self.pending_samples_end);
        let channels = self.current_frame.header.num_channels as usize;
        let full_slices_num_samples = QOA_SLICE_LEN * channels;
        if self.pending_samples.len() != full_slices_num_samples {
            self.pending_samples = vec![0_i16; full_slices_num_samples].into_boxed_slice();
        }
        self.next_pending_sample_idx = 0;

        for channel_idx in 0..channels {
            let mut slice = read_u64_be(&mut self.reader)?;

            let scale_factor = ((slice >> 60) & 0xf) as usize;
            let mut lms = self.lms[channel_idx];
            let mut data_idx = channel_idx;
            for _ in 0..QOA_SLICE_LEN {
                let prediction = lms.predict();
                let quantized = ((slice >> 57) & 0x7) as usize;
                let dequantized = QOA_DEQUANT_TAB[scale_factor][quantized];
                let reconstructed = (prediction + dequantized).clamp(-32768, 32767) as i16;

                self.pending_samples[data_idx] = reconstructed;
                data_idx += channels;
                slice <<= 3;

                lms.update(reconstructed, dequantized);
            }
            self.lms[channel_idx] = lms;
        }
        let num_samples_per_channel = self.current_frame.num_samples_per_channel_remaining;
        if (num_samples_per_channel as usize) < QOA_SLICE_LEN {
            let total_num_samples = num_samples_per_channel as usize * channels;
            self.pending_samples_end = total_num_samples;
            self.current_frame.num_samples_per_channel_remaining -= num_samples_per_channel;
        } else {
            self.pending_samples_end = full_slices_num_samples;
            self.current_frame.num_samples_per_channel_remaining -= QOA_SLICE_LEN as u16;
        }
        Ok(())
    }
}

impl QoaEncoder {
    /// Create a new QOA encoder with the specified parameters.
    pub fn new(desc: &QoaDesc) -> Result<Self, EncodeError> {
        if desc.channels == 0 || desc.channels > QOA_MAX_CHANNELS as u8 {
            return Err(EncodeError::InvalidChannels);
        }
        if desc.sample_rate == 0 {
            return Err(EncodeError::InvalidSampleRate);
        }
        if desc.samples == 0 {
            return Err(EncodeError::InvalidSamples);
        }

        let mut lms = [QoaLms::default(); QOA_MAX_CHANNELS];
        for c in 0..desc.channels as usize {
            lms[c] = QoaLms {
                history: [0; QOA_LMS_LEN],
                weights: [0, 0, -(1 << 13), 1 << 14],
            };
        }

        Ok(Self {
            channels: desc.channels,
            sample_rate: desc.sample_rate,
            samples: desc.samples,
            lms,
            prev_scalefactor: [0; QOA_MAX_CHANNELS],
        })
    }

    /// Encode all PCM audio data to QOA format in one shot.
    ///
    /// The length of `sample_data` must equal `samples * channels` as specified
    /// in the [`QoaDesc`] passed to [`QoaEncoder::new`].
    pub fn encode(&mut self, sample_data: &[i16]) -> Result<Vec<u8>, EncodeError> {
        if sample_data.len() != self.samples as usize * self.channels as usize {
            return Err(EncodeError::InvalidSamples);
        }

        let channels = self.channels as usize;
        let total = self.samples as usize;
        let num_slices = total.div_ceil(QOA_SLICE_LEN);
        let num_frames = total.div_ceil(QOA_FRAME_LEN);
        let encoded_size = QOA_HEADER_SIZE
            + num_frames * (8 + QOA_LMS_LEN * 4 * channels)
            + num_slices * 8 * channels;
        let mut buf = vec![0u8; encoded_size];
        let mut p = 0;

        buf[p..p + 4].copy_from_slice(&QOA_MAGIC.to_be_bytes());
        p += 4;
        buf[p..p + 4].copy_from_slice(&self.samples.to_be_bytes());
        p += 4;

        let mut sample_index = 0usize;
        while sample_index < total {
            let frame_len = (total - sample_index).min(QOA_FRAME_LEN);
            let start = sample_index * channels;
            let end = (sample_index + frame_len) * channels;
            p += self.encode_frame_to_buf(&sample_data[start..end], &mut buf[p..]);
            sample_index += frame_len;
        }

        buf.truncate(p);
        Ok(buf)
    }

    /// Write the 8-byte QOA file header.
    ///
    /// Call this once before any [`encode_frame`](QoaEncoder::encode_frame)
    /// calls when using the streaming API.
    pub fn write_header<W: io::Write>(&self, writer: &mut W) -> Result<(), EncodeError> {
        writer.write_all(&QOA_MAGIC.to_be_bytes())?;
        writer.write_all(&self.samples.to_be_bytes())?;
        Ok(())
    }

    /// Encode one frame of interleaved PCM samples and write it to `writer`.
    ///
    /// `sample_data` must contain interleaved samples for all channels, with at
    /// most `QOA_FRAME_LEN * channels` samples. Its length must be a multiple
    /// of `channels`. LMS state is preserved across calls, so frames can be
    /// encoded incrementally.
    ///
    /// Returns the number of samples per channel encoded in this frame.
    pub fn encode_frame<W: io::Write>(
        &mut self,
        sample_data: &[i16],
        writer: &mut W,
    ) -> Result<usize, EncodeError> {
        let channels = self.channels as usize;
        if sample_data.is_empty() || !sample_data.len().is_multiple_of(channels) {
            return Err(EncodeError::InvalidSamples);
        }
        let frame_len = sample_data.len() / channels;
        if frame_len > QOA_FRAME_LEN {
            return Err(EncodeError::InvalidSamples);
        }

        let slices = frame_len.div_ceil(QOA_SLICE_LEN);
        let frame_size = qoa_frame_size(channels, slices) as usize;

        let mut frame_buf = vec![0u8; frame_size];
        let written = self.encode_frame_to_buf(sample_data, &mut frame_buf);
        writer.write_all(&frame_buf[..written])?;
        Ok(frame_len)
    }

    fn encode_frame_to_buf(&mut self, sample_data: &[i16], buf: &mut [u8]) -> usize {
        let channels = self.channels as usize;
        let frame_len = sample_data.len() / channels;
        let slices = frame_len.div_ceil(QOA_SLICE_LEN);
        let frame_size = qoa_frame_size(channels, slices);
        let mut p = 0;

        let header = ((self.channels as u64) << 56)
            | ((self.sample_rate as u64) << 32)
            | ((frame_len as u64) << 16)
            | (frame_size as u64);
        buf[p..p + 8].copy_from_slice(&header.to_be_bytes());
        p += 8;

        for c in 0..channels {
            let mut history = 0u64;
            let mut weights = 0u64;
            for i in 0..QOA_LMS_LEN {
                history = (history << 16) | (self.lms[c].history[i] as u16 as u64);
                weights = (weights << 16) | (self.lms[c].weights[i] as u16 as u64);
            }
            buf[p..p + 8].copy_from_slice(&history.to_be_bytes());
            p += 8;
            buf[p..p + 8].copy_from_slice(&weights.to_be_bytes());
            p += 8;
        }

        for sample_index in (0..frame_len).step_by(QOA_SLICE_LEN) {
            for c in 0..channels {
                let slice_len = (frame_len - sample_index).min(QOA_SLICE_LEN);
                let slice_start = sample_index * channels + c;
                let slice_end = (sample_index + slice_len) * channels + c;

                let (best_slice, best_scalefactor, best_lms) = self.encode_slice(
                    sample_data,
                    slice_start,
                    slice_end,
                    channels,
                );

                self.prev_scalefactor[c] = best_scalefactor;
                self.lms[c] = best_lms;

                let mut slice_data = best_slice;
                if slice_len < QOA_SLICE_LEN {
                    slice_data <<= (QOA_SLICE_LEN - slice_len) * 3;
                }
                buf[p..p + 8].copy_from_slice(&slice_data.to_be_bytes());
                p += 8;
            }
        }
        p
    }

    fn encode_slice(
        &self,
        sample_data: &[i16],
        slice_start: usize,
        slice_end: usize,
        channels: usize,
    ) -> (u64, usize, QoaLms) {
        let mut samples = [0i32; QOA_SLICE_LEN];
        let mut slice_len = 0;
        for si in (slice_start..slice_end).step_by(channels) {
            samples[slice_len] = sample_data[si] as i32;
            slice_len += 1;
        }
        let slice_len = slice_len.min(QOA_SLICE_LEN);

        let channel_lms = &self.lms[slice_start % channels];
        let mut best_rank = u64::MAX;
        let mut best_slice = 0u64;
        let mut best_scalefactor = 0;
        let mut best_lms = QoaLms::default();

        let (first_predicted, first_penalty_sq) = channel_lms.predict_and_penalty_sq();
        let first_sample = samples[0];
        let first_residual = first_sample - first_predicted;

        let mut first_sample_results = [(0u64, 0i32, 0usize, u64::MAX); 16];
        let mut sf_order = [0u8; 16];
        let mut sf_count = 0usize;
        for sf in 0..16usize {
            let sf_quant_dequant = &QOA_QUANT_DEQUANT_TAB[sf];
            let scaled = qoa_div(first_residual, sf);
            let clamped = scaled.clamp(-8, 8);
            let packed = sf_quant_dequant[(clamped + 8) as usize];
            let quantized = (packed >> 32) as usize;
            let dequantized = packed as i32;
            let reconstructed = (first_predicted + dequantized).clamp(-32768, 32767);
            let error = (first_sample - reconstructed) as i64;
            let rank = (error * error) as u64 + first_penalty_sq;
            first_sample_results[sf] = (packed, reconstructed as i32, quantized, rank);

            let mut pos = sf_count;
            while pos > 0 && first_sample_results[sf_order[pos - 1] as usize].3 > rank {
                sf_order[pos] = sf_order[pos - 1];
                pos -= 1;
            }
            sf_order[pos] = sf as u8;
            sf_count += 1;
        }

        for &scalefactor_u8 in &sf_order {
            let scalefactor = scalefactor_u8 as usize;
            let sf_quant_dequant = &QOA_QUANT_DEQUANT_TAB[scalefactor];

            let mut lms = *channel_lms;
            let mut slice = scalefactor as u64;

            let (packed, reconstructed, quantized, first_rank) = first_sample_results[scalefactor];
            let dequantized = packed as i32;
            let mut current_rank = first_rank;

            if current_rank > best_rank {
                break;
            }

            lms.update(reconstructed as i16, dequantized);
            slice = (slice << 3) | quantized as u64;

            let mut valid = true;
            #[allow(clippy::needless_range_loop)]
            for i in 1..slice_len {
                let sample = samples[i];
                let (predicted, penalty_sq) = lms.predict_and_penalty_sq();
                let residual = sample - predicted;
                let scaled = qoa_div(residual, scalefactor);
                let clamped = scaled.clamp(-8, 8);
                let packed = sf_quant_dequant[(clamped + 8) as usize];
                let quantized = (packed >> 32) as usize;
                let dequantized = packed as i32;
                let reconstructed = (predicted + dequantized).clamp(-32768, 32767);

                let error = (sample - reconstructed) as i64;
                current_rank += (error * error) as u64 + penalty_sq;

                if current_rank > best_rank {
                    valid = false;
                    break;
                }

                lms.update(reconstructed as i16, dequantized);
                slice = (slice << 3) | quantized as u64;
            }

            if valid && current_rank < best_rank {
                best_rank = current_rank;
                best_slice = slice;
                best_scalefactor = scalefactor;
                best_lms = lms;
            }
        }

        (best_slice, best_scalefactor, best_lms)
    }
}

// QoaLms methods are already implemented above

/// Calculate frame size for QOA encoding
const fn qoa_frame_size(channels: usize, slices: usize) -> u16 {
    (8 + QOA_LMS_LEN * 4 * channels + 8 * slices * channels) as u16
}

/// QOA division with rounding away from zero.
///
/// Uses wrapping 32-bit arithmetic matching the C reference, which allows the
/// compiler to emit a single fused multiply-add. For large residuals this can
/// wrap, but the result only affects encoder scalefactor selection (a heuristic)
/// — the decoder always reads the exact quantized values from the bitstream.
#[inline(always)]
fn qoa_div(v: i32, scalefactor: usize) -> i32 {
    let reciprocal = QOA_RECIPROCAL_TAB[scalefactor];
    let n = v.wrapping_mul(reciprocal).wrapping_add(1 << 15) >> 16;
    n + ((v > 0) as i32 - (v < 0) as i32) - ((n > 0) as i32 - (n < 0) as i32)
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
        if self.next_pending_sample_idx < self.pending_samples_end {
            let sample = self.pending_samples[self.next_pending_sample_idx];
            self.next_pending_sample_idx += 1;
            return Some(Ok(QoaItem::Sample(sample)));
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
        debug_assert!(self.pending_samples_end > 0);
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

/// Encode PCM audio data to QOA format.
///
/// This is a convenience function that creates an encoder and encodes the provided
/// PCM audio data in one step.
pub fn encode_all(sample_data: &[i16], desc: &QoaDesc) -> Result<Vec<u8>, EncodeError> {
    let mut encoder = QoaEncoder::new(desc)?;
    encoder.encode(sample_data)
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
    fn predict(&self) -> i32 {
        let p01 = self.weights[0].wrapping_mul(self.history[0])
            .wrapping_add(self.weights[1].wrapping_mul(self.history[1]));
        let p23 = self.weights[2].wrapping_mul(self.history[2])
            .wrapping_add(self.weights[3].wrapping_mul(self.history[3]));
        p01.wrapping_add(p23) >> 13
    }

    #[inline(always)]
    fn predict_and_penalty_sq(&self) -> (i32, u64) {
        let w = self.weights;
        let h = self.history;
        let p01 = w[0].wrapping_mul(h[0]).wrapping_add(w[1].wrapping_mul(h[1]));
        let p23 = w[2].wrapping_mul(h[2]).wrapping_add(w[3].wrapping_mul(h[3]));
        let prediction = p01.wrapping_add(p23) >> 13;
        let s01 = w[0].wrapping_mul(w[0]).wrapping_add(w[1].wrapping_mul(w[1]));
        let s23 = w[2].wrapping_mul(w[2]).wrapping_add(w[3].wrapping_mul(w[3]));
        let penalty = ((s01.wrapping_add(s23) >> 18) - 0x8ff).max(0) as i64;
        (prediction, (penalty * penalty) as u64)
    }

    #[inline(always)]
    fn update(&mut self, sample: i16, residual: i32) {
        let delta = residual >> 4;
        self.weights[0] += if self.history[0] < 0 { -delta } else { delta };
        self.weights[1] += if self.history[1] < 0 { -delta } else { delta };
        self.weights[2] += if self.history[2] < 0 { -delta } else { delta };
        self.weights[3] += if self.history[3] < 0 { -delta } else { delta };
        self.history = [self.history[1], self.history[2], self.history[3], sample as i32];
    }
}

const QOA_QUANT_DEQUANT_TAB: [[u64; 17]; 16] = {
    let mut tab = [[0u64; 17]; 16];
    let mut sf = 0;
    while sf < 16 {
        let mut i = 0;
        while i < 17 {
            let quantized = QOA_QUANT_TAB[i] as u64;
            let dequantized = QOA_DEQUANT_TAB[sf][QOA_QUANT_TAB[i] as usize] as u32 as u64;
            tab[sf][i] = (quantized << 32) | dequantized;
            i += 1;
        }
        sf += 1;
    }
    tab
};

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

#[derive(Debug)]
pub enum DecodeError {
    NotQoaFile,
    NoSamples,
    InvalidFrameHeader,
    IncompatibleFrame,
    IoError(io::Error),
}

impl std::error::Error for DecodeError {}

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

impl std::error::Error for EncodeError {}

impl Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EncodeError::InvalidChannels => write!(f, "Invalid number of channels (must be 1-8)"),
            EncodeError::InvalidSampleRate => write!(f, "Invalid sample rate (must be > 0)"),
            EncodeError::InvalidSamples => write!(f, "Invalid number of samples (must be > 0)"),
            EncodeError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl From<io::Error> for EncodeError {
    fn from(inner: io::Error) -> Self {
        EncodeError::IoError(inner)
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
                            >= self.decoder.pending_samples_end
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

    #[test]
    fn test_encode_decode_simple() {
        // Create simple test data: sine wave
        let sample_rate = 44100;
        let duration = 0.1; // 100ms
        let samples_per_channel = (sample_rate as f64 * duration) as u32;
        let channels = 1;

        let mut samples = Vec::new();
        for i in 0..samples_per_channel {
            let t = i as f64 / sample_rate as f64;
            let sample = (t * 440.0 * 2.0 * std::f64::consts::PI).sin() * 16384.0;
            samples.push(sample as i16);
        }

        let desc = QoaDesc {
            channels,
            sample_rate,
            samples: samples_per_channel,
        };

        // Encode
        let encoded = encode_all(&samples, &desc).unwrap();

        // Verify we got some data
        assert!(encoded.len() > QOA_HEADER_SIZE);

        // Decode and verify
        let decoded = decode_all(Cursor::new(encoded)).unwrap();
        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.num_channels, channels);
        assert_eq!(decoded.samples.len(), samples.len());
    }

    #[test]
    fn test_encode_decode_stereo() {
        // Create stereo test data
        let sample_rate = 44100;
        let duration = 0.1; // 100ms
        let samples_per_channel = (sample_rate as f64 * duration) as u32;
        let channels = 2;

        let mut samples = Vec::new();
        for i in 0..samples_per_channel {
            let t = i as f64 / sample_rate as f64;
            // Left channel: sine wave
            let left = (t * 440.0 * 2.0 * std::f64::consts::PI).sin() * 16384.0;
            // Right channel: cosine wave
            let right = (t * 440.0 * 2.0 * std::f64::consts::PI).cos() * 16384.0;
            samples.push(left as i16);
            samples.push(right as i16);
        }

        let desc = QoaDesc {
            channels,
            sample_rate,
            samples: samples_per_channel,
        };

        // Encode
        let encoded = encode_all(&samples, &desc).unwrap();

        // Verify we got some data
        assert!(encoded.len() > QOA_HEADER_SIZE);

        // Decode and verify
        let decoded = decode_all(Cursor::new(encoded)).unwrap();
        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.num_channels, channels);
        assert_eq!(decoded.samples.len(), samples.len());
    }

    #[test]
    fn test_encoder_errors() {
        // Test invalid channels
        let desc = QoaDesc {
            channels: 0,
            sample_rate: 44100,
            samples: 1000,
        };
        let samples = vec![0i16; 1000];
        assert!(matches!(
            encode_all(&samples, &desc),
            Err(EncodeError::InvalidChannels)
        ));

        // Test invalid sample rate
        let desc = QoaDesc {
            channels: 1,
            sample_rate: 0,
            samples: 1000,
        };
        assert!(matches!(
            encode_all(&samples, &desc),
            Err(EncodeError::InvalidSampleRate)
        ));

        // Test invalid samples
        let desc = QoaDesc {
            channels: 1,
            sample_rate: 44100,
            samples: 0,
        };
        assert!(matches!(
            encode_all(&samples, &desc),
            Err(EncodeError::InvalidSamples)
        ));
    }

    #[test]
    fn test_round_trip_audio() {
        // Test that encode-decode preserves the basic structure
        let sample_rate = 44100;
        let samples_per_channel = 1000;
        let channels = 1;

        // Create test data with some variation - use a more gradual signal
        let mut samples = Vec::new();
        for i in 0..samples_per_channel {
            let sample = ((i % 200) as i16 - 100) * 100;
            samples.push(sample);
        }

        let desc = QoaDesc {
            channels,
            sample_rate,
            samples: samples_per_channel,
        };

        // Encode
        let encoded = encode_all(&samples, &desc).unwrap();

        // Decode
        let decoded = decode_all(Cursor::new(encoded)).unwrap();

        // Verify basic properties
        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.num_channels, channels);
        assert_eq!(decoded.samples.len(), samples.len());

        // Verify that the decoded audio has similar characteristics
        // (QOA is lossy, so we don't expect exact match)
        let mut max_diff = 0;
        for (original, decoded) in samples.iter().zip(decoded.samples.iter()) {
            let diff = (original - decoded).abs();
            max_diff = max_diff.max(diff);
        }

        // QOA is lossy, but the error should stay well within the i16 range
        assert!(
            max_diff < 8000,
            "Maximum difference too large: {}",
            max_diff
        );
    }

    #[test]
    fn test_full_file_round_trip() {
        // Decode the real fixture, re-encode, decode again, compare
        let original = decode_all(Cursor::new(QOA_BYTES)).unwrap();
        let desc = QoaDesc {
            channels: original.num_channels,
            sample_rate: original.sample_rate,
            samples: (original.samples.len() / original.num_channels as usize) as u32,
        };
        let encoded = encode_all(&original.samples, &desc).unwrap();
        let decoded = decode_all(Cursor::new(encoded)).unwrap();

        assert_eq!(decoded.num_channels, original.num_channels);
        assert_eq!(decoded.sample_rate, original.sample_rate);
        assert_eq!(decoded.samples.len(), original.samples.len());

        // Compute RMS error — QOA re-encoding a previously-decoded signal
        // goes through quantization twice, but should still be reasonable
        let mse: f64 = original
            .samples
            .iter()
            .zip(decoded.samples.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / original.samples.len() as f64;
        let rms = mse.sqrt();
        assert!(rms < 500.0, "RMS error too large: {:.1}", rms);
    }

    #[test]
    fn test_streaming_encode() {
        // Encode the fixture frame-by-frame via the streaming API,
        // then verify the output matches one-shot encoding
        let original = decode_all(Cursor::new(QOA_BYTES)).unwrap();
        let channels = original.num_channels as usize;
        let samples_per_channel = (original.samples.len() / channels) as u32;

        let desc = QoaDesc {
            channels: original.num_channels,
            sample_rate: original.sample_rate,
            samples: samples_per_channel,
        };

        // One-shot encode for reference
        let mut oneshot_enc = QoaEncoder::new(&desc).unwrap();
        let oneshot = oneshot_enc.encode(&original.samples).unwrap();

        // Streaming encode
        let mut streaming_enc = QoaEncoder::new(&desc).unwrap();
        let mut streamed = Vec::new();
        streaming_enc.write_header(&mut streamed).unwrap();

        let mut offset = 0usize;
        let total = samples_per_channel as usize;
        while offset < total {
            let frame_len = (total - offset).min(QOA_FRAME_LEN);
            let start = offset * channels;
            let end = (offset + frame_len) * channels;
            streaming_enc
                .encode_frame(&original.samples[start..end], &mut streamed)
                .unwrap();
            offset += frame_len;
        }

        assert_eq!(streamed, oneshot);
    }
}
