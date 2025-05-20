use std::env;
use std::fs::File;
use std::io::{self, Read, BufWriter};
use qoaudio::{QoaEncoder, QOA_SLICE_LEN};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <output_file.qoa> <num_channels> <sample_rate>", args[0]);
        eprintln!("Example: ffmpeg -i input.wav -f s16le -acodec pcm_s16le -ar 44100 -ac 2 - | {} output.qoa 2 44100", args[0]);
        return Err("Invalid arguments".into());
    }

    let output_path = &args[1];
    let num_channels: u8 = args[2].parse()?;
    let sample_rate: u32 = args[3].parse()?;

    let mut stdin = io::stdin();
    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    // QOA files store the total number of samples per channel in the header.
    // Since we are streaming, we don't know this upfront. We'll use 0,
    // which indicates streaming mode or unknown total samples for the decoder.
    let total_samples_per_channel_in_file: u32 = 0;

    let mut encoder = QoaEncoder::new(
        &mut writer,
        num_channels,
        sample_rate,
        total_samples_per_channel_in_file,
    )?;

    // Buffer to hold enough samples for a full frame for all channels.
    // Max samples per frame per channel is MAX_SLICES_PER_CHANNEL_PER_FRAME * QOA_SLICE_LEN.
    // However, we'll process data in smaller, more manageable chunks.
    // Let's define a chunk size, e.g., 10 slices worth of data per channel.
    const SLICES_PER_CHUNK: usize = 10;
    let samples_per_channel_per_chunk: usize = QOA_SLICE_LEN * SLICES_PER_CHUNK;
    // Calculate total samples and bytes for one chunk
    let chunk_total_samples: usize = samples_per_channel_per_chunk * num_channels as usize;
    let chunk_total_bytes: usize = chunk_total_samples * 2; // Each i16 sample is 2 bytes

    // Buffer to read raw bytes from stdin
    let mut byte_read_buffer: Vec<u8> = vec![0; chunk_total_bytes];
    // Buffer to hold interleaved i16 samples for the current chunk, pre-allocated
    let mut frame_buffer_interleaved: Vec<i16> = vec![0; chunk_total_samples];

    loop {
        let mut total_bytes_read_this_chunk = 0;
        // Read from stdin into the byte_read_buffer until it's full or EOF
        while total_bytes_read_this_chunk < chunk_total_bytes {
            match stdin.read(&mut byte_read_buffer[total_bytes_read_this_chunk..]) {
                Ok(0) => break, // EOF
                Ok(n) => total_bytes_read_this_chunk += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue, // Retry on interrupt
                Err(e) => return Err(e.into()), // Propagate other IO errors
            }
        }

        if total_bytes_read_this_chunk == 0 {
            break; // No more data from stdin
        }

        // Ensure we only process an even number of bytes to form complete i16 samples
        let bytes_to_convert = if total_bytes_read_this_chunk % 2 != 0 {
            eprintln!(
                "Warning: Read an odd number of bytes ({}). Discarding the last byte.",
                total_bytes_read_this_chunk
            );
            total_bytes_read_this_chunk - 1
        } else {
            total_bytes_read_this_chunk
        };

        let actual_samples_read_interleaved = bytes_to_convert / 2;

        if actual_samples_read_interleaved == 0 {
            if total_bytes_read_this_chunk > 0 {
                 eprintln!("Warning: Not enough bytes to form a single i16 sample after processing. Read {} byte(s).", total_bytes_read_this_chunk);
            }
            break; // No complete samples to process
        }

        // Convert bytes from byte_read_buffer to i16 samples in frame_buffer_interleaved
        for i in 0..actual_samples_read_interleaved {
            let byte_index = i * 2;
            // Assuming little-endian byte order for s16le from ffmpeg
            frame_buffer_interleaved[i] = i16::from_le_bytes([
                byte_read_buffer[byte_index],
                byte_read_buffer[byte_index + 1],
            ]);
        }

        // De-interleave the samples
        // Calculate precise capacity for this chunk's de-interleaved samples
        let max_samples_per_channel_this_chunk = (actual_samples_read_interleaved + num_channels as usize -1) / num_channels as usize;
        let mut samples_for_frame_per_channel: Vec<Vec<i16>> =
            vec![Vec::with_capacity(max_samples_per_channel_this_chunk); num_channels as usize];

        for i in 0..actual_samples_read_interleaved {
            samples_for_frame_per_channel[i % num_channels as usize]
                .push(frame_buffer_interleaved[i]);
        }

        // Pad with zeros if the last chunk is not full for all channels to meet QOA_SLICE_LEN multiple requirement
        let mut final_samples_per_channel_in_chunk = samples_for_frame_per_channel[0].len();
        if final_samples_per_channel_in_chunk == 0 { // Should not happen if actual_samples_read_interleaved > 0
            break;
        }

        if final_samples_per_channel_in_chunk % QOA_SLICE_LEN != 0 {
            let required_padding = QOA_SLICE_LEN - (final_samples_per_channel_in_chunk % QOA_SLICE_LEN);
            for chan_samples in samples_for_frame_per_channel.iter_mut() {
                // Ensure all channels have the same length before padding
                chan_samples.resize(final_samples_per_channel_in_chunk, 0);
                for _ in 0..required_padding {
                    chan_samples.push(0);
                }
            }
            final_samples_per_channel_in_chunk += required_padding;
        }
        
        // Ensure all channel vecs have the same final length after potential padding
        for chan_idx in 0..num_channels as usize {
            if samples_for_frame_per_channel[chan_idx].len() != final_samples_per_channel_in_chunk {
                 // This case implies uneven sample counts from stdin before EOF, which is tricky.
                 // For simplicity, we'll truncate or pad all to the length of the first channel's processed chunk.
                 // A more robust solution might involve more complex buffering or error handling.
                samples_for_frame_per_channel[chan_idx].resize(final_samples_per_channel_in_chunk, 0);
            }
        }


        if final_samples_per_channel_in_chunk > 0 {
             // If, after processing, a channel has zero samples for a frame, but others don't,
             // this indicates an issue or an edge case not perfectly handled by the padding above,
             // especially if EOF was hit mid-chunk for some channels but not others.
             // We must ensure all vectors passed to write_frame are non-empty if we proceed.
            let mut valid_frame = true;
            for chan_samples in &samples_for_frame_per_channel {
                if chan_samples.is_empty() && final_samples_per_channel_in_chunk > 0 {
                    valid_frame = false;
                    break;
                }
            }
            if valid_frame && final_samples_per_channel_in_chunk > 0 {
                 encoder.write_frame(&samples_for_frame_per_channel)?;
            } else if !valid_frame && total_bytes_read_this_chunk > 0 { // Changed bytes_read_for_chunk
                // This implies an incomplete frame at EOF that couldn't be padded correctly for all channels.
                // Depending on strictness, one might error or try to encode what's possible.
                // For now, we'll break, as QoaEncoder expects consistent sample counts.
                eprintln!("Warning: Incomplete frame at EOF, could not form valid QOA frame from remaining samples.");
                break;
            }
        }


        if total_bytes_read_this_chunk < chunk_total_bytes { // Changed condition variables
            break; // EOF reached and processed the last partial chunk
        }
    }

    encoder.finish()?;
    Ok(())
}

