use qoaudio::{encode_all, QoaDesc};
use std::fs::{File, write};
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 6 {
        eprintln!("Usage: encode <input_raw_pcm> <output_qoa> <channels> <sample_rate> <samples>");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  input_raw_pcm  - Path to raw PCM audio file (16-bit signed little-endian)");
        eprintln!("  output_qoa     - Path to output QOA file");
        eprintln!("  channels       - Number of channels (1 for mono, 2 for stereo, etc.)");
        eprintln!("  sample_rate    - Sample rate in Hz (e.g., 44100)");
        eprintln!("  samples        - Number of samples per channel");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  encode input.raw output.qoa 2 44100 88200");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let channels: u8 = args[3].parse().expect("channels must be a number between 1 and 8");
    let sample_rate: u32 = args[4].parse().expect("sample_rate must be a positive number");
    let samples_per_channel: u32 = args[5].parse().expect("samples must be a positive number");

    // Validate parameters
    if channels == 0 || channels > 8 {
        eprintln!("Error: channels must be between 1 and 8");
        std::process::exit(1);
    }
    if sample_rate == 0 {
        eprintln!("Error: sample_rate must be greater than 0");
        std::process::exit(1);
    }
    if samples_per_channel == 0 {
        eprintln!("Error: samples must be greater than 0");
        std::process::exit(1);
    }

    println!("Reading raw PCM data from: {}", input_path);

    // Read input file
    let mut input_file = File::open(input_path).expect("failed to open input file");
    let mut raw_data = Vec::new();
    input_file.read_to_end(&mut raw_data).expect("failed to read input file");

    // Convert bytes to i16 samples (assuming little-endian)
    let expected_bytes = samples_per_channel as usize * channels as usize * 2;
    if raw_data.len() != expected_bytes {
        eprintln!("Warning: File size ({} bytes) doesn't match expected size ({} bytes)",
                  raw_data.len(), expected_bytes);
        eprintln!("         Expected {} samples per channel × {} channels × 2 bytes per sample",
                  samples_per_channel, channels);
    }

    let sample_count = raw_data.len() / 2;
    let mut samples = Vec::with_capacity(sample_count);

    for chunk in raw_data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample);
    }

    println!("Encoding {} samples ({} per channel, {} channels) at {} Hz",
             samples.len(), samples_per_channel, channels, sample_rate);

    // Create QOA descriptor
    let desc = QoaDesc {
        channels,
        sample_rate,
        samples: samples_per_channel,
    };

    // Encode to QOA
    let encoded = match encode_all(&samples, &desc) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Encoding failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("Encoded {} bytes (compression ratio: {:.2}:1)",
             encoded.len(),
             raw_data.len() as f64 / encoded.len() as f64);

    // Write output file
    write(output_path, &encoded).expect("failed to write output file");

    println!("Successfully wrote QOA file to: {}", output_path);
}
