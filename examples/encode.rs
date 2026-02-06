use qoaudio::{encode_all, QoaDesc};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

#[cfg(feature = "hound")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: encode <input.wav> <output.qoa>");
        eprintln!();
        eprintln!("Encodes a WAV file to QOA format.");
        eprintln!("The input file must be a 16-bit PCM WAV file.");
        return Ok(());
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Validate input file exists
    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file '{}' does not exist", input_path);
        return Ok(());
    }

    println!("Loading WAV file: {}", input_path);

    // Load WAV file
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();

    // Validate WAV format
    if spec.bits_per_sample != 16 {
        eprintln!(
            "Error: Only 16-bit PCM WAV files are supported. Found {} bits per sample",
            spec.bits_per_sample
        );
        return Ok(());
    }

    if spec.channels == 0 || spec.channels > 8 {
        eprintln!(
            "Error: Unsupported number of channels: {}. Must be between 1 and 8",
            spec.channels
        );
        return Ok(());
    }

    if spec.sample_rate == 0 {
        eprintln!("Error: Invalid sample rate: {}", spec.sample_rate);
        return Ok(());
    }

    // Read all samples
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap_or(0)).collect();

    let samples_per_channel = samples.len() / spec.channels as usize;
    let duration = Duration::from_secs_f64(samples_per_channel as f64 / spec.sample_rate as f64);

    println!("Audio info:");
    println!("  Channels: {}", spec.channels);
    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Total samples: {}", samples.len());
    println!("  Samples per channel: {}", samples_per_channel);
    println!("  Duration: {:.2} seconds", duration.as_secs_f64());
    println!(
        "  Original size: {} bytes",
        std::fs::metadata(input_path)?.len()
    );

    // Create QOA description
    let desc = QoaDesc {
        channels: spec.channels as u8,
        sample_rate: spec.sample_rate,
        samples: samples_per_channel as u32,
    };

    println!("\nEncoding to QOA format...");

    // Encode to QOA
    let encoded_data = encode_all(&samples, &desc)?;

    println!("Encoding complete!");
    println!("  Compressed size: {} bytes", encoded_data.len());
    println!(
        "  Compression ratio: {:.2}x",
        samples.len() as f64 * 2.0 / encoded_data.len() as f64
    );

    // Write to output file
    let mut output_file = File::create(output_path)?;
    output_file.write_all(&encoded_data)?;

    println!("  Saved to: {}", output_path);

    Ok(())
}

// Stub implementation when hound feature is not enabled
#[cfg(not(feature = "hound"))]
fn main() {
    eprintln!("Error: This example requires the 'hound' feature to be enabled.");
    eprintln!("Run with: cargo run --example encode --features hound -- input.wav output.qoa");
}
