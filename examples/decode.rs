use qoaudio::open_and_decode_all;
use std::env;
use std::path::Path;

#[cfg(feature = "hound")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: decode <input.qoa> <output.wav>");
        eprintln!();
        eprintln!("Decodes a QOA file to 16-bit PCM WAV.");
        return Ok(());
    }

    let input_path = &args[1];
    let output_path = &args[2];

    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file '{}' does not exist", input_path);
        return Ok(());
    }

    println!("Decoding QOA file: {}", input_path);

    let decoded = open_and_decode_all(input_path)?;

    let samples_per_channel = decoded.samples.len() / decoded.num_channels as usize;

    println!("Audio info:");
    println!("  Channels: {}", decoded.num_channels);
    println!("  Sample rate: {} Hz", decoded.sample_rate);
    println!("  Samples per channel: {}", samples_per_channel);
    println!(
        "  Duration: {:.2} seconds",
        samples_per_channel as f64 / decoded.sample_rate as f64
    );

    let spec = hound::WavSpec {
        channels: decoded.num_channels as u16,
        sample_rate: decoded.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &sample in &decoded.samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    println!(
        "\nDecoded to: {} ({} bytes)",
        output_path,
        std::fs::metadata(output_path)?.len()
    );

    Ok(())
}

#[cfg(not(feature = "hound"))]
fn main() {
    eprintln!("Error: This example requires the 'hound' feature.");
    eprintln!("Run with: cargo run --example decode --features hound -- input.qoa output.wav");
}
