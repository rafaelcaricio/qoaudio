#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for channels + enough for one sample
    if data.len() < 4 {
        return;
    }

    let channels = (data[0] % 8) + 1; // 1-8
    let sample_rate = 44100u32;

    // Interpret remaining bytes as i16 samples
    let sample_bytes = &data[1..];
    let num_i16 = sample_bytes.len() / 2;
    let channels_usize = channels as usize;
    if num_i16 < channels_usize {
        return;
    }
    // Truncate to a multiple of channels
    let samples_per_channel = num_i16 / channels_usize;
    let total_samples = samples_per_channel * channels_usize;

    let samples: Vec<i16> = sample_bytes[..total_samples * 2]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let desc = qoaudio::QoaDesc {
        channels,
        sample_rate,
        samples: samples_per_channel as u32,
    };

    let Ok(encoded) = qoaudio::encode_all(&samples, &desc) else {
        return;
    };

    // The encoder output must always be decodable
    let decoded = qoaudio::decode_all(std::io::Cursor::new(encoded))
        .expect("encoder output must be valid QOA");

    assert_eq!(decoded.num_channels, channels);
    assert_eq!(decoded.sample_rate, sample_rate);
    assert_eq!(decoded.samples.len(), samples.len());
});
