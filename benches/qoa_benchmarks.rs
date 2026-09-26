use criterion::{black_box, criterion_group, criterion_main, Criterion};

static QOA_BYTES: &[u8] = include_bytes!("../fixtures/julien_baker_sprained_ankle.qoa");

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("decode_sprained_ankle", |b| {
        b.iter(|| {
            let Ok(decoder) = qoaudio::QoaDecoder::new(std::io::Cursor::new(QOA_BYTES)) else {
                panic!("QoaDecoder::new failed");
            };
            let count = decoder.take_while(|i| matches!(i, Ok(_))).count();
            assert_eq!(count, 2394122 * 2 + 468);
            black_box(count);
        })
    });

    c.bench_function("decode_into_sprained_ankle", |b| {
        b.iter(|| {
            let Ok(mut decoder) = qoaudio::QoaDecoder::new(std::io::Cursor::new(QOA_BYTES)) else {
                panic!("QoaDecoder::new failed");
            };
            let mut buf = [0_i16; 1024];
            let mut count = 0;
            loop {
                let written = decoder.decode_into(&mut buf).unwrap();
                count += written;
                black_box(&buf[..written]);
                if written == 0 && decoder.next_frame().unwrap().is_none() {
                    break;
                }
            }
            assert_eq!(count, 2394122 * 2);
        })
    });

    // Decode once to get PCM samples for encoding benchmark
    let decoded = qoaudio::decode_all(std::io::Cursor::new(QOA_BYTES)).unwrap();
    let desc = qoaudio::QoaDesc {
        channels: decoded.num_channels,
        sample_rate: decoded.sample_rate,
        samples: (decoded.samples.len() / decoded.num_channels as usize) as u32,
    };

    c.bench_function("encode_sprained_ankle", |b| {
        b.iter(|| {
            let encoded = qoaudio::encode_all(black_box(&decoded.samples), &desc).unwrap();
            black_box(encoded.len());
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
