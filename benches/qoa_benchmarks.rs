use criterion::{black_box, criterion_group, criterion_main, Criterion};

static QOA_BYTES: &[u8] = include_bytes!("../fixtures/julien_baker_sprained_ankle.qoa");

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("iter_sprained_ankle", |b| {
        b.iter(|| {
            let Ok(decoder) = qoaudio::QoaDecoder::new(std::io::Cursor::new(QOA_BYTES)) else {
                panic!("QoaDecoder::new failed");
            };
            let count = decoder.take_while(|i| matches!(i, Ok(_))).count();
            assert_eq!(count, 2394122 * 2 + 468);
            black_box(count);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
