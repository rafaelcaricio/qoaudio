# QOA - The Quite Ok Audio Format

A pure Rust, zero-dependency implementation of the [QOA](https://qoaformat.org) audio format with both decoding and encoding support.

> This code is based off [the reference C implementation](https://github.com/phoboslab/qoa) and was written before the release of the specification.

## Features

- **Decode** QOA files from any `io::Read` source — streaming, one sample at a time
- **Encode** 16-bit PCM audio to QOA — one-shot or frame-at-a-time streaming
- **Zero unsafe code** — the crate enforces `#![forbid(unsafe_code)]`
- **Zero required dependencies** — `rodio` and `hound` are optional features
- **Faster than C** — the encoder beats the C reference by ~27% while remaining 100% safe

## Performance

On Apple Silicon (M-series), encoding a 54-second stereo 44.1kHz file:

| Implementation | Decode | Encode |
|---|---|---|
| C reference (`qoa.h`, `gcc -O3`) | — | ~210 ms |
| **Rust** | ~46 ms | **~155 ms** |

The encoder's hot path is a brute-force search over 16 scalefactors × 20
samples per slice, dominated by a 4-element LMS dot product
(`predict`) and a self-dot product (`weights_penalty`). With full LTO and
`codegen-units = 1`, LLVM generates tight scalar code that pipelines
efficiently on wide out-of-order cores — outperforming manual SIMD
approaches for these small 4-element operations.

```bash
cargo bench
```

## Usage

### Decoding

```rust
use qoaudio::{QoaDecoder, QoaItem};
use std::io::BufReader;
use std::fs::File;

let file = File::open("audio.qoa").unwrap();
let decoder = QoaDecoder::new(BufReader::new(file)).unwrap();
for item in decoder {
    match item.unwrap() {
        QoaItem::Sample(s) => { /* process i16 sample */ }
        QoaItem::FrameHeader(h) => { /* new frame: h.num_channels, h.sample_rate */ }
    }
}
```

### Encoding (one-shot)

```rust
use qoaudio::{encode_all, QoaDesc};

let desc = QoaDesc {
    channels: 2,
    sample_rate: 44100,
    samples: samples_per_channel,
};
let encoded: Vec<u8> = encode_all(&pcm_samples, &desc).unwrap();
```

### Encoding (streaming, frame-at-a-time)

```rust
use qoaudio::{QoaEncoder, QoaDesc, QOA_FRAME_LEN};

let desc = QoaDesc { channels: 1, sample_rate: 44100, samples: total_samples };
let mut encoder = QoaEncoder::new(&desc).unwrap();
let mut output = Vec::new();
encoder.write_header(&mut output).unwrap();

for chunk in pcm_data.chunks(QOA_FRAME_LEN * desc.channels as usize) {
    encoder.encode_frame(chunk, &mut output).unwrap();
}
```

## Examples

**Play a QOA file** (requires `rodio` feature):
```bash
cargo run --release --example play --features rodio -- fixtures/julien_baker_sprained_ankle.qoa
```

**Encode a WAV file to QOA** (requires `hound` feature):
```bash
cargo run --release --example encode --features hound -- input.wav output.qoa
```

More audio samples can be found at [phoboslab.org/files/qoa-samples](https://phoboslab.org/files/qoa-samples/).

## License

Licensed under either of [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) or [MIT license](http://opensource.org/licenses/MIT) at your option.
