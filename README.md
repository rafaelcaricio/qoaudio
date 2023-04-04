# QOA - The Quite Ok Audio Format

This is a pure Rust (zero dependency) implementation of the [QOA](https://qoaformat.org) audio format.

> This code is based off [the reference C implementation](https://github.com/phoboslab/qoa) and was written before the release of the specification.

## Running the example

```bash
cargo run --release --example play --features="rodio" -- fixtures/julien_baker_sprained_ankle.qoa
```
We need to enable the `rodio` feature to play the audio. It is not a direct dependency of this crate, but we implement
integration with `rodio` for playback capability.

More audio samples can be found at [phoboslab.org/files/qoa-samples](https://phoboslab.org/files/qoa-samples/).
