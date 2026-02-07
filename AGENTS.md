# Repository Guidelines

## Overview
Pure Rust QOA (Quite OK Audio) encoder/decoder. Single-file crate (`src/lib.rs`, ~1250 lines) that enforces `#![forbid(unsafe_code)]`. The encoder beats the reference C implementation by ~27% while remaining 100% safe.

## Project Layout
```
src/lib.rs          — entire crate: decoder, encoder, LMS, rodio integration, tests
benches/            — Criterion benchmarks (decode + encode)
examples/play.rs    — audio playback (requires `rodio` feature)
examples/encode.rs  — WAV→QOA conversion (requires `hound` feature)
fixtures/           — test audio files (QOA format)
fuzz/               — libFuzzer targets (iter_all, encode_round_trip)
```

## Features & Toolchains
- `rodio` — optional playback integration
- `hound` — optional WAV reading for the encode example

## Build & Test Commands
```bash
cargo fmt
cargo clippy --features rodio,hound --tests
cargo test --features rodio,hound

# Benchmarks
cargo bench

# Fuzzing (requires cargo-fuzz + nightly)
cargo +nightly fuzz run iter_all
cargo +nightly fuzz run encode_round_trip
```

## Architecture & Hot Paths
The encoder's inner loop in `encode_slice` (~lines 485–555) dominates runtime: 16 scalefactor trials × up to 20 samples per slice, ~480K slices for the benchmark fixture. The critical functions are:

- **`QoaLms::predict()`** — 4-element dot product (weights · history), shifted right 13. Uses `wrapping_mul`/`wrapping_add`.
- **`QoaLms::weights_penalty()`** — self-dot product of weights for the encoder's heuristic.
- **`QoaLms::update()`** — LMS weight/history update, called per sample. Weights use branchless conditional add, history uses direct array assignment.
- **`qoa_div()`** — wrapping i32 division reciprocal, emits fused `madd`
- The gather loop at the top of `encode_slice` that converts interleaved samples to contiguous

## Performance Rules
1. **Always benchmark before and after** using `cargo bench`. The fixture is a 54-second stereo 44.1kHz file.
2. **Full LTO + `codegen-units = 1`** is the optimal profile. These settings allow LLVM to fully optimize scalar code, which outperforms manual SIMD for the small 4-element LMS operations.
3. **Wrapping arithmetic is intentional** in predict/weights_penalty — it matches C reference behavior. The values don't overflow in practice for well-formed audio.
4. **The `valid` flag in the inner loop is load-bearing** — removing it changes branch prediction behavior and regresses performance. Don't remove it.
5. Keep the `[i32; QOA_SLICE_LEN]` gather buffer and `.min(QOA_SLICE_LEN)` hint — they eliminate bounds checks from the inner loop.
6. **`weights_penalty()` is computed immediately after `predict()`** in the encode loop — both read from `self.weights` without intervening writes, enabling instruction-level parallelism.

## Coding Style
- `rustfmt` with defaults (4-space indent). Run `cargo fmt` before every commit.
- `snake_case` functions, `UpperCamelCase` types, `SCREAMING_SNAKE_CASE` constants.
- `#[allow(clippy::needless_range_loop)]` is used on the inner loop — index-based access benchmarks faster than iterators there.
- No `unsafe`. The crate root has `#![forbid(unsafe_code)]`.

## Testing
- Unit tests live in `mod tests` at the bottom of `src/lib.rs`.
- Round-trip tests verify encode→decode produces samples within tolerance (max diff 8000, RMS checked).
- The streaming encode test verifies frame-by-frame output matches one-shot.
- Fuzz targets: `iter_all` (decoder), `encode_round_trip` (encoder). Run after changes to parsing or encoding logic.

## Commits
Conventional Commits format: `type(scope): summary` (imperative, ≤72 chars). Common types: `feat`, `fix`, `perf`, `docs`, `test`, `chore`. Include motivation in the body for behavioral changes.
