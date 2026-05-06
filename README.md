# exaloglog

Space-efficient approximate distinct counting in pure Rust. Implements
**ExaLogLog (ELL)** by [Otmar Ertl (2024)][paper], achieving the same
estimation error as HyperLogLog with **43% less memory**.

[paper]: https://arxiv.org/abs/2402.13726

[![crates.io](https://img.shields.io/crates/v/exaloglog.svg)](https://crates.io/crates/exaloglog)
[![docs.rs](https://docs.rs/exaloglog/badge.svg)](https://docs.rs/exaloglog)

## Why

| Algorithm | Bytes for ~2% RMSE @ n=10⁶ |
| --- | --- |
| HyperLogLog (6-bit, `p=11`) | 1792 |
| UltraLogLog (`p=10`) | 1056 |
| **ExaLogLog `(t=2, d=20, p=8)`** | **896** |

The MVP (memory-variance product) for HLL with 6-bit registers is 6.48,
versus 3.67 for `ExaLogLog(t=2, d=20)` and 3.78 for the 32-bit-aligned
`ExaLogLogFast(t=2, d=24)` — a 43% and 41% improvement, respectively.

## Two variants

This crate ships two configurations of the algorithm. Both implement the same
public API and both expose the maximum-likelihood and martingale (HIP)
estimators.

### `ExaLogLog` (default — packed `d=20`, MVP=3.67)

28-bit registers, packed two-per-7-bytes. **43% smaller than HLL** at the
same RMSE. Use this unless you need lock-free concurrent updates.

### `ExaLogLogFast` (32-bit aligned, `d=24`, MVP=3.78)

32-bit registers, one register per `u32`. **41% smaller than HLL** at the
same RMSE. Slightly faster scalar inserts (~15-30%) and the variant
that exposes lock-free concurrent ingest via `add_hash_atomic(&self, hash)`.

### Sparse mode

`ExaLogLog::new(p)` starts in sparse mode and stores 32-bit hash tokens
until reaching `m · 7/8` distinct elements (the break-even point), then
promotes to dense. For low-cardinality sketches this means up to ~30×
less memory and *exact* distinct counts. Skip it with
`ExaLogLog::new_dense(p)` if you know `n` will be large.

### Reducing precision

Both variants support `reduce(new_p)`, returning a sketch at lower
precision identical to one that was built directly at `new_p`. Useful
when you committed to too high a `p` and need to compact existing
sketches.

## Usage

```rust
use exaloglog::ExaLogLog;

let mut sketch = ExaLogLog::new(12); // m = 2^12 = 4096 registers

for i in 0..1_000_000u64 {
    sketch.add(&i);
}

let estimate = sketch.estimate();
println!("estimated distinct: {estimate:.0}");
```

For deserialization or post-merge sketches, `estimate()` falls back to the
ML estimator (which works from register state alone). On a freshly built
sketch, `estimate_martingale()` is also available and has slightly lower
variance.

## Empirical accuracy

Run `cargo run --release --example accuracy` to reproduce. At `p = 12`
(packed, 14336 bytes) over 200 trials at each `n`:

| n | ML RMSE% | HIP RMSE% | ML bias% |
| ---: | ---: | ---: | ---: |
| 100 | 0.35 | 0.35 | +0.02 |
| 1k | 0.36 | 0.35 | +0.03 |
| 10k | 0.42 | 0.39 | -0.02 |
| 100k | 0.44 | 0.41 | +0.01 |
| 1M | 0.55 | 0.50 | +0.05 |

Theory predicts RMSE ≈ √(MVP / total_bits) = √(3.67 / 114688) ≈ 0.566%.
Empirical numbers track theory and the ML estimator is essentially unbiased.

## Throughput

Single-threaded, scalar implementation, on a recent Linux x86_64 machine:

| variant | `p=8` | `p=12` | `p=16` |
| --- | ---: | ---: | ---: |
| `ExaLogLog` (packed) | 101 M ins/s | 46 M ins/s | 39 M ins/s |
| `ExaLogLogFast` (aligned) | 146 M ins/s | 53 M ins/s | 45 M ins/s |

A SIMD-accelerated batch insert path is planned.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Issues and PRs welcome. By contributing you agree to license your
contribution under the same dual license.
