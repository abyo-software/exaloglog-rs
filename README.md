# exaloglog

Space-efficient approximate distinct counting in pure Rust. Implements
**ExaLogLog (ELL)** by [Otmar Ertl (2024)][paper], achieving the same
estimation error as HyperLogLog with **~43% less memory**.

[paper]: https://arxiv.org/abs/2402.13726

> **Status:** very early. v0 is the data structure, insert, merge, and a
> placeholder estimator. The maximum-likelihood estimator from the paper is
> being landed in successive commits.

## Why

| Algorithm | Memory for ~2% RMSE @ n=10⁶ |
| --- | --- |
| HyperLogLog (6-bit, `p=11`) | 1792 B |
| UltraLogLog (`p=10`) | 1056 B |
| **ExaLogLog `(t=2, d=20, p=8)`** | **936 B** |

ExaLogLog is mergeable, idempotent, has constant-time inserts, and supports
distinct counts up to the exa-scale. See the paper for the formal analysis.

## Configuration

This crate uses `ELL(t=2, d=24)`: 32-bit registers, 32-bit aligned for fast
access and trivial CAS-based concurrency. MVP = 3.78 (vs 6.48 for HLL with
6-bit registers, a 41% reduction). The `t=2, d=20` packed configuration
(MVP = 3.67) is planned.

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

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Issues and PRs welcome. By contributing you agree to license your contribution
under the same dual license.
