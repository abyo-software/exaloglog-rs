# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-06

### Added

- **Sparse mode** for `ExaLogLog` (packed). Sketches start in sparse
  mode and store hash tokens (paper §4.3) until the per-`m` break-even
  point, then auto-promote to dense. Sparse mode gives exact distinct
  counts for small `n` and reduces low-cardinality memory by up to ~30×.
- `ExaLogLog::new_dense(p)` skips sparse mode if you know `n` will
  exceed the break-even.
- `ExaLogLog::is_sparse()` and `ExaLogLog::densify()` exposed for
  introspection and explicit promotion.
- **Lock-free atomic insert** on `ExaLogLogFast` via
  `add_hash_atomic(&self, hash)`. Multiple threads can ingest into a
  shared sketch without external synchronization. Marks the martingale
  estimator unavailable for the sketch.
- **Reduction (Algorithm 6)** on both variants via `reduce(new_p)`,
  yielding a sketch at lower precision identical to one built directly
  at `new_p`. Useful for migration scenarios.
- Module-level documentation for both variants on using `add_hash` with
  custom hash functions (xxhash3, wyhash, etc.) — the recommended
  high-throughput path.
- `ExaLogLogFast::snapshot()` returns the current register values as a
  `Vec<u32>` (the registers are now atomic internally).

### Changed

- `ExaLogLogFast` now stores registers as `Box<[AtomicU32]>` instead of
  `Box<[u32]>`. Memory layout, alignment, and serialization format are
  unchanged.
- `ExaLogLogFast::registers()` was replaced by `snapshot()`. The old
  method couldn't return a meaningful `&[u32]` reference once registers
  were atomic.
- `ExaLogLog`'s wire format reserves the top bit of the format-version
  byte to signal sparse-mode payloads. Old `0.1.0` blobs (always dense)
  remain readable.

## [0.1.0] — 2026-05-06

Initial release.

### Added

- `ExaLogLog` — packed 28-bit ExaLogLog (`t = 2`, `d = 20`, MVP = 3.67).
  43% smaller than HLL with 6-bit registers at the same RMSE.
- `ExaLogLogFast` — 32-bit aligned ExaLogLog (`t = 2`, `d = 24`,
  MVP = 3.78). 41% smaller than HLL-6, friendlier to concurrent updates.
- Insert (Algorithm 2 of the paper).
- Per-register merge (Algorithm 5).
- Maximum-likelihood estimator with bisection solver (Algorithm 3 + ML
  equation from Section 3.2).
- Martingale (HIP) estimator (Algorithm 4) for non-distributed sketches.
- Byte-oriented serialization with explicit format magic, version, and
  parameter envelope.
- Worked examples: `cargo run --example accuracy` (RMSE/memory tables for
  both variants), `cargo run --example head_to_head` (vs reference HLL).
- Criterion benches for insert and estimate throughput.
