# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] — 2026-05-07

### Added

- **Optional `simd` feature** that enables an x86_64 AVX2 + BMI1 +
  LZCNT path for the batch hash → (register index, update value)
  computation used by `add_hashes_sorted`. The unsafe `core::arch`
  intrinsics are scoped to a single module and gated by runtime
  feature detection (`is_x86_feature_detected!`); on non-x86_64
  targets and on x86_64 CPUs missing the required ISA, the
  scalar path runs unchanged. The crate body still uses
  `#![deny(unsafe_code)]`; `simd_x86` is the only `unsafe` module
  and only with the `simd` feature on.
- Internal: `math::fill_iks` now batches the `(i, k)` computation
  with manual 4-way unrolling, giving LLVM a clean shape for
  auto-vectorization on aarch64 (CLZ).

### Notes

- On modern x86_64 (Ryzen 9 9950X, Ice Lake+) the scalar path already
  compiles `leading_zeros` to LZCNT, so the `simd` feature shows
  modest gains. Older CPUs that fall back to BSR see a larger win.
  An AVX-512 `vplzcntq` path is on the roadmap for fully vectorized
  lzcnt without per-lane extraction.

## [0.8.0] — 2026-05-07

### Added

- **Optional `rayon` feature** with `merge_many_par` and
  `merge_many_par_fast` free functions: parallel reduce-merge over a
  slice of sketches. Useful for rolling up many tenant sketches when
  you have spare cores. 4 new tests verify parity with serial merge.
- **`examples/fast_hashing.rs`**: side-by-side benchmark of the three
  insert paths — `add(&T)` with `DefaultHasher` (SipHash13),
  `add_hash(xxh3_64(bytes))`, and `add_hash(splitmix64)`. On byte
  inputs `xxh3` is ~1.6× faster than the standard library default.

## [0.7.0] — 2026-05-07

### Added

- **`add_hashes_sorted(&mut self, &[u64])`** on both variants: a
  cache-locality-optimized batch insert. Computes all `(i, k)` pairs,
  sorts by register index, and applies all updates to register `i`
  together before moving on. Cuts register R/W traffic from `O(N)` to
  `O(distinct registers in batch)` and turns a random-access pattern
  into a sequential one. Empirically ~10-25% faster than the scalar
  loop when the register array exceeds L1 cache (`p ≥ 16`). Below
  that, sort overhead dominates; use the regular `add_hashes`.
- **`merge_iter(impl IntoIterator<Item = &Self>)`** on both variants:
  merge many sketches into `self` in one call. Convenience for the
  common pattern of rolling up per-tenant sketches.

## [0.6.0] — 2026-05-07

### Added

- **Newton solver for the ML estimator** (paper Algorithm 8). Uses the
  recursive product computation (Eq. 22, 30) so the expensive
  `(1+x)^{2^l}` powers are evaluated by repeated squaring; converges
  in 5-10 iterations across the operating range. The public
  `estimate_ml` (and `estimate`) now route through Newton, with an
  automatic fallback to bisection on numerical failure.
- Validation: a new property test asserts Newton agrees with bisection
  to ~1e-3 across a synthetic battery of (β, α) configurations, and
  the existing accuracy bound tests now run against the Newton path.

### Fixed

- The Newton scaffolding shipped (disabled) in 0.4.0 had its
  termination condition inverted (`φ ≥ x'` should have been `φ ≤ x'`,
  per the sign of `f(x) = α·2^{u_max}·x − φ(x)`). Corrected and
  validated against the Java reference fixtures.

## [0.5.0] — 2026-05-07

### Added

- **Bit-for-bit parity with the Dynatrace Java reference** (paper's
  authoritative implementation at `dynatrace-research/exaloglog-paper`).
  `tests/java_parity.rs` checks that for every (d ∈ {20, 24}, p ∈ {4,
  8, 12}, n ∈ {100, 1000, 10000}) configuration, the Rust register
  state after inserting `splitmix64(0..n)` exactly matches the Java
  reference's `getState()`. 18 fixtures total. See `notes/java-parity.md`
  for the capture procedure.

## [0.4.0] — 2026-05-07

### Added

- **Batch insert APIs** on both variants: `add_hashes(&mut self, &[u64])`
  for single-threaded bulk ingestion and `add_hashes_atomic(&self, &[u64])`
  on `ExaLogLogFast` for lock-free concurrent batches. The sparse-mode
  path appends all tokens and then sort-dedupes once (`O(N log N)`)
  instead of individually binary-search-inserting (`O(N · (log N + N))`).
- **Property-based test suite** (`tests/properties.rs`) covering merge
  commutativity, merge associativity, identity-on-empty, insert-order
  independence, serialize round-trip, and `reduce(p)` parity with
  directly-built sketches. Eight properties × thirty random cases per
  `cargo test` invocation.
- **Worked examples**: `examples/sparse_demo.rs` (memory savings vs
  cardinality across the sparse → dense transition) and
  `examples/concurrent_ingest.rs` (multi-threaded `add_hash_atomic`
  scaling). The latter shows >400 M inserts/second at 8 threads on a
  recent x86_64 machine.

### Changed

- Internal: `solve_ml_newton` (paper Algorithm 8 scaffolding) is in the
  tree but not used in production. Single-bucket case validated against
  bisection; multi-bucket convergence is being debugged. Bisection
  remains the path `solve_ml` takes.

## [0.3.0] — 2026-05-07

### Added

- **Sparse mode for `ExaLogLogFast`**, mirroring the design used in
  `ExaLogLog`. New sketches start sparse; `new_dense(p)` skips straight
  to dense storage. Auto-promotes at the per-variant break-even point.
- `ExaLogLogFast::is_sparse()` and `ExaLogLogFast::densify()` exposed
  for explicit control.
- **Optional `serde` feature**. Behind `--features serde`, both
  `ExaLogLog` and `ExaLogLogFast` implement `Serialize` and
  `Deserialize`. The serde representation goes through the existing
  `to_bytes` / `from_bytes` byte format, so JSON, MessagePack, bincode,
  and CBOR all work without re-encoding.

### Changed

- `ExaLogLogFast::new(p)` now starts in sparse mode (was always dense).
  Use `ExaLogLogFast::new_dense(p)` to preserve the previous behavior;
  this is required for `add_hash_atomic` from a fresh sketch.
- `ExaLogLogFast::add_hash_atomic` now panics if called while the
  sketch is sparse, with a message pointing at `densify()` and
  `new_dense()`. The lock-free atomic invariant requires the dense
  storage layout.
- `ExaLogLogFast::snapshot()` now materializes registers from tokens
  in sparse mode rather than returning empty.

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
