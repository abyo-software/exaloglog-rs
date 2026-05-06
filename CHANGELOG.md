# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
