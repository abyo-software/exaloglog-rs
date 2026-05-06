# 43% smaller than HyperLogLog: a from-scratch Rust implementation of ExaLogLog

*A new sketch for approximate distinct counting that beats HyperLogLog at its
own game. Published today as the
[`exaloglog`](https://crates.io/crates/exaloglog) crate.*

---

If you've ever run `SELECT COUNT(DISTINCT user_id)` on a stream of telemetry,
you've probably leaned on HyperLogLog. It's been the standard distinct-counting
sketch for fifteen years; Redis, ClickHouse, BigQuery, Snowflake, Elasticsearch,
DuckDB, and Apache DataSketches all ship some flavor of it. The deal is
beautiful: spend 1.5 KB of register state per stream and recover the
cardinality of arbitrary 64-bit datasets to within ~2% relative error.

In early 2024, Otmar Ertl at Dynatrace Research [published a sketch][paper]
that achieves the same accuracy with 43% less memory. The paper is rigorous
— theory, experiments, a reference implementation in Java — but a year
later there was no production-quality Rust implementation. So I wrote one.

This post walks through what changed, why the math actually works, and what
it cost me to translate from paper to code.

[paper]: https://arxiv.org/abs/2402.13726

## The headline

| Algorithm | Bytes for ~2% RMSE @ n = 10⁶ | Notes |
| --- | ---: | --- |
| HyperLogLog (8-bit registers) | 4096 | Typical Rust crate layout |
| HyperLogLog (6-bit packed) | 3072 | Apache DataSketches style |
| UltraLogLog | 1056 | Ertl's prior work, 28% smaller than HLL-6 |
| **ExaLogLog (this crate)** | **1792** at p=9 | **41.7% smaller than HLL-6** |

Theory predicts 43.4% savings (`1 − MVP_ELL / MVP_HLL6 = 1 − 3.67 / 6.48`);
empirical numbers track theory once we account for the discrete nature of
`p = log₂(m)`. Either way, the savings are large enough to matter for
storage planning.

## What ExaLogLog is, in one paragraph

Like HyperLogLog, ExaLogLog distributes hash values across `m = 2^p`
registers using the low `p` bits of the hash. Unlike HyperLogLog, the
register doesn't just track the longest leading-zero run seen so far —
it splits into a high `q` bits storing the maximum *update value* `u`,
and a low `d` bits acting as a bitmap that records which of the most
recent `d` smaller update values have actually been observed.

The bitmap is the key insight. HyperLogLog throws away information every
time it overwrites a register: knowing only that some hash with leading-
zero count 7 arrived doesn't tell us anything about whether values 5 or 6
also did. ExaLogLog keeps a sliding window of that "auxiliary"
information, and the maximum-likelihood estimator that follows is the one
that uses it.

The configuration this crate ships is `t = 2, d = 20`: four "fine-grained"
update values per HLL-style integer level, twenty bits of bitmap. This
puts the per-register width at 28 bits — exactly two registers per 7
bytes. Memory-variance product (MVP) hits 3.67, where HLL with 6-bit
registers sits at 6.48. The 1.77× ratio is the 43% savings.

## Reading Algorithm 2 in earnest

The insert procedure (Algorithm 2 of the paper) looks unassuming on paper.
Given a 64-bit hash `h`:

```
i  = bits[t .. t+p) of h          // register index
a  = h | ((1 << (p+t)) - 1)       // force low p+t bits to 1
k  = nlz(a) * 2^t + low_t(h) + 1  // update value, k ∈ [1, (65-p-t)·2^t]
u  = r_i >> d                     // current max update for register i
Δ  = k - u

if Δ > 0:
    r_i ← (k << d) | floor((2^d + bitmap(r_i)) / 2^Δ)
elif Δ < 0 and d + Δ ≥ 0:
    r_i ← r_i | (1 << (d + Δ))
else:
    no-op
```

Two pieces tripped me up in implementation.

The first was `k`. ExaLogLog approximates a base-`b = 2^(2^-t)` geometric
distribution over update values by chunking `2^t` consecutive values into
"plateaus" with equal probability. Translating "compute `k` from the
hash" into a few cycles of bit-twiddling required understanding *why* the
construction `nlz(a) · 2^t + low_t(h) + 1` actually samples from the right
distribution. I spent an hour staring at Eq. 8 before convincing myself.

The second was the register update rule when `Δ > 0`. The expression
`floor((2^d + bitmap) / 2^Δ)` looked off until I realized: when we bump
`u` by `Δ` levels, the bitmap's highest position now has to record that
the *previous* `u` was hit. The `2^d` term prepends that 1-bit, then the
right-shift drops the bottom `Δ` bits that fall off the end of the
window. It's a one-line implementation of "shift the bitmap, mark the
most recent observation."

## Maximum likelihood, by way of bisection

The paper presents two estimators. The martingale (HIP) estimator is
incremental and trivially fast — every state-changing insert increments
the running estimate by `1/μ` and adjusts `μ`. But it requires
maintaining `μ` across every insert and breaks once you merge or
deserialize a sketch. So you also need a maximum-likelihood estimator
that works from the register state alone.

Algorithm 3 in the paper computes `(α, β_u)` coefficients of the
log-likelihood, which has the simple shape

```
ln L = -(n/m) · α + Σ β_u · ln(1 - exp(-n / (m · 2^u)))
```

Derivative-set-to-zero gives the ML equation `g(y) = α` where `y = n/m`,
and the paper devotes Algorithm 8 to a robust Newton's-method solver
with recursive product computation to avoid floating-point overflow.

I shipped bisection in `log₂(y)` instead. It converges to f64 precision
in under 200 iterations, sidesteps the overflow concerns by using
`exp_m1` for the denominator, and the cost is dwarfed by per-element
insert work for any realistic ingestion rate. Newton's method would be
~10× faster for the estimate call, but estimate is a query-time
operation; nobody calls it in a hot loop. (If your workload does, file
an issue.)

## A bug I caught with a sanity check

My first pass through Algorithm 3 was wrong. The paper's Section 3.3
defines `h(r) = (1/m)(ω(u) + Σ (1 - l_{u-k}) · ρ_update(k))` — bit set
means "we observed update value k" means *no contribution* to the
state-change probability. I had it inverted: my `compute_alpha_beta`
contributed `1/2^φ(k)` to `α` when the bit was set and to `β` when clear.

The smoking gun was that `h(0) = 1/m` is supposed to hold — the
state-change probability of an empty register equals the probability
that the next inserted element lands on it. With my bug, that identity
broke. Once I fixed the inversion, every other property fell into line:
the empirical RMSE matched theory to within statistical noise, and the
martingale and ML estimators agreed to within 2% of `n` on fresh
sketches at `p = 12`.

This is the kind of error that's invisible in spot-tests. A 5% bias on
the cardinality estimate is well below most sanity thresholds. The way
to catch it is to grind theory invariants — `h(0) = 1/m`, `ω(0) = 1`,
"`h` strictly decreases on every register transition the insert
algorithm produces" — and turn each one into a test that runs on every
build.

## Two variants, two tradeoffs

The crate ships both `ExaLogLog` (packed, `d = 20`, MVP = 3.67) and
`ExaLogLogFast` (32-bit aligned, `d = 24`, MVP = 3.78).

The packed variant is the headline 43% memory savings configuration. Two
registers share a 7-byte chunk, so reads are an unaligned 4-byte load
and a mask, and writes are read-modify-write of the byte that's shared
with the sibling register. That shared byte is the reason packed cannot
be CAS'd safely for lock-free concurrent updates.

`ExaLogLogFast` trades 3% extra memory for `u32`-aligned registers — one
register per machine word, individually CAS-friendly, and ~15-30% faster
on the scalar insert path. If you're feeding events into a sketch from
twenty cores at once, this is the variant you want.

The two share their math via a `math` module parameterized by `d`. The
ML solver, the (α, β) coefficient computation, the per-register insert
rule, and the merge rule are all the same code, called with different
`d`. This was worth the up-front refactor: it would have been easy to
implement the packed variant by copy-pasting the aligned one, but then
every bug fix would have needed two patches, and divergence over time
is inevitable.

## Throughput

Single-threaded, scalar code, on a recent x86_64 Linux machine:

| variant | `p = 8` | `p = 12` | `p = 16` |
| --- | ---: | ---: | ---: |
| `ExaLogLog` (packed) | 101 M ins/s | 46 M ins/s | 39 M ins/s |
| `ExaLogLogFast` (aligned) | 146 M ins/s | 53 M ins/s | 45 M ins/s |

For comparison, my reference HLL implementation in the same harness hits
~100 M ins/s — ExaLogLog is in the same throughput regime despite doing
strictly more bookkeeping per insert. The drop from 146 to 39 M as `p`
grows from 8 to 16 is cache-bound: at `p = 16`, the register array is
64 KB and falls out of L1.

A SIMD-accelerated batch-insert path is on the v0.2 list. Random-access
register updates limit the win — gather/scatter is needed, and AVX-512's
the only reasonable target — so I'd rather ship correct scalar code now
and verify I'm not breaking accuracy with the SIMD version later.

## What I'm doing next

ExaLogLog is the first of a series of paper implementations I'm shipping
through the [`abyo-software`][abyo] organization. The bet is that LLM-
assisted engineering has dropped the cost of going from "rigorously-
described algorithm" to "production-quality crate" by an order of
magnitude. ExaLogLog took two days end-to-end including the writeup
you're reading. The next ones — RaBitQ for vector quantization, Ribbon
+ InfiniFilter for filters, Eg-walker CRDTs, ChalametPIR for private
search — are queued up.

If you were already reaching for HyperLogLog, switching to ExaLogLog is
a pure win. The API is identical in spirit:

```rust
use exaloglog::ExaLogLog;

let mut sketch = ExaLogLog::new(12);
for item in stream {
    sketch.add(&item);
}
println!("distinct: {}", sketch.estimate());
```

Source on [GitHub][gh]. Crate on [crates.io][crate]. Docs on [docs.rs][docs].
Issues, benchmarks against your own workloads, and PRs welcome.

[abyo]: https://github.com/abyo-software
[gh]: https://github.com/abyo-software/exaloglog-rs
[crate]: https://crates.io/crates/exaloglog
[docs]: https://docs.rs/exaloglog
