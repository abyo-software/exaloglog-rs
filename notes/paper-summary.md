# ExaLogLog Paper Notes

Reference: Otmar Ertl, "ExaLogLog: Space-Efficient and Practical Approximate
Distinct Counting up to the Exa-Scale", arXiv:2402.13726 (2024-2025).
Reference Java implementation: https://github.com/dynatrace-research/exaloglog-paper

## Parameter Naming

| Symbol | Meaning |
| --- | --- |
| `p` | precision parameter (`m = 2^p` registers) |
| `t` | parameter of the approximated update value distribution |
| `d` | extra bitmap bits storing occurrences of update values in `[u-d, u-1]` |
| `q = 6 + t` | bits storing the maximum update value `u` |
| `b = 2^(2^-t)` | base of the geometric-like update value distribution |

The paper denotes a class of sketches `ELL(t, d)`. Recommended/empirical:

- `ELL(t=2, d=20)`: register width 28 bits → 7 bytes per pair. MVP = 3.67 (43% under HLL).
- `ELL(t=2, d=24)`: register width 32 bits → 4 bytes per register, 32-bit aligned. MVP = 3.78. Friendly to CAS-based concurrency.
- `ELL(t=1, d=9)`: register width 16 bits → 2 bytes. MVP ≈ 3.90, less efficient but byte-aligned.

This crate v0 implements `ELL(t=2, d=24)`.

## Insert (Algorithm 2)

Given 64-bit hash `h`:

1. `i = bits[t .. t+p)` of `h` — register index.
2. `a = h | ((1 << (p+t)) - 1)` — set the low `p+t` bits to 1 so that `nlz(a) ∈ [0, 64-p-t]`.
3. `nlz_a = leading_zeros(a)`.
4. `low_t = h & ((1 << t) - 1)`.
5. `k = nlz_a * 2^t + low_t + 1` — update value, `k ∈ [1, (65-p-t) * 2^t]`.
6. `u = r_i >> d` — current max update value for this register.
7. `Δ = k - u`.
   - If `Δ > 0`: `r_i ← (k << d) | floor((2^d + (r_i mod 2^d)) / 2^Δ)`.
   - Else if `Δ < 0` and `d + Δ ≥ 0`: `r_i ← r_i | (1 << (d + Δ))`.
   - Else: no change (idempotent).

Note: the register stores `u` in the high `q` bits and a length-`d` bitmap in
the low `d` bits indicating which of the most recent `d` smaller update values
have occurred. When `u` advances by `Δ`, the bitmap shifts right by `Δ` and a
1-bit is prepended at the top representing the previous value `u_old`.

## Merge (Algorithm 5)

Per-register merge `r ⊕ r'`:

```
u  = r  >> d
u' = r' >> d
if u > u' and u' > 0:
    return r  | floor((2^d + (r' mod 2^d)) / 2^(u - u'))
elif u' > u and u > 0:
    return r' | floor((2^d + (r  mod 2^d)) / 2^(u' - u))
else:  # u == u' or one side is empty
    return r | r'
```

## Estimation

Two estimators:

- **Martingale (HIP)** — Algorithm 4. Maintains `n̂` and inverse state-change probability `μ` incrementally during inserts. Optimal for non-distributed cases. Cannot be used after merging or on a deserialized sketch.
- **Maximum Likelihood (ML)** — Algorithm 8 (Newton's method). Works from the register state alone. Coefficients `α` and `β_u` are derived via Algorithm 3.

The shape of the log-likelihood (Eq. 15):

```
ln L = -(n/m) * α + Σ_{u=t+1}^{64-p} β_u * ln(1 - exp(-n / (m * 2^u)))
```

with `α = α' / 2^(64-p)` where `α'` is computed by Algorithm 3.

`φ(k) := min(t + 1 + ⌊(k-1)/2^t⌋, 64-p)` — maps a fine-grained update value `k` to the bucket index `j` whose probability is `1 / 2^φ(k)`.

`ω(u) := (2^t (1 - t + φ(u)) - u) / 2^φ(u)` — survival probability tail used in `h(r)` and elsewhere.

## Reducibility (Algorithm 6)

`ELL(t, d, p) → ELL(t, d', p')` for `d' ≤ d` and `p' ≤ p`. The reduction is
*lossless* in the sense that the resulting sketch matches what direct insertion
of the same elements at the smaller parameters would produce.

## Sparse Mode (Section 4.3)

For very small cardinalities, store a list of `(v+6)`-bit hash tokens instead
of a dense register array. Switch to dense mode at the break-even point.

## Hash Function

Recommended: WyHash, Komihash, or PolymurHash (high-quality 64-bit). The
default-hasher (SipHash) is acceptable but slower.

## Comparison Targets

| Algorithm | MVP (memory) | Bytes for n = 10^6, ~2% RMSE |
| --- | --- | --- |
| HLL 8-bit (p=11) | 9.66 | 2296 |
| HLL 6-bit (p=11) | 7.54 | 1792 |
| HLL 4-bit (p=11) | 5.60 | 1331 |
| UltraLogLog (p=10) | 4.78 | 1056 |
| HyperLogLogLog (p=11) | 4.64 | 1100 |
| **ELL(2, 24) p=8** | **3.93** | **1064** |
| **ELL(2, 20) p=8** | **3.86** | **936** |
| Conjectured lower bound | 1.98 | — |

(See Table 2 in the paper.)
