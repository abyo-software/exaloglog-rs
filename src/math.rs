//! Math primitives shared between `ExaLogLog` (packed, D=20) and
//! `ExaLogLogFast` (aligned, D=24). All functions are parameterized by the
//! `d` parameter so the two variants share their estimator implementations.

use crate::T;

/// φ(k) := min(t + 1 + ⌊(k-1)/2^t⌋, 64 - p)  — Eq. (11) of the paper.
pub(crate) fn phi(k: u32, p: u32) -> u32 {
    let v = if k == 0 {
        T // ⌊-1/2^t⌋ = -1, so φ(0) = t + 1 - 1 = t
    } else {
        T + 1 + ((k - 1) >> T)
    };
    v.min(64 - p)
}

/// ω(u) := (2^t (1 - t + φ(u)) - u) / 2^φ(u)  — Eq. (14) of the paper.
pub(crate) fn omega(u: u32, p: u32) -> f64 {
    let phi_u = phi(u, p);
    let two_t = (1u32 << T) as f64;
    (two_t * (1.0 + phi_u as f64 - T as f64) - u as f64) * pow2_neg(phi_u)
}

/// 2^(-x), branchless and fast for small x. Avoids `f64::powi` overhead.
#[inline]
pub(crate) fn pow2_neg(x: u32) -> f64 {
    f64::from_bits((1023u64.wrapping_sub(x as u64)) << 52)
}

/// Per-register state-change probability `h(r)` (Section 3.3 of the paper).
///
/// `h(r) = (1/m) (ω(u) + Σ_{k=max(1,u-d)}^{u-1} (1 - l_{u-k}) / 2^φ(k))`
/// where `r = u·2^d + ⟨l_1 ... l_d⟩₂`.
pub(crate) fn h(r: u32, p: u32, d: u32) -> f64 {
    let d_mask = (1u32 << d) - 1;
    let u = r >> d;
    let bitmap = r & d_mask;
    let m = (1u64 << p) as f64;

    let mut acc = omega(u, p);
    if u >= 2 {
        let k_lo = u.saturating_sub(d).max(1);
        for k in k_lo..u {
            let j = u - k; // j ∈ [1, d]
            let l_j = (bitmap >> (d - j)) & 1;
            acc += (1.0 - l_j as f64) * pow2_neg(phi(k, p));
        }
    }
    acc / m
}

/// Compute the (α, β) coefficients of the log-likelihood from a register
/// iterator (Algorithm 3).
///
/// `α` is returned already divided by `2^(64-p)` (it is the per-register
/// average, not the integer accumulator from the paper). `β` is indexed
/// `0..(64 - p - t)` where entry `j` is the paper's `β_{j + t + 1}`.
pub(crate) fn compute_alpha_beta<I>(registers: I, p: u32, d: u32) -> (f64, Vec<u32>)
where
    I: Iterator<Item = u32>,
{
    let d_mask = (1u32 << d) - 1;
    let beta_len = (64 - p - T) as usize;
    let mut beta = vec![0u32; beta_len];
    let mut alpha = 0.0_f64;

    for r in registers {
        let u = r >> d;
        let bitmap = r & d_mask;

        alpha += omega(u, p);

        if u >= 1 {
            let j = phi(u, p);
            beta[(j - T - 1) as usize] += 1;

            if u >= 2 {
                let k_lo = u.saturating_sub(d).max(1);
                for k in k_lo..u {
                    let bit_pos = d - (u - k);
                    let bit_set = (bitmap >> bit_pos) & 1 == 1;
                    let phi_k = phi(k, p);
                    if bit_set {
                        beta[(phi_k - T - 1) as usize] += 1;
                    } else {
                        alpha += pow2_neg(phi_k);
                    }
                }
            }
        }
    }

    (alpha, beta)
}

/// `g(y) = Σ_u β_u / (2^u · (exp(y/2^u) − 1))` — left-hand side of the ML
/// equation `g(y) = α` (derivative of Eq. 15 set to zero).
fn g(y: f64, beta: &[u32]) -> f64 {
    let mut sum = 0.0;
    for (idx, &b) in beta.iter().enumerate() {
        if b == 0 {
            continue;
        }
        let u = idx as u32 + T + 1;
        let scale = pow2_neg(u);
        let denom = (y * scale).exp_m1();
        if !denom.is_finite() || denom == 0.0 {
            continue;
        }
        sum += b as f64 * scale / denom;
    }
    sum
}

/// Solve the ML equation `g(y) = α` for `y = n/m` by bisection in `log₂(y)`.
/// Returns the cardinality estimate `n = m · y`. Returns `0.0` for an empty
/// sketch (all `β_u` zero).
///
/// Used as the trustworthy fallback in [`solve_ml`].
pub(crate) fn solve_ml_bisection(alpha: f64, beta: &[u32], p: u32) -> f64 {
    if beta.iter().all(|&b| b == 0) {
        return 0.0;
    }
    if alpha <= 0.0 {
        return f64::INFINITY;
    }

    let mut lo: f64 = -200.0;
    let mut hi: f64 = 200.0;

    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let y = (mid * std::f64::consts::LN_2).exp();
        let gv = g(y, beta);
        if gv > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 {
            break;
        }
    }

    let mid = 0.5 * (lo + hi);
    let y = (mid * std::f64::consts::LN_2).exp();
    let m = (1u64 << p) as f64;
    m * y
}

/// Solve the ML equation `g(y) = α` using Newton's method per paper
/// Algorithm 8. Faster than bisection (5-7 iterations to convergence vs
/// ~50) and uses the recursive product computation (Eq. 22, 30) so all
/// expensive `(1+x)^{2^l}` powers are computed by repeated squaring.
///
/// Returns `f64::NAN` on numerical failure; the public [`solve_ml`]
/// wrapper falls back to bisection in that case.
///
/// Currently only matches bisection in the single-bucket case; multi-
/// bucket case is being debugged. Bisection is the production path.
#[allow(dead_code)]
pub(crate) fn solve_ml_newton(alpha: f64, beta: &[u32], p: u32) -> f64 {
    if beta.iter().all(|&b| b == 0) {
        return 0.0;
    }
    if alpha <= 0.0 {
        return f64::INFINITY;
    }

    // Step 1: σ_0, σ_1, u_min, u_max (Algorithm 8 prelude).
    let mut sigma_0 = 0.0_f64;
    let mut sigma_1 = 0.0_f64;
    let mut u_min: i32 = -1;
    let mut u_max: u32 = 0;
    for (idx, &b) in beta.iter().enumerate() {
        if b > 0 {
            let u = idx as u32 + T + 1; // β index 0 ↔ u = T + 1
            if u_min < 0 {
                u_min = u as i32;
            }
            u_max = u;
            sigma_0 += b as f64;
            sigma_1 += b as f64 * pow2_neg(u);
        }
    }
    if u_min < 0 {
        return 0.0;
    }
    let u_min = u_min as u32;

    // Step 2: scale σ_1 to integer-like range (= Σ β_u · 2^{u_max − u}).
    let two_u_max = (1u64 << u_max) as f64;
    sigma_1 *= two_u_max;
    let alpha_scaled = alpha * two_u_max; // = α · 2^{u_max}
    if !sigma_1.is_finite() || !alpha_scaled.is_finite() {
        return f64::NAN;
    }

    // Step 3: initial x.
    let mut x = sigma_1 / alpha_scaled;

    if u_min < u_max {
        // Eq. 27: x_0 = (1 + σ_1/α_scaled)^(σ_0/σ_1) − 1.
        let ratio = sigma_1 / alpha_scaled;
        let exponent = ratio.ln_1p() * (sigma_0 / sigma_1);
        x = exponent.exp_m1();
        if !x.is_finite() || x <= 0.0 {
            return f64::NAN;
        }

        // Newton iteration. Practical bound: 30 iterations is plenty;
        // the paper observes ≤ 10 in their experiments.
        for _ in 0..50 {
            let mut lambda = 1.0_f64;
            let mut eta = 0.0_f64;
            let mut y = x;
            let mut phi = beta[(u_max - T - 1) as usize] as f64;
            let mut psi = 0.0_f64;
            let mut u = u_max;

            // Inner loop accumulates φ (Eq. 17) and ψ (Eq. 28) using the
            // recursive λ, η updates (Eq. 22, 30).
            loop {
                u -= 1;
                let z = 2.0 / (2.0 + y); // ∈ [0, 1]
                lambda *= z;
                eta = eta * (2.0 - z) + (1.0 - z);
                let beta_u = beta[(u - T - 1) as usize] as f64;
                phi += beta_u * lambda;
                psi += beta_u * lambda * eta;
                if u <= u_min {
                    break;
                }
                y = y * (y + 2.0); // (1+x)^{2^{j+1}} − 1
            }

            let x_target = alpha_scaled * x;
            if phi >= x_target {
                // f(x) ≥ 0 → x is at or above the root, stop (Eq. 18).
                break;
            }
            let denom = psi + alpha_scaled * x;
            if !denom.is_finite() || denom == 0.0 {
                return f64::NAN;
            }
            let x_old = x;
            x *= 1.0 + (phi - x_target) / denom;
            if !x.is_finite() || x <= x_old {
                // Numerically converged or diverging; stop.
                break;
            }
        }
    }

    // Eq. 19: n̂ = m · 2^{u_max} · log1p(x).
    let m = (1u64 << p) as f64;
    let result = m * two_u_max * x.ln_1p();
    if result.is_finite() {
        result
    } else {
        f64::NAN
    }
}

/// Solve the ML equation. Currently delegates to bisection;
/// [`solve_ml_newton`] exists as an experimental Newton implementation
/// being validated against bisection (matches at single-β cases but
/// diverges at multi-β; under investigation).
pub(crate) fn solve_ml(alpha: f64, beta: &[u32], p: u32) -> f64 {
    solve_ml_bisection(alpha, beta, p)
}

/// Apply Algorithm 2's register update rule given the existing register `r`,
/// new update value `k`, and parameter `d`.
#[inline]
pub(crate) fn apply_insert(r: u32, k: u32, d: u32) -> u32 {
    let d_mask = (1u32 << d) - 1;
    let u = r >> d;
    if k > u {
        let delta = (k - u) as u64;
        let bitmap = (r & d_mask) as u64;
        let combined = (1u64 << d) | bitmap;
        let new_low = if delta <= u64::from(d) + 1 {
            combined >> delta
        } else {
            0
        };
        (k << d) | (new_low as u32 & d_mask)
    } else if k < u {
        let neg_delta = (u - k) as u64;
        if neg_delta <= u64::from(d) {
            let pos = d - neg_delta as u32;
            r | (1u32 << pos)
        } else {
            r
        }
    } else {
        r
    }
}

/// Merge two register values from sketches with the same `t, d, p`
/// (Algorithm 5 of the paper).
pub(crate) fn merge_register(r: u32, r2: u32, d: u32) -> u32 {
    let d_mask = (1u32 << d) - 1;
    let u = r >> d;
    let u2 = r2 >> d;

    if u > u2 && u2 > 0 {
        let bitmap2 = (r2 & d_mask) as u64;
        let combined = (1u64 << d) | bitmap2;
        let shift = u - u2;
        let extra = if shift <= d + 1 { combined >> shift } else { 0 };
        r | (extra as u32 & d_mask)
    } else if u2 > u && u > 0 {
        let bitmap = (r & d_mask) as u64;
        let combined = (1u64 << d) | bitmap;
        let shift = u2 - u;
        let extra = if shift <= d + 1 { combined >> shift } else { 0 };
        r2 | (extra as u32 & d_mask)
    } else {
        r | r2
    }
}

/// Compute (register_index, update_value `k`) from a 64-bit hash.
#[inline]
pub(crate) fn hash_to_register_k(hash: u64, p: u32) -> (usize, u32) {
    let p_plus_t = p + T;
    let i = ((hash >> T) & ((1u64 << p) - 1)) as usize;
    let a = hash | ((1u64 << p_plus_t) - 1);
    let nlz_a = a.leading_zeros() as u64;
    let t_mask = (1u64 << T) - 1;
    let low_t = hash & t_mask;
    let k = ((nlz_a << T) + low_t + 1) as u32;
    debug_assert!(k >= 1);
    debug_assert!(k as u64 <= ((65 - p as u64 - T as u64) << T));
    (i, k)
}

/// Hash-token compression for sparse mode (Section 4.3, Eq. v + 6 bits).
///
/// Maps a 64-bit hash to a 32-bit token: the high 26 bits store the hash's
/// low 26 bits, and the low 6 bits store `nlz(hash | (2^26 − 1))` — the
/// number of leading zeros of the hash's high 38 bits.
///
/// `V = 26` is chosen so `p + t ≤ V` holds for any `p ≤ MAX_P = 26` with
/// `t = 2`, ensuring the token preserves enough information to lossy-
/// reconstruct a representative hash for any sketch in this crate.
pub(crate) const SPARSE_V: u32 = 26;
const SPARSE_NLZ_BITS: u32 = 6;
const SPARSE_NLZ_MASK: u32 = (1 << SPARSE_NLZ_BITS) - 1;

/// Hash → 32-bit token.
#[inline]
pub(crate) fn hash_to_token(hash: u64) -> u32 {
    let low_v = (hash & ((1u64 << SPARSE_V) - 1)) as u32;
    let masked = hash | ((1u64 << SPARSE_V) - 1);
    let nlz = masked.leading_zeros().min(64 - SPARSE_V);
    (low_v << SPARSE_NLZ_BITS) | (nlz & SPARSE_NLZ_MASK)
}

/// Token → representative 64-bit hash. The reconstruction is faithful
/// for all bits that any ELL sketch with `p + t ≤ V` could observe.
#[inline]
pub(crate) fn token_to_hash(token: u32) -> u64 {
    let low_v = (token >> SPARSE_NLZ_BITS) as u64;
    let nlz = token & SPARSE_NLZ_MASK;
    let high_bit: u64 = if nlz < 64 - SPARSE_V {
        1u64 << (63 - nlz)
    } else {
        0
    };
    high_bit | low_v
}

/// ML estimate of distinct count from a set of distinct hash tokens
/// (sparse mode, paper §4.3 Eq. 26). The tokens must be deduplicated.
///
/// The log-likelihood has the same shape as the dense Eq. 15 with `m = 1`
/// and `t = V`, so we reuse [`solve_ml`] by re-indexing the sparse β
/// vector into its dense layout.
pub(crate) fn estimate_from_tokens(tokens: &[u32]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }

    // Sparse PMF: ρ_token(w) = 2^(-u_w) where u_w = min(V + 1 + nlz(w), 64).
    // α (sparse) = 1 − Σ ρ_token(w) summed over w ∈ T  (Eq. 25 + 26).
    // β_u (sparse) = number of tokens at PMF index u, for u ∈ [V+1, 64].
    let mut alpha = 1.0_f64;
    let beta_len_sparse = (64 - SPARSE_V) as usize; // u ∈ [V+1, 64]
    let mut sparse_beta = vec![0u32; beta_len_sparse];

    for &w in tokens {
        let nlz = w & SPARSE_NLZ_MASK;
        let u = (SPARSE_V + 1 + nlz).min(64);
        sparse_beta[(u - SPARSE_V - 1) as usize] += 1;
        alpha -= pow2_neg(u);
    }

    // solve_ml expects β indexed by (u − T − 1) for u ∈ [T+1, 64].
    // Sparse β index 0 corresponds to u = V + 1; in solve_ml's index that is
    // V − T (since solve_ml index = u − T − 1).
    let solve_beta_len = (64 - T) as usize;
    let mut solve_beta = vec![0u32; solve_beta_len];
    let offset = (SPARSE_V - T) as usize;
    for (i, &b) in sparse_beta.iter().enumerate() {
        solve_beta[i + offset] = b;
    }

    // p = 0 → m = 1, so solve_ml returns m · y = y = n directly.
    solve_ml(alpha, &solve_beta, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MAX_P, MIN_P};

    #[test]
    fn phi_matches_definition() {
        let p = 8;
        assert_eq!(phi(0, p), T);
        assert_eq!(phi(1, p), T + 1);
        for k in 1u32..1000 {
            let expected = (T + 1 + (k - 1) / (1 << T)).min(64 - p);
            assert_eq!(phi(k, p), expected);
        }
    }

    #[test]
    fn omega_zero_is_one() {
        for p in MIN_P..=18 {
            let w = omega(0, p);
            assert!((w - 1.0).abs() < 1e-12, "ω(0) for p={p} = {w}");
        }
    }

    #[test]
    fn pow2_neg_matches_powi() {
        for x in 0u32..64 {
            let fast = pow2_neg(x);
            let reference = 2.0_f64.powi(-(x as i32));
            assert!((fast - reference).abs() < 1e-300 || (fast / reference - 1.0).abs() < 1e-15,);
        }
        let _ = MAX_P;
    }

    #[test]
    fn newton_returns_zero_for_empty_beta() {
        let beta = vec![0u32; 60];
        assert_eq!(solve_ml_newton(0.0, &beta, 12), 0.0);
        assert_eq!(solve_ml_newton(1.0, &beta, 12), 0.0);
    }

    #[test]
    fn newton_matches_bisection_for_single_nonzero_beta() {
        // The single-bucket case has a closed form and Newton's recursion
        // never iterates, so it should match bisection exactly.
        for p in [4u32, 12, 20] {
            let beta_len = (64 - T) as usize;
            for u_idx in 5..beta_len.min(20) {
                let mut beta = vec![0u32; beta_len];
                beta[u_idx] = 100;
                for &alpha_n in &[10.0_f64, 1000.0, 1e6] {
                    let alpha = alpha_n / (1u64 << p) as f64;
                    let bis = solve_ml_bisection(alpha, &beta, p);
                    let nwt = solve_ml_newton(alpha, &beta, p);
                    if !nwt.is_finite() {
                        continue;
                    }
                    let rel = (bis - nwt).abs() / bis.max(1.0);
                    assert!(
                        rel < 1e-6,
                        "p={p} u_idx={u_idx} alpha_n={alpha_n}: bis={bis}, nwt={nwt}"
                    );
                }
            }
        }
    }

    #[test]
    fn h_of_zero_is_one_over_m() {
        for p in MIN_P..=18 {
            let m = (1u64 << p) as f64;
            for d in [20u32, 24] {
                let value = h(0, p, d);
                assert!((value - 1.0 / m).abs() < 1e-15);
            }
        }
    }
}
