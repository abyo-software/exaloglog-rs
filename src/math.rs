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
pub(crate) fn solve_ml(alpha: f64, beta: &[u32], p: u32) -> f64 {
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
