//! ExaLogLog: space-efficient approximate distinct counting.
//!
//! Implementation of the ExaLogLog (ELL) algorithm by Otmar Ertl, "ExaLogLog:
//! Space-Efficient and Practical Approximate Distinct Counting up to the
//! Exa-Scale", arXiv:2402.13726 (2024). Compared to HyperLogLog with 6-bit
//! registers, ExaLogLog achieves the same estimation error with ~43% less
//! memory.
//!
//! This v0 uses the configuration `ELL(t=2, d=24)`: 32-bit registers, 32-bit
//! aligned, MVP ≈ 3.78 (HLL-6 has MVP 6.48). The packed `t=2, d=20`
//! configuration (28-bit registers, MVP 3.67) is planned.
//!
//! # Estimators
//!
//! Two estimators are described in the paper:
//!
//! - **Martingale (HIP)** — incremental, optimal for non-distributed cases.
//!   Implemented here. Cannot be used after a merge or after deserialization.
//! - **Maximum likelihood** — works from the register state alone. Coming in a
//!   subsequent commit.
//!
//! # Example
//!
//! ```
//! use exaloglog::ExaLogLog;
//!
//! let mut sketch = ExaLogLog::new(12);
//! for i in 0..100_000u64 {
//!     sketch.add(&i);
//! }
//! let estimate = sketch.estimate();
//! assert!((estimate - 100_000.0).abs() / 100_000.0 < 0.05);
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use std::hash::{DefaultHasher, Hash, Hasher};

/// `t` parameter (Eq. 8). Fixed at 2 in v0.
const T: u32 = 2;
/// `d` parameter — extra bitmap bits per register.
const D: u32 = 24;
/// `q = 6 + t` — bits storing the maximum update value `u`.
const Q: u32 = 6 + T;
/// Total bits per register (`q + d` = 32 in v0).
const REGISTER_BITS: u32 = Q + D;
const _: () = assert!(REGISTER_BITS == 32);

const D_MASK: u32 = (1u32 << D) - 1;
const T_MASK: u64 = (1u64 << T) - 1;

/// Minimum precision parameter.
pub const MIN_P: u32 = 3;
/// Maximum precision parameter.
pub const MAX_P: u32 = 26;

/// Error returned when two sketches cannot be merged.
#[derive(Debug, PartialEq, Eq)]
pub enum MergeError {
    /// The two sketches were created with different `p` values.
    PrecisionMismatch {
        /// Precision of `self`.
        lhs: u32,
        /// Precision of `other`.
        rhs: u32,
    },
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::PrecisionMismatch { lhs, rhs } => {
                write!(f, "precision mismatch: lhs p={lhs}, rhs p={rhs}")
            }
        }
    }
}

impl std::error::Error for MergeError {}

/// An ExaLogLog distinct-count sketch with parameters `t = 2`, `d = 24`.
#[derive(Clone, Debug)]
pub struct ExaLogLog {
    p: u32,
    registers: Box<[u32]>,
    /// Running martingale (HIP) cardinality estimate.
    martingale: f64,
    /// Sum over registers of `h(r_i)` — the inverse state-change probability.
    mu: f64,
    /// Set when the running estimate has been invalidated (e.g. after merge).
    martingale_invalid: bool,
}

impl ExaLogLog {
    /// Create an empty sketch with `2^p` registers.
    ///
    /// `p` must be in `[MIN_P, MAX_P]`.
    pub fn new(p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&p),
            "precision p={p} out of range [{MIN_P}, {MAX_P}]"
        );
        let m = 1usize << p;
        Self {
            p,
            registers: vec![0u32; m].into_boxed_slice(),
            martingale: 0.0,
            // Initially every register is 0, h(0) = 1/m for each, so μ = m * (1/m) = 1.
            mu: 1.0,
            martingale_invalid: false,
        }
    }

    /// Precision parameter `p`. The sketch has `2^p` registers.
    pub fn precision(&self) -> u32 {
        self.p
    }

    /// Number of registers (`m = 2^p`).
    pub fn num_registers(&self) -> usize {
        self.registers.len()
    }

    /// In-memory size of the register array, in bytes.
    pub fn register_bytes(&self) -> usize {
        self.registers.len() * 4
    }

    /// Read-only view of the register array (primarily for tests / inspection).
    pub fn registers(&self) -> &[u32] {
        &self.registers
    }

    /// Insert a 64-bit hash value (Algorithm 2 of the paper).
    pub fn add_hash(&mut self, hash: u64) {
        let p = self.p;
        let p_plus_t = p + T;

        // Register index: bits [t, t+p) of the hash.
        let i = ((hash >> T) & ((1u64 << p) - 1)) as usize;

        // a = hash with low (p+t) bits forced to 1, so nlz(a) ∈ [0, 64-p-t].
        let a = hash | ((1u64 << p_plus_t) - 1);
        let nlz_a = a.leading_zeros() as u64;

        // k = nlz(a) * 2^t + low_t + 1, ∈ [1, (65-p-t) * 2^t]
        let low_t = hash & T_MASK;
        let k = (nlz_a << T) + low_t + 1;
        debug_assert!(k >= 1);
        debug_assert!(k <= ((65 - p as u64 - T as u64) << T));
        debug_assert!(k <= u32::from(u8::MAX) as u64);

        let r = self.registers[i];
        let u = r >> D;

        let new_r = if (k as u32) > u {
            let delta = k - u as u64;
            let bitmap = (r & D_MASK) as u64;
            // floor((2^d + bitmap) / 2^Δ)
            let combined = (1u64 << D) | bitmap;
            let new_low = if delta <= u64::from(D) + 1 {
                combined >> delta
            } else {
                0
            };
            ((k as u32) << D) | (new_low as u32 & D_MASK)
        } else if (k as u32) < u {
            let neg_delta = u as u64 - k;
            if neg_delta <= u64::from(D) {
                // d + Δ ≥ 0 holds; set bit at position D - |Δ|.
                let pos = D - neg_delta as u32;
                r | (1u32 << pos)
            } else {
                r
            }
        } else {
            r
        };

        self.update_register(i, new_r);
    }

    fn update_register(&mut self, i: usize, new_r: u32) {
        let old_r = self.registers[i];
        if old_r == new_r {
            return;
        }
        if !self.martingale_invalid {
            // Algorithm 4: n_martingale += 1/μ; μ -= h(r_old) - h(r_new).
            // h is monotonically decreasing in r, so h(old) - h(new) ≥ 0.
            self.martingale += 1.0 / self.mu;
            self.mu -= h(old_r, self.p) - h(new_r, self.p);
            // Numerical safety: clamp μ to a small positive value.
            if self.mu < 1e-300 {
                self.mu = 1e-300;
            }
        }
        self.registers[i] = new_r;
    }

    /// Insert any hashable value, using the standard library default hasher.
    pub fn add<H: Hash + ?Sized>(&mut self, item: &H) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        self.add_hash(hasher.finish());
    }

    /// Best available cardinality estimate.
    ///
    /// Uses the maximum-likelihood estimator (Algorithm 3 + log-likelihood
    /// Eq. 15), which works from the register state alone and so is valid
    /// after merges and deserialization.
    pub fn estimate(&self) -> f64 {
        self.estimate_ml()
    }

    /// Maximum-likelihood estimate of the cardinality.
    ///
    /// Computes the coefficients `α` and `β_u` per Algorithm 3, then solves
    /// the ML equation `g(y) = α` by bisection in `log₂(y)` where `y = n/m`
    /// and `g(y) = Σ β_u / (2^u · (exp(y/2^u) − 1))`. The dedicated Newton's-
    /// method solver from Algorithm 8 is planned but bisection already
    /// converges to f64 precision.
    pub fn estimate_ml(&self) -> f64 {
        let (alpha, beta) = self.compute_alpha_beta();
        solve_ml(alpha, &beta, self.p)
    }

    /// Martingale (HIP) estimate, if the running state is still valid.
    ///
    /// Returns `None` after a merge or any operation that breaks the
    /// incremental state. Has slightly lower variance than ML on freshly
    /// built sketches; the paper reports up to 33% smaller MVP than HLL.
    pub fn estimate_martingale(&self) -> Option<f64> {
        if self.martingale_invalid {
            None
        } else {
            Some(self.martingale)
        }
    }

    /// Compute α and β_u coefficients of the log-likelihood (Algorithm 3).
    ///
    /// Returns α already divided by `2^(64-p)`. Per register `i` with
    /// max-update-value `u_i` and bitmap `l_1...l_d`:
    ///
    /// - α gets `ω(u_i)` for the "no update values > u_i seen" event.
    /// - α gets `1/2^φ(k)` for each `k ∈ [max(1, u_i - d), u_i - 1]` whose
    ///   bitmap bit is clear (that update value has not been hit).
    /// - β bucket `φ(u_i)` is incremented for the "u_i was the max" event.
    /// - β bucket `φ(k)` is incremented for each `k ∈ [max(1, u_i - d),
    ///   u_i - 1]` whose bitmap bit is set (k has been hit).
    ///
    /// `β` is indexed `0..(64 - p - t)` where entry `j` is the paper's
    /// `β_{j + t + 1}`.
    fn compute_alpha_beta(&self) -> (f64, Vec<u32>) {
        let p = self.p;
        let beta_len = (64 - p - T) as usize;
        let mut beta = vec![0u32; beta_len];
        let mut alpha = 0.0_f64;

        for &r in self.registers.iter() {
            let u = r >> D;
            let bitmap = r & D_MASK;

            alpha += omega(u, p);

            if u >= 1 {
                let j = phi(u, p);
                beta[(j - T - 1) as usize] += 1;

                if u >= 2 {
                    let k_lo = u.saturating_sub(D).max(1);
                    for k in k_lo..u {
                        let bit_pos = D - (u - k);
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

    /// Merge another sketch into `self` (Algorithm 5).
    ///
    /// Both sketches must have the same precision `p`.
    pub fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.p != other.p {
            return Err(MergeError::PrecisionMismatch {
                lhs: self.p,
                rhs: other.p,
            });
        }
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = merge_register(*a, *b);
        }
        // Merging breaks the running martingale state. ML estimator (TODO) can
        // recover an estimate from the registers alone.
        self.martingale_invalid = true;
        self.martingale = f64::NAN;
        self.mu = f64::NAN;
        Ok(())
    }

    /// Reset the sketch to empty.
    pub fn clear(&mut self) {
        for r in self.registers.iter_mut() {
            *r = 0;
        }
        self.martingale = 0.0;
        self.mu = 1.0;
        self.martingale_invalid = false;
    }
}

/// φ(k) := min(t + 1 + ⌊(k-1)/2^t⌋, 64 - p)  — Eq. (11) of the paper.
fn phi(k: u32, p: u32) -> u32 {
    let v = if k == 0 {
        T // ⌊-1/2^t⌋ = -1, so φ(0) = t + 1 - 1 = t
    } else {
        T + 1 + ((k - 1) >> T)
    };
    v.min(64 - p)
}

/// ω(u) := (2^t (1 - t + φ(u)) - u) / 2^φ(u)  — Eq. (14) of the paper.
fn omega(u: u32, p: u32) -> f64 {
    let phi_u = phi(u, p);
    let two_t = (1u32 << T) as f64;
    (two_t * (1.0 + phi_u as f64 - T as f64) - u as f64) * pow2_neg(phi_u)
}

/// Per-register state-change probability `h(r)` — Section 3.3.
///
/// `h(r) = (1/m) (ω(u) + Σ_{k=max(1,u-d)}^{u-1} (1 - l_{u-k}) / 2^φ(k))`
/// where `r = u·2^d + ⟨l_1 ... l_d⟩₂` (the `l_j` are most-significant-first
/// inside the d-bit bitmap).
fn h(r: u32, p: u32) -> f64 {
    let u = r >> D;
    let bitmap = r & D_MASK;
    let m = (1u64 << p) as f64;

    let mut acc = omega(u, p);
    if u >= 2 {
        let k_lo = u.saturating_sub(D).max(1);
        for k in k_lo..u {
            let j = u - k; // j ∈ [1, d]
            let l_j = (bitmap >> (D - j)) & 1;
            acc += (1.0 - l_j as f64) * pow2_neg(phi(k, p));
        }
    }
    acc / m
}

/// 2^(-x), branchless and fast for small x. Avoids `f64::powi` overhead.
#[inline]
fn pow2_neg(x: u32) -> f64 {
    f64::from_bits((1023u64.wrapping_sub(x as u64)) << 52)
}

/// `g(y) = Σ_u β_u / (2^u · (exp(y/2^u) − 1))` — left-hand side of the ML
/// equation `g(y) = α` (derivative of Eq. 15 set to zero).
///
/// Monotonically decreasing in `y` from `+∞` (as `y → 0⁺`) to `0`
/// (as `y → ∞`), so the equation has at most one root in `(0, ∞)`.
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
            // For huge y/2^u, exp_m1 → +inf and the term → 0. For very tiny y
            // (denom underflowed to 0), the term is effectively +∞, but we
            // bracket the search to keep y away from that regime.
            continue;
        }
        sum += b as f64 * scale / denom;
    }
    sum
}

/// Solve the ML equation `g(y) = α` for `y = n/m` by bisection in `log₂(y)`.
///
/// Returns the cardinality estimate `n = m · y`. Returns `0.0` for an empty
/// sketch (all `β_u` zero).
fn solve_ml(alpha: f64, beta: &[u32], p: u32) -> f64 {
    if beta.iter().all(|&b| b == 0) {
        return 0.0;
    }
    if alpha <= 0.0 {
        return f64::INFINITY;
    }

    // log₂(y) bracket. Wide enough for any practical cardinality and within
    // the safe range for f64 exponentiation. The actual ML root lies in
    // [-log₂(m), log₂(2^64)] in practical use; we bracket much wider for
    // safety and let bisection converge.
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

/// Merge two register values from sketches with the same `t, d, p`
/// (Algorithm 5 of the paper).
fn merge_register(r: u32, r2: u32) -> u32 {
    let u = r >> D;
    let u2 = r2 >> D;

    if u > u2 && u2 > 0 {
        let bitmap2 = (r2 & D_MASK) as u64;
        let combined = (1u64 << D) | bitmap2;
        let shift = u - u2;
        let extra = if shift <= D + 1 { combined >> shift } else { 0 };
        r | (extra as u32 & D_MASK)
    } else if u2 > u && u > 0 {
        let bitmap = (r & D_MASK) as u64;
        let combined = (1u64 << D) | bitmap;
        let shift = u2 - u;
        let extra = if shift <= D + 1 { combined >> shift } else { 0 };
        r2 | (extra as u32 & D_MASK)
    } else {
        // u == u2, or one register is zero; bitwise OR is correct.
        r | r2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // ω(0) = sum of all ρ_update(k) for k ≥ 1 = 1.
        for p in MIN_P..=18 {
            let w = omega(0, p);
            assert!((w - 1.0).abs() < 1e-12, "ω(0) for p={p} = {w}");
        }
    }

    #[test]
    fn h_of_zero_register_is_one_over_m() {
        for p in MIN_P..=18 {
            let m = (1u64 << p) as f64;
            let value = h(0, p);
            let expected = 1.0 / m;
            assert!(
                (value - expected).abs() < 1e-15,
                "h(0) for p={p}: {value} vs {expected}"
            );
        }
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let s = ExaLogLog::new(12);
        assert_eq!(s.estimate(), 0.0);
    }

    #[test]
    fn idempotent_inserts_do_not_change_state() {
        let mut s = ExaLogLog::new(12);
        for _ in 0..1000 {
            s.add_hash(0xDEAD_BEEF_CAFE_BABE);
        }
        let registers_changed = s.registers().iter().filter(|&&r| r != 0).count();
        assert_eq!(registers_changed, 1);
        let est_after = s.estimate();
        // After many idempotent inserts, the running estimate equals the
        // estimate after a single insert.
        assert!(est_after > 0.0 && est_after < 5.0, "estimate = {est_after}");
    }

    #[test]
    fn distinct_inserts_grow_estimate() {
        let mut s = ExaLogLog::new(12);
        for i in 0..1000u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate();
        assert!(
            (est - 1000.0).abs() / 1000.0 < 0.10,
            "n=1000, estimate={est}"
        );
    }

    #[test]
    fn estimate_within_error_bounds_at_various_sizes() {
        // For ELL(t=2, d=24) with martingale, theoretical RMSE ≈ sqrt(MVP / m_bits)
        // = sqrt(3.78 / (32 * 2^p)) (using martingale MVP = 2.77, but conservatively
        // we just check that error stays under a generous bound).
        let p = 12;
        for &n in &[100u64, 1000, 10_000, 100_000] {
            let mut s = ExaLogLog::new(p);
            for i in 0..n {
                s.add_hash(splitmix64(i));
            }
            let est = s.estimate();
            let rel_err = (est - n as f64).abs() / n as f64;
            assert!(rel_err < 0.05, "n={n}, est={est}, rel_err={rel_err}");
        }
    }

    #[test]
    fn merge_of_disjoint_sets_estimates_union() {
        let p = 12;
        let mut a = ExaLogLog::new(p);
        let mut b = ExaLogLog::new(p);
        let mut combined = ExaLogLog::new(p);

        for i in 0..50_000u64 {
            a.add_hash(splitmix64(i));
            combined.add_hash(splitmix64(i));
        }
        for i in 50_000..100_000u64 {
            b.add_hash(splitmix64(i));
            combined.add_hash(splitmix64(i));
        }

        a.merge(&b).unwrap();

        // Martingale state is invalidated by merge.
        assert_eq!(a.estimate_martingale(), None);

        // Register state must match a single sketch of the union.
        assert_eq!(a.registers(), combined.registers());

        // The ML estimator works from registers alone and recovers a good
        // estimate of the union cardinality.
        let est = a.estimate();
        let rel_err = (est - 100_000.0).abs() / 100_000.0;
        assert!(
            rel_err < 0.05,
            "post-merge ML estimate = {est}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn ml_estimate_within_error_bounds() {
        // ML works from register state alone; verify it across a few
        // cardinalities. Theoretical RMSE for ELL(2, 24) at p=12 is
        // sqrt(MVP / total_bits) = sqrt(3.78 / (32 · 4096)) ≈ 0.54%.
        // Allow 5% headroom from a single random sample.
        let p = 12;
        for &n in &[100u64, 1_000, 10_000, 100_000, 1_000_000] {
            let mut s = ExaLogLog::new(p);
            for i in 0..n {
                s.add_hash(splitmix64(i));
            }
            let est = s.estimate_ml();
            let rel_err = (est - n as f64).abs() / n as f64;
            assert!(
                rel_err < 0.05,
                "ML at n={n}: est={est}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn ml_and_martingale_agree_on_fresh_sketch() {
        // Both estimators should give similar values on a sketch built by
        // streaming inserts (no merge). They use different statistics and
        // both target the true cardinality.
        let p = 12;
        let n = 50_000u64;
        let mut s = ExaLogLog::new(p);
        for i in 0..n {
            s.add_hash(splitmix64(i));
        }
        let mart = s.estimate_martingale().unwrap();
        let ml = s.estimate_ml();
        let rel_diff = (mart - ml).abs() / n as f64;
        assert!(
            rel_diff < 0.02,
            "ML vs martingale disagree: ml={ml}, mart={mart}"
        );
    }

    #[test]
    fn ml_estimate_zero_on_empty_sketch() {
        for p in [3u32, 8, 12, 18] {
            let s = ExaLogLog::new(p);
            assert_eq!(s.estimate_ml(), 0.0);
        }
    }

    #[test]
    fn merge_idempotent_with_self() {
        let p = 10;
        let mut a = ExaLogLog::new(p);
        for i in 0..10_000u64 {
            a.add_hash(splitmix64(i));
        }
        let regs_before = a.registers().to_vec();
        let snapshot = a.clone();
        a.merge(&snapshot).unwrap();
        assert_eq!(a.registers(), regs_before.as_slice());
    }

    #[test]
    fn merge_precision_mismatch_errors() {
        let mut a = ExaLogLog::new(10);
        let b = ExaLogLog::new(11);
        let err = a.merge(&b).unwrap_err();
        assert_eq!(err, MergeError::PrecisionMismatch { lhs: 10, rhs: 11 });
    }

    #[test]
    fn h_strictly_decreases_on_real_state_change() {
        // For every register transition the insert algorithm produces, h must
        // strictly decrease. This is the property the martingale estimator
        // depends on.
        let p = 10;
        let mut s = ExaLogLog::new(p);
        for i in 0..200_000u64 {
            let r_before = s.registers().to_vec();
            s.add_hash(splitmix64(i));
            for (j, (&old_r, &new_r)) in
                r_before.iter().zip(s.registers().iter()).enumerate()
            {
                if old_r != new_r {
                    let h_old = h(old_r, p);
                    let h_new = h(new_r, p);
                    assert!(
                        h_new < h_old,
                        "h did not decrease at register {j}: \
                         r {old_r:#010x} → {new_r:#010x}, h {h_old} → {h_new}"
                    );
                }
            }
        }
    }

    #[test]
    fn pow2_neg_matches_powi() {
        for x in 0u32..64 {
            let fast = pow2_neg(x);
            let reference = 2.0_f64.powi(-(x as i32));
            assert!(
                (fast - reference).abs() < 1e-300 || (fast / reference - 1.0).abs() < 1e-15,
                "pow2_neg({x}) = {fast}, expected {reference}"
            );
        }
    }

    /// SplitMix64 — a simple high-quality 64-bit hash, used to spread sequential
    /// `u64` keys uniformly. Avoids depending on the order-sensitive default
    /// hasher inside numerical tests.
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
}
