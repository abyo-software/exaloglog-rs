//! `ExaLogLogFast`: 32-bit-aligned ExaLogLog (`ELL(t=2, d=24)`).
//!
//! Each register occupies exactly 32 bits, so register access is a single
//! aligned `u32` load. MVP = 3.78, ~41% smaller than HLL with 6-bit
//! registers. Slightly worse memory efficiency than [`crate::ExaLogLog`]
//! (the packed `d=20` variant, MVP = 3.67), but easier to make concurrent
//! and SIMD-friendly because each register is its own cache-aligned word.
//!
//! Use this variant when:
//!
//! - You need atomic / CAS-based concurrent updates per register.
//! - You're willing to trade ~3% extra memory for fast scalar access.
//!
//! Use [`crate::ExaLogLog`] otherwise.

use std::hash::{DefaultHasher, Hash, Hasher};

use crate::math;
use crate::{DeserializeError, FORMAT_VERSION, MAGIC, MergeError};
use crate::{MAX_P, MIN_P, T};

const D: u32 = 24;
const HEADER_LEN: usize = 8;

/// 32-bit aligned ExaLogLog. See module docs.
#[derive(Clone, Debug)]
pub struct ExaLogLogFast {
    p: u32,
    registers: Box<[u32]>,
    martingale: f64,
    mu: f64,
    martingale_invalid: bool,
}

impl ExaLogLogFast {
    /// Create an empty sketch with `2^p` registers.
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
            mu: 1.0,
            martingale_invalid: false,
        }
    }

    /// Precision parameter.
    pub fn precision(&self) -> u32 {
        self.p
    }

    /// Number of registers (`2^p`).
    pub fn num_registers(&self) -> usize {
        self.registers.len()
    }

    /// In-memory size of the register array in bytes.
    pub fn register_bytes(&self) -> usize {
        self.registers.len() * 4
    }

    /// Read-only view of the register array.
    pub fn registers(&self) -> &[u32] {
        &self.registers
    }

    /// `d` parameter (24).
    pub fn d_parameter() -> u32 {
        D
    }

    /// Insert a 64-bit hash value (Algorithm 2).
    pub fn add_hash(&mut self, hash: u64) {
        let (i, k) = math::hash_to_register_k(hash, self.p);
        let r = self.registers[i];
        let new_r = math::apply_insert(r, k, D);
        self.update_register(i, new_r);
    }

    fn update_register(&mut self, i: usize, new_r: u32) {
        let old_r = self.registers[i];
        if old_r == new_r {
            return;
        }
        if !self.martingale_invalid {
            self.martingale += 1.0 / self.mu;
            self.mu -= math::h(old_r, self.p, D) - math::h(new_r, self.p, D);
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

    /// Best available cardinality estimate (ML estimator).
    pub fn estimate(&self) -> f64 {
        self.estimate_ml()
    }

    /// Maximum-likelihood estimate.
    pub fn estimate_ml(&self) -> f64 {
        let (alpha, beta) = math::compute_alpha_beta(self.registers.iter().copied(), self.p, D);
        math::solve_ml(alpha, &beta, self.p)
    }

    /// Martingale (HIP) estimate, if the running state is still valid.
    pub fn estimate_martingale(&self) -> Option<f64> {
        if self.martingale_invalid {
            None
        } else {
            Some(self.martingale)
        }
    }

    /// Merge another sketch into `self` (Algorithm 5).
    pub fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.p != other.p {
            return Err(MergeError::PrecisionMismatch {
                lhs: self.p,
                rhs: other.p,
            });
        }
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = math::merge_register(*a, *b, D);
        }
        self.martingale_invalid = true;
        self.martingale = f64::NAN;
        self.mu = f64::NAN;
        Ok(())
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        for r in self.registers.iter_mut() {
            *r = 0;
        }
        self.martingale = 0.0;
        self.mu = 1.0;
        self.martingale_invalid = false;
    }

    /// Serialize. See module docs for layout.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.registers.len() * 4);
        out.extend_from_slice(&MAGIC);
        out.push(FORMAT_VERSION);
        out.push(T as u8);
        out.push(D as u8);
        out.push(self.p as u8);
        for &r in self.registers.iter() {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out
    }

    /// Deserialize from a byte slice produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        if bytes.len() < HEADER_LEN {
            return Err(DeserializeError::TooShort {
                got: bytes.len(),
                need: HEADER_LEN,
            });
        }
        if bytes[0..4] != MAGIC {
            return Err(DeserializeError::BadMagic);
        }
        if bytes[4] != FORMAT_VERSION {
            return Err(DeserializeError::UnsupportedVersion(bytes[4]));
        }
        let t = bytes[5];
        let d = bytes[6];
        if u32::from(t) != T || u32::from(d) != D {
            return Err(DeserializeError::ParameterMismatch { t, d });
        }
        let p = bytes[7];
        if !(MIN_P..=MAX_P).contains(&u32::from(p)) {
            return Err(DeserializeError::InvalidPrecision(p));
        }
        let m = 1usize << p;
        let expected_len = HEADER_LEN + m * 4;
        if bytes.len() != expected_len {
            return Err(DeserializeError::LengthMismatch {
                got: bytes.len(),
                expected: expected_len,
            });
        }

        let mut registers = vec![0u32; m].into_boxed_slice();
        for (i, r) in registers.iter_mut().enumerate() {
            let off = HEADER_LEN + i * 4;
            *r = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        }

        Ok(Self {
            p: u32::from(p),
            registers,
            martingale: f64::NAN,
            mu: f64::NAN,
            martingale_invalid: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::h;

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let s = ExaLogLogFast::new(12);
        assert_eq!(s.estimate(), 0.0);
    }

    #[test]
    fn idempotent_inserts_do_not_change_state() {
        let mut s = ExaLogLogFast::new(12);
        for _ in 0..1000 {
            s.add_hash(0xDEAD_BEEF_CAFE_BABE);
        }
        let changed = s.registers().iter().filter(|&&r| r != 0).count();
        assert_eq!(changed, 1);
    }

    #[test]
    fn h_strictly_decreases_on_real_state_change() {
        let p = 10;
        let mut s = ExaLogLogFast::new(p);
        for i in 0..200_000u64 {
            let r_before = s.registers().to_vec();
            s.add_hash(splitmix64(i));
            for (j, (&old_r, &new_r)) in r_before.iter().zip(s.registers().iter()).enumerate() {
                if old_r != new_r {
                    let h_old = h(old_r, p, D);
                    let h_new = h(new_r, p, D);
                    assert!(h_new < h_old, "register {j}: h {h_old} → {h_new}");
                }
            }
        }
    }

    #[test]
    fn ml_estimate_within_error_bounds() {
        let p = 12;
        for &n in &[100u64, 1_000, 10_000, 100_000, 1_000_000] {
            let mut s = ExaLogLogFast::new(p);
            for i in 0..n {
                s.add_hash(splitmix64(i));
            }
            let est = s.estimate_ml();
            let rel_err = (est - n as f64).abs() / n as f64;
            assert!(rel_err < 0.05, "n={n}: est={est}, rel_err={rel_err}");
        }
    }

    #[test]
    fn ml_and_martingale_agree() {
        let p = 12;
        let n = 50_000u64;
        let mut s = ExaLogLogFast::new(p);
        for i in 0..n {
            s.add_hash(splitmix64(i));
        }
        let mart = s.estimate_martingale().unwrap();
        let ml = s.estimate_ml();
        let rel_diff = (mart - ml).abs() / n as f64;
        assert!(rel_diff < 0.02);
    }

    #[test]
    fn merge_disjoint_recovers_union() {
        let p = 12;
        let mut a = ExaLogLogFast::new(p);
        let mut b = ExaLogLogFast::new(p);
        let mut combined = ExaLogLogFast::new(p);
        for i in 0..50_000u64 {
            a.add_hash(splitmix64(i));
            combined.add_hash(splitmix64(i));
        }
        for i in 50_000..100_000u64 {
            b.add_hash(splitmix64(i));
            combined.add_hash(splitmix64(i));
        }
        a.merge(&b).unwrap();
        assert_eq!(a.registers(), combined.registers());
        assert_eq!(a.estimate_martingale(), None);
        let est = a.estimate();
        let rel_err = (est - 100_000.0).abs() / 100_000.0;
        assert!(rel_err < 0.05, "post-merge estimate = {est}");
    }

    #[test]
    fn merge_precision_mismatch() {
        let mut a = ExaLogLogFast::new(10);
        let b = ExaLogLogFast::new(11);
        assert_eq!(
            a.merge(&b),
            Err(MergeError::PrecisionMismatch { lhs: 10, rhs: 11 })
        );
    }

    #[test]
    fn serialize_roundtrip() {
        let p = 12;
        let mut s = ExaLogLogFast::new(p);
        for i in 0..50_000u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let bytes = s.to_bytes();
        assert_eq!(bytes.len(), 8 + 4 * (1 << p));
        let restored = ExaLogLogFast::from_bytes(&bytes).unwrap();
        assert_eq!(restored.registers(), s.registers());
        assert_eq!(restored.estimate_martingale(), None);
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }
}
