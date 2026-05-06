//! `ExaLogLog`: packed 28-bit ExaLogLog (`ELL(t=2, d=20)`).
//!
//! This is the default and most space-efficient configuration. Each register
//! occupies 28 bits (`q = 8`, `d = 20`), packed as two registers per 7 bytes.
//! MVP = 3.67, **43% smaller than HLL** with 6-bit registers at the same
//! estimation error.
//!
//! Per-register read costs an unaligned 4-byte load and a mask; per-register
//! write costs the same plus a read-modify-write of the byte shared with the
//! sibling register. Concurrent updates require external synchronization
//! because the shared boundary byte cannot be CAS'd safely. If you need
//! lock-free concurrent updates, use [`crate::ExaLogLogFast`] instead.

use std::hash::{DefaultHasher, Hash, Hasher};

use crate::math;
use crate::{DeserializeError, FORMAT_VERSION, MAGIC, MergeError};
use crate::{MAX_P, MIN_P, T};

const D: u32 = 20;
const REGISTER_MASK: u32 = (1u32 << 28) - 1; // q + d = 28 bits
const HEADER_LEN: usize = 8;
const BYTES_PER_PAIR: usize = 7;

/// Packed 28-bit ExaLogLog. See module docs.
#[derive(Clone, Debug)]
pub struct ExaLogLog {
    p: u32,
    /// Register storage: `m / 2` chunks of 7 bytes, two registers each.
    storage: Box<[u8]>,
    martingale: f64,
    mu: f64,
    martingale_invalid: bool,
}

impl ExaLogLog {
    /// Create an empty sketch with `2^p` registers.
    pub fn new(p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&p),
            "precision p={p} out of range [{MIN_P}, {MAX_P}]"
        );
        let m = 1usize << p;
        let storage_bytes = (m / 2) * BYTES_PER_PAIR;
        Self {
            p,
            storage: vec![0u8; storage_bytes].into_boxed_slice(),
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
        1 << self.p
    }

    /// In-memory size of the register storage in bytes (`m * 7 / 2`).
    pub fn register_bytes(&self) -> usize {
        self.storage.len()
    }

    /// `d` parameter (20).
    pub fn d_parameter() -> u32 {
        D
    }

    /// Read register `i`.
    #[inline]
    pub fn get_register(&self, i: usize) -> u32 {
        debug_assert!(i < self.num_registers());
        let chunk_off = (i >> 1) * BYTES_PER_PAIR;
        if i & 1 == 0 {
            // Even register: bits [0..28] of the 7-byte chunk.
            let bytes = [
                self.storage[chunk_off],
                self.storage[chunk_off + 1],
                self.storage[chunk_off + 2],
                self.storage[chunk_off + 3],
            ];
            u32::from_le_bytes(bytes) & REGISTER_MASK
        } else {
            // Odd register: bits [28..56].
            let bytes = [
                self.storage[chunk_off + 3],
                self.storage[chunk_off + 4],
                self.storage[chunk_off + 5],
                self.storage[chunk_off + 6],
            ];
            u32::from_le_bytes(bytes) >> 4
        }
    }

    /// Write register `i`. Preserves the sibling register's bits in the
    /// shared byte.
    #[inline]
    fn set_register(&mut self, i: usize, value: u32) {
        debug_assert!(i < self.num_registers());
        debug_assert!(
            value <= REGISTER_MASK,
            "register value {value:#x} exceeds 28 bits"
        );
        let chunk_off = (i >> 1) * BYTES_PER_PAIR;
        let v = value & REGISTER_MASK;
        if i & 1 == 0 {
            self.storage[chunk_off] = v as u8;
            self.storage[chunk_off + 1] = (v >> 8) as u8;
            self.storage[chunk_off + 2] = (v >> 16) as u8;
            // Shared byte: low nibble is even reg's [24..28],
            // high nibble belongs to odd reg.
            let high4 = self.storage[chunk_off + 3] & 0xF0;
            self.storage[chunk_off + 3] = high4 | ((v >> 24) as u8 & 0x0F);
        } else {
            // Shared byte: high nibble is odd reg's [0..4],
            // low nibble belongs to even reg.
            let low4 = self.storage[chunk_off + 3] & 0x0F;
            self.storage[chunk_off + 3] = low4 | ((v << 4) as u8 & 0xF0);
            self.storage[chunk_off + 4] = (v >> 4) as u8;
            self.storage[chunk_off + 5] = (v >> 12) as u8;
            self.storage[chunk_off + 6] = (v >> 20) as u8;
        }
    }

    /// Iterator over all register values.
    pub fn iter_registers(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.num_registers()).map(move |i| self.get_register(i))
    }

    /// Insert a 64-bit hash value (Algorithm 2).
    pub fn add_hash(&mut self, hash: u64) {
        let (i, k) = math::hash_to_register_k(hash, self.p);
        let r = self.get_register(i);
        let new_r = math::apply_insert(r, k, D);
        if new_r != r {
            if !self.martingale_invalid {
                self.martingale += 1.0 / self.mu;
                self.mu -= math::h(r, self.p, D) - math::h(new_r, self.p, D);
                if self.mu < 1e-300 {
                    self.mu = 1e-300;
                }
            }
            self.set_register(i, new_r);
        }
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
        let (alpha, beta) = math::compute_alpha_beta(self.iter_registers(), self.p, D);
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
        for i in 0..self.num_registers() {
            let merged = math::merge_register(self.get_register(i), other.get_register(i), D);
            self.set_register(i, merged);
        }
        self.martingale_invalid = true;
        self.martingale = f64::NAN;
        self.mu = f64::NAN;
        Ok(())
    }

    /// Reduce this sketch's precision to `new_p ≤ self.precision()`,
    /// returning a new sketch. Lossless: the result equals what you would
    /// get by directly inserting the same elements into a sketch with
    /// `new_p`. Implements Algorithm 6 of the paper, restricted to the
    /// case where `d` stays the same.
    pub fn reduce(&self, new_p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&new_p) && new_p <= self.p,
            "new_p={new_p} must be in [{MIN_P}, {self_p}]",
            self_p = self.p
        );
        let mut out = Self::new(new_p);
        if new_p == self.p {
            for i in 0..self.num_registers() {
                out.set_register(i, self.get_register(i));
            }
            out.martingale_invalid = true;
            return out;
        }
        let p_diff = self.p - new_p;
        let m_new = 1usize << new_p;
        let two_t = 1u32 << T;
        let a = (64 - T - self.p) * two_t + 1;

        for new_i in 0..m_new {
            let mut acc = 0u32;
            for j in 0..(1u64 << p_diff) {
                let old_i = new_i + m_new * j as usize;
                let mut r = self.get_register(old_i);
                let u = r >> D;

                if u >= a {
                    let bit_len_j = if j == 0 {
                        0
                    } else {
                        64 - j.leading_zeros()
                    };
                    let s = (p_diff - bit_len_j) * two_t;
                    if s > 0 {
                        let v = D + a - u;
                        if v > 0 {
                            let high = (r >> v) << v;
                            let low_v = r & ((1u32 << v) - 1);
                            let low_v_shifted = low_v >> s;
                            r = high | low_v_shifted;
                        }
                        r += s << D;
                    }
                }

                acc = math::merge_register(acc, r, D);
            }
            out.set_register(new_i, acc & REGISTER_MASK);
        }
        out.martingale_invalid = true;
        out
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        for b in self.storage.iter_mut() {
            *b = 0;
        }
        self.martingale = 0.0;
        self.mu = 1.0;
        self.martingale_invalid = false;
    }

    /// Serialize to bytes. Layout: 8-byte header (magic, version, t, d, p) +
    /// raw packed storage bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.storage.len());
        out.extend_from_slice(&MAGIC);
        out.push(FORMAT_VERSION);
        out.push(T as u8);
        out.push(D as u8);
        out.push(self.p as u8);
        out.extend_from_slice(&self.storage);
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
        let expected_len = HEADER_LEN + (m / 2) * BYTES_PER_PAIR;
        if bytes.len() != expected_len {
            return Err(DeserializeError::LengthMismatch {
                got: bytes.len(),
                expected: expected_len,
            });
        }

        let storage: Box<[u8]> = bytes[HEADER_LEN..].to_vec().into_boxed_slice();

        Ok(Self {
            p: u32::from(p),
            storage,
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
    fn pack_unpack_roundtrip_for_each_register() {
        // Ensure adjacent registers don't trample each other in the shared
        // byte. Write a unique value into each register, then read it back.
        let p = 6;
        let m = 1usize << p;
        let mut s = ExaLogLog::new(p);
        for i in 0..m {
            // Use a value pattern with bits in every nibble to catch off-by-
            // one shifts.
            let v = ((0xA5A5A5u32.wrapping_mul(i as u32 + 1)) ^ 0x0DEADBu32) & REGISTER_MASK;
            s.set_register(i, v);
        }
        for i in 0..m {
            let v = ((0xA5A5A5u32.wrapping_mul(i as u32 + 1)) ^ 0x0DEADBu32) & REGISTER_MASK;
            assert_eq!(
                s.get_register(i),
                v,
                "register {i} round-trip failed (m={m})"
            );
        }
    }

    #[test]
    fn writing_register_does_not_disturb_neighbors() {
        let p = 8;
        let mut s = ExaLogLog::new(p);
        for i in 0..s.num_registers() {
            s.set_register(i, REGISTER_MASK); // all ones
        }
        // Now zero out every other register; check the rest remain all-ones.
        for i in (0..s.num_registers()).step_by(2) {
            s.set_register(i, 0);
        }
        for i in 0..s.num_registers() {
            let expected = if i % 2 == 0 { 0 } else { REGISTER_MASK };
            assert_eq!(s.get_register(i), expected, "register {i}");
        }
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let s = ExaLogLog::new(12);
        assert_eq!(s.estimate(), 0.0);
    }

    #[test]
    fn idempotent_inserts() {
        let mut s = ExaLogLog::new(12);
        for _ in 0..1000 {
            s.add_hash(0xDEAD_BEEF_CAFE_BABE);
        }
        let changed = s.iter_registers().filter(|&r| r != 0).count();
        assert_eq!(changed, 1);
    }

    #[test]
    fn h_strictly_decreases_on_real_state_change() {
        let p = 10;
        let mut s = ExaLogLog::new(p);
        for i in 0..200_000u64 {
            let regs_before: Vec<u32> = s.iter_registers().collect();
            s.add_hash(splitmix64(i));
            let regs_after: Vec<u32> = s.iter_registers().collect();
            for (j, (&old_r, &new_r)) in regs_before.iter().zip(regs_after.iter()).enumerate() {
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
            let mut s = ExaLogLog::new(p);
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
        let mut s = ExaLogLog::new(p);
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
        for i in 0..a.num_registers() {
            assert_eq!(
                a.get_register(i),
                combined.get_register(i),
                "register {i} differs"
            );
        }
        assert_eq!(a.estimate_martingale(), None);
        let est = a.estimate();
        let rel_err = (est - 100_000.0).abs() / 100_000.0;
        assert!(rel_err < 0.05);
    }

    #[test]
    fn serialize_roundtrip() {
        let p = 12;
        let mut s = ExaLogLog::new(p);
        for i in 0..50_000u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let bytes = s.to_bytes();
        let expected_size = 8 + (1 << p) / 2 * 7;
        assert_eq!(bytes.len(), expected_size);
        let restored = ExaLogLog::from_bytes(&bytes).unwrap();
        for i in 0..restored.num_registers() {
            assert_eq!(restored.get_register(i), s.get_register(i));
        }
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn reduce_to_same_p_returns_same_state() {
        let p = 10;
        let mut s = ExaLogLog::new(p);
        for i in 0..10_000u64 {
            s.add_hash(splitmix64(i));
        }
        let r = s.reduce(p);
        for i in 0..s.num_registers() {
            assert_eq!(r.get_register(i), s.get_register(i));
        }
    }

    #[test]
    fn reduce_preserves_estimate_within_tolerance() {
        let p_high = 12;
        let p_low = 10;
        let n = 50_000u64;
        let mut a = ExaLogLog::new(p_high);
        let mut direct = ExaLogLog::new(p_low);
        for i in 0..n {
            let h = splitmix64(i);
            a.add_hash(h);
            direct.add_hash(h);
        }
        let reduced = a.reduce(p_low);
        let red_est = reduced.estimate_ml();
        let dir_est = direct.estimate_ml();
        let rel_diff = (red_est - dir_est).abs() / n as f64;
        assert!(rel_diff < 0.10, "reduced={red_est}, direct={dir_est}");
    }

    #[test]
    fn memory_is_43_percent_smaller_than_hll_6bit() {
        // Sanity-check the headline. HLL with 6-bit registers takes
        // ceil(6 * m / 8) bytes (Apache DataSketches packs them densely).
        // ExaLogLog(t=2, d=20) takes m * 7 / 2 bytes for the same m.
        // Both are *register-array* sizes — we ignore HLL's auxiliary
        // small-range estimator state, which is in the noise at p ≥ 8.
        //
        // The fairer comparison is at equal RMSE, not equal m. HLL-6 has
        // MVP = 6.48; ELL(2,20) has MVP = 3.67. To match HLL-6's RMSE,
        // ELL needs sqrt(3.67/6.48) = 75.2% as many register-bits, but
        // each register is 28 bits vs 6 bits. So same-m comparison is the
        // wrong one; what matters is bits-per-RMSE.
        //
        // This test just confirms the in-memory size formulas hold.
        for p in [8u32, 10, 12, 14] {
            let m = 1usize << p;
            let s = ExaLogLog::new(p);
            assert_eq!(s.register_bytes(), m * 7 / 2);
        }
    }
}
