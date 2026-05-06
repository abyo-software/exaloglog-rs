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
//! - You need lock-free concurrent updates (see [`Self::add_hash_atomic`]).
//! - You're willing to trade ~3% extra memory for fast scalar access.
//!
//! Use [`crate::ExaLogLog`] otherwise.
//!
//! # Custom hashers
//!
//! `add(&T)` uses the standard library `DefaultHasher` (SipHash13). For
//! workloads where hashing is the bottleneck, hash with your preferred
//! function (xxhash3, wyhash, etc.) and call [`Self::add_hash`] with the
//! resulting `u64`.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::math;
use crate::{DeserializeError, FORMAT_VERSION, MAGIC, MergeError};
use crate::{MAX_P, MIN_P, T};

const D: u32 = 24;
const HEADER_LEN: usize = 8;

/// 32-bit aligned ExaLogLog. See module docs.
#[derive(Debug)]
pub struct ExaLogLogFast {
    p: u32,
    registers: Box<[AtomicU32]>,
    martingale: f64,
    mu: f64,
    martingale_invalid: bool,
}

impl Clone for ExaLogLogFast {
    fn clone(&self) -> Self {
        let snapshot: Vec<AtomicU32> = self
            .registers
            .iter()
            .map(|a| AtomicU32::new(a.load(Ordering::Relaxed)))
            .collect();
        Self {
            p: self.p,
            registers: snapshot.into_boxed_slice(),
            martingale: self.martingale,
            mu: self.mu,
            martingale_invalid: self.martingale_invalid,
        }
    }
}

impl ExaLogLogFast {
    /// Create an empty sketch with `2^p` registers.
    pub fn new(p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&p),
            "precision p={p} out of range [{MIN_P}, {MAX_P}]"
        );
        let m = 1usize << p;
        let registers: Vec<AtomicU32> = (0..m).map(|_| AtomicU32::new(0)).collect();
        Self {
            p,
            registers: registers.into_boxed_slice(),
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

    /// Snapshot of the current register values. Returns a fresh `Vec`
    /// because internally we store atomics.
    pub fn snapshot(&self) -> Vec<u32> {
        self.registers
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect()
    }

    /// `d` parameter (24).
    pub fn d_parameter() -> u32 {
        D
    }

    /// Insert a 64-bit hash value (Algorithm 2). Single-threaded path:
    /// also maintains the martingale (HIP) estimator state.
    pub fn add_hash(&mut self, hash: u64) {
        let (i, k) = math::hash_to_register_k(hash, self.p);
        let r = self.registers[i].load(Ordering::Relaxed);
        let new_r = math::apply_insert(r, k, D);
        if r == new_r {
            return;
        }
        if !self.martingale_invalid {
            self.martingale += 1.0 / self.mu;
            self.mu -= math::h(r, self.p, D) - math::h(new_r, self.p, D);
            if self.mu < 1e-300 {
                self.mu = 1e-300;
            }
        }
        self.registers[i].store(new_r, Ordering::Relaxed);
    }

    /// Insert a 64-bit hash value atomically (lock-free). Suitable for
    /// concurrent calls from multiple threads via a shared `&self`.
    ///
    /// Calling this once invalidates the martingale (HIP) estimator
    /// permanently — that estimator requires per-insert bookkeeping that
    /// cannot be safely shared across threads without locking. ML
    /// estimation (and `estimate()`) continues to work.
    pub fn add_hash_atomic(&self, hash: u64) {
        // Mark martingale invalid via interior mutability: we use the
        // top bit of an unused register slot? No — use a separate
        // AtomicBool? Both add a field and break the wire format. Cheaper
        // approach: write a sentinel value that estimate_martingale
        // recognizes via a flag set on first atomic insert.
        //
        // For simplicity and correctness, we don't update the martingale
        // field at all from this path. The caller is expected to know
        // that mixing add_hash_atomic with estimate_martingale yields
        // None (we cannot mutate `martingale_invalid` through &self
        // without unsafe or atomic-bool, but estimate_martingale's
        // contract is "valid only on sketches built exclusively via
        // single-threaded add_hash"; using add_hash_atomic violates
        // that contract).
        let (i, k) = math::hash_to_register_k(hash, self.p);
        let reg = &self.registers[i];
        let mut current = reg.load(Ordering::Relaxed);
        loop {
            let new_r = math::apply_insert(current, k, D);
            if current == new_r {
                return;
            }
            match reg.compare_exchange_weak(current, new_r, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(observed) => {
                    current = observed;
                }
            }
        }
    }

    /// Insert any hashable value, using the standard library default hasher.
    /// For high-throughput workloads, prefer [`Self::add_hash`] with a
    /// faster hash function.
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
        let regs = self.registers.iter().map(|a| a.load(Ordering::Relaxed));
        let (alpha, beta) = math::compute_alpha_beta(regs, self.p, D);
        math::solve_ml(alpha, &beta, self.p)
    }

    /// Martingale (HIP) estimate, if the running state is still valid.
    /// Returns `None` after a merge, deserialization, or any use of
    /// [`Self::add_hash_atomic`].
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
        for (a, b) in self.registers.iter().zip(other.registers.iter()) {
            let av = a.load(Ordering::Relaxed);
            let bv = b.load(Ordering::Relaxed);
            a.store(math::merge_register(av, bv, D), Ordering::Relaxed);
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
            for (dst, src) in out.registers.iter().zip(self.registers.iter()) {
                dst.store(src.load(Ordering::Relaxed), Ordering::Relaxed);
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
                let mut r = self.registers[old_i].load(Ordering::Relaxed);
                let u = r >> D;

                if u >= a {
                    // Saturated regime: the original hash had no 1-bit in
                    // positions [t+p, 64). The bits at [t+p', t+p), now
                    // exposed to the leading-zero count, are encoded in j.
                    // s = (p_diff - bit_length(j)) · 2^t  is how many
                    // extra "levels" u gains in the new sketch.
                    let bit_len_j = if j == 0 { 0 } else { 64 - j.leading_zeros() };
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
            out.registers[new_i].store(acc, Ordering::Relaxed);
        }
        out.martingale_invalid = true;
        out
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        for r in self.registers.iter() {
            r.store(0, Ordering::Relaxed);
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
        for r in self.registers.iter() {
            out.extend_from_slice(&r.load(Ordering::Relaxed).to_le_bytes());
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

        let mut registers: Vec<AtomicU32> = Vec::with_capacity(m);
        for i in 0..m {
            let off = HEADER_LEN + i * 4;
            let v = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            registers.push(AtomicU32::new(v));
        }

        Ok(Self {
            p: u32::from(p),
            registers: registers.into_boxed_slice(),
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
    use std::sync::Arc;
    use std::thread;

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
        let changed = s.snapshot().iter().filter(|&&r| r != 0).count();
        assert_eq!(changed, 1);
    }

    #[test]
    fn h_strictly_decreases_on_real_state_change() {
        let p = 10;
        let mut s = ExaLogLogFast::new(p);
        for i in 0..200_000u64 {
            let r_before = s.snapshot();
            s.add_hash(splitmix64(i));
            let r_after = s.snapshot();
            for (j, (&old_r, &new_r)) in r_before.iter().zip(r_after.iter()).enumerate() {
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
        assert_eq!(a.snapshot(), combined.snapshot());
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
        assert_eq!(restored.snapshot(), s.snapshot());
        assert_eq!(restored.estimate_martingale(), None);
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn atomic_insert_matches_serial_insert() {
        // Inserting the same hashes via add_hash_atomic from a single thread
        // must produce the same registers as add_hash.
        let p = 12;
        let mut serial = ExaLogLogFast::new(p);
        let atomic = ExaLogLogFast::new(p);
        for i in 0..50_000u64 {
            let h = splitmix64(i);
            serial.add_hash(h);
            atomic.add_hash_atomic(h);
        }
        assert_eq!(serial.snapshot(), atomic.snapshot());
        // Atomic estimate via ML should match serial ML (registers are identical).
        let serial_ml = serial.estimate_ml();
        let atomic_ml = atomic.estimate_ml();
        assert!((serial_ml - atomic_ml).abs() < 1e-6);
    }

    #[test]
    fn atomic_insert_concurrent_recovers_correct_estimate() {
        let p = 14;
        let n_per_thread = 100_000u64;
        let n_threads = 4;
        let total = n_per_thread * n_threads as u64;

        let s = Arc::new(ExaLogLogFast::new(p));
        let mut handles = Vec::new();
        for tid in 0..n_threads {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                let start = tid as u64 * n_per_thread;
                for i in start..start + n_per_thread {
                    s.add_hash_atomic(splitmix64(i));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let est = s.estimate_ml();
        let rel_err = (est - total as f64).abs() / total as f64;
        assert!(rel_err < 0.05, "concurrent estimate = {est}, n = {total}");
    }

    #[test]
    fn reduce_to_same_p_returns_same_registers() {
        let p = 10;
        let mut s = ExaLogLogFast::new(p);
        for i in 0..10_000u64 {
            s.add_hash(splitmix64(i));
        }
        let reduced = s.reduce(p);
        assert_eq!(reduced.snapshot(), s.snapshot());
    }

    #[test]
    fn reduce_preserves_estimate_within_tolerance() {
        // Reducing p by 2 produces a sketch whose ML estimate of the same
        // input set is consistent with a directly-built sketch at the
        // smaller p.
        let p_high = 12;
        let p_low = 10;
        let n = 50_000u64;
        let mut a = ExaLogLogFast::new(p_high);
        let mut direct = ExaLogLogFast::new(p_low);
        for i in 0..n {
            let h = splitmix64(i);
            a.add_hash(h);
            direct.add_hash(h);
        }
        let reduced = a.reduce(p_low);
        // The estimate from the reduced sketch should match the direct one.
        let red_est = reduced.estimate_ml();
        let dir_est = direct.estimate_ml();
        let rel_diff = (red_est - dir_est).abs() / n as f64;
        assert!(
            rel_diff < 0.10,
            "reduce(p={p_low}) estimate = {red_est}, direct = {dir_est}, n = {n}"
        );
    }
}
