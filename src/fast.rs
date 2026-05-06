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
//! # Sparse mode
//!
//! Like [`crate::ExaLogLog`], `ExaLogLogFast` starts in sparse mode (a
//! sorted list of 32-bit hash tokens) and promotes to the dense register
//! array at the break-even point — `m` distinct tokens, the count at
//! which the sparse list equals the dense register array in size.
//!
//! Sparse mode is single-threaded only. [`Self::add_hash_atomic`] panics
//! if called while sparse — the design relies on storing tokens in a
//! mutable `Vec`, which can't be safely shared across threads via
//! `&self`. Call [`Self::densify`] (or use [`Self::new_dense`]) before
//! handing the sketch out to multiple threads.
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

/// Format-version flag bit indicating sparse-mode payload in `to_bytes`.
const FORMAT_FLAG_SPARSE: u8 = 0x80;

/// Number of distinct tokens at which sparse memory equals dense (`m`
/// tokens at 4 bytes each = `m · 4` bytes, same as the dense register
/// array of `m` `AtomicU32`s).
fn sparse_capacity(p: u32) -> usize {
    1usize << p
}

/// 32-bit aligned ExaLogLog with automatic sparse↔dense storage. See
/// module docs.
#[derive(Debug)]
pub struct ExaLogLogFast {
    p: u32,
    storage: FastStorage,
    martingale: f64,
    mu: f64,
    martingale_invalid: bool,
}

#[derive(Debug)]
enum FastStorage {
    Sparse(Vec<u32>),
    Dense(Box<[AtomicU32]>),
}

impl Clone for ExaLogLogFast {
    fn clone(&self) -> Self {
        let storage = match &self.storage {
            FastStorage::Sparse(v) => FastStorage::Sparse(v.clone()),
            FastStorage::Dense(regs) => {
                let snap: Vec<AtomicU32> = regs
                    .iter()
                    .map(|a| AtomicU32::new(a.load(Ordering::Relaxed)))
                    .collect();
                FastStorage::Dense(snap.into_boxed_slice())
            }
        };
        Self {
            p: self.p,
            storage,
            martingale: self.martingale,
            mu: self.mu,
            martingale_invalid: self.martingale_invalid,
        }
    }
}

impl ExaLogLogFast {
    /// Create an empty sketch starting in sparse mode. Auto-promotes to
    /// dense at the break-even point.
    pub fn new(p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&p),
            "precision p={p} out of range [{MIN_P}, {MAX_P}]"
        );
        Self {
            p,
            storage: FastStorage::Sparse(Vec::new()),
            martingale: 0.0,
            mu: 1.0,
            martingale_invalid: true, // sparse mode doesn't track HIP
        }
    }

    /// Create an empty sketch directly in dense mode (skips sparse).
    /// Use this if you know the cardinality will exceed the break-even
    /// point, or if you need atomic concurrent inserts immediately.
    pub fn new_dense(p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&p),
            "precision p={p} out of range [{MIN_P}, {MAX_P}]"
        );
        let m = 1usize << p;
        let regs: Vec<AtomicU32> = (0..m).map(|_| AtomicU32::new(0)).collect();
        Self {
            p,
            storage: FastStorage::Dense(regs.into_boxed_slice()),
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

    /// In-memory size of the storage in bytes.
    pub fn register_bytes(&self) -> usize {
        match &self.storage {
            FastStorage::Sparse(v) => v.capacity() * 4,
            FastStorage::Dense(regs) => regs.len() * 4,
        }
    }

    /// Returns `true` if the sketch is currently in sparse mode.
    pub fn is_sparse(&self) -> bool {
        matches!(self.storage, FastStorage::Sparse(_))
    }

    /// Snapshot of the current dense register values, materializing the
    /// sparse representation on the fly if needed. Always returns a
    /// fresh `Vec`.
    pub fn snapshot(&self) -> Vec<u32> {
        match &self.storage {
            FastStorage::Dense(regs) => {
                regs.iter().map(|a| a.load(Ordering::Relaxed)).collect()
            }
            FastStorage::Sparse(tokens) => {
                let mut regs = vec![0u32; 1usize << self.p];
                for &tok in tokens {
                    let h = math::token_to_hash(tok);
                    let (i, k) = math::hash_to_register_k(h, self.p);
                    regs[i] = math::apply_insert(regs[i], k, D);
                }
                regs
            }
        }
    }

    /// `d` parameter (24).
    pub fn d_parameter() -> u32 {
        D
    }

    /// Force promotion to dense mode if currently sparse. Idempotent.
    pub fn densify(&mut self) {
        if matches!(self.storage, FastStorage::Dense(_)) {
            return;
        }
        let tokens = match std::mem::replace(&mut self.storage, FastStorage::Sparse(Vec::new())) {
            FastStorage::Sparse(t) => t,
            FastStorage::Dense(_) => unreachable!(),
        };
        let m = 1usize << self.p;
        let regs: Vec<AtomicU32> = (0..m).map(|_| AtomicU32::new(0)).collect();
        self.storage = FastStorage::Dense(regs.into_boxed_slice());
        // Insert tokens into dense.
        let dense_regs = match &self.storage {
            FastStorage::Dense(r) => r,
            FastStorage::Sparse(_) => unreachable!(),
        };
        for tok in tokens {
            let h = math::token_to_hash(tok);
            let (i, k) = math::hash_to_register_k(h, self.p);
            let old = dense_regs[i].load(Ordering::Relaxed);
            let new_r = math::apply_insert(old, k, D);
            if new_r != old {
                dense_regs[i].store(new_r, Ordering::Relaxed);
            }
        }
        self.martingale_invalid = true;
        self.martingale = f64::NAN;
        self.mu = f64::NAN;
    }

    /// Insert a 64-bit hash value (Algorithm 2). Single-threaded path.
    pub fn add_hash(&mut self, hash: u64) {
        match &mut self.storage {
            FastStorage::Sparse(tokens) => {
                let token = math::hash_to_token(hash);
                if let Err(idx) = tokens.binary_search(&token) {
                    tokens.insert(idx, token);
                    if tokens.len() > sparse_capacity(self.p) {
                        self.densify();
                    }
                }
            }
            FastStorage::Dense(regs) => {
                let (i, k) = math::hash_to_register_k(hash, self.p);
                let r = regs[i].load(Ordering::Relaxed);
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
                regs[i].store(new_r, Ordering::Relaxed);
            }
        }
    }

    /// Insert a 64-bit hash value atomically (lock-free). Suitable for
    /// concurrent calls from multiple threads via a shared `&self`.
    ///
    /// Panics if the sketch is currently in sparse mode — sparse storage
    /// is `Vec`-backed and cannot be safely shared across threads. Call
    /// [`Self::densify`] first, or construct with [`Self::new_dense`].
    pub fn add_hash_atomic(&self, hash: u64) {
        let regs = match &self.storage {
            FastStorage::Dense(r) => r,
            FastStorage::Sparse(_) => panic!(
                "add_hash_atomic requires dense mode; call .densify() or use new_dense()"
            ),
        };
        let (i, k) = math::hash_to_register_k(hash, self.p);
        let reg = &regs[i];
        let mut current = reg.load(Ordering::Relaxed);
        loop {
            let new_r = math::apply_insert(current, k, D);
            if current == new_r {
                return;
            }
            match reg.compare_exchange_weak(
                current,
                new_r,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
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
        match &self.storage {
            FastStorage::Sparse(tokens) => math::estimate_from_tokens(tokens),
            FastStorage::Dense(regs) => {
                let regs_iter = regs.iter().map(|a| a.load(Ordering::Relaxed));
                let (alpha, beta) = math::compute_alpha_beta(regs_iter, self.p, D);
                math::solve_ml(alpha, &beta, self.p)
            }
        }
    }

    /// Martingale (HIP) estimate, if the running state is still valid.
    /// Returns `None` in sparse mode and after any merge, deserialization,
    /// or use of [`Self::add_hash_atomic`].
    pub fn estimate_martingale(&self) -> Option<f64> {
        if self.martingale_invalid {
            None
        } else {
            Some(self.martingale)
        }
    }

    /// Merge another sketch into `self` (Algorithm 5). Both sketches must
    /// share the same precision `p`. Modes are handled symmetrically with
    /// [`crate::ExaLogLog`]: sparse + sparse takes the union of token sets;
    /// any other combination densifies and uses Algorithm 5.
    pub fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.p != other.p {
            return Err(MergeError::PrecisionMismatch {
                lhs: self.p,
                rhs: other.p,
            });
        }
        match (&mut self.storage, &other.storage) {
            (FastStorage::Sparse(self_t), FastStorage::Sparse(other_t)) => {
                let mut merged = Vec::with_capacity(self_t.len() + other_t.len());
                let mut a = self_t.iter().copied().peekable();
                let mut b = other_t.iter().copied().peekable();
                loop {
                    match (a.peek().copied(), b.peek().copied()) {
                        (Some(x), Some(y)) if x < y => {
                            merged.push(x);
                            a.next();
                        }
                        (Some(x), Some(y)) if x > y => {
                            merged.push(y);
                            b.next();
                        }
                        (Some(x), Some(_)) => {
                            merged.push(x);
                            a.next();
                            b.next();
                        }
                        (Some(x), None) => {
                            merged.push(x);
                            a.next();
                        }
                        (None, Some(y)) => {
                            merged.push(y);
                            b.next();
                        }
                        (None, None) => break,
                    }
                }
                self.storage = FastStorage::Sparse(merged);
                if let FastStorage::Sparse(ref t) = self.storage {
                    if t.len() > sparse_capacity(self.p) {
                        self.densify();
                    }
                }
            }
            _ => {
                self.densify();
                let regs = match &self.storage {
                    FastStorage::Dense(r) => r,
                    FastStorage::Sparse(_) => unreachable!(),
                };
                match &other.storage {
                    FastStorage::Sparse(other_t) => {
                        for &tok in other_t {
                            let h = math::token_to_hash(tok);
                            let (i, k) = math::hash_to_register_k(h, self.p);
                            let old = regs[i].load(Ordering::Relaxed);
                            let new_r = math::apply_insert(old, k, D);
                            if new_r != old {
                                regs[i].store(new_r, Ordering::Relaxed);
                            }
                        }
                    }
                    FastStorage::Dense(other_regs) => {
                        for (a, b) in regs.iter().zip(other_regs.iter()) {
                            let av = a.load(Ordering::Relaxed);
                            let bv = b.load(Ordering::Relaxed);
                            a.store(math::merge_register(av, bv, D), Ordering::Relaxed);
                        }
                    }
                }
                self.martingale_invalid = true;
                self.martingale = f64::NAN;
                self.mu = f64::NAN;
            }
        }
        Ok(())
    }

    /// Reduce this sketch's precision to `new_p ≤ self.precision()`,
    /// returning a new sketch (Algorithm 6, restricted to keeping `d`).
    pub fn reduce(&self, new_p: u32) -> Self {
        assert!(
            (MIN_P..=MAX_P).contains(&new_p) && new_p <= self.p,
            "new_p={new_p} must be in [{MIN_P}, {self_p}]",
            self_p = self.p
        );
        // For sparse mode, reducing p is just inserting tokens into a
        // fresh sketch at the smaller p (tokens don't depend on p).
        if let FastStorage::Sparse(tokens) = &self.storage {
            let mut out = Self::new(new_p);
            for &tok in tokens {
                out.add_hash(math::token_to_hash(tok));
            }
            return out;
        }
        let mut out = Self::new_dense(new_p);
        let dense = match &self.storage {
            FastStorage::Dense(r) => r,
            FastStorage::Sparse(_) => unreachable!(),
        };
        if new_p == self.p {
            let out_regs = match &out.storage {
                FastStorage::Dense(r) => r,
                FastStorage::Sparse(_) => unreachable!(),
            };
            for (dst, src) in out_regs.iter().zip(dense.iter()) {
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
                let mut r = dense[old_i].load(Ordering::Relaxed);
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
            let out_regs = match &out.storage {
                FastStorage::Dense(r) => r,
                FastStorage::Sparse(_) => unreachable!(),
            };
            out_regs[new_i].store(acc, Ordering::Relaxed);
        }
        out.martingale_invalid = true;
        out
    }

    /// Reset to empty (sparse).
    pub fn clear(&mut self) {
        self.storage = FastStorage::Sparse(Vec::new());
        self.martingale = 0.0;
        self.mu = 1.0;
        self.martingale_invalid = true;
    }

    /// Serialize. Layout: 4-byte magic, 1-byte format version (top bit
    /// set in sparse mode), 1-byte t, 1-byte d, 1-byte p, then either:
    ///
    /// - dense: raw u32 register bytes (LE);
    /// - sparse: 4-byte token count followed by `count` LE u32 tokens.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.storage {
            FastStorage::Dense(regs) => {
                let mut out = Vec::with_capacity(HEADER_LEN + regs.len() * 4);
                out.extend_from_slice(&MAGIC);
                out.push(FORMAT_VERSION);
                out.push(T as u8);
                out.push(D as u8);
                out.push(self.p as u8);
                for r in regs.iter() {
                    out.extend_from_slice(&r.load(Ordering::Relaxed).to_le_bytes());
                }
                out
            }
            FastStorage::Sparse(tokens) => {
                let mut out = Vec::with_capacity(HEADER_LEN + 4 + tokens.len() * 4);
                out.extend_from_slice(&MAGIC);
                out.push(FORMAT_VERSION | FORMAT_FLAG_SPARSE);
                out.push(T as u8);
                out.push(D as u8);
                out.push(self.p as u8);
                out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
                for &tok in tokens {
                    out.extend_from_slice(&tok.to_le_bytes());
                }
                out
            }
        }
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
        let raw_version = bytes[4];
        let is_sparse = raw_version & FORMAT_FLAG_SPARSE != 0;
        let version = raw_version & !FORMAT_FLAG_SPARSE;
        if version != FORMAT_VERSION {
            return Err(DeserializeError::UnsupportedVersion(raw_version));
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

        if is_sparse {
            if bytes.len() < HEADER_LEN + 4 {
                return Err(DeserializeError::TooShort {
                    got: bytes.len(),
                    need: HEADER_LEN + 4,
                });
            }
            let count =
                u32::from_le_bytes(bytes[HEADER_LEN..HEADER_LEN + 4].try_into().unwrap()) as usize;
            let expected = HEADER_LEN + 4 + count * 4;
            if bytes.len() != expected {
                return Err(DeserializeError::LengthMismatch {
                    got: bytes.len(),
                    expected,
                });
            }
            let mut tokens = Vec::with_capacity(count);
            for i in 0..count {
                let off = HEADER_LEN + 4 + i * 4;
                tokens.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            }
            tokens.sort_unstable();
            tokens.dedup();
            return Ok(Self {
                p: u32::from(p),
                storage: FastStorage::Sparse(tokens),
                martingale: f64::NAN,
                mu: f64::NAN,
                martingale_invalid: true,
            });
        }

        let m = 1usize << p;
        let expected_len = HEADER_LEN + m * 4;
        if bytes.len() != expected_len {
            return Err(DeserializeError::LengthMismatch {
                got: bytes.len(),
                expected: expected_len,
            });
        }

        let mut regs: Vec<AtomicU32> = Vec::with_capacity(m);
        for i in 0..m {
            let off = HEADER_LEN + i * 4;
            let v = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            regs.push(AtomicU32::new(v));
        }

        Ok(Self {
            p: u32::from(p),
            storage: FastStorage::Dense(regs.into_boxed_slice()),
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
        assert!(s.is_sparse());
    }

    #[test]
    fn sparse_mode_is_exact_for_small_n() {
        for &n in &[1u64, 5, 10, 50, 100, 500] {
            let mut s = ExaLogLogFast::new(12);
            for i in 0..n {
                s.add_hash(splitmix64(i));
            }
            assert!(s.is_sparse());
            let est = s.estimate_ml();
            let rel_err = (est - n as f64).abs() / n as f64;
            assert!(rel_err < 0.20, "n={n}: est={est}");
        }
    }

    #[test]
    fn auto_promotes_at_break_even() {
        let p = 8;
        let break_even = sparse_capacity(p);
        let mut s = ExaLogLogFast::new(p);
        for i in 0..(break_even as u64 + 100) {
            s.add_hash(splitmix64(i));
        }
        assert!(!s.is_sparse());
    }

    #[test]
    fn idempotent_inserts() {
        let mut s = ExaLogLogFast::new(12);
        for _ in 0..1000 {
            s.add_hash(0xDEAD_BEEF_CAFE_BABE);
        }
        assert!(s.is_sparse());
        match &s.storage {
            FastStorage::Sparse(t) => assert_eq!(t.len(), 1),
            FastStorage::Dense(_) => panic!("should be sparse"),
        }
    }

    #[test]
    fn dense_constructor_skips_sparse() {
        let s = ExaLogLogFast::new_dense(12);
        assert!(!s.is_sparse());
    }

    #[test]
    fn h_strictly_decreases_on_real_state_change() {
        let p = 10;
        let mut s = ExaLogLogFast::new_dense(p);
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
            assert!(rel_err < 0.05, "n={n}: est={est}");
        }
    }

    #[test]
    fn ml_and_martingale_agree() {
        let p = 12;
        let n = 50_000u64;
        let mut s = ExaLogLogFast::new_dense(p);
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
        let mut a = ExaLogLogFast::new_dense(p);
        let mut b = ExaLogLogFast::new_dense(p);
        let mut combined = ExaLogLogFast::new_dense(p);
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
    }

    #[test]
    fn merge_sparse_with_sparse() {
        let p = 12;
        let mut a = ExaLogLogFast::new(p);
        let mut b = ExaLogLogFast::new(p);
        for i in 0..50u64 {
            a.add_hash(splitmix64(i));
        }
        for i in 30..80u64 {
            b.add_hash(splitmix64(i));
        }
        a.merge(&b).unwrap();
        let est = a.estimate_ml();
        let rel_err = (est - 80.0).abs() / 80.0;
        assert!(rel_err < 0.05);
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
    fn serialize_roundtrip_dense() {
        let p = 12;
        let mut s = ExaLogLogFast::new_dense(p);
        for i in 0..50_000u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let bytes = s.to_bytes();
        assert_eq!(bytes.len(), 8 + 4 * (1 << p));
        let restored = ExaLogLogFast::from_bytes(&bytes).unwrap();
        assert!(!restored.is_sparse());
        assert_eq!(restored.snapshot(), s.snapshot());
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn serialize_roundtrip_sparse() {
        let p = 12;
        let mut s = ExaLogLogFast::new(p);
        for i in 0..50u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let bytes = s.to_bytes();
        let restored = ExaLogLogFast::from_bytes(&bytes).unwrap();
        assert!(restored.is_sparse());
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn atomic_insert_matches_serial_insert() {
        let p = 12;
        let mut serial = ExaLogLogFast::new_dense(p);
        let atomic = ExaLogLogFast::new_dense(p);
        for i in 0..50_000u64 {
            let h = splitmix64(i);
            serial.add_hash(h);
            atomic.add_hash_atomic(h);
        }
        assert_eq!(serial.snapshot(), atomic.snapshot());
    }

    #[test]
    fn atomic_insert_concurrent_recovers_correct_estimate() {
        let p = 14;
        let n_per_thread = 100_000u64;
        let n_threads = 4;
        let total = n_per_thread * n_threads as u64;
        let s = Arc::new(ExaLogLogFast::new_dense(p));
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
        assert!(rel_err < 0.05);
    }

    #[test]
    #[should_panic(expected = "add_hash_atomic requires dense mode")]
    fn atomic_insert_panics_in_sparse_mode() {
        let s = ExaLogLogFast::new(12);
        s.add_hash_atomic(0xDEAD_BEEF);
    }

    #[test]
    fn densify_then_atomic_works() {
        let mut s = ExaLogLogFast::new(12);
        for i in 0..10u64 {
            s.add_hash(splitmix64(i));
        }
        s.densify();
        // Now add_hash_atomic should work.
        s.add_hash_atomic(splitmix64(100));
    }

    #[test]
    fn reduce_to_same_p_returns_same_state() {
        let p = 10;
        let mut s = ExaLogLogFast::new_dense(p);
        for i in 0..10_000u64 {
            s.add_hash(splitmix64(i));
        }
        let r = s.reduce(p);
        assert_eq!(r.snapshot(), s.snapshot());
    }

    #[test]
    fn reduce_preserves_estimate_within_tolerance() {
        let p_high = 12;
        let p_low = 10;
        let n = 50_000u64;
        let mut a = ExaLogLogFast::new_dense(p_high);
        let mut direct = ExaLogLogFast::new_dense(p_low);
        for i in 0..n {
            let h = splitmix64(i);
            a.add_hash(h);
            direct.add_hash(h);
        }
        let reduced = a.reduce(p_low);
        let red_est = reduced.estimate_ml();
        let dir_est = direct.estimate_ml();
        let rel_diff = (red_est - dir_est).abs() / n as f64;
        assert!(rel_diff < 0.10);
    }
}
