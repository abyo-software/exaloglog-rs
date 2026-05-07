//! ExaLogLog: space-efficient approximate distinct counting.
//!
//! Implementation of the ExaLogLog (ELL) algorithm by Otmar Ertl, "ExaLogLog:
//! Space-Efficient and Practical Approximate Distinct Counting up to the
//! Exa-Scale", arXiv:2402.13726 (2024).
//!
//! Compared to HyperLogLog with 6-bit registers, ExaLogLog achieves the same
//! estimation error with **~43% less memory** (`ExaLogLog`, packed `d=20`)
//! or **~41% less memory** with 32-bit aligned registers
//! ([`ExaLogLogFast`], `d=24`) which are easier to access concurrently.
//!
//! Both variants:
//! - support distinct counts up to the exa-scale (10¹⁹+);
//! - are mergeable, idempotent, and reducible;
//! - have constant-time inserts independent of `p`;
//! - implement martingale (HIP) and maximum-likelihood estimators.
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
#![deny(unsafe_code)]

mod fast;
mod math;
mod packed;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[allow(unsafe_code)]
mod simd_x86;
#[cfg(feature = "rayon")]
mod rayon_impl;
#[cfg(feature = "serde")]
mod serde_impl;

#[cfg(feature = "rayon")]
pub use rayon_impl::{merge_many_par, merge_many_par_fast};

pub use fast::ExaLogLogFast;
pub use packed::ExaLogLog;

/// `t` parameter of the approximated update value distribution. Fixed at 2
/// in this build — both [`ExaLogLog`] (`d=20`) and [`ExaLogLogFast`] (`d=24`)
/// use it, and it's the configuration the paper recommends in practice.
pub const T: u32 = 2;

/// Minimum supported precision parameter `p`.
pub const MIN_P: u32 = 3;
/// Maximum supported precision parameter `p`.
pub const MAX_P: u32 = 26;

/// Magic prefix used by the serializers. ASCII `"ELL\0"`.
pub(crate) const MAGIC: [u8; 4] = *b"ELL\0";
/// Format version. Bumped on incompatible layout changes.
pub(crate) const FORMAT_VERSION: u8 = 1;

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

/// Error returned when deserializing a byte slice into an ExaLogLog sketch.
#[derive(Debug, PartialEq, Eq)]
pub enum DeserializeError {
    /// Byte slice is shorter than the minimum header length.
    TooShort {
        /// Bytes received.
        got: usize,
        /// Bytes required (at minimum).
        need: usize,
    },
    /// Magic prefix did not match.
    BadMagic,
    /// Format version is not supported by this build.
    UnsupportedVersion(u8),
    /// The encoded `t` or `d` parameter does not match what this type expects.
    /// `ExaLogLog` accepts `(t=2, d=20)`; `ExaLogLogFast` accepts `(t=2, d=24)`.
    ParameterMismatch {
        /// Encoded `t`.
        t: u8,
        /// Encoded `d`.
        d: u8,
    },
    /// Encoded `p` is outside the supported range.
    InvalidPrecision(u8),
    /// Length of the byte slice does not match what `p` implies.
    LengthMismatch {
        /// Bytes received.
        got: usize,
        /// Bytes expected, given the encoded `p`.
        expected: usize,
    },
}

impl std::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeserializeError::TooShort { got, need } => {
                write!(f, "byte slice too short: got {got}, need at least {need}")
            }
            DeserializeError::BadMagic => write!(f, "bad magic prefix"),
            DeserializeError::UnsupportedVersion(v) => {
                write!(f, "unsupported format version: {v}")
            }
            DeserializeError::ParameterMismatch { t, d } => {
                write!(f, "parameter mismatch: encoded t={t}, d={d}")
            }
            DeserializeError::InvalidPrecision(p) => {
                write!(f, "invalid precision p={p} (allowed: {MIN_P}..={MAX_P})")
            }
            DeserializeError::LengthMismatch { got, expected } => {
                write!(f, "length mismatch: got {got} bytes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for DeserializeError {}
