//! Optional Rayon-powered parallel reductions, gated behind the `rayon`
//! feature.
//!
//! For a slice of sketches at the same precision, [`merge_many_par`]
//! reduces them in parallel using Rayon's work-stealing thread pool.
//! Useful when rolling up many tenant sketches into a single sketch and
//! you have spare cores.

use rayon::prelude::*;

use crate::{ExaLogLog, ExaLogLogFast};

/// Merge a slice of [`ExaLogLog`] sketches in parallel, returning the
/// rolled-up sketch. All sketches must share the same precision `p`;
/// the function panics on mismatch (use [`ExaLogLog::merge`] in a loop
/// if you need fine-grained error handling).
///
/// Returns `None` for an empty slice.
pub fn merge_many_par(sketches: &[ExaLogLog]) -> Option<ExaLogLog> {
    sketches.par_iter().cloned().reduce_with(|mut a, b| {
        a.merge(&b)
            .expect("merge_many_par: precision mismatch across sketches");
        a
    })
}

/// Merge a slice of [`ExaLogLogFast`] sketches in parallel.
pub fn merge_many_par_fast(sketches: &[ExaLogLogFast]) -> Option<ExaLogLogFast> {
    sketches.par_iter().cloned().reduce_with(|mut a, b| {
        a.merge(&b)
            .expect("merge_many_par_fast: precision mismatch across sketches");
        a
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn make_packed(p: u32, range: std::ops::Range<u64>) -> ExaLogLog {
        let mut s = ExaLogLog::new_dense(p);
        for i in range {
            s.add_hash(splitmix64(i));
        }
        s
    }

    fn make_fast(p: u32, range: std::ops::Range<u64>) -> ExaLogLogFast {
        let mut s = ExaLogLogFast::new_dense(p);
        for i in range {
            s.add_hash(splitmix64(i));
        }
        s
    }

    #[test]
    fn merge_many_par_packed_matches_serial() {
        let p = 10;
        let sketches: Vec<ExaLogLog> =
            (0..8u64).map(|t| make_packed(p, t * 1000..(t + 1) * 1000)).collect();
        let mut serial = sketches[0].clone();
        for s in &sketches[1..] {
            serial.merge(s).unwrap();
        }
        let parallel = merge_many_par(&sketches).unwrap();
        for i in 0..serial.num_registers() {
            assert_eq!(
                serial.get_register(i),
                parallel.get_register(i),
                "register {i}"
            );
        }
    }

    #[test]
    fn merge_many_par_fast_matches_serial() {
        let p = 10;
        let sketches: Vec<ExaLogLogFast> =
            (0..8u64).map(|t| make_fast(p, t * 1000..(t + 1) * 1000)).collect();
        let mut serial = sketches[0].clone();
        for s in &sketches[1..] {
            serial.merge(s).unwrap();
        }
        let parallel = merge_many_par_fast(&sketches).unwrap();
        assert_eq!(serial.snapshot(), parallel.snapshot());
    }

    #[test]
    fn merge_many_par_empty_returns_none() {
        let empty: Vec<ExaLogLog> = vec![];
        assert!(merge_many_par(&empty).is_none());
    }

    #[test]
    fn merge_many_par_single_returns_clone() {
        let s = make_packed(8, 0..1000);
        let result = merge_many_par(std::slice::from_ref(&s)).unwrap();
        for i in 0..s.num_registers() {
            assert_eq!(result.get_register(i), s.get_register(i));
        }
    }
}
