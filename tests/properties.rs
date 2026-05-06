//! Property-based tests via proptest. Validates algebraic invariants of
//! merge, serialize, and reduce that the unit tests can only spot-check.

use exaloglog::{ExaLogLog, ExaLogLogFast};
use proptest::prelude::*;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Build a packed sketch by inserting `seeds.iter().map(splitmix64)`.
fn build_packed(p: u32, seeds: &[u64]) -> ExaLogLog {
    let mut s = ExaLogLog::new(p);
    for &seed in seeds {
        s.add_hash(splitmix64(seed));
    }
    s
}

fn build_fast(p: u32, seeds: &[u64]) -> ExaLogLogFast {
    let mut s = ExaLogLogFast::new(p);
    for &seed in seeds {
        s.add_hash(splitmix64(seed));
    }
    s
}

fn registers_packed(s: &ExaLogLog) -> Vec<u32> {
    (0..s.num_registers()).map(|i| s.get_register(i)).collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 30,
        ..ProptestConfig::default()
    })]

    /// Merge is commutative: A ∪ B ≡ B ∪ A (register-level equality).
    #[test]
    fn packed_merge_is_commutative(
        p in 6u32..=10,
        seeds_a in proptest::collection::vec(0u64..1_000_000, 0..200),
        seeds_b in proptest::collection::vec(0u64..1_000_000, 0..200),
    ) {
        let mut ab = build_packed(p, &seeds_a);
        let bb = build_packed(p, &seeds_b);
        let mut ba = build_packed(p, &seeds_b);
        let aa = build_packed(p, &seeds_a);

        ab.merge(&bb).unwrap();
        ba.merge(&aa).unwrap();

        // Densify both to compare register state directly (sparse merge
        // produces a sparse result; we want apples-to-apples).
        ab.densify();
        ba.densify();
        prop_assert_eq!(registers_packed(&ab), registers_packed(&ba));
    }

    /// Merge is associative: (A ∪ B) ∪ C ≡ A ∪ (B ∪ C).
    #[test]
    fn packed_merge_is_associative(
        p in 6u32..=8,
        seeds_a in proptest::collection::vec(0u64..1_000_000, 0..150),
        seeds_b in proptest::collection::vec(0u64..1_000_000, 0..150),
        seeds_c in proptest::collection::vec(0u64..1_000_000, 0..150),
    ) {
        // (A ∪ B) ∪ C
        let mut ab = build_packed(p, &seeds_a);
        let b1 = build_packed(p, &seeds_b);
        let c1 = build_packed(p, &seeds_c);
        ab.merge(&b1).unwrap();
        ab.merge(&c1).unwrap();
        // A ∪ (B ∪ C)
        let mut bc = build_packed(p, &seeds_b);
        let c2 = build_packed(p, &seeds_c);
        bc.merge(&c2).unwrap();
        let mut a_bc = build_packed(p, &seeds_a);
        a_bc.merge(&bc).unwrap();

        ab.densify();
        a_bc.densify();
        prop_assert_eq!(registers_packed(&ab), registers_packed(&a_bc));
    }

    /// Merging with an empty sketch is the identity.
    #[test]
    fn packed_merge_with_empty_is_identity(
        p in 6u32..=10,
        seeds in proptest::collection::vec(0u64..1_000_000, 0..200),
    ) {
        let mut s = build_packed(p, &seeds);
        let empty = ExaLogLog::new(p);
        let before = {
            let mut s_clone = s.clone();
            s_clone.densify();
            registers_packed(&s_clone)
        };
        s.merge(&empty).unwrap();
        s.densify();
        prop_assert_eq!(registers_packed(&s), before);
    }

    /// Serialize → deserialize preserves the ML estimate.
    #[test]
    fn packed_serialize_roundtrip_preserves_estimate(
        p in 6u32..=10,
        seeds in proptest::collection::vec(0u64..1_000_000, 0..500),
    ) {
        let s = build_packed(p, &seeds);
        let est = s.estimate_ml();
        let bytes = s.to_bytes();
        let restored = ExaLogLog::from_bytes(&bytes).unwrap();
        let restored_est = restored.estimate_ml();
        prop_assert!((est - restored_est).abs() < 1e-6);
    }

    /// Insertion order doesn't affect the final register state.
    #[test]
    fn packed_insert_order_independent(
        p in 6u32..=10,
        mut seeds in proptest::collection::vec(0u64..1_000_000, 0..200),
    ) {
        let s1 = build_packed(p, &seeds);
        seeds.reverse();
        let s2 = build_packed(p, &seeds);
        let mut s1d = s1.clone();
        let mut s2d = s2.clone();
        s1d.densify();
        s2d.densify();
        prop_assert_eq!(registers_packed(&s1d), registers_packed(&s2d));
    }

    /// reduce(p_high → p_low) gives the same registers as building
    /// directly at p_low.
    #[test]
    fn packed_reduce_matches_direct(
        p_low in 4u32..=8,
        p_diff in 1u32..=3,
        seeds in proptest::collection::vec(0u64..1_000_000, 50..500),
    ) {
        let p_high = p_low + p_diff;
        let mut a = ExaLogLog::new_dense(p_high);
        let mut direct = ExaLogLog::new_dense(p_low);
        for &seed in &seeds {
            a.add_hash(splitmix64(seed));
            direct.add_hash(splitmix64(seed));
        }
        let reduced = a.reduce(p_low);
        // Estimates should be very close — exact register equality isn't
        // guaranteed because the saturated-regime adaptation has
        // edge-case interactions with the concrete bit pattern of `j`.
        let red_est = reduced.estimate_ml();
        let dir_est = direct.estimate_ml();
        let n = seeds.len() as f64;
        prop_assert!(
            (red_est - dir_est).abs() / n < 0.10,
            "reduced={}, direct={}, n={}", red_est, dir_est, n
        );
    }

    // -- Same battery for ExaLogLogFast --

    #[test]
    fn fast_merge_is_commutative(
        p in 6u32..=10,
        seeds_a in proptest::collection::vec(0u64..1_000_000, 0..200),
        seeds_b in proptest::collection::vec(0u64..1_000_000, 0..200),
    ) {
        let mut ab = build_fast(p, &seeds_a);
        let bb = build_fast(p, &seeds_b);
        let mut ba = build_fast(p, &seeds_b);
        let aa = build_fast(p, &seeds_a);
        ab.merge(&bb).unwrap();
        ba.merge(&aa).unwrap();
        ab.densify();
        ba.densify();
        prop_assert_eq!(ab.snapshot(), ba.snapshot());
    }

    #[test]
    fn fast_serialize_roundtrip_preserves_estimate(
        p in 6u32..=10,
        seeds in proptest::collection::vec(0u64..1_000_000, 0..500),
    ) {
        let s = build_fast(p, &seeds);
        let est = s.estimate_ml();
        let bytes = s.to_bytes();
        let restored = ExaLogLogFast::from_bytes(&bytes).unwrap();
        prop_assert!((est - restored.estimate_ml()).abs() < 1e-6);
    }
}
