//! x86_64 SIMD batch hash → (i, k) computation, gated behind the
//! `simd` feature.
//!
//! The hot path computes for each 64-bit hash:
//!
//! ```text
//! i = (hash >> t) & ((1 << p) - 1)
//! a = hash | ((1 << (p + t)) - 1)
//! nlz = a.leading_zeros() as u64
//! low_t = hash & ((1 << t) - 1)
//! k = (nlz << t) + low_t + 1
//! ```
//!
//! Most of this maps cleanly to AVX2 — shifts, masks, ORs, adds — but
//! `leading_zeros` of u64 lacks a single AVX2 instruction. The native
//! u64 `LZCNT` (BMI1) is fast on its own, so we extract each lane,
//! call `_lzcnt_u64`, repack, and continue in vector registers. Net
//! win is a single fused multiply-add-shift sequence per chunk of 4
//! hashes; the leading-zero call dominates per-hash work but the rest
//! pipelines cleanly with the next chunk.

use core::arch::x86_64::*;

const T: u32 = 2;

/// AVX2 + BMI1 + LZCNT variant of [`crate::math::fill_iks`]. The caller
/// guarantees the runtime CPU has these feature sets via the
/// `is_x86_feature_detected!` check in [`fill_iks`].
#[target_feature(enable = "avx2,bmi1,lzcnt")]
unsafe fn fill_iks_avx2(hashes: &[u64], p: u32, output: &mut Vec<(u32, u32)>) {
    // SAFETY: target_feature above enables the instruction set we use
    // (avx2 + bmi1 + lzcnt). The dispatcher in `fill_iks` confirms it
    // at runtime before calling. Within this body the loads/stores
    // operate on 32-byte slices and stack arrays we own, so the
    // per-intrinsic alignment/range invariants hold.
    unsafe {
        let mask_p = (1u64 << p) - 1;
        let mask_t = (1u64 << T) - 1;
        let mask_pt = (1u64 << (p + T)) - 1;

        let mask_p_v = _mm256_set1_epi64x(mask_p as i64);
        let mask_t_v = _mm256_set1_epi64x(mask_t as i64);
        let mask_pt_v = _mm256_set1_epi64x(mask_pt as i64);
        let one_v = _mm256_set1_epi64x(1);

        let chunks = hashes.chunks_exact(4);
        let rem = chunks.remainder();
        output.reserve(hashes.len());

        for chunk in chunks {
            let h_v = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let shifted = _mm256_srli_epi64(h_v, T as i32);
            let i_v = _mm256_and_si256(shifted, mask_p_v);
            let a_v = _mm256_or_si256(h_v, mask_pt_v);

            let mut a_arr = [0u64; 4];
            _mm256_storeu_si256(a_arr.as_mut_ptr() as *mut __m256i, a_v);
            let nlz_arr: [u64; 4] = [
                _lzcnt_u64(a_arr[0]),
                _lzcnt_u64(a_arr[1]),
                _lzcnt_u64(a_arr[2]),
                _lzcnt_u64(a_arr[3]),
            ];
            let nlz_v = _mm256_loadu_si256(nlz_arr.as_ptr() as *const __m256i);

            let low_t_v = _mm256_and_si256(h_v, mask_t_v);
            let nlz_shl = _mm256_slli_epi64(nlz_v, T as i32);
            let nlz_plus = _mm256_add_epi64(nlz_shl, low_t_v);
            let k_v = _mm256_add_epi64(nlz_plus, one_v);

            let mut i_arr = [0u64; 4];
            let mut k_arr = [0u64; 4];
            _mm256_storeu_si256(i_arr.as_mut_ptr() as *mut __m256i, i_v);
            _mm256_storeu_si256(k_arr.as_mut_ptr() as *mut __m256i, k_v);
            for j in 0..4 {
                output.push((i_arr[j] as u32, k_arr[j] as u32));
            }
        }

        for &h in rem {
            let p_plus_t = p + T;
            let i = ((h >> T) & ((1u64 << p) - 1)) as u32;
            let a = h | ((1u64 << p_plus_t) - 1);
            let nlz = a.leading_zeros() as u64;
            let low_t = h & ((1u64 << T) - 1);
            let k = ((nlz << T) + low_t + 1) as u32;
            output.push((i, k));
        }
    }
}

/// Public entry point: dispatches to AVX2 if available, otherwise falls
/// through to the scalar [`crate::math::fill_iks`].
pub(crate) fn fill_iks(hashes: &[u64], p: u32, output: &mut Vec<(u32, u32)>) {
    if hashes.len() >= 4
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("lzcnt")
    {
        // SAFETY: feature detection above confirms the target_feature
        // set required by fill_iks_avx2. The function's body only uses
        // those instructions.
        unsafe { fill_iks_avx2(hashes, p, output) };
    } else {
        crate::math::fill_iks(hashes, p, output);
    }
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

    #[test]
    fn simd_matches_scalar() {
        for p in [3u32, 8, 12, 18, 26] {
            let hashes: Vec<u64> = (0..1000u64).map(splitmix64).collect();
            let mut scalar_out = Vec::new();
            crate::math::fill_iks(&hashes, p, &mut scalar_out);
            let mut simd_out = Vec::new();
            fill_iks(&hashes, p, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "p={p}");
        }
    }

    #[test]
    fn simd_handles_partial_chunks() {
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 17] {
            let hashes: Vec<u64> = (0..n as u64).map(splitmix64).collect();
            let mut scalar_out = Vec::new();
            crate::math::fill_iks(&hashes, 12, &mut scalar_out);
            let mut simd_out = Vec::new();
            fill_iks(&hashes, 12, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "n={n}");
        }
    }
}
