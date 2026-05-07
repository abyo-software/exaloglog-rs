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

// AVX-512 intrinsics are stable since rustc 1.89. The `simd` feature is
// opt-in and effectively raises the crate's MSRV for users who enable it;
// the rest of the crate still compiles on the declared 1.85 baseline.
#![allow(clippy::incompatible_msrv)]

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

/// AVX-512 F + CD variant: 8-wide batch with native vectorized
/// `vplzcntq` for u64 leading_zeros, no per-lane extraction. Available
/// on Intel Ice Lake+ and AMD Zen 4+.
#[target_feature(enable = "avx512f,avx512cd")]
unsafe fn fill_iks_avx512(hashes: &[u64], p: u32, output: &mut Vec<(u32, u32)>) {
    // SAFETY: target_feature enables AVX-512F + CD; the dispatcher
    // confirms at runtime. All loads/stores operate on 64-byte slices
    // and stack arrays we own.
    unsafe {
        let mask_p = (1u64 << p) - 1;
        let mask_t = (1u64 << T) - 1;
        let mask_pt = (1u64 << (p + T)) - 1;

        let mask_p_v = _mm512_set1_epi64(mask_p as i64);
        let mask_t_v = _mm512_set1_epi64(mask_t as i64);
        let mask_pt_v = _mm512_set1_epi64(mask_pt as i64);
        let one_v = _mm512_set1_epi64(1);

        let chunks = hashes.chunks_exact(8);
        let rem = chunks.remainder();
        output.reserve(hashes.len());

        for chunk in chunks {
            let h_v = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
            let shifted = _mm512_srli_epi64(h_v, T);
            let i_v = _mm512_and_si512(shifted, mask_p_v);
            let a_v = _mm512_or_si512(h_v, mask_pt_v);
            // Native 8-wide leading-zeros count — the whole point of
            // the AVX-512 path.
            let nlz_v = _mm512_lzcnt_epi64(a_v);
            let low_t_v = _mm512_and_si512(h_v, mask_t_v);
            let nlz_shl = _mm512_slli_epi64(nlz_v, T);
            let nlz_plus = _mm512_add_epi64(nlz_shl, low_t_v);
            let k_v = _mm512_add_epi64(nlz_plus, one_v);

            let mut i_arr = [0u64; 8];
            let mut k_arr = [0u64; 8];
            _mm512_storeu_si512(i_arr.as_mut_ptr() as *mut __m512i, i_v);
            _mm512_storeu_si512(k_arr.as_mut_ptr() as *mut __m512i, k_v);
            for j in 0..8 {
                output.push((i_arr[j] as u32, k_arr[j] as u32));
            }
        }

        // Handle remainder via scalar path (chunk size 1..7).
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

/// Public entry point: dispatches to the best available SIMD path,
/// falling through to the scalar [`crate::math::fill_iks`].
pub(crate) fn fill_iks(hashes: &[u64], p: u32, output: &mut Vec<(u32, u32)>) {
    if hashes.len() >= 8
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512cd")
    {
        // SAFETY: feature detection confirms AVX-512F + CD.
        unsafe { fill_iks_avx512(hashes, p, output) };
    } else if hashes.len() >= 4
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("lzcnt")
    {
        // SAFETY: feature detection confirms AVX2 + BMI1 + LZCNT.
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
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25] {
            let hashes: Vec<u64> = (0..n as u64).map(splitmix64).collect();
            let mut scalar_out = Vec::new();
            crate::math::fill_iks(&hashes, 12, &mut scalar_out);
            let mut simd_out = Vec::new();
            fill_iks(&hashes, 12, &mut simd_out);
            assert_eq!(simd_out, scalar_out, "n={n}");
        }
    }
}
