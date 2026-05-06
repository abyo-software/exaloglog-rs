//! Head-to-head comparison: ExaLogLog vs HyperLogLog.
//!
//! Implements a textbook HLL (with both 6-bit and 8-bit register storage) so
//! we have a clean, correctly-configured baseline to compare against. Using
//! one of the existing crates was tempting but the most popular pure-Rust
//! HLL on stable (`hyperloglog` 1.0.3) has a long-standing bug: it derives
//! its precision parameter from the requested error rate using `f64::ln`
//! instead of `log2`, so a sketch built with `error_rate=0.01` actually has
//! ~3× the error you'd expect. We sidestep that by configuring `p` directly
//! here.
//!
//! Two comparisons:
//!
//! 1. **Equal `m` (register count):** same number of registers, different
//!    register widths.
//! 2. **Equal target RMSE:** find the smallest configuration of each
//!    algorithm that meets a target RMSE; compare bytes.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example head_to_head
//! ```

use exaloglog::ExaLogLog;
use rand::{rngs::StdRng, Rng, SeedableRng};

const TRIALS: usize = 200;
const N: u64 = 1_000_000;

fn main() {
    println!(
        "Head-to-head: ExaLogLog vs HyperLogLog\n\
         {TRIALS} trials at n = {N} per config.\n"
    );

    println!("=== Equal-m comparison: same register count m=2^p ===\n");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "p", "ELL_B", "HLL6_B", "HLL8_B", "ELL_RMSE%", "HLL6_RMSE%", "HLL8_RMSE%"
    );
    println!("{}", "-".repeat(72));

    for p in [8u32, 10, 12, 14] {
        let m = 1u64 << p;
        let ell_bytes = ((m as usize) * 7) / 2; // 28 bits per register, packed
        let hll6_bytes = ((m as usize * 6) + 7) / 8; // 6 bits packed
        let hll8_bytes = m as usize; // 8 bits per register

        let ell_rmse = run_ell(p);
        let hll6_rmse = run_hll(p, HllReg::Bits6);
        let hll8_rmse = run_hll(p, HllReg::Bits8);

        println!(
            "{:<6} {:>10} {:>10} {:>10} {:>10.3} {:>10.3} {:>10.3}",
            p,
            ell_bytes,
            hll6_bytes,
            hll8_bytes,
            ell_rmse * 100.0,
            hll6_rmse * 100.0,
            hll8_rmse * 100.0,
        );
    }

    println!();
    println!("=== Equal-RMSE comparison: smallest config per algorithm ===\n");
    println!("How small can each algorithm get while meeting a target RMSE?");
    println!();
    println!(
        "{:<10} {:>8} {:>10} {:>10} {:>10} {:>16}",
        "target", "ELL_p", "ELL_B", "HLL6_B", "HLL8_B", "ELL_savings_vs_HLL6"
    );
    println!("{}", "-".repeat(70));

    for target_pct in [2.0_f64, 1.5, 1.0, 0.7, 0.5] {
        let target = target_pct / 100.0;
        let (ell_p, ell_b) = smallest_meeting(target, |p| run_ell(p), |p| {
            (((1u64 << p) as usize) * 7) / 2
        });
        let (_, hll6_b) = smallest_meeting(target, |p| run_hll(p, HllReg::Bits6), |p| {
            ((1usize << p) * 6 + 7) / 8
        });
        let (_, hll8_b) = smallest_meeting(target, |p| run_hll(p, HllReg::Bits8), |p| {
            1usize << p
        });
        let saving = 100.0 * (1.0 - ell_b as f64 / hll6_b as f64);
        println!(
            "{:<9.2}% {:>8} {:>10} {:>10} {:>10} {:>15.1}%",
            target_pct, ell_p, ell_b, hll6_b, hll8_b, saving
        );
    }

    println!();
    println!("ELL_B  = ExaLogLog packed (28 bits/register, 7 bytes per pair)");
    println!("HLL6_B = HyperLogLog with 6-bit packed registers (Apache DataSketches style)");
    println!("HLL8_B = HyperLogLog with 8-bit registers (typical Rust crate layout)");
}

#[derive(Copy, Clone)]
enum HllReg {
    Bits6,
    Bits8,
}

fn run_ell(p: u32) -> f64 {
    let mut sq = 0.0;
    for trial in 0..TRIALS {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_0000_0000 ^ trial as u64);
        let mut s = ExaLogLog::new(p);
        for _ in 0..N {
            let h: u64 = rng.r#gen();
            s.add_hash(h);
        }
        let est = s.estimate_ml();
        let err = (est - N as f64) / N as f64;
        sq += err * err;
    }
    (sq / TRIALS as f64).sqrt()
}

fn run_hll(p: u32, reg: HllReg) -> f64 {
    let mut sq = 0.0;
    for trial in 0..TRIALS {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_0000_0000 ^ trial as u64);
        let mut h = Hll::new(p, reg);
        for _ in 0..N {
            let v: u64 = rng.r#gen();
            h.add_hash(v);
        }
        let est = h.estimate();
        let err = (est - N as f64) / N as f64;
        sq += err * err;
    }
    (sq / TRIALS as f64).sqrt()
}

fn smallest_meeting<F, B>(target: f64, mut measure: F, mut byte_fn: B) -> (u32, usize)
where
    F: FnMut(u32) -> f64,
    B: FnMut(u32) -> usize,
{
    for p in 4u32..=18 {
        let r = measure(p);
        if r <= target * 1.05 {
            return (p, byte_fn(p));
        }
    }
    (18, byte_fn(18))
}

// ============================================================
// Reference HyperLogLog implementation (textbook version).
//
// Standard HLL with the bias-corrected estimator. We model both 6-bit and
// 8-bit register storage to compare bytes, but the algorithm is the same;
// at p ≤ 14 and reasonable cardinalities, neither register width can
// saturate (max possible value is 64 - p ≤ 60, well within both 6 and 8
// bits).
// ============================================================

struct Hll {
    p: u32,
    m: usize,
    registers: Vec<u8>,
    /// Storage variant — only affects reported memory size, not behavior.
    _reg: HllReg,
}

impl Hll {
    fn new(p: u32, reg: HllReg) -> Self {
        let m = 1usize << p;
        Self {
            p,
            m,
            registers: vec![0u8; m],
            _reg: reg,
        }
    }

    fn add_hash(&mut self, hash: u64) {
        let i = (hash & ((1u64 << self.p) - 1)) as usize;
        let w = hash >> self.p;
        // ρ(w) = 1 + position of leftmost 1-bit within the (64 - p)-bit
        // window, with ρ = 65 - p when all of w is zero.
        let rho = if w == 0 {
            (65 - self.p) as u8
        } else {
            (w.leading_zeros() - self.p) as u8 + 1
        };
        if rho > self.registers[i] {
            self.registers[i] = rho;
        }
    }

    fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let alpha = match self.p {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let mut sum = 0.0;
        let mut zeros = 0;
        for &r in &self.registers {
            sum += 2f64.powi(-(r as i32));
            if r == 0 {
                zeros += 1;
            }
        }
        let raw = alpha * m * m / sum;

        // Small-range correction: linear counting when too many empty
        // registers.
        if raw <= 2.5 * m && zeros > 0 {
            return m * (m / zeros as f64).ln();
        }
        // Large-range correction (32-bit hash regime — irrelevant for our
        // 64-bit case but kept for completeness).
        let two_pow_32 = (1u64 << 32) as f64;
        if raw > two_pow_32 / 30.0 {
            return -two_pow_32 * (1.0 - raw / two_pow_32).ln();
        }
        raw
    }
}
