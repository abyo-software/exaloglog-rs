//! Empirical RMSE vs memory for both ExaLogLog variants.
//!
//! Runs many simulation rounds at fixed (true) cardinality and prints the
//! observed relative-RMSE alongside the in-memory size of the register
//! storage. Confirms the implementation matches theory and produces the
//! evidence for the headline memory-savings claim.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example accuracy
//! ```

use exaloglog::{ExaLogLog, ExaLogLogFast};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

fn main() {
    let cardinalities = [100u64, 1_000, 10_000, 100_000, 1_000_000];
    let precisions = [8u32, 10, 12, 14];
    let trials = 200;

    println!(
        "ExaLogLog accuracy — {trials} trials per row, ML and martingale (HIP) estimators\n"
    );

    println!("=== ExaLogLog (packed, t=2, d=20, MVP=3.67) ===");
    print_header();
    for &p in &precisions {
        for &n in &cardinalities {
            let s = ExaLogLog::new(p);
            let bytes = s.register_bytes();
            drop(s);
            let stats = run_packed(p, n, trials);
            print_row(p, n, bytes, stats);
        }
        println!();
    }

    println!("=== ExaLogLogFast (32-bit aligned, t=2, d=24, MVP=3.78) ===");
    print_header();
    for &p in &precisions {
        for &n in &cardinalities {
            let s = ExaLogLogFast::new(p);
            let bytes = s.register_bytes();
            drop(s);
            let stats = run_fast(p, n, trials);
            print_row(p, n, bytes, stats);
        }
        println!();
    }

    println!("=== Reference: HLL with 6-bit registers, MVP=6.48 ===");
    println!("Memory for the same RMSE that ExaLogLog reaches at given (p, bytes):");
    println!("  bytes_HLL_for_same_rmse = bytes_ELL · (MVP_HLL / MVP_ELL)");
    println!();
    let mvp_hll6 = 6.48;
    let mvp_packed = 3.67;
    let mvp_fast = 3.78;
    println!("{:<6} {:>12} {:>12} {:>12} {:>12}", "p", "ELL_bytes", "HLL_for_same", "saving_%", "fast_bytes");
    for &p in &precisions {
        let m = 1u64 << p;
        let packed_bytes = (m as f64 * 7.0 / 2.0) as u64;
        let fast_bytes = m * 4;
        let hll_for_same = (packed_bytes as f64 * mvp_hll6 / mvp_packed) as u64;
        let saving_pct = 100.0 * (1.0 - packed_bytes as f64 / hll_for_same as f64);
        let _ = mvp_fast;
        println!(
            "{:<6} {:>12} {:>12} {:>11.1}% {:>12}",
            p, packed_bytes, hll_for_same, saving_pct, fast_bytes
        );
    }
    println!();
    println!("memory_B = register storage size in bytes");
    println!("ml_rmse% = sqrt(mean((est_ml - n)^2)) / n");
    println!("hip_rmse% = same, using martingale estimator");
}

#[derive(Default, Clone, Copy)]
struct Stats {
    ml_rmse: f64,
    hip_rmse: f64,
    ml_bias: f64,
}

fn print_header() {
    println!(
        "{:<6} {:<8} {:<10} {:>10} {:>10} {:>10}",
        "p", "n", "memory_B", "ml_rmse%", "hip_rmse%", "ml_bias%"
    );
    println!("{}", "-".repeat(66));
}

fn print_row(p: u32, n: u64, bytes: usize, s: Stats) {
    println!(
        "{:<6} {:<8} {:<10} {:>10.3} {:>10.3} {:>+10.3}",
        p,
        pretty_count(n),
        bytes,
        s.ml_rmse * 100.0,
        s.hip_rmse * 100.0,
        s.ml_bias * 100.0,
    );
}

fn run_packed(p: u32, n: u64, trials: usize) -> Stats {
    let mut ml_sq = 0.0;
    let mut hip_sq = 0.0;
    let mut ml_sum = 0.0;
    for trial in 0..trials {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_0000_0000 ^ trial as u64);
        let mut s = ExaLogLog::new(p);
        for _ in 0..n {
            let h: u64 = rng.r#gen();
            s.add_hash(h);
        }
        let ml = s.estimate_ml();
        let hip = s.estimate_martingale().expect("HIP valid for fresh sketch");
        let ml_err = (ml - n as f64) / n as f64;
        let hip_err = (hip - n as f64) / n as f64;
        ml_sq += ml_err * ml_err;
        hip_sq += hip_err * hip_err;
        ml_sum += ml_err;
    }
    Stats {
        ml_rmse: (ml_sq / trials as f64).sqrt(),
        hip_rmse: (hip_sq / trials as f64).sqrt(),
        ml_bias: ml_sum / trials as f64,
    }
}

fn run_fast(p: u32, n: u64, trials: usize) -> Stats {
    let mut ml_sq = 0.0;
    let mut hip_sq = 0.0;
    let mut ml_sum = 0.0;
    for trial in 0..trials {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_0000_0000 ^ trial as u64);
        let mut s = ExaLogLogFast::new(p);
        for _ in 0..n {
            let h: u64 = rng.r#gen();
            s.add_hash(h);
        }
        let ml = s.estimate_ml();
        let hip = s.estimate_martingale().expect("HIP valid for fresh sketch");
        let ml_err = (ml - n as f64) / n as f64;
        let hip_err = (hip - n as f64) / n as f64;
        ml_sq += ml_err * ml_err;
        hip_sq += hip_err * hip_err;
        ml_sum += ml_err;
    }
    Stats {
        ml_rmse: (ml_sq / trials as f64).sqrt(),
        hip_rmse: (hip_sq / trials as f64).sqrt(),
        ml_bias: ml_sum / trials as f64,
    }
}

fn pretty_count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{}M", n / 1_000_000),
        n if n >= 1_000 => format!("{}k", n / 1_000),
        n => n.to_string(),
    }
}
