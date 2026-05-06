//! Empirical RMSE vs memory for ExaLogLog.
//!
//! Runs many simulation rounds at fixed (true) cardinality and prints the
//! observed relative-RMSE alongside the in-memory size of the register array.
//! This is the central evidence for the headline claim: ExaLogLog reaches
//! the same accuracy as HyperLogLog with ~40% less memory.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example accuracy
//! ```

use exaloglog::ExaLogLog;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

fn main() {
    let cardinalities = [100u64, 1_000, 10_000, 100_000, 1_000_000];
    let precisions = [8u32, 10, 12, 14];
    let trials = 200;

    println!(
        "ExaLogLog (t=2, d=24) — empirical RMSE over {trials} trials per row\n"
    );
    println!("{:<6} {:<8} {:<10} {:>10} {:>10} {:>10}",
        "p", "n", "memory_B", "ml_rmse%", "hip_rmse%", "ml_bias%");
    println!("{}", "-".repeat(66));

    for &p in &precisions {
        let bytes = (1usize << p) * 4;
        for &n in &cardinalities {
            let (ml_rmse, hip_rmse, ml_bias) = run_trials(p, n, trials);
            println!(
                "{:<6} {:<8} {:<10} {:>10.3} {:>10.3} {:>+10.3}",
                p,
                pretty_count(n),
                bytes,
                ml_rmse * 100.0,
                hip_rmse * 100.0,
                ml_bias * 100.0,
            );
        }
        println!();
    }

    println!("memory_B = register array size in bytes (= 4 * 2^p)");
    println!("ml_rmse%  = sqrt(mean((est_ml - n)^2)) / n");
    println!("hip_rmse% = same, using martingale (HIP) estimator");
    println!("ml_bias%  = mean(est_ml - n) / n");
}

fn run_trials(p: u32, n: u64, trials: usize) -> (f64, f64, f64) {
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
    let ml_rmse = (ml_sq / trials as f64).sqrt();
    let hip_rmse = (hip_sq / trials as f64).sqrt();
    let ml_bias = ml_sum / trials as f64;
    (ml_rmse, hip_rmse, ml_bias)
}

fn pretty_count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{}M", n / 1_000_000),
        n if n >= 1_000 => format!("{}k", n / 1_000),
        n => n.to_string(),
    }
}
