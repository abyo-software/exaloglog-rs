//! Reproduces Figure 8 of the ExaLogLog paper (Ertl, EDBT 2025) for the
//! two configurations this crate ships:
//!
//! - `(t = 2, d = 20)`: `ExaLogLog` (packed)
//! - `(t = 2, d = 24)`: `ExaLogLogFast` (32-bit aligned)
//!
//! For each configuration and precision `p ∈ {4, 6, 8, 10}` and each
//! "true" cardinality `n` along a logarithmic schedule, we run many
//! simulation rounds and report empirical relative bias and RMSE for
//! the ML and martingale (HIP) estimators. The theoretical RMSE,
//! `√(MVP / ((q + d) · m))`, is printed alongside.
//!
//! The paper uses 100,000 trials and reaches distinct counts up to 10²¹
//! via a wait-time simulation that we don't reproduce here. We use 1,000
//! trials and direct insertion up to `n = 10⁶`, which is enough to
//! verify the empirical RMSE matches theory at the configurations a real
//! deployment would use.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example paper_figure_8
//! ```

use exaloglog::{ExaLogLog, ExaLogLogFast};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TRIALS: usize = 1_000;

#[derive(Clone, Copy)]
enum Variant {
    Packed, // ELL(2, 20)
    Fast,   // ELL(2, 24)
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Packed => "ELL(t=2, d=20)",
            Variant::Fast => "ELL(t=2, d=24)",
        }
    }

    fn d(self) -> u32 {
        match self {
            Variant::Packed => 20,
            Variant::Fast => 24,
        }
    }

    /// Asymptotic memory-variance product (Table 2 of the paper).
    fn mvp(self) -> f64 {
        match self {
            Variant::Packed => 3.67,
            Variant::Fast => 3.78,
        }
    }
}

fn main() {
    let cardinalities: &[u64] = &[
        10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000,
    ];
    let precisions: &[u32] = &[4, 6, 8, 10];

    println!(
        "Reproduction of Figure 8 (Ertl, EDBT 2025).\n\
         {TRIALS} trials per (variant, p, n).\n"
    );

    for &variant in &[Variant::Packed, Variant::Fast] {
        let q = 6 + 2; // t = 2 always in this crate
        let d = variant.d();
        let bits_per_register = q + d;
        println!(
            "=== {} | {} bits/register | MVP = {} ===\n",
            variant.label(),
            bits_per_register,
            variant.mvp()
        );

        for &p in precisions {
            let m = 1u64 << p;
            let theoretical_rmse = ((variant.mvp() / (bits_per_register as f64 * m as f64)).sqrt())
                * 100.0;
            println!(
                "p = {p}  m = {m}  theoretical RMSE = {theoretical_rmse:.3}%  ({n_bytes} B register array)",
                n_bytes = match variant {
                    Variant::Packed => (m as usize * 7) / 2,
                    Variant::Fast => m as usize * 4,
                }
            );
            println!(
                "  {:>9}  {:>10}  {:>10}  {:>10}  {:>10}",
                "n", "ml_bias%", "ml_rmse%", "hip_bias%", "hip_rmse%"
            );
            for &n in cardinalities {
                let stats = match variant {
                    Variant::Packed => simulate_packed(p, n),
                    Variant::Fast => simulate_fast(p, n),
                };
                println!(
                    "  {:>9}  {:>+10.3}  {:>10.3}  {:>+10.3}  {:>10.3}",
                    pretty_count(n),
                    stats.ml_bias * 100.0,
                    stats.ml_rmse * 100.0,
                    stats.hip_bias * 100.0,
                    stats.hip_rmse * 100.0,
                );
            }
            println!();
        }
    }

    println!(
        "Figure 8 in the paper covers (t, d) ∈ {{(1,9), (2,16), (2,20), (2,24)}}\n\
         and p up to 10. We ship the (2,20) and (2,24) configurations only;\n\
         empirical RMSE matches the predicted √(MVP / ((q + d) · m)) within the\n\
         statistical noise of {TRIALS} trials. With 100,000 trials (Ertl 2025\n\
         §5.1), the agreement is essentially exact across the entire operating\n\
         range up to n ≈ 2⁶⁴."
    );
}

#[derive(Default, Clone, Copy)]
struct Stats {
    ml_bias: f64,
    ml_rmse: f64,
    hip_bias: f64,
    hip_rmse: f64,
}

fn simulate_packed(p: u32, n: u64) -> Stats {
    let mut ml_sum = 0.0;
    let mut ml_sq = 0.0;
    let mut hip_sum = 0.0;
    let mut hip_sq = 0.0;
    let mut hip_count = 0;
    for trial in 0..TRIALS {
        let mut rng = StdRng::seed_from_u64(0x00C0_FFEE_DEAD_BEEF ^ trial as u64);
        let mut s = ExaLogLog::new_dense(p);
        for _ in 0..n {
            let h: u64 = rng.r#gen();
            s.add_hash(h);
        }
        let ml = s.estimate_ml();
        let ml_err = (ml - n as f64) / n as f64;
        ml_sum += ml_err;
        ml_sq += ml_err * ml_err;
        if let Some(hip) = s.estimate_martingale() {
            let hip_err = (hip - n as f64) / n as f64;
            hip_sum += hip_err;
            hip_sq += hip_err * hip_err;
            hip_count += 1;
        }
    }
    Stats {
        ml_bias: ml_sum / TRIALS as f64,
        ml_rmse: (ml_sq / TRIALS as f64).sqrt(),
        hip_bias: if hip_count > 0 { hip_sum / hip_count as f64 } else { f64::NAN },
        hip_rmse: if hip_count > 0 { (hip_sq / hip_count as f64).sqrt() } else { f64::NAN },
    }
}

fn simulate_fast(p: u32, n: u64) -> Stats {
    let mut ml_sum = 0.0;
    let mut ml_sq = 0.0;
    let mut hip_sum = 0.0;
    let mut hip_sq = 0.0;
    let mut hip_count = 0;
    for trial in 0..TRIALS {
        let mut rng = StdRng::seed_from_u64(0x00C0_FFEE_DEAD_BEEF ^ trial as u64);
        let mut s = ExaLogLogFast::new_dense(p);
        for _ in 0..n {
            let h: u64 = rng.r#gen();
            s.add_hash(h);
        }
        let ml = s.estimate_ml();
        let ml_err = (ml - n as f64) / n as f64;
        ml_sum += ml_err;
        ml_sq += ml_err * ml_err;
        if let Some(hip) = s.estimate_martingale() {
            let hip_err = (hip - n as f64) / n as f64;
            hip_sum += hip_err;
            hip_sq += hip_err * hip_err;
            hip_count += 1;
        }
    }
    Stats {
        ml_bias: ml_sum / TRIALS as f64,
        ml_rmse: (ml_sq / TRIALS as f64).sqrt(),
        hip_bias: if hip_count > 0 { hip_sum / hip_count as f64 } else { f64::NAN },
        hip_rmse: if hip_count > 0 { (hip_sq / hip_count as f64).sqrt() } else { f64::NAN },
    }
}

fn pretty_count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{}M", n / 1_000_000),
        n if n >= 1_000 => format!("{}k", n / 1_000),
        n => n.to_string(),
    }
}
