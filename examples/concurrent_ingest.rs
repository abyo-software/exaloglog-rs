//! Concurrent ingestion via `add_hash_atomic`. Demonstrates that multiple
//! threads can safely share an `Arc<ExaLogLogFast>` and the resulting
//! cardinality estimate is accurate.
//!
//! `cargo run --release --example concurrent_ingest`

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use exaloglog::ExaLogLogFast;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn main() {
    let p = 14;
    let n_per_thread = 1_000_000u64;
    let thread_counts = [1usize, 2, 4, 8];

    println!(
        "Concurrent atomic ingest into ExaLogLogFast at p = {p}\n\
         {n_per_thread} elements per thread.\n"
    );
    println!(
        "{:>8}  {:>15}  {:>10}  {:>15}  {:>10}",
        "threads", "total_n", "est", "throughput_M/s", "rel_err"
    );
    println!("{}", "-".repeat(64));

    for &n_threads in &thread_counts {
        let total = n_per_thread * n_threads as u64;
        let s = Arc::new(ExaLogLogFast::new_dense(p));

        let start = Instant::now();
        let mut handles = Vec::new();
        for tid in 0..n_threads {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                let base = tid as u64 * n_per_thread;
                for i in base..base + n_per_thread {
                    s.add_hash_atomic(splitmix64(i));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();

        let est = s.estimate_ml();
        let throughput = total as f64 / elapsed.as_secs_f64() / 1e6;
        let rel_err = (est - total as f64) / total as f64 * 100.0;
        println!(
            "{:>8}  {:>15}  {:>10.0}  {:>14.1}M  {:>+9.2}%",
            n_threads, total, est, throughput, rel_err
        );
    }

    println!();
    println!("Notes:");
    println!("- new_dense() is required: sparse mode isn't shareable via &self.");
    println!("- Each register is a single AtomicU32; updates are lock-free CAS.");
    println!("- The martingale (HIP) estimator is unavailable on sketches that");
    println!("  see add_hash_atomic; ML works on registers alone.");
}
