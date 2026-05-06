//! Demonstrates the memory savings from sparse mode at low cardinality.
//!
//! `cargo run --release --example sparse_demo`

use exaloglog::ExaLogLog;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn main() {
    let p = 12;
    let dense_bytes = ExaLogLog::new_dense(p).register_bytes();
    println!(
        "ExaLogLog at p = {p} (dense register array would take {dense_bytes} bytes)\n"
    );

    println!(
        "{:>10}  {:>15}  {:>10}  {:>10}  {:>20}",
        "n", "storage_bytes", "is_sparse", "estimate", "rel_err_vs_n"
    );
    println!("{}", "-".repeat(72));

    let mut s = ExaLogLog::new(p);
    let checkpoints = [
        1u64, 10, 100, 1_000, 2_000, 3_000, 3_500, 3_600, 4_000, 10_000, 100_000,
    ];
    let mut last = 0u64;

    for &n in &checkpoints {
        for i in last..n {
            s.add_hash(splitmix64(i));
        }
        last = n;

        let bytes = s.register_bytes();
        let is_sparse = s.is_sparse();
        let est = s.estimate_ml();
        let rel_err = (est - n as f64).abs() / n as f64 * 100.0;

        println!(
            "{:>10}  {:>15}  {:>10}  {:>10.0}  {:>19.2}%",
            n, bytes, is_sparse, est, rel_err
        );
    }

    println!();
    println!("Below break-even (~m·7/8 = 3584 distinct elements at p=12) the");
    println!("sketch keeps a sorted list of 32-bit hash tokens — small and");
    println!("growing linearly with n. Above break-even it has promoted to the");
    println!("dense register array and storage is fixed.");
}
