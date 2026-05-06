//! Insertion throughput micro-benchmarks.
//!
//! Run with `cargo bench`. ExaLogLog does ~constant-time inserts independent
//! of `p` and the sketch size (Algorithm 2: a few CPU instructions per
//! element), and we want to keep it that way as we add SIMD and other
//! optimizations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use exaloglog::ExaLogLog;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_hash");
    let n = 100_000u64;
    group.throughput(Throughput::Elements(n));
    for p in [8u32, 12, 16] {
        group.bench_function(format!("ell_p{p}_n{n}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLog::new(p);
                for i in 0..n {
                    s.add_hash(black_box(splitmix64(i)));
                }
                black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate");
    for p in [8u32, 12, 16] {
        let mut s = ExaLogLog::new(p);
        for i in 0..100_000u64 {
            s.add_hash(splitmix64(i));
        }
        group.bench_function(format!("ml_p{p}"), |b| {
            b.iter(|| black_box(s.estimate_ml()));
        });
        group.bench_function(format!("hip_p{p}"), |b| {
            b.iter(|| black_box(s.estimate_martingale().unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_estimate);
criterion_main!(benches);
