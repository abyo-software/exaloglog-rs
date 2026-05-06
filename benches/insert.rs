//! Insertion and estimation throughput micro-benchmarks for both variants.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use exaloglog::{ExaLogLog, ExaLogLogFast};

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
        group.bench_function(format!("packed_p{p}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLog::new(p);
                for i in 0..n {
                    s.add_hash(black_box(splitmix64(i)));
                }
                black_box(s)
            });
        });
        group.bench_function(format!("fast_p{p}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLogFast::new(p);
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
        let mut packed = ExaLogLog::new(p);
        let mut fast = ExaLogLogFast::new(p);
        for i in 0..100_000u64 {
            packed.add_hash(splitmix64(i));
            fast.add_hash(splitmix64(i));
        }
        group.bench_function(format!("packed_ml_p{p}"), |b| {
            b.iter(|| black_box(packed.estimate_ml()));
        });
        group.bench_function(format!("fast_ml_p{p}"), |b| {
            b.iter(|| black_box(fast.estimate_ml()));
        });
    }
    group.finish();
}

fn bench_sorted_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    let n = 1_000_000u64;
    group.throughput(Throughput::Elements(n));
    let hashes: Vec<u64> = (0..n).map(splitmix64).collect();
    for p in [12u32, 16] {
        group.bench_function(format!("packed_loop_p{p}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLog::new_dense(p);
                for &h in &hashes {
                    s.add_hash(h);
                }
                black_box(s)
            });
        });
        group.bench_function(format!("packed_sorted_p{p}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLog::new_dense(p);
                s.add_hashes_sorted(&hashes);
                black_box(s)
            });
        });
        group.bench_function(format!("fast_loop_p{p}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLogFast::new_dense(p);
                for &h in &hashes {
                    s.add_hash(h);
                }
                black_box(s)
            });
        });
        group.bench_function(format!("fast_sorted_p{p}"), |b| {
            b.iter(|| {
                let mut s = ExaLogLogFast::new_dense(p);
                s.add_hashes_sorted(&hashes);
                black_box(s)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_estimate, bench_sorted_batch);
criterion_main!(benches);
