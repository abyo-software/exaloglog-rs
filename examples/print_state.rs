//! Diagnostic: prints raw register state bytes for parity testing
//! against the reference Java implementation. Inserts splitmix64-derived
//! hashes and dumps the dense register array as hex.

use exaloglog::{ExaLogLog, ExaLogLogFast};

fn splitmix64(x: i64) -> u64 {
    let mut x = x as u64;
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: u32 = args[1].parse().unwrap();
    let d: u32 = args[2].parse().unwrap();
    let p: u32 = args[3].parse().unwrap();
    let n: u32 = args[4].parse().unwrap();
    assert_eq!(t, 2, "this driver only supports t=2");

    if d == 24 {
        let mut s = ExaLogLogFast::new_dense(p);
        for i in 0..n as i64 {
            s.add_hash(splitmix64(i));
        }
        // Dense storage of ExaLogLogFast: m * 4 bytes, LE u32.
        let snapshot = s.snapshot();
        let mut buf = Vec::with_capacity(snapshot.len() * 4);
        for r in &snapshot {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        for byte in buf {
            print!("{byte:02x}");
        }
        println!();
    } else if d == 20 {
        let mut s = ExaLogLog::new_dense(p);
        for i in 0..n as i64 {
            s.add_hash(splitmix64(i));
        }
        // Dense storage of ExaLogLog: m * 7 / 2 bytes, packed two regs per 7.
        let bytes = s.to_bytes();
        // Strip the 8-byte header to get raw storage.
        for byte in &bytes[8..] {
            print!("{byte:02x}");
        }
        println!();
    } else {
        panic!("only d=20 or d=24 supported");
    }
}
