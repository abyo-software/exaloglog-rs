//! Optional `serde` support, gated behind the `serde` feature.
//!
//! Sketches serialize as the wire-format bytes produced by their
//! `to_bytes` method, and deserialize via `from_bytes`. This keeps the
//! serde representation aligned with on-disk storage and avoids a second
//! ad-hoc encoding.

use serde::de::{Deserializer, Error};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::{ExaLogLog, ExaLogLogFast};

impl Serialize for ExaLogLog {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.to_bytes())
    }
}

impl<'de> Deserialize<'de> for ExaLogLog {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Vec::<u8>::deserialize(deserializer)?;
        ExaLogLog::from_bytes(&bytes).map_err(|e| Error::custom(e.to_string()))
    }
}

impl Serialize for ExaLogLogFast {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.to_bytes())
    }
}

impl<'de> Deserialize<'de> for ExaLogLogFast {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Vec::<u8>::deserialize(deserializer)?;
        ExaLogLogFast::from_bytes(&bytes).map_err(|e| Error::custom(e.to_string()))
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
    fn bincode_roundtrip_packed_sparse() {
        let mut s = ExaLogLog::new(12);
        for i in 0..50u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let encoded = bincode::serialize(&s).unwrap();
        let restored: ExaLogLog = bincode::deserialize(&encoded).unwrap();
        assert!(restored.is_sparse());
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn bincode_roundtrip_packed_dense() {
        let mut s = ExaLogLog::new_dense(10);
        for i in 0..10_000u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let encoded = bincode::serialize(&s).unwrap();
        let restored: ExaLogLog = bincode::deserialize(&encoded).unwrap();
        assert!(!restored.is_sparse());
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn bincode_roundtrip_fast_dense() {
        let mut s = ExaLogLogFast::new_dense(10);
        for i in 0..10_000u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let encoded = bincode::serialize(&s).unwrap();
        let restored: ExaLogLogFast = bincode::deserialize(&encoded).unwrap();
        assert!(!restored.is_sparse());
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn bincode_roundtrip_fast_sparse() {
        let mut s = ExaLogLogFast::new(12);
        for i in 0..50u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let encoded = bincode::serialize(&s).unwrap();
        let restored: ExaLogLogFast = bincode::deserialize(&encoded).unwrap();
        assert!(restored.is_sparse());
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }

    #[test]
    fn json_roundtrip_packed() {
        let mut s = ExaLogLog::new(12);
        for i in 0..50u64 {
            s.add_hash(splitmix64(i));
        }
        let est = s.estimate_ml();
        let json = serde_json::to_string(&s).unwrap();
        let restored: ExaLogLog = serde_json::from_str(&json).unwrap();
        assert!((restored.estimate_ml() - est).abs() < 1e-6);
    }
}
