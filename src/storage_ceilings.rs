//! Typed persistence `DoS` ceilings for corpus-scale artifacts.
//!
//! These bounds are explicit product policy, not host autodetection. Defaults
//! are sized for million-document workstation corpora while remaining finite.
//! Protocol/format caps (manifest, single WAL frame, lock owner) stay
//! hardcoded elsewhere.

use crate::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize};

/// Default document-snapshot write/recovery ceiling (8 GiB).
pub const DEFAULT_SNAPSHOT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Default derived-index-cache payload ceiling (8 GiB).
pub const DEFAULT_INDEX_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Extra allowance for the on-disk index-cache framing beyond the payload.
pub const INDEX_CACHE_FILE_OVERHEAD_BYTES: u64 = 4_096;
/// Default committed WAL replay ceiling (8 GiB).
pub const DEFAULT_WAL_REPLAY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Default `DiskANN` sidecar file ceiling (512 MiB).
pub const DEFAULT_DISKANN_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Explicit ceilings for durable corpus artifacts.
///
/// Callers raise or tighten these through [`crate::CollectionOptions`] or
/// process [`crate::ConfigBuilder`]. The engine never invents values from
/// host RAM or free disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[allow(clippy::struct_field_names)]
pub struct StorageCeilings {
    #[serde(rename = "max_snapshot_bytes")]
    snapshot_bytes: u64,
    #[serde(rename = "max_index_cache_bytes")]
    index_cache_bytes: u64,
    #[serde(rename = "max_wal_replay_bytes")]
    wal_replay_bytes: u64,
    #[serde(rename = "max_diskann_file_bytes")]
    diskann_file_bytes: u64,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
#[allow(clippy::struct_field_names)]
struct StorageCeilingsWire {
    #[serde(rename = "max_snapshot_bytes")]
    snapshot_bytes: Option<u64>,
    #[serde(rename = "max_index_cache_bytes")]
    index_cache_bytes: Option<u64>,
    #[serde(rename = "max_wal_replay_bytes")]
    wal_replay_bytes: Option<u64>,
    #[serde(rename = "max_diskann_file_bytes")]
    diskann_file_bytes: Option<u64>,
}

impl<'de> Deserialize<'de> for StorageCeilings {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StorageCeilingsWire::deserialize(deserializer)?;
        let mut ceilings = Self::default();
        if let Some(limit) = wire.snapshot_bytes {
            ceilings = ceilings
                .try_with_max_snapshot_bytes(limit)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(limit) = wire.index_cache_bytes {
            ceilings = ceilings
                .try_with_max_index_cache_bytes(limit)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(limit) = wire.wal_replay_bytes {
            ceilings = ceilings
                .try_with_max_wal_replay_bytes(limit)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(limit) = wire.diskann_file_bytes {
            ceilings = ceilings
                .try_with_max_diskann_file_bytes(limit)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(ceilings)
    }
}

impl Default for StorageCeilings {
    fn default() -> Self {
        Self {
            snapshot_bytes: DEFAULT_SNAPSHOT_BYTES,
            index_cache_bytes: DEFAULT_INDEX_CACHE_BYTES,
            wal_replay_bytes: DEFAULT_WAL_REPLAY_BYTES,
            diskann_file_bytes: DEFAULT_DISKANN_FILE_BYTES,
        }
    }
}

impl StorageCeilings {
    /// Product defaults for a new policy object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Caps one atomic document snapshot write or recovery.
    pub fn try_with_max_snapshot_bytes(mut self, limit: u64) -> Result<Self> {
        self.snapshot_bytes = positive_limit(limit, "max_snapshot_bytes")?;
        Ok(self)
    }

    /// Caps the derived index-cache payload (HNSW/scalar/FTS blob).
    pub fn try_with_max_index_cache_bytes(mut self, limit: u64) -> Result<Self> {
        self.index_cache_bytes = positive_limit(limit, "max_index_cache_bytes")?;
        Ok(self)
    }

    /// Caps committed WAL bytes replayed during open/recovery.
    pub fn try_with_max_wal_replay_bytes(mut self, limit: u64) -> Result<Self> {
        self.wal_replay_bytes = positive_limit(limit, "max_wal_replay_bytes")?;
        Ok(self)
    }

    /// Caps one `DiskANN` sidecar file.
    pub fn try_with_max_diskann_file_bytes(mut self, limit: u64) -> Result<Self> {
        self.diskann_file_bytes = positive_limit(limit, "max_diskann_file_bytes")?;
        Ok(self)
    }

    pub fn max_snapshot_bytes(self) -> u64 {
        self.snapshot_bytes
    }

    pub fn max_index_cache_bytes(self) -> u64 {
        self.index_cache_bytes
    }

    /// On-disk index-cache file ceiling, including framing overhead.
    pub fn max_index_cache_file_bytes(self) -> u64 {
        self.index_cache_bytes
            .saturating_add(INDEX_CACHE_FILE_OVERHEAD_BYTES)
    }

    pub fn max_wal_replay_bytes(self) -> u64 {
        self.wal_replay_bytes
    }

    pub fn max_diskann_file_bytes(self) -> u64 {
        self.diskann_file_bytes
    }
}

fn positive_limit(limit: u64, name: &str) -> Result<u64> {
    if limit == 0 {
        return Err(Error::invalid_argument(format!(
            "{name} must be positive; omit the field to keep the product default"
        )));
    }
    Ok(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_published_product_ceilings() {
        let ceilings = StorageCeilings::default();
        assert_eq!(ceilings.max_snapshot_bytes(), DEFAULT_SNAPSHOT_BYTES);
        assert_eq!(ceilings.max_index_cache_bytes(), DEFAULT_INDEX_CACHE_BYTES);
        assert_eq!(
            ceilings.max_index_cache_file_bytes(),
            DEFAULT_INDEX_CACHE_BYTES + INDEX_CACHE_FILE_OVERHEAD_BYTES
        );
        assert_eq!(ceilings.max_wal_replay_bytes(), DEFAULT_WAL_REPLAY_BYTES);
        assert_eq!(
            ceilings.max_diskann_file_bytes(),
            DEFAULT_DISKANN_FILE_BYTES
        );
    }

    #[test]
    fn zero_ceilings_are_rejected() {
        assert!(StorageCeilings::new()
            .try_with_max_snapshot_bytes(0)
            .is_err());
        assert!(StorageCeilings::new()
            .try_with_max_index_cache_bytes(0)
            .is_err());
        assert!(StorageCeilings::new()
            .try_with_max_wal_replay_bytes(0)
            .is_err());
        assert!(StorageCeilings::new()
            .try_with_max_diskann_file_bytes(0)
            .is_err());
    }
}
