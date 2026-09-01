//! Collection telemetry and health snapshots.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexStat {
    pub name: String,
    pub index_type: crate::types::IndexType,
    pub completeness: f32,
    pub source_revision: u64,
    pub document_count: u64,
    /// Deterministic encoded index payload estimate. This excludes allocator,
    /// map-node, and authoritative document-storage overhead.
    #[serde(default)]
    pub estimated_payload_bytes: Option<u64>,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub collection_name: String,
    pub revision: u64,
    pub doc_count: u64,
    pub query_count: u64,
    pub fts_query_count: u64,
    pub fts_index_query_count: u64,
    pub ann_query_count: u64,
    /// Successful ANN queries whose immutable Vamana/DiskANN base was traversed
    /// from the native sector sidecar instead of the in-memory graph.
    #[serde(default)]
    pub diskann_query_count: u64,
    /// Native sidecar sectors loaded by successful positioned graph queries.
    #[serde(default)]
    pub diskann_sector_read_count: u64,
    pub exact_query_count: u64,
    pub filtered_query_count: u64,
    pub scalar_index_query_count: u64,
    pub radius_query_count: u64,
    pub candidates_scanned: u64,
    pub indexed_field_count: usize,
    pub indexes: Vec<IndexStat>,
    /// Whether this handle restored its index generation from the optional
    /// on-disk derived-index cache during `Collection::open`.
    #[serde(default)]
    pub index_cache_hit: bool,
    pub read_only: bool,
    pub wal_active_seq: u64,
    pub wal_checkpoint_seq: u64,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
}

#[derive(Debug, Default)]
pub(crate) struct StatsRegistry {
    pub query_count: AtomicU64,
    pub fts_query_count: AtomicU64,
    pub fts_index_query_count: AtomicU64,
    pub ann_query_count: AtomicU64,
    pub diskann_query_count: AtomicU64,
    pub diskann_sector_read_count: AtomicU64,
    pub exact_query_count: AtomicU64,
    pub filtered_query_count: AtomicU64,
    pub scalar_index_query_count: AtomicU64,
    pub radius_query_count: AtomicU64,
    pub candidates_scanned: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryObservation {
    pub kind: QueryKind,
    pub diskann: bool,
    pub diskann_sector_reads: u64,
    pub filtered: bool,
    pub index_usage: IndexUsage,
    pub radius: bool,
    pub candidates: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexUsage {
    None,
    Scalar,
    Fts,
    ScalarFts,
}

impl IndexUsage {
    pub(crate) fn new(scalar: bool, fts: bool) -> Self {
        match (scalar, fts) {
            (false, false) => Self::None,
            (true, false) => Self::Scalar,
            (false, true) => Self::Fts,
            (true, true) => Self::ScalarFts,
        }
    }

    fn scalar(self) -> bool {
        matches!(self, Self::Scalar | Self::ScalarFts)
    }

    fn fts(self) -> bool {
        matches!(self, Self::Fts | Self::ScalarFts)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryKind {
    Exact,
    Fts,
    Ann,
    AnnFts,
}

impl StatsRegistry {
    pub fn record_query(&self, observation: QueryObservation) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        if matches!(observation.kind, QueryKind::Fts | QueryKind::AnnFts) {
            self.fts_query_count.fetch_add(1, Ordering::Relaxed);
        }
        if observation.index_usage.fts() {
            self.fts_index_query_count.fetch_add(1, Ordering::Relaxed);
        }
        if matches!(observation.kind, QueryKind::Ann | QueryKind::AnnFts) {
            self.ann_query_count.fetch_add(1, Ordering::Relaxed);
        }
        if observation.diskann {
            self.diskann_query_count.fetch_add(1, Ordering::Relaxed);
            self.diskann_sector_read_count
                .fetch_add(observation.diskann_sector_reads, Ordering::Relaxed);
        }
        self.exact_query_count.fetch_add(1, Ordering::Relaxed);
        if observation.filtered {
            self.filtered_query_count.fetch_add(1, Ordering::Relaxed);
        }
        if observation.index_usage.scalar() {
            self.scalar_index_query_count
                .fetch_add(1, Ordering::Relaxed);
        }
        if observation.radius {
            self.radius_query_count.fetch_add(1, Ordering::Relaxed);
        }
        self.candidates_scanned
            .fetch_add(observation.candidates, Ordering::Relaxed);
    }
}
