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
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub collection_name: String,
    pub revision: u64,
    pub doc_count: u64,
    pub query_count: u64,
    pub fts_query_count: u64,
    pub ann_query_count: u64,
    pub exact_query_count: u64,
    pub filtered_query_count: u64,
    pub radius_query_count: u64,
    pub candidates_scanned: u64,
    pub indexed_field_count: usize,
    pub indexes: Vec<IndexStat>,
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
    pub ann_query_count: AtomicU64,
    pub exact_query_count: AtomicU64,
    pub filtered_query_count: AtomicU64,
    pub radius_query_count: AtomicU64,
    pub candidates_scanned: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryObservation {
    pub kind: QueryKind,
    pub filtered: bool,
    pub radius: bool,
    pub candidates: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryKind {
    Exact,
    Fts,
}

impl StatsRegistry {
    pub fn record_query(&self, observation: QueryObservation) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        if observation.kind == QueryKind::Fts {
            self.fts_query_count.fetch_add(1, Ordering::Relaxed);
        }
        self.exact_query_count.fetch_add(1, Ordering::Relaxed);
        if observation.filtered {
            self.filtered_query_count.fetch_add(1, Ordering::Relaxed);
        }
        if observation.radius {
            self.radius_query_count.fetch_add(1, Ordering::Relaxed);
        }
        self.candidates_scanned
            .fetch_add(observation.candidates, Ordering::Relaxed);
    }
}
