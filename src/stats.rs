//! Collection telemetry and health snapshots.

use crate::collection::CollectionResourceLimits;
use crate::config::IoBackend;
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
    /// Successful Vamana/`DiskANN` sidecar queries served from an immutable
    /// mmap snapshot instead of positioned file reads.
    #[serde(default)]
    pub diskann_mmap_query_count: u64,
    /// Native sidecar sectors staged in request-local caches by successful
    /// positioned or mmap-snapshot graph queries.
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
    /// Resolved sidecar backend for this collection handle.
    #[serde(default)]
    pub io_backend: crate::config::IoBackend,
    pub read_only: bool,
    pub wal_active_seq: u64,
    pub wal_checkpoint_seq: u64,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
    /// Deterministic serialized size of the authoritative document map.
    #[serde(default)]
    pub accounted_document_bytes: u64,
    /// Sum of deterministic derived-index payload estimates.
    #[serde(default)]
    pub estimated_index_bytes: u64,
    /// Authoritative document accounting plus derived-index estimates.
    #[serde(default)]
    pub accounted_bytes: u64,
    /// Collection-local limits captured when this handle was opened.
    #[serde(default)]
    pub resource_limits: CollectionResourceLimits,
    /// Operations rejected by this handle's resource policy.
    #[serde(default)]
    pub resource_limit_rejections: u64,
}

/// Readiness assessment for one collection handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectionHealthStatus {
    /// The authoritative and derived generations agree and every configured
    /// index is complete.
    Healthy,
    /// Authoritative storage is usable, but at least one derived index is
    /// incomplete, missing, or stale.
    Degraded,
    /// The in-memory and durable authoritative revisions disagree.
    Unhealthy,
    /// The shared collection handle has been closed.
    Closed,
}

/// Machine-readable collection readiness and checkpoint-lag snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionHealth {
    pub status: CollectionHealthStatus,
    pub revision: u64,
    pub storage_revision: u64,
    pub doc_count: u64,
    pub index_count: usize,
    pub ready_index_count: usize,
    pub read_only: bool,
    /// WAL work awaiting an authoritative snapshot checkpoint. Pending WAL is
    /// normal for interval/manual durability and does not by itself degrade
    /// readiness.
    pub checkpoint_pending: bool,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
    /// Whether an explicitly owned background maintenance runtime currently
    /// holds this collection's single scheduler claim.
    pub maintenance_active: bool,
    /// Stable, actionable explanations for non-healthy states.
    pub reasons: Vec<String>,
}

impl CollectionHealth {
    /// Returns true only while the collection is open and fully ready.
    pub fn is_healthy(&self) -> bool {
        self.status == CollectionHealthStatus::Healthy
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CollectionHealthInput<'a> {
    pub is_open: bool,
    pub revision: u64,
    pub storage_revision: u64,
    pub doc_count: u64,
    pub indexes: &'a [IndexStat],
    pub read_only: bool,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
    pub maintenance_active: bool,
}

pub(crate) fn assess_collection_health(input: CollectionHealthInput<'_>) -> CollectionHealth {
    let ready_index_count = input
        .indexes
        .iter()
        .filter(|index| {
            index.state == "ready"
                && is_complete(index.completeness)
                && index.source_revision == input.revision
        })
        .count();
    let mut reasons = Vec::new();
    let status = if input.is_open {
        let mut status = CollectionHealthStatus::Healthy;
        if input.storage_revision != input.revision {
            reasons.push(format!(
                "storage revision {} does not match collection revision {}",
                input.storage_revision, input.revision
            ));
            status = CollectionHealthStatus::Unhealthy;
        }
        for index in input.indexes {
            if index.state != "ready" {
                reasons.push(format!(
                    "index '{}' is in '{}' state",
                    index.name, index.state
                ));
            } else if !is_complete(index.completeness) {
                reasons.push(format!(
                    "index '{}' completeness is {}",
                    index.name, index.completeness
                ));
            } else if index.source_revision != input.revision {
                reasons.push(format!(
                    "index '{}' source revision {} does not match collection revision {}",
                    index.name, index.source_revision, input.revision
                ));
            } else {
                continue;
            }
            if status == CollectionHealthStatus::Healthy {
                status = CollectionHealthStatus::Degraded;
            }
        }
        status
    } else {
        reasons.push("collection is closed".to_string());
        CollectionHealthStatus::Closed
    };

    CollectionHealth {
        status,
        revision: input.revision,
        storage_revision: input.storage_revision,
        doc_count: input.doc_count,
        index_count: input.indexes.len(),
        ready_index_count,
        read_only: input.read_only,
        checkpoint_pending: input.wal_ops_since_checkpoint > 0
            || input.wal_bytes_since_checkpoint > 0,
        wal_ops_since_checkpoint: input.wal_ops_since_checkpoint,
        wal_bytes_since_checkpoint: input.wal_bytes_since_checkpoint,
        maintenance_active: input.maintenance_active,
        reasons,
    }
}

fn is_complete(completeness: f32) -> bool {
    (completeness - 1.0).abs() <= f32::EPSILON
}

#[derive(Debug, Default)]
pub(crate) struct StatsRegistry {
    pub query_count: AtomicU64,
    pub fts_query_count: AtomicU64,
    pub fts_index_query_count: AtomicU64,
    pub ann_query_count: AtomicU64,
    pub diskann_query_count: AtomicU64,
    pub diskann_mmap_query_count: AtomicU64,
    pub diskann_sector_read_count: AtomicU64,
    pub exact_query_count: AtomicU64,
    pub filtered_query_count: AtomicU64,
    pub scalar_index_query_count: AtomicU64,
    pub radius_query_count: AtomicU64,
    pub candidates_scanned: AtomicU64,
    pub resource_limit_rejections: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryObservation {
    pub kind: QueryKind,
    pub diskann_io_backend: Option<IoBackend>,
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
        if let Some(io_backend) = observation.diskann_io_backend {
            self.diskann_query_count.fetch_add(1, Ordering::Relaxed);
            if io_backend == IoBackend::Mmap {
                self.diskann_mmap_query_count
                    .fetch_add(1, Ordering::Relaxed);
            }
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

    pub fn record_resource_limit_rejection(&self) {
        self.resource_limit_rejections
            .fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IndexType;

    fn index(source_revision: u64, completeness: f32, state: &str) -> IndexStat {
        IndexStat {
            name: "embedding".to_string(),
            index_type: IndexType::Hnsw,
            completeness,
            source_revision,
            document_count: 2,
            estimated_payload_bytes: Some(128),
            state: state.to_string(),
        }
    }

    fn assess(indexes: &[IndexStat], storage_revision: u64) -> CollectionHealth {
        assess_collection_health(CollectionHealthInput {
            is_open: true,
            revision: 7,
            storage_revision,
            doc_count: 2,
            indexes,
            read_only: false,
            wal_ops_since_checkpoint: 0,
            wal_bytes_since_checkpoint: 0,
            maintenance_active: false,
        })
    }

    #[test]
    fn stale_or_non_finite_index_completeness_is_degraded() {
        let stale = assess(&[index(6, 1.0, "ready")], 7);
        assert_eq!(stale.status, CollectionHealthStatus::Degraded);
        assert_eq!(stale.ready_index_count, 0);
        assert!(stale.reasons[0].contains("source revision 6"));

        let incomplete = assess(&[index(7, f32::NAN, "ready")], 7);
        assert_eq!(incomplete.status, CollectionHealthStatus::Degraded);
        assert_eq!(incomplete.ready_index_count, 0);
        assert!(incomplete.reasons[0].contains("completeness"));
    }

    #[test]
    fn authoritative_revision_disagreement_is_unhealthy() {
        let health = assess(&[index(7, 1.0, "ready")], 6);
        assert_eq!(health.status, CollectionHealthStatus::Unhealthy);
        assert_eq!(health.ready_index_count, 1);
        assert!(health.reasons[0].contains("storage revision 6"));
    }

    #[test]
    fn closed_status_is_explicit_even_for_a_consistent_snapshot() {
        let ready = [index(7, 1.0, "ready")];
        let health = assess_collection_health(CollectionHealthInput {
            is_open: false,
            revision: 7,
            storage_revision: 7,
            doc_count: 2,
            indexes: &ready,
            read_only: false,
            wal_ops_since_checkpoint: 0,
            wal_bytes_since_checkpoint: 0,
            maintenance_active: false,
        });
        assert_eq!(health.status, CollectionHealthStatus::Closed);
        assert_eq!(health.ready_index_count, 1);
        assert!(!health.is_healthy());
    }
}
