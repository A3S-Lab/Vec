//! Deterministic collection-local resource admission and accounting.

use crate::doc::DocumentMap;
use crate::error::{Error, Result};
use crate::index::IndexRegistry;
use crate::schema::CollectionSchema;
use serde::{Deserialize, Deserializer, Serialize};

/// Collection-local logical resource budgets.
///
/// These limits bound published authoritative/derived logical payloads and the
/// explicit candidate/batch admission units. `max_accounted_bytes` is
/// deterministic engine accounting, not a promise about allocator overhead,
/// temporary construction memory, or process RSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub struct CollectionResourceLimits {
    #[serde(rename = "max_documents")]
    documents: Option<u64>,
    #[serde(rename = "max_accounted_bytes")]
    accounted_bytes: Option<u64>,
    #[serde(rename = "max_query_candidates")]
    query_candidates: Option<u64>,
    #[serde(rename = "max_write_batch_documents")]
    write_batch_documents: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct CollectionResourceLimitsWire {
    #[serde(rename = "max_documents")]
    documents: Option<u64>,
    #[serde(rename = "max_accounted_bytes")]
    accounted_bytes: Option<u64>,
    #[serde(rename = "max_query_candidates")]
    query_candidates: Option<u64>,
    #[serde(rename = "max_write_batch_documents")]
    write_batch_documents: Option<u64>,
}

impl<'de> Deserialize<'de> for CollectionResourceLimits {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CollectionResourceLimitsWire::deserialize(deserializer)?;
        let mut limits = Self::new();
        if let Some(limit) = wire.documents {
            limits = limits
                .try_with_max_documents(limit)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(limit) = wire.accounted_bytes {
            limits = limits
                .try_with_max_accounted_bytes(limit)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(limit) = wire.query_candidates {
            limits = limits
                .try_with_max_query_candidates(limit)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(limit) = wire.write_batch_documents {
            limits = limits
                .try_with_max_write_batch_documents(limit)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(limits)
    }
}

impl CollectionResourceLimits {
    /// Creates an unbounded policy. Callers opt into each limit explicitly.
    pub fn new() -> Self {
        Self::default()
    }

    /// Limits the document count of every published generation.
    pub fn try_with_max_documents(mut self, limit: u64) -> Result<Self> {
        self.documents = Some(positive_limit(limit, "max_documents")?);
        Ok(self)
    }

    /// Limits authoritative serialized bytes plus derived payload estimates.
    pub fn try_with_max_accounted_bytes(mut self, limit: u64) -> Result<Self> {
        self.accounted_bytes = Some(positive_limit(limit, "max_accounted_bytes")?);
        Ok(self)
    }

    /// Limits planned exact/refinement candidates for one query operation.
    pub fn try_with_max_query_candidates(mut self, limit: u64) -> Result<Self> {
        self.query_candidates = Some(positive_limit(limit, "max_query_candidates")?);
        Ok(self)
    }

    /// Limits documents targeted by one explicit or filter-derived write batch.
    pub fn try_with_max_write_batch_documents(mut self, limit: u64) -> Result<Self> {
        self.write_batch_documents = Some(positive_limit(limit, "max_write_batch_documents")?);
        Ok(self)
    }

    /// Returns the retained-document limit, or `None` when unbounded.
    pub fn max_documents(self) -> Option<u64> {
        self.documents
    }

    /// Returns the logical accounted-byte limit, or `None` when unbounded.
    pub fn max_accounted_bytes(self) -> Option<u64> {
        self.accounted_bytes
    }

    /// Returns the per-query candidate limit, or `None` when unbounded.
    pub fn max_query_candidates(self) -> Option<u64> {
        self.query_candidates
    }

    /// Returns the per-write document limit, or `None` when unbounded.
    pub fn max_write_batch_documents(self) -> Option<u64> {
        self.write_batch_documents
    }

    pub(super) fn enforce_write_batch(self, documents: usize) -> Result<()> {
        let documents = u64::try_from(documents)
            .map_err(|_| Error::resource_exhausted("write batch exceeds u64 documents"))?;
        if self
            .write_batch_documents
            .is_some_and(|limit| documents > limit)
        {
            return Err(Error::resource_exhausted(format!(
                "write batch document count {documents} exceeds configured limit {}",
                self.write_batch_documents.unwrap_or(u64::MAX)
            )));
        }
        Ok(())
    }

    pub(super) fn enforce_query_candidates(self, candidates: u64) -> Result<()> {
        if self
            .query_candidates
            .is_some_and(|limit| candidates > limit)
        {
            return Err(Error::resource_exhausted(format!(
                "query candidate count {candidates} exceeds configured limit {}",
                self.query_candidates.unwrap_or(u64::MAX)
            )));
        }
        Ok(())
    }

    pub(super) fn enforce_state(
        self,
        schema: &CollectionSchema,
        docs: &DocumentMap,
        indexes: &IndexRegistry,
    ) -> Result<ResourceUsage> {
        let document_count = u64::try_from(docs.len())
            .map_err(|_| Error::resource_exhausted("collection exceeds u64 documents"))?;
        if self.documents.is_some_and(|limit| document_count > limit) {
            return Err(Error::resource_exhausted(format!(
                "collection document count {document_count} exceeds configured limit {}",
                self.documents.unwrap_or(u64::MAX)
            )));
        }

        let usage = ResourceUsage::measure(schema, docs, indexes)?;
        if self
            .accounted_bytes
            .is_some_and(|limit| usage.total > limit)
        {
            return Err(Error::resource_exhausted(format!(
                "collection accounted bytes {} exceeds configured limit {}",
                usage.total,
                self.accounted_bytes.unwrap_or(u64::MAX)
            )));
        }
        Ok(usage)
    }
}

fn positive_limit(limit: u64, name: &str) -> Result<u64> {
    if limit == 0 {
        Err(Error::invalid_argument(format!(
            "resource limit '{name}' must be positive"
        )))
    } else {
        Ok(limit)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResourceUsage {
    pub documents: u64,
    pub indexes: u64,
    pub total: u64,
}

impl ResourceUsage {
    pub(super) fn measure(
        schema: &CollectionSchema,
        docs: &DocumentMap,
        indexes: &IndexRegistry,
    ) -> Result<Self> {
        let document_bytes = bincode::serialized_size(docs)
            .map_err(|error| Error::internal(format!("account document bytes: {error}")))?;
        let estimated_index_bytes = indexes
            .stats(schema)
            .into_iter()
            .filter_map(|index| index.estimated_payload_bytes)
            .fold(0_u64, u64::saturating_add);
        let accounted_bytes = document_bytes.saturating_add(estimated_index_bytes);
        Ok(Self {
            documents: document_bytes,
            indexes: estimated_index_bytes,
            total: accounted_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_explicitly_unbounded() {
        let limits = CollectionResourceLimits::default();
        assert_eq!(limits.max_documents(), None);
        assert_eq!(limits.max_accounted_bytes(), None);
        assert_eq!(limits.max_query_candidates(), None);
        assert_eq!(limits.max_write_batch_documents(), None);
    }

    #[test]
    fn deserialization_preserves_the_typed_invariants() {
        let limits: CollectionResourceLimits =
            serde_json::from_str(r#"{"max_documents":10,"max_query_candidates":20}"#)
                .expect("partial policies must default omitted limits to unbounded");
        assert_eq!(limits.max_documents(), Some(10));
        assert_eq!(limits.max_accounted_bytes(), None);
        assert_eq!(limits.max_query_candidates(), Some(20));
        assert!(
            serde_json::from_str::<CollectionResourceLimits>(r#"{"max_documents":0}"#).is_err()
        );
        assert!(
            serde_json::from_str::<CollectionResourceLimits>(r#"{"max_documentz":10}"#).is_err()
        );
    }

    #[test]
    fn public_policy_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CollectionResourceLimits>();
    }
}
