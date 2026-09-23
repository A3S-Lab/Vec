//! Deterministic collection-local resource admission and accounting.

use crate::doc::{Doc, DocumentMap};
use crate::error::{Error, Result};
use crate::index::IndexRegistry;
use crate::schema::CollectionSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;

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
        let usage = ResourceUsage::measure(schema, docs, indexes)?;
        self.admit(document_count, usage)
    }

    pub(super) fn admit(self, document_count: u64, usage: ResourceUsage) -> Result<ResourceUsage> {
        if self.documents.is_some_and(|limit| document_count > limit) {
            return Err(Error::resource_exhausted(format!(
                "collection document count {document_count} exceeds configured limit {}",
                self.documents.unwrap_or(u64::MAX)
            )));
        }
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
        _schema: &CollectionSchema,
        docs: &DocumentMap,
        indexes: &IndexRegistry,
    ) -> Result<Self> {
        Ok(Self::from_parts(
            document_map_bytes(docs)?,
            indexes.accounted_payload_bytes(),
        ))
    }

    /// Updates document accounting from the changed primary keys only.
    ///
    /// `bincode` encodes an `OrdMap` as one length prefix plus each key and
    /// value. The prefix width does not grow with the entry count, so a
    /// generation can add and remove entry sizes without walking documents
    /// that this batch did not touch. A drift or overflow falls back to a
    /// full measurement.
    pub(super) fn after_documents(
        &self,
        previous_docs: &DocumentMap,
        next_docs: &DocumentMap,
        changed_ids: &BTreeSet<String>,
        indexes: &IndexRegistry,
    ) -> Result<Self> {
        let mut documents = self.documents;
        for id in changed_ids {
            let before = document_entry_bytes(id, previous_docs.get(id))?;
            let after = document_entry_bytes(id, next_docs.get(id))?;
            if before > documents {
                return Ok(Self::from_parts(
                    document_map_bytes(next_docs)?,
                    indexes.accounted_payload_bytes(),
                ));
            }
            documents -= before;
            documents = match documents.checked_add(after) {
                Some(documents) => documents,
                None => {
                    return Ok(Self::from_parts(
                        document_map_bytes(next_docs)?,
                        indexes.accounted_payload_bytes(),
                    ));
                }
            };
        }
        Ok(Self::from_parts(
            documents,
            indexes.accounted_payload_bytes(),
        ))
    }

    fn from_parts(documents: u64, indexes: u64) -> Self {
        Self {
            documents,
            indexes,
            total: documents.saturating_add(indexes),
        }
    }
}

fn document_map_bytes(docs: &DocumentMap) -> Result<u64> {
    bincode::serialized_size(docs)
        .map_err(|error| Error::internal(format!("account document bytes: {error}")))
}

fn document_entry_bytes(id: &str, doc: Option<&Arc<Doc>>) -> Result<u64> {
    let Some(doc) = doc else {
        return Ok(0);
    };
    let key = bincode::serialized_size(id)
        .map_err(|error| Error::internal(format!("account document key: {error}")))?;
    let value = bincode::serialized_size(doc)
        .map_err(|error| Error::internal(format!("account document value: {error}")))?;
    Ok(key.saturating_add(value))
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

    #[test]
    fn document_map_bincode_size_is_the_header_plus_entries() {
        use crate::doc::{Doc, DocumentMap};
        use std::sync::Arc;

        let mut docs = DocumentMap::new();
        let mut running = bincode::serialized_size(&docs).expect("empty map size");
        for index in 0..6 {
            let mut doc = Doc::with_pk(format!("doc-{index}")).expect("primary key");
            let coordinate = f32::from(u8::try_from(index).expect("fixture index"));
            doc.add_vector_f32("embedding", &[coordinate, -1.0, 0.25])
                .expect("vector");
            if index % 2 == 0 {
                doc.add_string("body", "kept").expect("field");
            }
            let id = doc.get_pk().expect("pk").to_string();
            let stored = Arc::new(doc);
            running += super::document_entry_bytes(&id, Some(&stored)).expect("entry");
            docs.insert(id, stored);
            assert_eq!(
                bincode::serialized_size(&docs).expect("map size"),
                running,
                "insert {index}"
            );
        }
        let removed_id = "doc-2".to_string();
        let removed = docs.get(&removed_id).expect("present").clone();
        running -= super::document_entry_bytes(&removed_id, Some(&removed)).expect("removed entry");
        docs.remove(&removed_id);
        assert_eq!(bincode::serialized_size(&docs).expect("map size"), running);

        let replaced_id = "doc-4".to_string();
        let previous = docs.get(&replaced_id).expect("present").clone();
        let mut replacement = Doc::with_pk(replaced_id.clone()).expect("primary key");
        replacement
            .add_vector_f32("embedding", &[9.0, 8.0, 7.0])
            .expect("vector");
        let stored = Arc::new(replacement);
        running -= super::document_entry_bytes(&replaced_id, Some(&previous)).expect("old entry");
        running += super::document_entry_bytes(&replaced_id, Some(&stored)).expect("new entry");
        docs.insert(replaced_id, stored);
        assert_eq!(bincode::serialized_size(&docs).expect("map size"), running);
    }
}
