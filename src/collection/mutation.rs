//! Serialized document mutations and atomic generation publication.

use super::checkpoint::maybe_checkpoint;
use super::query_engine::{matches_filter, parse_filter_expression};
use super::validation::{merge_patch, prepare_mutation_batch};
use super::{ensure_same_generation, ensure_writable, next_revision, Collection};
use crate::doc::Doc;
use crate::error::{Error, ErrorCode, Result};
use crate::index::IndexRegistry;
use crate::storage::WalOperation;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

/// Per-document outcome in a batch write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocWriteResult {
    pub success: bool,
    pub code: ErrorCode,
    pub message: String,
}

impl DocWriteResult {
    pub fn is_success(&self) -> bool {
        self.success
    }
}

/// Result of a batch write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    pub success_count: u64,
    pub error_count: u64,
    pub results: Vec<DocWriteResult>,
}

impl Collection {
    pub fn insert(&self, docs: &[&Doc]) -> Result<WriteResult> {
        self.mutate_documents(docs, Mutation::Insert)
    }

    pub fn update(&self, docs: &[&Doc]) -> Result<WriteResult> {
        self.mutate_documents(docs, Mutation::Update)
    }

    pub fn upsert(&self, docs: &[&Doc]) -> Result<WriteResult> {
        self.mutate_documents(docs, Mutation::Upsert)
    }

    pub fn delete(&self, pks: &[&str]) -> Result<WriteResult> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;
        if let Err(error) = current
            .options
            .resource_limits
            .enforce_write_batch(pks.len())
        {
            current.stats.record_resource_limit_rejection();
            return Err(error);
        }
        let mut results = Vec::with_capacity(pks.len());
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for pk in pks {
            if pk.is_empty() {
                results.push(write_error(
                    ErrorCode::InvalidArgument,
                    "primary key is empty",
                ));
            } else if !seen.insert(*pk) {
                results.push(write_error(
                    ErrorCode::AlreadyExists,
                    "duplicate primary key in delete batch",
                ));
            } else if current.docs.contains_key(*pk) {
                ids.push((*pk).to_string());
                results.push(write_success());
            } else {
                results.push(write_error(
                    ErrorCode::NotFound,
                    format!("document '{pk}' not found"),
                ));
            }
        }
        if !ids.is_empty() {
            self.publish_deletion(&current, &ids)?;
        }
        Ok(write_result(results))
    }

    pub fn delete_by_filter(&self, filter: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;
        let parsed_filter = parse_filter_expression(filter)?;
        let indexed = current
            .indexes
            .scalar_candidates(current.revision, &parsed_filter);
        let ids: Vec<String> = if let Some(indexed) = indexed {
            indexed
                .ids()
                .filter_map(|id| current.docs.get(id))
                .filter(|doc| matches_filter(doc, Some(&parsed_filter)))
                .filter_map(|doc| doc.get_pk().map(str::to_string))
                .collect()
        } else {
            current
                .docs
                .values()
                .filter(|doc| matches_filter(doc, Some(&parsed_filter)))
                .filter_map(|doc| doc.get_pk().map(str::to_string))
                .collect()
        };
        if ids.is_empty() {
            return Ok(());
        }
        if let Err(error) = current
            .options
            .resource_limits
            .enforce_write_batch(ids.len())
        {
            current.stats.record_resource_limit_rejection();
            return Err(error);
        }
        self.publish_deletion(&current, &ids)
    }

    fn publish_deletion(&self, current: &super::CollectionState, ids: &[String]) -> Result<()> {
        let config = current.config.clone();
        let revision = next_revision(current.revision)?;
        let mut next_docs = current.docs.as_ref().clone();
        for id in ids {
            next_docs.remove(id);
        }
        let changed_ids = ids.iter().cloned().collect::<BTreeSet<_>>();
        let incremental_indexes = current.indexes.apply_document_changes(
            &current.schema,
            current.docs.as_ref(),
            &next_docs,
            revision,
            &changed_ids,
        )?;
        let (next_indexes, next_resource_usage) = match current
            .options
            .resource_limits
            .enforce_state(&current.schema, &next_docs, &incremental_indexes)
        {
            Ok(usage) => (incremental_indexes, usage),
            Err(error) if error.code == ErrorCode::ResourceExhausted => {
                // A tombstone overlay can be larger than the removed payload.
                // Compact before rejecting so deletion remains a practical way
                // to return the collection below its configured budget.
                let compacted = IndexRegistry::build(&current.schema, &next_docs, revision)?;
                match current.options.resource_limits.enforce_state(
                    &current.schema,
                    &next_docs,
                    &compacted,
                ) {
                    Ok(usage) => (compacted, usage),
                    Err(error) => {
                        current.stats.record_resource_limit_rejection();
                        return Err(error);
                    }
                }
            }
            Err(error) => return Err(error),
        };
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, current)?;
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(
            revision,
            WalOperation::Delete { ids: ids.to_vec() },
            &config,
        )?;
        state.docs = Arc::new(next_docs);
        state.indexes = Arc::new(next_indexes);
        state.revision = revision;
        state.resource_usage = next_resource_usage;
        maybe_checkpoint(&mut storage, &state, &config)?;
        Ok(())
    }

    fn mutate_documents(&self, docs: &[&Doc], mutation: Mutation) -> Result<WriteResult> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;

        if let Err(error) = current
            .options
            .resource_limits
            .enforce_write_batch(docs.len())
        {
            current.stats.record_resource_limit_rejection();
            return Err(error);
        }

        let (accepted, outcomes) = prepare_mutation_batch(&current, docs, mutation);
        if accepted.is_empty() {
            return Ok(write_result(outcomes));
        }
        let operation = match mutation {
            Mutation::Insert => WalOperation::Insert {
                docs: accepted.clone(),
            },
            Mutation::Update => WalOperation::Update {
                docs: accepted.clone(),
            },
            Mutation::Upsert => WalOperation::Upsert {
                docs: accepted.clone(),
            },
        };
        let config = current.config.clone();
        let revision = next_revision(current.revision)?;
        let mut next_docs = current.docs.as_ref().clone();
        for doc in &accepted {
            let Some(pk) = doc.get_pk().map(str::to_string) else {
                continue;
            };
            match mutation {
                Mutation::Insert | Mutation::Upsert => {
                    next_docs.insert(pk, Arc::new(doc.clone()));
                }
                Mutation::Update => {
                    if let Some(existing) = next_docs.get_mut(&pk) {
                        merge_patch(Arc::make_mut(existing), doc)?;
                    }
                }
            }
        }
        let changed_ids = accepted
            .iter()
            .filter_map(|doc| doc.get_pk().map(str::to_string))
            .collect::<BTreeSet<_>>();
        let next_indexes = current.indexes.apply_document_changes(
            &current.schema,
            current.docs.as_ref(),
            &next_docs,
            revision,
            &changed_ids,
        )?;
        let next_resource_usage = match current.options.resource_limits.enforce_state(
            &current.schema,
            &next_docs,
            &next_indexes,
        ) {
            Ok(usage) => usage,
            Err(error) => {
                current.stats.record_resource_limit_rejection();
                return Err(error);
            }
        };
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, &current)?;
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(revision, operation, &config)?;
        state.docs = Arc::new(next_docs);
        state.indexes = Arc::new(next_indexes);
        state.revision = revision;
        state.resource_usage = next_resource_usage;
        maybe_checkpoint(&mut storage, &state, &config)?;
        Ok(write_result(outcomes))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Mutation {
    Insert,
    Update,
    Upsert,
}

pub(super) fn write_success() -> DocWriteResult {
    DocWriteResult {
        success: true,
        code: ErrorCode::Unknown,
        message: String::new(),
    }
}

pub(super) fn write_error(code: ErrorCode, message: impl Into<String>) -> DocWriteResult {
    DocWriteResult {
        success: false,
        code,
        message: message.into(),
    }
}

fn write_result(results: Vec<DocWriteResult>) -> WriteResult {
    let success_count = results.iter().filter(|result| result.success).count() as u64;
    WriteResult {
        success_count,
        error_count: results.len() as u64 - success_count,
        results,
    }
}
