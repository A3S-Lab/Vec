//! Checkpoint and optional derived-index cache maintenance.

use super::{next_revision, CollectionState};
use crate::config::{ConfigBuilder, Durability};
use crate::doc::Doc;
use crate::error::Result;
use crate::index::IndexRegistry;
use crate::schema::CollectionSchema;
use crate::storage::StorageHandle;
use crate::storage::WalOperation;

pub(super) fn commit_prepared_schema_change(
    storage: &mut StorageHandle,
    state: &mut CollectionState,
    next: CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    if next.revision != next_revision(state.revision)? {
        return Err(crate::error::Error::failed_precondition(
            "prepared schema generation no longer follows collection state",
        ));
    }
    // Schema-only revisions (index lifecycle changes and metadata-only
    // alterations) must not copy the entire document set into one WAL frame.
    // Pointer identity is the collection's immutable-generation signal: any
    // schema operation that changed documents receives a new document map.
    let revision = next.revision;
    let schema = next.schema.clone();
    let docs = (!std::sync::Arc::ptr_eq(&next.docs, &state.docs)).then(|| {
        next.docs
            .values()
            .map(|doc| doc.as_ref().clone())
            .collect::<Vec<Doc>>()
    });
    let operation = match docs.as_ref() {
        Some(docs) => WalOperation::Schema {
            schema: schema.clone(),
            docs: docs.clone(),
        },
        None => WalOperation::SchemaOnly {
            schema: schema.clone(),
        },
    };
    storage.append(revision, operation, config)?;

    // The WAL + manifest pair is the commit point. Publish the same state in
    // memory before checkpoint maintenance so a checkpoint error cannot leave
    // this process behind the already committed revision.
    *state = next;
    let checkpoint_docs = docs.unwrap_or_else(|| {
        state
            .docs
            .values()
            .map(|doc| doc.as_ref().clone())
            .collect()
    });
    let sync = !matches!(config.durability, Durability::Manual);
    storage.checkpoint(&schema, &checkpoint_docs, revision, sync)?;
    persist_index_cache(storage, &schema, &state.indexes, revision, sync);
    Ok(())
}

pub(super) fn maybe_checkpoint(
    storage: &mut StorageHandle,
    state: &CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    let should =
        matches!(config.durability, Durability::Interval) && storage.should_checkpoint(config);
    if should {
        let docs: Vec<Doc> = state
            .docs
            .values()
            .map(|doc| doc.as_ref().clone())
            .collect();
        storage.checkpoint(&state.schema, &docs, state.revision, true)?;
        persist_index_cache(storage, &state.schema, &state.indexes, state.revision, true);
    }
    Ok(())
}

pub(super) fn persist_index_cache(
    storage: &StorageHandle,
    schema: &CollectionSchema,
    indexes: &IndexRegistry,
    revision: u64,
    sync: bool,
) {
    if !indexes.has_cacheable_indexes() {
        return;
    }
    let identity = storage.index_cache_identity();
    let Ok(diskann_bytes) = indexes.diskann_bytes(schema, revision, &identity) else {
        return;
    };
    if let Some(diskann_bytes) = diskann_bytes {
        if storage.write_diskann_file(&diskann_bytes, sync).is_err() {
            return;
        }
    }
    let Ok(bytes) = indexes.cache_bytes(schema, revision, &identity) else {
        return;
    };
    let _cache_result = storage.write_index_cache(&bytes, sync);
}
