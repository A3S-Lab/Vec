//! Checkpoint and optional derived-index cache maintenance.

use super::{next_revision, CollectionState};
use crate::config::{ConfigBuilder, Durability};
use crate::doc::Doc;
use crate::error::Result;
use crate::index::IndexRegistry;
use crate::schema::CollectionSchema;
use crate::storage::StorageHandle;
use crate::storage::WalOperation;

pub(super) fn append_prepared_schema_change(
    storage: &mut StorageHandle,
    previous_docs: &std::sync::Arc<crate::doc::DocumentMap>,
    previous_revision: u64,
    next: &CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    if next.revision != next_revision(previous_revision)? {
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
    let docs = (!std::sync::Arc::ptr_eq(&next.docs, previous_docs)).then(|| {
        next.docs
            .values()
            .map(|doc| doc.as_ref().clone())
            .collect::<Vec<Doc>>()
    });
    let operation = match docs.as_ref() {
        Some(docs) => WalOperation::Schema {
            schema,
            docs: docs.clone(),
        },
        None => WalOperation::SchemaOnly { schema },
    };
    storage.append(revision, operation, config)?;
    Ok(())
}

pub(super) fn publish_prepared_schema_change(
    storage: &mut StorageHandle,
    state: &mut CollectionState,
    next: CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    // The WAL + manifest pair is already the commit point. Publish that state
    // before checkpoint maintenance so a checkpoint error cannot leave this
    // process behind the committed revision.
    let revision = next.revision;
    let schema = next.schema.clone();
    *state = next;
    let sync = !matches!(config.durability, Durability::Manual);
    storage.checkpoint(&schema, state.docs.as_ref(), revision, sync)?;
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
        storage.checkpoint(&state.schema, state.docs.as_ref(), state.revision, true)?;
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
    // The sidecar is derived. A failed DiskANN write must still persist the
    // cache of the other indexes; reopen keeps the in-memory graph for that field.
    if let Ok(Some(diskann_bytes)) =
        indexes.diskann_bytes(schema, revision, &identity, storage.ceilings)
    {
        let _sidecar = storage.write_diskann_file(&diskann_bytes, sync);
    }
    let Ok(bytes) = indexes.cache_bytes(schema, revision, &identity, storage.ceilings) else {
        return;
    };
    let _cache_result = storage.write_index_cache(&bytes, sync);
}
