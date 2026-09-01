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
    let revision = next.revision;
    let schema = next.schema.clone();
    let docs: Vec<Doc> = next.docs.values().map(|doc| doc.as_ref().clone()).collect();
    storage.append(
        revision,
        WalOperation::Schema {
            schema: schema.clone(),
            docs: docs.clone(),
        },
        config,
    )?;

    // The WAL + manifest pair is the commit point. Publish the same state in
    // memory before checkpoint maintenance so a checkpoint error cannot leave
    // this process behind the already committed revision.
    *state = next;
    let sync = !matches!(config.durability, Durability::Manual);
    storage.checkpoint(&schema, &docs, revision, sync)?;
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
