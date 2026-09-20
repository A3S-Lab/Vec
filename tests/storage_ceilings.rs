//! Explicit storage `DoS` ceilings — typed policy, not host autodetection.

use a3s_vec::{
    initialize, shutdown, Collection, CollectionOptions, CollectionSchema, ConfigBuilder, DataType,
    Doc, Durability, ErrorCode, FieldSchema, StorageCeilings,
};
use std::sync::{Mutex, MutexGuard};
use tempfile::tempdir;

static PROCESS_CONFIG_LOCK: Mutex<()> = Mutex::new(());

fn schema() -> CollectionSchema {
    CollectionSchema::builder("storage-ceilings")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 8)
                .expect("embedding schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn vector_doc(id: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_vector_f32("embedding", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("vector must be valid");
    doc
}

fn options_with(ceilings: StorageCeilings) -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
        .set_storage_ceilings(ceilings)
        .expect("storage ceilings must be accepted");
    options
}

#[test]
fn zero_storage_ceilings_are_rejected() {
    for result in [
        StorageCeilings::new().try_with_max_snapshot_bytes(0),
        StorageCeilings::new().try_with_max_index_cache_bytes(0),
        StorageCeilings::new().try_with_max_wal_replay_bytes(0),
        StorageCeilings::new().try_with_max_diskann_file_bytes(0),
    ] {
        let error = result.expect_err("zero must not silently disable a typed ceiling");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
}

#[test]
fn tiny_snapshot_ceiling_rejects_flush_fail_closed() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let ceilings = StorageCeilings::new()
        .try_with_max_snapshot_bytes(64)
        .expect("tiny snapshot ceiling must be valid");
    let collection = Collection::create(
        path.to_str().expect("path must be UTF-8"),
        &schema(),
        Some(&options_with(ceilings)),
    )
    .expect("create with empty snapshot must succeed under a tiny ceiling");

    let docs: Vec<Doc> = (0..32).map(|i| vector_doc(&format!("doc-{i}"))).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection
        .insert(&refs)
        .expect("insert into memory must succeed before flush");
    let error = collection
        .flush()
        .expect_err("oversized snapshot must fail closed");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert!(
        error.message.contains("snapshot"),
        "error should name the snapshot ceiling: {}",
        error.message
    );
}

#[test]
fn reopen_with_tighter_snapshot_ceiling_fails_closed() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let path_str = path.to_str().expect("path must be UTF-8");

    let collection = Collection::create(
        path_str,
        &schema(),
        Some(&options_with(StorageCeilings::new())),
    )
    .expect("create must succeed");
    let docs: Vec<Doc> = (0..16).map(|i| vector_doc(&format!("doc-{i}"))).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert must succeed");
    collection
        .flush()
        .expect("flush under default ceilings must succeed");
    drop(collection);

    let tight = StorageCeilings::new()
        .try_with_max_snapshot_bytes(128)
        .expect("tight ceiling must be valid");
    let error = Collection::open(path_str, Some(&options_with(tight)))
        .expect_err("reopen under a ceiling below the snapshot must fail");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
}

#[test]
fn process_default_ceilings_apply_without_collection_override() {
    let process = ConfigBuilder::new().storage_ceilings(
        StorageCeilings::new()
            .try_with_max_snapshot_bytes(96)
            .expect("process ceiling must be valid"),
    );
    let _guard = ProcessConfigGuard::install(&process);
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    // No set_storage_ceilings — process default must apply.
    let collection = Collection::create(
        path.to_str().expect("path must be UTF-8"),
        &schema(),
        Some(&options),
    )
    .expect("create must succeed");
    assert_eq!(
        collection
            .stats()
            .expect("stats must succeed")
            .storage_ceilings
            .max_snapshot_bytes(),
        96
    );

    let docs: Vec<Doc> = (0..32).map(|i| vector_doc(&format!("doc-{i}"))).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert must succeed");
    let error = collection
        .flush()
        .expect_err("process-default tiny snapshot ceiling must reject flush");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
}

#[test]
fn collection_override_wins_over_process_default() {
    let process = ConfigBuilder::new().storage_ceilings(
        StorageCeilings::new()
            .try_with_max_snapshot_bytes(64)
            .expect("tight process ceiling must be valid"),
    );
    let _guard = ProcessConfigGuard::install(&process);
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let roomy = StorageCeilings::new()
        .try_with_max_snapshot_bytes(8 * 1024 * 1024)
        .expect("roomy ceiling must be valid");
    let collection = Collection::create(
        path.to_str().expect("path must be UTF-8"),
        &schema(),
        Some(&options_with(roomy)),
    )
    .expect("create must succeed");
    assert_eq!(
        collection
            .stats()
            .expect("stats must succeed")
            .storage_ceilings
            .max_snapshot_bytes(),
        8 * 1024 * 1024
    );
    let docs: Vec<Doc> = (0..8).map(|i| vector_doc(&format!("doc-{i}"))).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert must succeed");
    collection
        .flush()
        .expect("collection override must allow flush despite tight process default");
}

struct ProcessConfigGuard {
    _lock: MutexGuard<'static, ()>,
}

impl ProcessConfigGuard {
    fn install(config: &ConfigBuilder) -> Self {
        let lock = PROCESS_CONFIG_LOCK
            .lock()
            .expect("process config lock must not be poisoned");
        initialize(Some(config)).expect("process config must initialize");
        Self { _lock: lock }
    }
}

impl Drop for ProcessConfigGuard {
    fn drop(&mut self) {
        let _ = shutdown();
    }
}
