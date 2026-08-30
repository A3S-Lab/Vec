use super::{snapshot, wal, StorageHandle, WalOperation};
use crate::config::{ConfigBuilder, Durability};
use crate::doc::Doc;
use crate::error::ErrorCode;
use crate::schema::{CollectionSchema, FieldSchema};
use crate::types::DataType;
use tempfile::tempdir;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("storage-tests")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("test field schema must be valid"),
        )
        .build()
        .expect("test collection schema must be valid")
}

fn doc(id: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("test primary key must be valid");
    doc.add_string("title", "stored title")
        .expect("test field value must be valid");
    doc
}

#[test]
fn orphaned_snapshot_generation_does_not_replace_manifest_state() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let schema = schema();
    let stored_doc = doc("doc-1");
    let mut storage =
        StorageHandle::create(&root, &schema, false).expect("storage must be created");
    storage
        .append(
            1,
            WalOperation::Insert {
                docs: vec![stored_doc.clone()],
            },
            &ConfigBuilder::default(),
        )
        .expect("WAL append must commit");

    snapshot::write(&root, &schema, &[stored_doc], 2, 1, true)
        .expect("orphaned generation must be writable");
    drop(storage);

    let (recovered, recovered_schema, docs) =
        StorageHandle::open(&root, false).expect("old manifest state must recover");
    assert_eq!(recovered.manifest.generation, 1);
    assert_eq!(recovered.manifest.revision, 1);
    assert_eq!(recovered_schema, schema);
    assert_eq!(docs.len(), 1);
}

#[test]
fn partial_uncommitted_wal_tail_is_ignored_and_replaced_by_the_next_commit() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let schema = schema();
    let storage = StorageHandle::create(&root, &schema, false).expect("storage must be created");
    let record = wal::WalRecord::new(
        1,
        WalOperation::Insert {
            docs: vec![doc("uncommitted")],
        },
    )
    .expect("test WAL record must be valid");
    let frame_bytes = wal::append(&root, storage.manifest.wal_active_seq, 0, &record, true)
        .expect("uncommitted WAL tail must be written");
    let wal_path = wal::segment_path(&root, storage.manifest.wal_active_seq);
    std::fs::OpenOptions::new()
        .write(true)
        .open(wal_path)
        .expect("uncommitted WAL tail must be writable by the test")
        .set_len(frame_bytes - 3)
        .expect("test must leave a partial final frame");
    drop(storage);

    let (mut recovered, _, docs) =
        StorageHandle::open(&root, false).expect("committed state must recover");
    assert!(docs.is_empty());
    assert_eq!(recovered.manifest.revision, 0);

    recovered
        .append(
            1,
            WalOperation::Insert {
                docs: vec![doc("committed")],
            },
            &ConfigBuilder::default(),
        )
        .expect("next committed append must replace the tail");
    drop(recovered);

    let (recovered, _, docs) =
        StorageHandle::open(&root, false).expect("new committed state must recover");
    assert_eq!(recovered.manifest.revision, 1);
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].get_pk(), Some("committed"));
}

#[test]
fn schema_wal_record_recovers_schema_and_backfilled_documents() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let initial_schema = schema();
    let mut next_schema = initial_schema.clone();
    next_schema
        .add_field(
            &FieldSchema::new("category", DataType::String, false, 0)
                .expect("test field schema must be valid"),
        )
        .expect("test schema change must be valid");
    let mut stored_doc = doc("doc-1");
    stored_doc
        .add_string("category", "reference")
        .expect("backfilled value must be valid");

    let mut storage =
        StorageHandle::create(&root, &initial_schema, false).expect("storage must be created");
    storage
        .append(
            1,
            WalOperation::Schema {
                schema: next_schema.clone(),
                docs: vec![stored_doc],
            },
            &ConfigBuilder::default(),
        )
        .expect("schema WAL record must commit");
    drop(storage);

    let (recovered, recovered_schema, docs) =
        StorageHandle::open(&root, false).expect("schema WAL must recover");
    assert_eq!(recovered.manifest.revision, 1);
    assert_eq!(recovered_schema, next_schema);
    assert_eq!(docs.len(), 1);
    assert_eq!(
        docs[0]
            .get_string("category")
            .expect("recovered field type must match"),
        Some("reference".to_string())
    );
}

#[test]
fn committed_wal_checksum_corruption_is_rejected() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let mut storage =
        StorageHandle::create(&root, &schema(), false).expect("storage must be created");
    storage
        .append(
            1,
            WalOperation::Insert {
                docs: vec![doc("doc-1")],
            },
            &ConfigBuilder::default(),
        )
        .expect("WAL append must commit");
    let wal_path = wal::segment_path(&root, storage.manifest.wal_active_seq);
    drop(storage);

    let mut bytes = std::fs::read(&wal_path).expect("committed WAL must be readable");
    let last = bytes
        .last_mut()
        .expect("committed WAL frame must contain a payload");
    *last ^= 0xff;
    std::fs::write(&wal_path, bytes).expect("test must corrupt committed WAL payload");

    let error = StorageHandle::open(&root, false)
        .expect_err("checksum corruption inside committed WAL must fail recovery");
    assert_eq!(error.code, ErrorCode::InternalError);
    assert!(error.message.contains("WAL checksum mismatch"));
}

#[test]
fn truncated_committed_wal_is_rejected() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let mut storage =
        StorageHandle::create(&root, &schema(), false).expect("storage must be created");
    storage
        .append(
            1,
            WalOperation::Insert {
                docs: vec![doc("doc-1")],
            },
            &ConfigBuilder::default(),
        )
        .expect("WAL append must commit");
    let committed_bytes = storage.manifest.wal_bytes_since_checkpoint;
    let wal_path = wal::segment_path(&root, storage.manifest.wal_active_seq);
    drop(storage);

    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal_path)
        .expect("committed WAL must be writable by the test")
        .set_len(committed_bytes - 1)
        .expect("test must truncate committed WAL");

    let error = StorageHandle::open(&root, false)
        .expect_err("a physical WAL shorter than the manifest boundary must fail recovery");
    assert_eq!(error.code, ErrorCode::InternalError);
    assert!(error
        .message
        .contains("shorter than the committed boundary"));
}

#[test]
fn oversized_snapshot_is_rejected_before_allocation() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let storage = StorageHandle::create(&root, &schema(), false).expect("storage must be created");
    let generation = storage.manifest.generation;
    drop(storage);

    let snapshot_path = root
        .join("segments")
        .join(format!("snapshot-{generation:020}.json"));
    std::fs::OpenOptions::new()
        .write(true)
        .open(snapshot_path)
        .expect("snapshot must be writable by the test")
        .set_len(snapshot::MAX_SNAPSHOT_BYTES + 1)
        .expect("test must create an oversized sparse snapshot");

    let error = StorageHandle::open(&root, false)
        .expect_err("oversized snapshots must be rejected before deserialization");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert!(error.message.contains("recovery limit"));
}

#[test]
fn interval_checkpoint_limits_are_consumed_by_storage() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let mut storage =
        StorageHandle::create(&root, &schema(), false).expect("storage must be created");
    let operation_limit = ConfigBuilder::default()
        .durability(Durability::Interval)
        .wal_max_ops(2);

    storage
        .append(
            1,
            WalOperation::Insert {
                docs: vec![doc("doc-1")],
            },
            &operation_limit,
        )
        .expect("first WAL append must commit");
    assert!(!storage.should_checkpoint(&operation_limit));
    storage
        .append(
            2,
            WalOperation::Insert {
                docs: vec![doc("doc-2")],
            },
            &operation_limit,
        )
        .expect("second WAL append must commit");
    assert!(storage.should_checkpoint(&operation_limit));

    let byte_limit = ConfigBuilder::default()
        .durability(Durability::Interval)
        .wal_max_bytes(1);
    assert!(storage.should_checkpoint(&byte_limit));
}
