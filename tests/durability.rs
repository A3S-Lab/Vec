use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema,
};
use tempfile::tempdir;

fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("durability")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("test field schema must be valid"),
        )
        .build()
        .expect("test collection schema must be valid")
}

fn test_doc(id: &str, title: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("test primary key must be valid");
    doc.add_string("title", title)
        .expect("test field value must be valid");
    doc
}

#[test]
fn read_only_create_is_rejected_without_touching_the_filesystem() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let mut options = CollectionOptions::new().expect("default options must be valid");
    options
        .set_read_only(true)
        .expect("read-only mode must be configurable");

    let error = Collection::create(
        collection_path
            .to_str()
            .expect("temporary path must be valid UTF-8"),
        &test_schema(),
        Some(&options),
    )
    .expect_err("creating a collection through a read-only handle must fail");

    assert_eq!(error.code, ErrorCode::PermissionDenied);
    assert!(!collection_path.exists());
}

#[test]
fn read_only_close_is_side_effect_free() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");

    let collection = Collection::create(path, &test_schema(), None)
        .expect("writable collection must be created");
    let doc = test_doc("doc-1", "durable title");
    let result = collection
        .insert(&[&doc])
        .expect("document insert must succeed");
    assert_eq!(result.success_count, 1);
    collection.close().expect("writable close must checkpoint");

    let manifest_before = std::fs::read(collection_path.join("manifest.json"))
        .expect("manifest must exist after writable close");
    let mut options = CollectionOptions::new().expect("default options must be valid");
    options
        .set_read_only(true)
        .expect("read-only mode must be configurable");

    let collection =
        Collection::open(path, Some(&options)).expect("read-only collection must open");
    let docs = collection
        .fetch(&["doc-1"])
        .expect("read-only fetch must succeed");
    assert_eq!(docs.len(), 1);
    collection
        .close()
        .expect("closing a read-only collection must not checkpoint");

    let manifest_after = std::fs::read(collection_path.join("manifest.json"))
        .expect("manifest must remain available");
    assert_eq!(manifest_after, manifest_before);
}

#[test]
fn recovery_advances_revision_for_uncheckpointed_wal_records() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");

    let collection =
        Collection::create(path, &test_schema(), None).expect("collection must be created");
    let doc = test_doc("doc-1", "replayed title");
    collection
        .insert(&[&doc])
        .expect("document insert must succeed");
    assert_eq!(
        collection.stats().expect("stats must be readable").revision,
        1
    );

    // Dropping the last handle models a process exit after the durable WAL
    // append but before an explicit checkpoint.
    drop(collection);

    let recovered = Collection::open(path, None).expect("collection must recover from WAL");
    assert_eq!(recovered.count().expect("count must be readable"), 1);
    assert_eq!(
        recovered.stats().expect("stats must be readable").revision,
        1
    );
    recovered.close().expect("recovered collection must close");
}

#[test]
fn read_only_open_does_not_recreate_a_missing_lock_file() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");
    Collection::create(path, &test_schema(), None)
        .expect("collection must be created")
        .close()
        .expect("writable collection must close");

    let lock_path = collection_path.join(".a3s-vec.lock");
    std::fs::remove_file(&lock_path).expect("test lock file must be removable");
    let mut options = CollectionOptions::new().expect("default options must be valid");
    options
        .set_read_only(true)
        .expect("read-only mode must be configurable");

    Collection::open(path, Some(&options))
        .expect_err("read-only open must require the existing lock authority");
    assert!(!lock_path.exists());
}

#[test]
fn every_document_mutation_recovers_at_a_monotonic_revision() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");

    let collection =
        Collection::create(path, &test_schema(), None).expect("collection must be created");
    let first = test_doc("doc-1", "first");
    collection.insert(&[&first]).expect("insert must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("insert must recover");
    assert_eq!(collection.stats().expect("stats must load").revision, 1);
    let updated_doc = test_doc("doc-1", "updated");
    collection
        .update(&[&updated_doc])
        .expect("update must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("update must recover");
    assert_eq!(collection.stats().expect("stats must load").revision, 2);
    assert_eq!(
        collection
            .fetch(&["doc-1"])
            .expect("updated document must load")[0]
            .get_string("title")
            .expect("title type must match"),
        Some("updated".to_string())
    );
    let replacement = test_doc("doc-1", "upserted");
    let second = test_doc("doc-2", "second");
    collection
        .upsert(&[&replacement, &second])
        .expect("upsert batch must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("upsert must recover");
    assert_eq!(collection.stats().expect("stats must load").revision, 3);
    assert_eq!(collection.count().expect("count must load"), 2);
    collection.delete(&["doc-1"]).expect("delete must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("delete must recover");
    assert_eq!(collection.stats().expect("stats must load").revision, 4);
    assert_eq!(collection.count().expect("count must load"), 1);
    assert_eq!(
        collection
            .fetch(&["doc-2"])
            .expect("survivor must load")
            .len(),
        1
    );
    collection.close().expect("collection must close");
}

#[test]
fn schema_changes_and_backfills_survive_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");
    let collection =
        Collection::create(path, &test_schema(), None).expect("collection must be created");
    let stored_doc = test_doc("doc-1", "schema test");
    collection
        .insert(&[&stored_doc])
        .expect("insert must succeed");
    let category =
        FieldSchema::new("category", DataType::String, false, 0).expect("field must be valid");
    collection
        .add_column(&category, Some("'reference'"))
        .expect("column add and backfill must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("schema addition must recover");
    assert!(collection
        .schema()
        .expect("schema must load")
        .has_field("category"));
    assert_eq!(
        collection.fetch(&["doc-1"]).expect("document must load")[0]
            .get_string("category")
            .expect("category type must match"),
        Some("reference".to_string())
    );
    collection
        .rename_column("category", "kind")
        .expect("column rename must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("schema rename must recover");
    let schema = collection.schema().expect("schema must load");
    assert!(!schema.has_field("category"));
    assert!(schema.has_field("kind"));
    collection
        .drop_column("kind")
        .expect("column drop must succeed");
    drop(collection);

    let collection = Collection::open(path, None).expect("schema drop must recover");
    assert!(!collection
        .schema()
        .expect("schema must load")
        .has_field("kind"));
    assert_eq!(collection.stats().expect("stats must load").revision, 4);
    collection.close().expect("collection must close");
}

#[test]
fn checkpoints_publish_one_generation_specific_snapshot() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");
    let collection =
        Collection::create(path, &test_schema(), None).expect("collection must be created");
    let stored_doc = test_doc("doc-1", "snapshot test");
    collection
        .insert(&[&stored_doc])
        .expect("insert must succeed");
    let committed_manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(collection_path.join("manifest.json"))
            .expect("committed manifest must be readable"),
    )
    .expect("committed manifest must be valid JSON");
    let wal_sequence = committed_manifest["wal_active_seq"]
        .as_u64()
        .expect("WAL sequence must be an integer");
    let wal = std::fs::read(
        collection_path
            .join("wal")
            .join(format!("wal-{wal_sequence:020}.bin")),
    )
    .expect("WAL frame must be readable");
    assert_eq!(&wal[..4], b"A3VW");
    assert_eq!(u16::from_le_bytes([wal[4], wal[5]]), 3);
    collection.flush().expect("first checkpoint must succeed");
    collection.flush().expect("second checkpoint must succeed");

    assert!(!collection_path.join("snapshot.json").exists());
    let snapshots = std::fs::read_dir(collection_path.join("segments"))
        .expect("snapshot directory must exist")
        .collect::<std::io::Result<Vec<_>>>()
        .expect("snapshot directory entries must be readable");
    assert_eq!(snapshots.len(), 1);
    assert!(snapshots[0]
        .file_name()
        .to_string_lossy()
        .starts_with("snapshot-"));
    assert_eq!(
        snapshots[0]
            .path()
            .extension()
            .and_then(|value| value.to_str()),
        Some("bin")
    );
    assert!(!std::fs::read(snapshots[0].path())
        .expect("snapshot must be readable")
        .is_empty());

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(collection_path.join("manifest.json")).expect("manifest must be readable"),
    )
    .expect("manifest must be valid JSON");
    assert_eq!(manifest["format_version"], 4);
    assert_eq!(manifest["revision"], 1);
    assert_eq!(manifest["checkpoint_revision"], 1);
    collection.close().expect("collection must close");
}

#[test]
fn format_two_is_rejected_before_vector_payloads_can_be_reinterpreted() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path
        .to_str()
        .expect("temporary path must be valid UTF-8");
    Collection::create(path, &test_schema(), None)
        .expect("collection must be created")
        .close()
        .expect("collection must close");

    let manifest_path = collection_path.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("manifest must be readable"))
            .expect("manifest must be valid JSON");
    manifest["format_version"] = serde_json::json!(2);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("manifest must serialize"),
    )
    .expect("old manifest fixture must be written");

    let error = Collection::open(path, None).expect_err("format 2 must fail closed");
    assert_eq!(error.code, ErrorCode::NotSupported);
}
