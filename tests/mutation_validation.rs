#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Fail-closed mutation validation oracles (schema owns meaning / I2).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, ErrorCode,
    FieldSchema, IndexParams, MetricType, WriteResult,
};
use tempfile::tempdir;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

fn schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("embedding");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let mut tag = FieldSchema::new("tag", DataType::String, false, 0).expect("tag");
    tag.set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach invert");
    let nullable = FieldSchema::new("note", DataType::String, true, 0).expect("note");
    CollectionSchema::builder("mutation-validation")
        .add_field(tag)
        .add_field(nullable)
        .add_field(embedding)
        .build()
        .expect("schema")
}

fn dense_doc(id: &str, tag: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("pk");
    doc.add_string("tag", tag).expect("tag");
    doc.add_vector_f32("embedding", &[1.0, 0.0, 0.0, 0.0])
        .expect("vector");
    doc
}

fn assert_failed(result: &WriteResult, code: ErrorCode) {
    assert_eq!(result.success_count, 0);
    assert_eq!(result.error_count, 1);
    assert_eq!(result.results[0].code, code);
    assert!(!result.results[0].is_success());
}

#[test]
fn insert_update_upsert_reject_schema_and_existence_violations() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("mv").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");

    let ok = dense_doc("a", "rust");
    collection.insert(&[&ok]).expect("insert a");

    // Duplicate insert
    let dup = collection
        .insert(&[&ok])
        .expect("dup insert returns outcomes");
    assert_failed(&dup, ErrorCode::AlreadyExists);

    // Update missing
    let missing = dense_doc("missing", "go");
    let upd = collection.update(&[&missing]).expect("update missing");
    assert_failed(&upd, ErrorCode::NotFound);

    // Upsert ok
    let upserted = collection.upsert(&[&missing]).expect("upsert");
    assert_eq!(upserted.success_count, 1);

    // Unknown scalar field
    let mut unknown = Doc::with_pk("b").expect("pk");
    unknown.add_string("tag", "ok").expect("tag");
    unknown.add_string("ghost", "nope").expect("ghost");
    unknown
        .add_vector_f32("embedding", &[0.0, 1.0, 0.0, 0.0])
        .expect("vector");
    let bad_field = collection.insert(&[&unknown]).expect("unknown field");
    assert_failed(&bad_field, ErrorCode::InvalidArgument);

    // Dimension mismatch
    let mut dim = Doc::with_pk("c").expect("pk");
    dim.add_string("tag", "ok").expect("tag");
    dim.add_vector_f32("embedding", &[1.0, 0.0]).expect("short");
    let bad_dim = collection.insert(&[&dim]).expect("dim");
    assert_failed(&bad_dim, ErrorCode::InvalidArgument);

    // Duplicate primary key inside one batch
    let left = dense_doc("batch", "a");
    let right = dense_doc("batch", "b");
    let batch = collection.insert(&[&left, &right]).expect("batch");
    assert_eq!(batch.success_count, 1);
    assert_eq!(batch.error_count, 1);
    assert_eq!(batch.results[1].code, ErrorCode::AlreadyExists);

    // Type mismatch on scalar
    let mut typed = Doc::with_pk("typed").expect("pk");
    typed.add_i32("tag", 1).expect("wrong type");
    typed
        .add_vector_f32("embedding", &[0.0, 0.0, 1.0, 0.0])
        .expect("vector");
    let bad_type = collection.insert(&[&typed]).expect("type");
    assert_failed(&bad_type, ErrorCode::InvalidArgument);

    // Rebuild / optimize keep indexes coherent after rejected mutations
    collection.optimize().expect("optimize");
    collection
        .rebuild_index("embedding")
        .expect("rebuild embedding");
    collection.rebuild_index("tag").expect("rebuild tag");
    assert!(collection.rebuild_index("missing").is_err());
}

#[test]
fn delete_and_filter_mutations_and_nullable_fields_fail_closed() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("mv2").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");

    let a = dense_doc("a", "rust");
    let b = dense_doc("b", "go");
    collection.insert(&[&a, &b]).expect("insert");

    // Nullable field may be omitted on upsert.
    let mut with_note = Doc::with_pk("c").expect("pk");
    with_note.add_string("tag", "rust").expect("tag");
    with_note.add_string("note", "hello").expect("note");
    with_note
        .add_vector_f32("embedding", &[0.0, 0.0, 0.0, 1.0])
        .expect("vector");
    collection.upsert(&[&with_note]).expect("upsert note");

    collection
        .delete_by_filter("tag = \"go\"")
        .expect("delete by filter");

    let missing_delete = collection.delete(&["missing"]).expect("delete missing");
    assert_eq!(missing_delete.error_count, 1);

    // Non-nullable null rejected.
    let mut null_tag = Doc::with_pk("d").expect("pk");
    null_tag
        .set_field_value("tag", a3s_vec::FieldValue::Null)
        .expect("set null");
    null_tag
        .add_vector_f32("embedding", &[0.0, 1.0, 0.0, 0.0])
        .expect("vector");
    let bad_null = collection.insert(&[&null_tag]).expect("null tag");
    assert_failed(&bad_null, ErrorCode::InvalidArgument);

    // Missing required non-nullable scalar on insert
    let mut missing_tag = Doc::with_pk("e").expect("pk");
    missing_tag
        .add_vector_f32("embedding", &[1.0, 0.0, 0.0, 0.0])
        .expect("vector");
    let bad_missing = collection.insert(&[&missing_tag]).expect("missing tag");
    assert_failed(&bad_missing, ErrorCode::InvalidArgument);

    // Unknown vector field
    let mut ghost_vec = dense_doc("f", "rust");
    ghost_vec
        .set_vector_value(
            "ghost",
            a3s_vec::VectorValue::Fp32(vec![1.0, 0.0, 0.0, 0.0]),
        )
        .expect("ghost");
    let bad_vec = collection.insert(&[&ghost_vec]).expect("ghost vector");
    assert_failed(&bad_vec, ErrorCode::InvalidArgument);

    // Empty filter delete is a no-op success path.
    collection
        .delete_by_filter("tag = \"does-not-exist\"")
        .expect("empty delete");
    assert_eq!(collection.count().expect("count"), 2);

    // Duplicate primary keys inside one batch accept the first and reject the rest.
    let left = dense_doc("batch-a", "rust");
    let right = dense_doc("batch-a", "go");
    let batch = collection.insert(&[&left, &right]).expect("batch outcomes");
    assert_eq!(batch.success_count, 1);
    assert_eq!(batch.error_count, 1);
    assert_eq!(batch.results[1].code, ErrorCode::AlreadyExists);
    assert_eq!(collection.count().expect("count after duplicate batch"), 3);

    // Delete batch rejects empty/NUL/duplicate keys without applying them.
    let delete_batch = collection
        .delete(&["", "batch-a", "batch-a", "missing-again"])
        .expect("delete batch");
    assert_eq!(delete_batch.success_count, 1);
    assert_eq!(delete_batch.error_count, 3);
    assert_eq!(delete_batch.results[0].code, ErrorCode::InvalidArgument);
    assert_eq!(delete_batch.results[2].code, ErrorCode::AlreadyExists);
    assert_eq!(delete_batch.results[3].code, ErrorCode::NotFound);
    assert_eq!(collection.count().expect("count after delete batch"), 2);

    let nul_delete = collection.delete(&["has\0nul"]).expect("nul delete");
    assert_eq!(nul_delete.error_count, 1);
    assert_eq!(nul_delete.results[0].code, ErrorCode::InvalidArgument);
}
