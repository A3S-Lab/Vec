#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Schema rename / alter fail-closed and document rewrite contracts (I2).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    IndexParams, MetricType, SearchQuery,
};
use tempfile::tempdir;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

fn schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let tag = FieldSchema::new("tag", DataType::String, false, 0).expect("tag");
    CollectionSchema::builder("schema-evo")
        .add_field(tag)
        .add_field(embedding)
        .build()
        .expect("schema")
}

#[test]
fn rename_scalar_and_vector_columns_rewrites_documents_and_rejects_conflicts() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("evo").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");

    let mut doc = Doc::with_pk("a").expect("pk");
    doc.add_string("tag", "rust").expect("tag");
    doc.add_vector_f32("embedding", &[1.0, 0.0]).expect("vec");
    collection.insert(&[&doc]).expect("insert");

    assert!(collection.rename_column("", "x").is_err());
    assert!(collection.rename_column("tag", "").is_err());
    assert!(collection.rename_column("missing", "other").is_err());
    collection.rename_column("tag", "tag").expect("noop");
    assert!(collection.rename_column("tag", "embedding").is_err());

    collection
        .rename_column("tag", "label")
        .expect("rename scalar");
    collection
        .rename_column("embedding", "vector")
        .expect("rename vector");

    let query = SearchQuery::new("vector", &[1.0, 0.0], 4).expect("query");
    let hits = collection.query(&query).expect("query after rename");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].get_pk(), Some("a"));
    match hits[0].field("label") {
        Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "rust"),
        other => panic!("expected renamed label, got {other:?}"),
    }
}

#[test]
fn alter_and_add_column_paths_cover_schema_mutations() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("evo2").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");
    let mut doc = Doc::with_pk("a").expect("pk");
    doc.add_string("tag", "rust").expect("tag");
    doc.add_vector_f32("embedding", &[1.0, 0.0]).expect("vec");
    collection.insert(&[&doc]).expect("insert");

    let note = FieldSchema::new("note", DataType::String, true, 0).expect("note");
    collection
        .add_column(&note, None)
        .expect("add nullable column");

    // Adding a duplicate field fails closed.
    assert!(collection.add_column(&note, None).is_err());

    // Dropping an unknown field fails closed.
    assert!(collection.drop_column("missing").is_err());

    collection.drop_column("note").expect("drop added column");
}

#[test]
fn alter_column_rejects_type_changes_and_accepts_compatible_updates() {
    use a3s_vec::AlterColumnOption;

    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("evo3").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");
    let mut doc = Doc::with_pk("a").expect("pk");
    doc.add_string("tag", "rust").expect("tag");
    doc.add_vector_f32("embedding", &[1.0, 0.0]).expect("vec");
    collection.insert(&[&doc]).expect("insert");

    let wrong_type = FieldSchema::new("tag", DataType::Int32, false, 0).expect("int");
    assert!(collection
        .alter_column(&wrong_type, AlterColumnOption::default())
        .is_err());

    let missing = FieldSchema::new("ghost", DataType::String, true, 0).expect("ghost");
    assert!(collection
        .alter_column(&missing, AlterColumnOption::default())
        .is_err());

    let nullable_tag = FieldSchema::new("tag", DataType::String, true, 0).expect("nullable");
    collection
        .alter_column(&nullable_tag, AlterColumnOption::default())
        .expect("nullable flip");
}

#[test]
fn closed_and_read_only_collections_reject_mutations() {
    let temporary = tempdir().expect("temp");
    let path = temporary.path().join("ro");
    {
        let collection =
            Collection::create(path.to_str().expect("utf8"), &schema(), Some(&options()))
                .expect("create");
        let mut doc = Doc::with_pk("a").expect("pk");
        doc.add_string("tag", "rust").expect("tag");
        doc.add_vector_f32("embedding", &[1.0, 0.0]).expect("vec");
        collection.insert(&[&doc]).expect("insert");
        collection.close().expect("close");
    }

    let mut ro_options = options();
    ro_options.set_read_only(true).expect("ro");
    let reopened =
        Collection::open(path.to_str().expect("utf8"), Some(&ro_options)).expect("open read-only");
    let mut doc = Doc::with_pk("b").expect("pk");
    doc.add_string("tag", "go").expect("tag");
    doc.add_vector_f32("embedding", &[0.0, 1.0]).expect("vec");
    assert!(reopened.insert(&[&doc]).is_err());
    assert!(reopened.drop_column("tag").is_err());
    assert!(reopened.optimize().is_err());
    assert!(reopened.rename_column("tag", "label").is_err());
    assert!(reopened
        .add_column(
            &FieldSchema::new("note", DataType::String, true, 0).expect("note"),
            None
        )
        .is_err());
}

#[test]
fn destroy_removes_collection_directory() {
    let temporary = tempdir().expect("temp");
    let path = temporary.path().join("destroy-me");
    let collection = Collection::create(path.to_str().expect("utf8"), &schema(), Some(&options()))
        .expect("create");
    assert!(path.exists());
    collection.destroy().expect("destroy");
    assert!(!path.exists());
}
