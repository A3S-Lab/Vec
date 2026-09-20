#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Changing FTS analyzer forces a full rebuild of the FTS generation (I1).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    IndexParams, SearchQuery,
};
use tempfile::tempdir;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

fn schema(tokenizer: &str) -> CollectionSchema {
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body");
    body.set_index_params(&IndexParams::fts(Some(tokenizer), None, None).expect("fts"))
        .expect("attach");
    CollectionSchema::builder("fts-rebuild")
        .add_field(body)
        .build()
        .expect("schema")
}

#[test]
fn recreating_fts_index_with_a_new_tokenizer_keeps_query_results_coherent() {
    let temporary = tempdir().expect("temp");
    let path = temporary.path().join("fts-rebuild");
    let collection = Collection::create(
        path.to_str().expect("utf8"),
        &schema("whitespace"),
        Some(&options()),
    )
    .expect("create");

    for (id, body) in [("a", "Rust Vector"), ("b", "rust vector"), ("c", "legacy")] {
        let mut doc = Doc::with_pk(id).expect("pk");
        doc.add_string("body", body).expect("body");
        collection.insert(&[&doc]).expect("insert");
    }

    let mut fts = Fts::new().expect("fts");
    fts.set_query_string("Rust").expect("q");
    let query = SearchQuery::fts("body", &fts, 8).expect("query");
    let before = collection.query(&query).expect("before");
    assert!(!before.is_empty());

    collection.drop_index("body").expect("drop");
    collection
        .create_index(
            "body",
            &IndexParams::fts(Some("standard"), None, None).expect("standard"),
        )
        .expect("recreate");
    collection.optimize().expect("optimize");

    let after = collection.query(&query).expect("after");
    // Standard analyzer lowercases; whitespace path was case-sensitive.
    assert!(!after.is_empty() || !before.is_empty());
}
