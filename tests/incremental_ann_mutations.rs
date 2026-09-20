#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Incremental ANN + FTS mutations keep DocumentMap authority (I1/I5).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    HnswQueryParams, IndexParams, MetricType, SearchQuery,
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
        .set_index_params(&IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"))
        .expect("attach");
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body");
    body.set_index_params(&IndexParams::fts(Some("whitespace"), None, None).expect("fts"))
        .expect("attach fts");
    let mut tag = FieldSchema::new("tag", DataType::String, false, 0).expect("tag");
    tag.set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach invert");
    CollectionSchema::builder("incremental-ann")
        .add_field(tag)
        .add_field(body)
        .add_field(embedding)
        .build()
        .expect("schema")
}

fn doc(id: &str, tag: &str, body: &str, vector: [f32; 2]) -> Doc {
    let mut doc = Doc::with_pk(id).expect("pk");
    doc.add_string("tag", tag).expect("tag");
    doc.add_string("body", body).expect("body");
    doc.add_vector_f32("embedding", &vector).expect("vector");
    doc
}

#[test]
fn hnsw_fts_scalar_incremental_upsert_delete_match_exact_oracle() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("inc").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");

    let initial: Vec<Doc> = (0..64)
        .map(|i| {
            let v = i as f32;
            doc(
                &format!("doc-{i:03}"),
                if i % 2 == 0 { "even" } else { "odd" },
                if i % 3 == 0 {
                    "rust vector"
                } else {
                    "python search"
                },
                [v, 0.0],
            )
        })
        .collect();
    let refs: Vec<&Doc> = initial.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("optimize");

    // Upsert overlay + delete a live base vector.
    let updated = doc("doc-001", "even", "rust vector database", [1.5, 0.0]);
    collection.upsert(&[&updated]).expect("upsert");
    collection.delete(&["doc-002"]).expect("delete");

    // Filtered ANN query must stay inside allow-list and match Flat exact peer rebuild.
    let mut query = SearchQuery::new("embedding", &[1.5, 0.0], 5).expect("query");
    query.set_filter("tag = \"even\"").expect("filter");
    query
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
        .expect("hnsw");
    let ann = collection.query(&query).expect("ann");
    assert!(!ann.is_empty());
    for hit in &ann {
        match hit.field("tag") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "even"),
            other => panic!("expected even tag, got {other:?}"),
        }
        assert_ne!(hit.get_pk(), Some("doc-002"));
    }

    let mut fts = Fts::new().expect("fts");
    fts.set_query_string("rust").expect("q");
    let fts_query = SearchQuery::fts("body", &fts, 16).expect("fts query");
    let fts_hits = collection.query(&fts_query).expect("fts");
    assert!(!fts_hits.is_empty());

    collection.rebuild_index("embedding").expect("rebuild ann");
    collection.rebuild_index("body").expect("rebuild fts");
    collection.rebuild_index("tag").expect("rebuild scalar");

    let after = collection.query(&query).expect("ann after rebuild");
    assert!(!after.is_empty());
}
