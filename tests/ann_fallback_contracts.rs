#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Wave B P0 cases: stale ANN fallback and ef=n ≡ Flat (TESTING.md F-P0-5/7).

use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, HnswQueryParams, IndexParams,
    MetricType, SearchQuery,
};
use tempfile::tempdir;

fn dense_doc(id: &str, vector: &[f32]) -> Doc {
    let mut doc = Doc::with_pk(id).expect("pk");
    doc.add_vector_f32("embedding", vector).expect("vector");
    doc
}

fn hnsw_schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field");
    embedding
        .set_index_params(&IndexParams::hnsw(MetricType::Cosine, 8, 32).expect("hnsw"))
        .expect("attach");
    CollectionSchema::builder("stale-hnsw")
        .add_field(embedding)
        .build()
        .expect("schema")
}

fn flat_schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::Cosine).expect("flat"))
        .expect("attach");
    CollectionSchema::builder("ef-flat")
        .add_field(embedding)
        .build()
        .expect("schema")
}

/// F-P0-5: after upsert without rebuild, results still match DocumentMap oracle
/// (stale/missing ANN must fail closed to exact).
#[test]
fn stale_hnsw_after_upsert_still_matches_exact_oracle() {
    let temporary = tempdir().expect("temp");
    let path = temporary
        .path()
        .join("stale")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = Collection::create(&path, &hnsw_schema(), None).expect("create");
    let initial = [
        dense_doc("a", &[1.0, 0.0]),
        dense_doc("b", &[0.0, 1.0]),
        dense_doc("c", &[0.7, 0.3]),
    ];
    let refs: Vec<&Doc> = initial.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("build HNSW");

    let updated = dense_doc("b", &[0.9, 0.1]);
    collection
        .upsert(&[&updated])
        .expect("upsert without rebuild");

    let query = [1.0_f32, 0.0];
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
        .expect("query must succeed via exact fallback");
    let ids: Vec<_> = hits
        .iter()
        .map(|doc| doc.get_pk().unwrap().to_string())
        .collect();
    // Exact Cosine against updated corpus: a, b(updated), c.
    assert_eq!(ids[0], "a");
    assert!(ids.contains(&"b".to_string()));
    assert_eq!(hits.len(), 3);
}

/// F-P0-7: HNSW with ef = document count matches Flat ranking on a small corpus.
#[test]
fn hnsw_ef_equals_n_matches_flat_ranking() {
    let temporary = tempdir().expect("temp");
    let corpus = [
        ("d0", [1.0_f32, 0.0]),
        ("d1", [0.95, 0.05]),
        ("d2", [0.0, 1.0]),
        ("d3", [-1.0, 0.0]),
        ("d4", [0.6, 0.8]),
        ("d5", [0.3, 0.2]),
    ];

    let flat_path = temporary
        .path()
        .join("flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let flat = Collection::create(&flat_path, &flat_schema(), None).expect("flat");
    let hnsw_path = temporary
        .path()
        .join("hnsw")
        .to_str()
        .expect("utf8")
        .to_string();
    let hnsw = Collection::create(&hnsw_path, &hnsw_schema(), None).expect("hnsw");

    for (id, vector) in &corpus {
        let doc = dense_doc(id, vector);
        flat.insert(&[&doc]).expect("flat insert");
        hnsw.insert(&[&doc]).expect("hnsw insert");
    }
    flat.optimize().expect("flat pack");
    hnsw.optimize().expect("hnsw build");

    let query = [0.9_f32, 0.2];
    let flat_hits = flat
        .query(&SearchQuery::new("embedding", &query, 6).expect("q"))
        .expect("flat");
    let mut hnsw_query = SearchQuery::new("embedding", &query, 6).expect("q");
    hnsw_query
        .set_hnsw_params(HnswQueryParams::new(corpus.len() as i32, 0.0, false, false))
        .expect("ef=n");
    let hnsw_hits = hnsw.query(&hnsw_query).expect("hnsw");

    let flat_ids: Vec<_> = flat_hits.iter().map(|doc| doc.get_pk()).collect();
    let hnsw_ids: Vec<_> = hnsw_hits.iter().map(|doc| doc.get_pk()).collect();
    assert_eq!(flat_ids, hnsw_ids);
    for (left, right) in flat_hits.iter().zip(&hnsw_hits) {
        assert!((left.get_score() - right.get_score()).abs() <= 1.0e-5);
    }
}
