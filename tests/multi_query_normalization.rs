#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Multi-query fusion + score normalization contracts (I1 / deterministic fusion).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    IndexParams, MetricType, MultiQuery, SubQuery,
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
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body");
    body.set_index_params(&IndexParams::fts(Some("whitespace"), None, None).expect("fts"))
        .expect("attach fts");
    CollectionSchema::builder("multi-norm")
        .add_field(body)
        .add_field(embedding)
        .build()
        .expect("schema")
}

#[test]
fn weighted_and_rrf_fusion_with_normalization_are_deterministic() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("mq").to_str().expect("utf8"),
        &schema(),
        Some(&options()),
    )
    .expect("create");
    for (id, body, vector) in [
        ("a", "rust vector", [1.0_f32, 0.0]),
        ("b", "rust database", [0.9, 0.1]),
        ("c", "python vector", [0.0, 1.0]),
        ("d", "legacy index", [0.5, 0.5]),
    ] {
        let mut doc = Doc::with_pk(id).expect("pk");
        doc.add_string("body", body).expect("body");
        doc.add_vector_f32("embedding", &vector).expect("vector");
        collection.insert(&[&doc]).expect("insert");
    }
    collection.optimize().expect("optimize");

    let mut vector_branch = SubQuery::new().expect("sub");
    vector_branch.set_field_name("embedding").expect("field");
    vector_branch.set_query_vector(&[1.0, 0.0]).expect("vector");
    vector_branch.set_num_candidates(4).expect("k");

    let mut fts_branch = SubQuery::new().expect("sub");
    fts_branch.set_field_name("body").expect("field");
    let mut fts = Fts::new().expect("fts");
    fts.set_query_string("rust").expect("q");
    fts_branch.set_fts(&fts).expect("fts");
    fts_branch.set_num_candidates(4).expect("k");

    for (norm, rerank) in [
        ("minmax", "weighted"),
        ("zscore", "weighted"),
        ("none", "rrf"),
        ("minmax", "rrf"),
    ] {
        let mut multi = MultiQuery::new().expect("multi");
        multi.add_sub_query(&vector_branch).expect("add");
        multi.add_sub_query(&fts_branch).expect("add");
        multi.set_topk(3).expect("topk");
        multi.set_normalization(norm).expect("norm");
        match rerank {
            "weighted" => multi.set_rerank_weighted(&[0.6, 0.4]).expect("w"),
            "rrf" => multi.set_rerank_rrf(60).expect("rrf"),
            _ => unreachable!(),
        }
        let first = collection.multi_query(&multi).expect("first");
        let second = collection.multi_query(&multi).expect("second");
        let ids = |docs: &[Doc]| {
            docs.iter()
                .map(|doc| {
                    (
                        doc.get_pk().unwrap_or_default().to_string(),
                        doc.get_score().to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&first), ids(&second), "norm={norm} rerank={rerank}");
        assert!(!first.is_empty());
    }

    // Unknown normalization fails closed through the query engine.
    let mut multi = MultiQuery::new().expect("multi");
    multi.add_sub_query(&vector_branch).expect("add");
    multi.set_topk(2).expect("topk");
    // Bypass MultiQuery setter to force an unknown method into the engine.
    multi.normalization = Some("bogus".into());
    assert!(collection.multi_query(&multi).is_err());
}
