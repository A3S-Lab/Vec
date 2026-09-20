#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! MultiQuery / SubQuery builder contract coverage (TESTING.md F-P1-7).

use a3s_vec::{
    ErrorCode, FlatQueryParams, Fts, FtsQueryParams, HnswQueryParams, MultiQuery, RerankMethod,
    SubQuery,
};

#[test]
fn sub_query_routes_and_controls_are_exclusive_and_typed() {
    let mut dense = SubQuery::new().expect("sub");
    dense.set_field_name("embedding").expect("field");
    dense.set_num_candidates(7).expect("n");
    dense.set_query_vector(&[1.0, 0.0]).expect("dense");
    assert_eq!(dense.num_candidates(), 7);
    assert!(dense.set_num_candidates(0).is_err());
    assert!(dense.set_field_name("").is_err());
    assert!(dense.set_query_vector(&[]).is_err());
    assert!(dense.set_query_vector(&[f32::NAN]).is_err());

    let mut binary = SubQuery::new().expect("sub");
    binary.set_field_name("bits").expect("field");
    binary.set_binary_vector(&[1, 2, 3, 4]).expect("bits");
    assert!(binary.set_binary_vector(&[]).is_err());

    let mut sparse = SubQuery::new().expect("sub");
    sparse.set_field_name("sparse").expect("field");
    sparse
        .set_sparse_vector(&[0, 2], &[1.0, 0.5])
        .expect("sparse");
    sparse.set_sparse_indices(&[0, 3]).expect("indices");
    sparse.set_sparse_values(&[0.25, 0.75]).expect("values");
    assert!(sparse.set_sparse_vector(&[], &[]).is_err());
    assert!(sparse.set_sparse_values(&[f32::INFINITY]).is_err());

    let mut fts = SubQuery::new().expect("sub");
    fts.set_field_name("body").expect("field");
    let mut expression = Fts::new().expect("fts");
    expression.set_match_string("alpha").expect("match");
    fts.set_fts(&expression).expect("set fts");
    assert!(fts.set_fts(&Fts::new().expect("empty")).is_err());

    dense
        .set_hnsw_params(HnswQueryParams::new(16, 0.0, false, false))
        .expect("hnsw");
    assert_eq!(
        dense
            .set_flat_params(FlatQueryParams::new(false, 1.0))
            .unwrap_err()
            .code,
        ErrorCode::NotSupported
    );
    dense
        .set_ivf_params(a3s_vec::IvfQueryParams::new(4, false, 1.0))
        .expect("ivf");
    dense
        .set_ivf_rabitq_params(a3s_vec::IvfRabitqQueryParams::new(4, 0.0, false, false))
        .expect("rabitq");
    dense
        .set_diskann_params(a3s_vec::DiskannQueryParams::new(32))
        .expect("diskann");
    dense
        .set_fts_params(FtsQueryParams::new(None).expect("fts params"))
        .expect("fts params accepted on builder");

    let mut sparse_mismatch = SubQuery::new().expect("sub");
    assert!(sparse_mismatch.set_sparse_vector(&[0, 1], &[1.0]).is_err());
    sparse_mismatch
        .set_sparse_values(&[1.0, 2.0])
        .expect("values first");
    assert!(sparse_mismatch.set_sparse_indices(&[0]).is_err());
    assert!(sparse_mismatch.set_sparse_indices(&[]).is_err());
}

#[test]
fn multi_query_fusion_controls_validate_and_default() {
    let mut multi = MultiQuery::new().expect("multi");
    assert_eq!(multi.topk(), 10);
    assert_eq!(multi.sub_query_count(), 0);
    assert!(matches!(
        multi.rerank,
        RerankMethod::Weighted { ref weights } if weights.is_empty()
    ));

    let mut branch = SubQuery::new().expect("sub");
    branch.set_field_name("embedding").expect("field");
    branch.set_query_vector(&[1.0, 0.0]).expect("vector");
    multi.add_sub_query(&branch).expect("add");
    assert_eq!(multi.sub_query_count(), 1);

    multi.set_topk(5).expect("topk");
    assert!(multi.set_topk(0).is_err());
    multi.set_filter("bucket == 1").expect("filter");
    multi.set_filter("   ").expect("empty clears");
    multi.set_include_vector(true).expect("include");
    assert!(multi.include_vector());
    multi
        .set_output_fields(&["embedding", "bucket"])
        .expect("fields");
    assert!(multi.set_output_fields(&[""]).is_err());
    multi.set_rerank_rrf(60).expect("rrf");
    assert!(matches!(
        multi.rerank,
        RerankMethod::ReciprocalRank { rank_constant } if (rank_constant - 60.0).abs() < f64::EPSILON
    ));
    assert!(multi.set_rerank_rrf(0).is_err());
    multi.set_rerank_weighted(&[0.7, 0.3]).expect("weighted");
    assert!(multi.set_rerank_weighted(&[]).is_err());
    assert!(multi.set_rerank_weighted(&[-1.0]).is_err());
    multi.set_normalization("minmax").expect("norm");
    multi.set_normalization("ZSCORE").expect("case");
    multi.set_normalization("none").expect("none");
    assert!(multi.set_normalization("bogus").is_err());
}

#[test]
fn multi_query_rrf_and_weighted_execute_deterministically() {
    use a3s_vec::{
        Collection, CollectionSchema, DataType, Doc, FieldSchema, IndexParams, MetricType,
        SearchQuery,
    };
    use tempfile::tempdir;

    let temporary = tempdir().expect("temp");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::Ip).expect("flat"))
        .expect("index");
    let schema = CollectionSchema::builder("multi-oracle")
        .add_field(embedding)
        .build()
        .expect("schema");
    let collection = Collection::create(
        temporary.path().join("c").to_str().expect("utf8"),
        &schema,
        None,
    )
    .expect("create");
    for (id, vector) in [("a", [1.0_f32, 0.0]), ("b", [0.0, 1.0]), ("c", [0.7, 0.7])] {
        let mut doc = Doc::with_pk(id).expect("pk");
        doc.add_vector_f32("embedding", &vector).expect("vec");
        collection.insert(&[&doc]).expect("insert");
    }
    collection.optimize().expect("optimize");

    let mut left = SubQuery::new().expect("left");
    left.set_field_name("embedding").expect("field");
    left.set_query_vector(&[1.0, 0.0]).expect("q");
    let mut right = SubQuery::new().expect("right");
    right.set_field_name("embedding").expect("field");
    right.set_query_vector(&[0.0, 1.0]).expect("q");

    let mut rrf = MultiQuery::new().expect("rrf");
    rrf.add_sub_query(&left).expect("l");
    rrf.add_sub_query(&right).expect("r");
    rrf.set_topk(3).expect("topk");
    rrf.set_rerank_rrf(60).expect("rrf");
    let rrf_hits = collection.multi_query(&rrf).expect("rrf exec");
    assert_eq!(rrf_hits.len(), 3);
    let rrf_ids: Vec<_> = rrf_hits.iter().map(|doc| doc.get_pk()).collect();
    let again = collection.multi_query(&rrf).expect("rrf repeat");
    let again_ids: Vec<_> = again.iter().map(|doc| doc.get_pk()).collect();
    assert_eq!(rrf_ids, again_ids);

    let mut weighted = MultiQuery::new().expect("w");
    weighted.add_sub_query(&left).expect("l");
    weighted.add_sub_query(&right).expect("r");
    weighted.set_topk(2).expect("topk");
    weighted.set_rerank_weighted(&[1.0, 0.0]).expect("w");
    let weighted_hits = collection.multi_query(&weighted).expect("w exec");
    assert_eq!(weighted_hits.len(), 2);
    // Weighting fully toward the left IP query should prefer "a".
    assert_eq!(weighted_hits[0].get_pk(), Some("a"));

    let direct = collection
        .query(&SearchQuery::new("embedding", &[1.0, 0.0], 1).expect("q"))
        .expect("direct");
    assert_eq!(direct[0].get_pk(), Some("a"));

    let empty = MultiQuery::new().expect("empty");
    assert!(collection
        .multi_query(&empty)
        .expect_err("empty multi-query must fail")
        .message
        .contains("at least one"));
}
