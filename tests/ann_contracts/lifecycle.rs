#[test]
fn index_lifecycle_tracks_mutations_rebuilds_and_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("lifecycle");
    let path = path.to_str().expect("UTF-8 path");
    let collection = Collection::create(path, &schema(), None).expect("collection must be created");
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("HNSW descriptor must be valid"),
        )
        .expect("empty HNSW index creation must succeed");
    insert_docs(&collection, 24);

    let mut replacement = doc(7);
    replacement
        .add_vector_f32("embedding", &[10.0; 8])
        .expect("replacement vector must be valid");
    collection
        .upsert(&[&replacement])
        .expect("indexed upsert must succeed");
    collection
        .delete(&["doc-0003"])
        .expect("indexed delete must succeed");

    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.doc_count, 23);
    assert_eq!(stats.indexes[0].document_count, 23);
    assert_eq!(stats.indexes[0].source_revision, stats.revision);
    collection
        .rebuild_index("embedding")
        .expect("targeted rebuild must succeed");
    collection
        .optimize()
        .expect("optimize must rebuild ANN indexes");
    collection.close().expect("collection must close cleanly");

    let reopened = Collection::open(path, None).expect("collection must reopen");
    let stats = reopened.stats().expect("stats must be available");
    assert_eq!(stats.indexes[0].source_revision, stats.revision);
    assert_eq!(stats.indexes[0].document_count, 23);
    let mut query = SearchQuery::new("embedding", &[10.0; 8], 1).expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query
        .set_hnsw_params(HnswQueryParams::new(23, 0.0, false, true))
        .expect("HNSW controls must be accepted");
    let result = reopened
        .query(&query)
        .expect("reopened ANN query must succeed");
    assert_eq!(result[0].get_pk(), Some("doc-0007"));

    reopened
        .drop_index("embedding")
        .expect("index drop must succeed");
    assert!(reopened
        .stats()
        .expect("stats must be available")
        .indexes
        .is_empty());
}

#[test]
fn quantized_indexes_keep_authoritative_vectors_and_exact_results() {
    for quantize in [QuantizeType::Fp16, QuantizeType::Int8, QuantizeType::Int4] {
        let temporary = tempdir().expect("temporary directory must be available");
        let collection = Collection::create(
            temporary
                .path()
                .join(format!("quantized-{quantize:?}"))
                .to_str()
                .expect("UTF-8 path"),
            &schema(),
            None,
        )
        .expect("collection must be created");
        insert_docs(&collection, 48);
        let original = collection.fetch(&["doc-0013"]).expect("fetch must succeed")[0]
            .vector("embedding")
            .cloned();
        let mut query =
            SearchQuery::new("embedding", &vector_for(13), 10).expect("query must be valid");
        let exact = collection.query(&query).expect("exact query must succeed");

        let params = IndexParams::hnsw_with_quantize(MetricType::Cosine, 8, 32, quantize)
            .expect("quantized HNSW descriptor must be valid");
        collection
            .create_index("embedding", &params)
            .expect("quantized HNSW index must build");
        query
            .set_hnsw_params(HnswQueryParams::new(48, 0.0, false, true))
            .expect("HNSW controls must be accepted");
        let result = collection.query(&query).expect("ANN query must succeed");
        assert_same_ranking(&exact, &result);
        assert_eq!(
            collection.fetch(&["doc-0013"]).expect("fetch must succeed")[0]
                .vector("embedding")
                .cloned(),
            original
        );
    }
}

#[test]
fn hnsw_recall_fixture_uses_a_bounded_candidate_set() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("recall")
            .to_str()
            .expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 256);
    let mut query =
        SearchQuery::new("embedding", &vector_for(177), 10).expect("query must be valid");
    let exact = collection.query(&query).expect("exact query must succeed");
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index must build");
    query
        .set_hnsw_params(HnswQueryParams::new(32, 0.0, false, true))
        .expect("HNSW controls must be accepted");
    let before = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");
    let approximate = collection.query(&query).expect("ANN query must succeed");
    let after = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");

    let exact_ids = ids(&exact);
    let hits = ids(&approximate)
        .iter()
        .filter(|id| exact_ids.contains(id))
        .count();
    assert!(hits >= 8, "recall@10 was only {hits}/10");
    assert_eq!(after.ann_query_count - before.ann_query_count, 1);
    assert!(after.candidates_scanned - before.candidates_scanned <= 32);
    assert!(after.candidates_scanned - before.candidates_scanned < 256);
}

#[test]
fn hnsw_rabitq_recall_fixture_uses_codes_and_a_bounded_candidate_set() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("hnsw-rabitq-recall")
            .to_str()
            .expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 256);
    let mut query =
        SearchQuery::new("embedding", &vector_for(177), 10).expect("query must be valid");
    let exact = collection.query(&query).expect("exact query must succeed");
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw_rabitq(MetricType::Cosine, 16, 96)
                .expect("HNSW RaBitQ descriptor must be valid"),
        )
        .expect("HNSW RaBitQ index must build");
    query
        .set_hnsw_params(HnswQueryParams::new(32, 0.0, false, true))
        .expect("HNSW RaBitQ controls must be accepted");
    let before = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");
    let approximate = collection
        .query(&query)
        .expect("HNSW RaBitQ query must succeed");
    let after = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");

    let exact_ids = ids(&exact);
    let hits = ids(&approximate)
        .iter()
        .filter(|id| exact_ids.contains(id))
        .count();
    assert!(hits >= 8, "recall@10 was only {hits}/10");
    assert_eq!(after.ann_query_count - before.ann_query_count, 1);
    assert!(after.candidates_scanned - before.candidates_scanned <= 32);
    assert!(after.candidates_scanned - before.candidates_scanned < 256);
}

#[test]
fn ivf_and_soar_recall_fixtures_use_unique_bounded_candidate_sets() {
    for use_soar in [false, true] {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join(format!("ivf-recall-{use_soar}"));
        let collection = Collection::create(path.to_str().expect("UTF-8 path"), &schema(), None)
            .expect("collection must be created");
        insert_docs(&collection, 256);
        let mut query =
            SearchQuery::new("embedding", &vector_for(177), 10).expect("query must be valid");
        let exact = collection.query(&query).expect("exact query must succeed");
        collection
            .create_index(
                "embedding",
                &IndexParams::ivf(MetricType::Cosine, 16, 8, use_soar)
                    .expect("IVF descriptor must be valid"),
            )
            .expect("IVF index must build");
        query
            .set_ivf_params(IvfQueryParams::new(4, true, 8.0))
            .expect("IVF controls must be accepted");
        let before = collection
            .stats_snapshot()
            .expect("stats snapshot must be available");
        let approximate = collection.query(&query).expect("ANN query must succeed");
        let after = collection
            .stats_snapshot()
            .expect("stats snapshot must be available");

        let exact_ids = ids(&exact);
        let hits = ids(&approximate)
            .iter()
            .filter(|id| exact_ids.contains(id))
            .count();
        assert!(hits >= 8, "SOAR={use_soar} recall@10 was only {hits}/10");
        assert_eq!(after.ann_query_count - before.ann_query_count, 1);
        assert!(after.candidates_scanned - before.candidates_scanned <= 80);
        assert!(after.candidates_scanned - before.candidates_scanned < 256);
        collection.close().expect("collection must close");
    }
}

#[test]
fn ivf_rabitq_recall_fixture_uses_codes_and_a_bounded_refiner_set() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("ivf-rabitq-recall")
            .to_str()
            .expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 256);
    let mut query =
        SearchQuery::new("embedding", &vector_for(177), 10).expect("query must be valid");
    let exact = collection.query(&query).expect("exact query must succeed");
    collection
        .create_index(
            "embedding",
            &IndexParams::ivf_rabitq(MetricType::Cosine, 16, 7, 128)
                .expect("IVF RaBitQ descriptor must be valid"),
        )
        .expect("IVF RaBitQ index must build");
    let mut params = IvfRabitqQueryParams::new(4, 0.0, false, true);
    params
        .set_scale_factor(8.0)
        .expect("RaBitQ refinement scale must be valid");
    query
        .set_ivf_rabitq_params(params)
        .expect("IVF RaBitQ controls must be accepted");
    let before = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");
    let approximate = collection
        .query(&query)
        .expect("IVF RaBitQ query must succeed");
    let after = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");

    let exact_ids = ids(&exact);
    let hits = ids(&approximate)
        .iter()
        .filter(|id| exact_ids.contains(id))
        .count();
    assert!(hits >= 8, "recall@10 was only {hits}/10");
    assert_eq!(after.ann_query_count - before.ann_query_count, 1);
    assert!(after.candidates_scanned - before.candidates_scanned <= 80);
    assert!(after.candidates_scanned - before.candidates_scanned < 256);
}

#[test]
fn vamana_recall_fixture_uses_a_bounded_candidate_set() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("vamana-recall")
            .to_str()
            .expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 256);
    let mut query =
        SearchQuery::new("embedding", &vector_for(177), 10).expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    let exact = collection.query(&query).expect("exact query must succeed");
    collection
        .create_index(
            "embedding",
            &IndexParams::vamana(MetricType::L2, 16, 64, 1.2)
                .expect("Vamana descriptor must be valid"),
        )
        .expect("Vamana index must build");
    query
        .set_diskann_params(DiskannQueryParams::new(32))
        .expect("Vamana controls must be accepted");
    let before = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");
    let approximate = collection.query(&query).expect("Vamana query must succeed");
    let after = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");

    let exact_ids = ids(&exact);
    let hits = ids(&approximate)
        .iter()
        .filter(|id| exact_ids.contains(id))
        .count();
    assert!(hits >= 8, "recall@10 was only {hits}/10");
    assert_eq!(after.ann_query_count - before.ann_query_count, 1);
    assert!(after.candidates_scanned - before.candidates_scanned <= 32);
    assert!(after.candidates_scanned - before.candidates_scanned < 256);
}

#[test]
fn unsupported_ann_field_and_mismatched_quantizer_contracts_fail_before_mutation() {
    let mut sparse = FieldSchema::new("sparse", DataType::SparseVectorFp32, false, 0)
        .expect("sparse schema must be valid");
    let error = sparse
        .set_index_params(
            &IndexParams::hnsw(MetricType::Cosine, 8, 32)
                .expect("descriptor must be syntactically valid"),
        )
        .expect_err("HNSW must reject sparse vectors");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!sparse.has_index());

    let mut dense = FieldSchema::new("dense", DataType::VectorFp32, false, 8)
        .expect("dense schema must be valid");
    let params = IndexParams::hnsw_with_quantize(MetricType::Cosine, 8, 32, QuantizeType::Rabitq)
        .expect("descriptor must be syntactically valid");
    let error = dense
        .set_index_params(&params)
        .expect_err("RaBitQ must use its dedicated index family");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(!dense.has_index());

    dense
        .set_index_params(
            &IndexParams::hnsw_rabitq(MetricType::Cosine, 8, 32)
                .expect("HNSW RaBitQ descriptor must be valid"),
        )
        .expect("HNSW RaBitQ must be a live dense-vector configuration");
    assert_eq!(dense.index_type(), IndexType::HnswRabitq);

    let error = IndexParams::ivf_rabitq(MetricType::L2, 8, 10, 0)
        .expect_err("RaBitQ total_bits above nine must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = IndexParams::hnsw_rabitq_with_options(MetricType::L2, 8, 32, 7, 0, 0)
        .expect_err("HNSW RaBitQ requires at least one cluster");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

