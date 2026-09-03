#[test]
fn hnsw_exhaustive_search_matches_the_exact_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary.path().join("hnsw").to_str().expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 96);

    let mut query =
        SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
    let exact = collection.query(&query).expect("exact query must succeed");

    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::Cosine, 12, 64).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index creation must succeed");
    query
        .set_hnsw_params(HnswQueryParams::new(96, 0.0, false, true))
        .expect("HNSW controls must be accepted");
    let exhaustive = collection.query(&query).expect("HNSW query must succeed");
    assert_same_ranking(&exact, &exhaustive);

    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes.len(), 1);
    assert_eq!(stats.indexes[0].index_type, IndexType::Hnsw);
    assert_eq!(stats.indexes[0].state, "ready");
    assert_eq!(stats.indexes[0].source_revision, stats.revision);
    assert_eq!(stats.indexes[0].document_count, 96);
}

#[test]
fn ivf_exhaustive_search_matches_the_exact_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary.path().join("ivf").to_str().expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 96);

    let mut query =
        SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
    let exact = collection.query(&query).expect("exact query must succeed");

    collection
        .create_index(
            "embedding",
            &IndexParams::ivf(MetricType::Cosine, 12, 7, false)
                .expect("IVF descriptor must be valid"),
        )
        .expect("IVF index creation must succeed");
    query
        .set_ivf_params(IvfQueryParams::new(12, true, 1.0))
        .expect("IVF controls must be accepted");
    let exhaustive = collection.query(&query).expect("IVF query must succeed");
    assert_same_ranking(&exact, &exhaustive);
}

#[test]
fn ivf_soar_matches_the_exact_oracle_and_reopens_from_cache() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("ivf-soar");
    let collection = Collection::create(path.to_str().expect("UTF-8 path"), &schema(), None)
        .expect("collection must be created");
    insert_docs(&collection, 96);

    let mut query =
        SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
    let exact = collection.query(&query).expect("exact query must succeed");
    collection
        .create_index(
            "embedding",
            &IndexParams::ivf(MetricType::Cosine, 12, 7, true)
                .expect("IVF SOAR descriptor must be valid"),
        )
        .expect("IVF SOAR index creation must succeed");
    query
        .set_ivf_params(IvfQueryParams::new(12, true, 1.0))
        .expect("IVF controls must be accepted");
    let indexed = collection
        .query(&query)
        .expect("IVF SOAR query must succeed");
    assert_same_ranking(&exact, &indexed);

    collection.close().expect("collection must close");
    let reopened =
        Collection::open(path.to_str().expect("UTF-8 path"), None).expect("collection must reopen");
    assert!(
        reopened
            .stats()
            .expect("stats must be available")
            .index_cache_hit
    );
    let restored = reopened
        .query(&query)
        .expect("restored IVF SOAR query must succeed");
    assert_same_ranking(&indexed, &restored);
    reopened.close().expect("reopened collection must close");
}

#[test]
fn vamana_exhaustive_search_matches_the_exact_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("vamana")
            .to_str()
            .expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 96);

    let mut query =
        SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    let exact = collection.query(&query).expect("exact query must succeed");

    collection
        .create_index(
            "embedding",
            &IndexParams::vamana(MetricType::L2, 16, 96, 1.2)
                .expect("Vamana descriptor must be valid"),
        )
        .expect("Vamana index creation must succeed");
    query
        .set_diskann_params(DiskannQueryParams::new(96))
        .expect("Vamana search controls must be accepted");
    let exhaustive = collection.query(&query).expect("Vamana query must succeed");
    assert_same_ranking(&exact, &exhaustive);

    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes.len(), 1);
    assert_eq!(stats.indexes[0].index_type, IndexType::Vamana);
    assert_eq!(stats.indexes[0].state, "ready");
    assert_eq!(stats.indexes[0].source_revision, stats.revision);
    assert_eq!(stats.indexes[0].document_count, 96);
}

#[test]
fn vamana_non_l2_metrics_match_the_exact_oracle() {
    for (label, metric) in [
        ("ip", MetricType::Ip),
        ("cosine", MetricType::Cosine),
        ("mips-l2", MetricType::MipsL2),
    ] {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join(label);
        let collection = Collection::create(path.to_str().expect("UTF-8 path"), &schema(), None)
            .expect("collection must be created");
        insert_docs(&collection, 128);
        let mut query =
            SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!(label));
        let exact = collection.query(&query).expect("exact query must succeed");
        collection
            .create_index(
                "embedding",
                &IndexParams::vamana(metric, 16, 128, 1.2)
                    .expect("Vamana descriptor must be valid"),
            )
            .expect("non-L2 Vamana index must build");
        query
            .set_diskann_params(DiskannQueryParams::new(128))
            .expect("Vamana controls must be accepted");
        let indexed = collection
            .query(&query)
            .expect("indexed query must succeed");
        assert_same_ranking(&exact, &indexed);
        collection.flush().expect("sidecar must flush");
        collection.close().expect("collection must close");
        let reopened = Collection::open(path.to_str().expect("UTF-8 path"), None)
            .expect("collection must reopen");
        let restored = reopened
            .query(&query)
            .expect("reopened Vamana query must succeed");
        assert_same_ranking(&exact, &restored);
        reopened.close().expect("reopened collection must close");
    }
}

#[test]
fn vamana_scalar_quantization_and_pruning_controls_survive_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("vamana-quantized");
    let path_string = path.to_str().expect("UTF-8 path");
    let collection =
        Collection::create(path_string, &schema(), None).expect("collection must be created");
    insert_docs(&collection, 128);

    let mut query =
        SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("cosine"));
    let exact = collection.query(&query).expect("exact query must succeed");
    let mut params = IndexParams::vamana_with_options(MetricType::Cosine, 16, 128, 1.2, 32, true)
        .expect("Vamana options must be valid");
    params
        .set_quantize_type(QuantizeType::Int8)
        .expect("scalar quantization must be valid");
    collection
        .create_index("embedding", &params)
        .expect("quantized Vamana index must build");
    query
        .set_diskann_params(DiskannQueryParams::new(128))
        .expect("Vamana query controls must be accepted");
    let indexed = collection
        .query(&query)
        .expect("quantized Vamana query must succeed");
    assert_same_ranking(&exact, &indexed);
    collection.flush().expect("quantized Vamana cache must flush");
    collection.close().expect("collection must close");

    let reopened = Collection::open(path_string, None).expect("collection must reopen");
    assert!(
        reopened
            .stats()
            .expect("stats must be available")
            .index_cache_hit
    );
    let restored = reopened
        .query(&query)
        .expect("reopened quantized Vamana query must succeed");
    assert_same_ranking(&exact, &restored);
    reopened.close().expect("reopened collection must close");
}

#[test]
fn diskann_non_l2_full_vector_path_matches_after_sidecar_reopen() {
    for (label, metric) in [
        ("ip", MetricType::Ip),
        ("cosine", MetricType::Cosine),
        ("mips-l2", MetricType::MipsL2),
    ] {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join(label);
        let path_string = path.to_str().expect("UTF-8 path");
        let collection =
            Collection::create(path_string, &schema(), None).expect("collection must be created");
        insert_docs(&collection, 128);
        let mut query =
            SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!(label));
        let exact = collection.query(&query).expect("exact query must succeed");
        collection
            .create_index(
                "embedding",
                &IndexParams::diskann(metric, 16, 128, 0)
                    .expect("full-vector DiskANN descriptor must be valid"),
            )
            .expect("non-L2 DiskANN index must build");
        query
            .set_diskann_params(DiskannQueryParams::new(128))
            .expect("DiskANN controls must be accepted");
        let before_close = collection
            .query(&query)
            .expect("indexed query must succeed");
        assert_same_ranking(&exact, &before_close);
        collection.flush().expect("sidecar must flush");
        collection.close().expect("collection must close");
        let reopened = Collection::open(path_string, None).expect("collection must reopen");
        let after_reopen = reopened.query(&query).expect("reopened query must succeed");
        assert_same_ranking(&exact, &after_reopen);
        reopened.close().expect("reopened collection must close");
    }
}

#[test]
fn diskann_pq_similarity_metrics_match_the_exact_oracle() {
    for (label, metric) in [
        ("ip", MetricType::Ip),
        ("cosine", MetricType::Cosine),
        ("mips-l2", MetricType::MipsL2),
    ] {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join(label);
        let collection = Collection::create(path.to_str().expect("UTF-8 path"), &schema(), None)
            .expect("collection must be created");
        insert_docs(&collection, 128);
        let mut query =
            SearchQuery::new("embedding", &vector_for(37), 15).expect("query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!(label));
        let exact = collection.query(&query).expect("exact query must succeed");
        collection
            .create_index(
                "embedding",
                &IndexParams::diskann(metric, 16, 128, 4)
                    .expect("PQ DiskANN descriptor must be valid"),
            )
            .expect("similarity PQ DiskANN index must build");
        query
            .set_diskann_params(DiskannQueryParams::new(128))
            .expect("DiskANN controls must be accepted");
        let indexed = collection
            .query(&query)
            .expect("indexed query must succeed");
        assert_same_ranking(&exact, &indexed);
        collection.flush().expect("sidecar must flush");
        collection.close().expect("collection must close");
        let reopened = Collection::open(path.to_str().expect("UTF-8 path"), None)
            .expect("collection must reopen");
        let restored = reopened
            .query(&query)
            .expect("reopened PQ query must succeed");
        assert_same_ranking(&exact, &restored);
        reopened.close().expect("reopened collection must close");
    }
}

#[test]
fn non_l2_vamana_and_diskann_keep_bounded_candidate_work() {
    for (index_label, is_diskann) in [("vamana", false), ("diskann-pq", true)] {
        for (label, metric) in [
            ("ip", MetricType::Ip),
            ("cosine", MetricType::Cosine),
            ("mips-l2", MetricType::MipsL2),
        ] {
            let temporary = tempdir().expect("temporary directory must be available");
            let path = temporary.path().join(format!("{index_label}-{label}"));
            let collection =
                Collection::create(path.to_str().expect("UTF-8 path"), &schema(), None)
                    .expect("collection must be created");
            insert_docs(&collection, 256);
            let index_params = if is_diskann {
                IndexParams::diskann(metric, 24, 96, 4).expect("DiskANN descriptor must be valid")
            } else {
                IndexParams::vamana(metric, 24, 96, 1.2).expect("Vamana descriptor must be valid")
            };
            collection
                .create_index("embedding", &index_params)
                .expect("graph index must build");
            for query_index in [0, 17, 37, 71, 103, 137, 177, 223] {
                let mut query = SearchQuery::new("embedding", &vector_for(query_index), 10)
                    .expect("query must be valid");
                query
                    .params
                    .insert("metric".into(), serde_json::json!(label));
                let exact = collection.query(&query).expect("exact query must succeed");
                query
                    .set_diskann_params(DiskannQueryParams::new(96))
                    .expect("graph controls must be accepted");
                let before = collection
                    .stats_snapshot()
                    .expect("stats snapshot must be available");
                let approximate = collection.query(&query).expect("graph query must succeed");
                let after = collection
                    .stats_snapshot()
                    .expect("stats snapshot must be available");
                let exact_ids = ids(&exact);
                let hits = ids(&approximate)
                    .iter()
                    .filter(|id| exact_ids.contains(id))
                    .count();
                assert!(
                    hits >= 5,
                    "{index_label} {label} query={query_index} recall@10 was only {hits}/10"
                );
                assert!(after.candidates_scanned - before.candidates_scanned <= 96);
                assert!(after.candidates_scanned - before.candidates_scanned < 256);
            }
            collection.close().expect("graph collection must close");
        }
    }
}
