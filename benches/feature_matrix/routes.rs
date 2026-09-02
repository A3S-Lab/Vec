fn dense_query(config: Config, metric: &str) -> SearchQuery {
    let mut query = SearchQuery::new("embedding", &vector_for(17, config.dimensions), 10)
        .expect("dense query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!(metric));
    query
}

#[allow(clippy::too_many_lines)]
fn measure_full_routes(collection: &Collection, config: Config) {
    let samples = config.rounds * config.queries;
    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let query = dense_query(config, metric);
        measure(&format!("dense_{metric}"), config, samples, |sample| {
            let mut query = query.clone();
            query
                .set_query_vector(&vector_for(sample % config.documents, config.dimensions))
                .expect("query vector must be valid");
            let result = collection
                .query(black_box(&query))
                .expect("dense query must succeed");
            assert_eq!(result.len(), 10.min(config.documents));
            u64::try_from(result.len()).expect("result count fits u64")
        });
    }

    let mut include_doc_id = dense_query(config, "cosine");
    include_doc_id
        .set_include_doc_id(true)
        .expect("include-doc-id control must be valid");
    measure("dense_include_doc_id", config, samples, |sample| {
        let mut query = include_doc_id.clone();
        query
            .set_query_vector(&vector_for(sample % config.documents, config.dimensions))
            .expect("query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("include-doc-id query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        assert!(result.iter().all(|doc| doc.doc_id().is_some()));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let mut query = SearchQuery::by_id("embedding", "doc-00017", 10)
            .expect("dense source-ID query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!(metric));
        measure(
            &format!("dense_source_id_{metric}"),
            config,
            samples,
            |_| {
                let result = collection
                    .query(black_box(&query))
                    .expect("dense source-ID query must succeed");
                assert_eq!(result.len(), 10.min(config.documents));
                u64::try_from(result.len()).expect("result count fits u64")
            },
        );
    }

    let sparse = SearchQuery::sparse("sparse", &[1, 8], &[1.0, 0.25], 10)
        .expect("sparse query must be valid");
    measure("sparse", config, samples, |sample| {
        let mut query = sparse.clone();
        let first = u32::try_from(sample % 16).expect("sparse index fits u32");
        query.sparse_vector = Some(vec![(first, 1.0), ((first + 7) % 32, 0.25)]);
        let result = collection
            .query(black_box(&query))
            .expect("sparse query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let sparse_source =
        SearchQuery::by_id("sparse", "doc-00017", 10).expect("sparse source-ID query");
    measure("sparse_source_id", config, samples, |_| {
        let result = collection
            .query(black_box(&sparse_source))
            .expect("sparse source-ID query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let binary32 = SearchQuery::binary("bits32", &binary_for(17, 4), 10)
        .expect("Binary32 query must be valid");
    measure("binary32_exact", config, samples, |sample| {
        let mut query = binary32.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 4))
            .expect("Binary32 query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("Binary32 exact query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary32_radius = binary32.clone();
    binary32_radius
        .set_radius(1.0)
        .expect("Binary32 radius must be valid");
    measure("binary32_radius", config, samples, |sample| {
        let mut query = binary32_radius.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 4))
            .expect("Binary32 radius query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("Binary32 radius query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary32_projection = binary32.clone();
    binary32_projection
        .set_include_vector(true)
        .expect("Binary32 include-vector control must be valid");
    binary32_projection
        .set_include_doc_id(true)
        .expect("Binary32 include-doc-id control must be valid");
    binary32_projection
        .set_output_fields(&["category", "bits32"])
        .expect("Binary32 projection must be valid");
    measure("binary32_projection", config, samples, |sample| {
        let mut query = binary32_projection.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 4))
            .expect("Binary32 projection query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("Binary32 projection query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        assert!(result.iter().all(|doc| {
            doc.doc_id().is_some()
                && doc.get_vector_binary32("bits32").ok().flatten().is_some()
                && doc.vector("bits64").is_none()
        }));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let binary64 = SearchQuery::binary("bits64", &binary_for(17, 8), 10)
        .expect("Binary64 query must be valid");
    measure("binary64_exact", config, samples, |sample| {
        let mut query = binary64.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 8))
            .expect("Binary64 query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("Binary64 exact query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary64_radius = binary64.clone();
    binary64_radius
        .set_radius(1.0)
        .expect("Binary64 radius must be valid");
    measure("binary64_radius", config, samples, |sample| {
        let mut query = binary64_radius.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 8))
            .expect("Binary64 radius query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("Binary64 radius query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary64_projection = binary64.clone();
    binary64_projection
        .set_include_vector(true)
        .expect("Binary64 include-vector control must be valid");
    binary64_projection
        .set_include_doc_id(true)
        .expect("Binary64 include-doc-id control must be valid");
    binary64_projection
        .set_output_fields(&["category", "bits64"])
        .expect("Binary64 projection must be valid");
    measure("binary64_projection", config, samples, |sample| {
        let mut query = binary64_projection.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 8))
            .expect("Binary64 projection query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("Binary64 projection query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        assert!(result.iter().all(|doc| {
            doc.doc_id().is_some()
                && doc.get_vector_binary64("bits64").ok().flatten().is_some()
                && doc.vector("bits32").is_none()
        }));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let binary_source =
        SearchQuery::by_id("bits32", "doc-00017", 10).expect("binary source-ID query");
    measure("binary_source_id", config, samples, |_| {
        let result = collection
            .query(black_box(&binary_source))
            .expect("binary source-ID query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });
    let binary64_source =
        SearchQuery::by_id("bits64", "doc-00017", 10).expect("Binary64 source-ID query");
    measure("binary64_source_id", config, samples, |_| {
        let result = collection
            .query(black_box(&binary64_source))
            .expect("Binary64 source-ID query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary_filtered = binary32.clone();
    binary_filtered
        .set_filter("category == 'even'")
        .expect("binary scalar filter must be valid");
    measure("binary_scalar_filter", config, samples, |sample| {
        let mut query = binary_filtered.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 4))
            .expect("filtered Binary32 query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("filtered binary query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });
    let mut binary64_filtered = binary64.clone();
    binary64_filtered
        .set_filter("category == 'even'")
        .expect("Binary64 scalar filter must be valid");
    measure("binary64_scalar_filter", config, samples, |sample| {
        let mut query = binary64_filtered.clone();
        query
            .set_binary_vector(&binary_for(sample % config.documents, 8))
            .expect("filtered Binary64 query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("filtered Binary64 query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut fts_payload = Fts::new().expect("FTS payload must be valid");
    fts_payload
        .set_query_string("rust AND vector")
        .expect("FTS expression must be valid");
    let fts_query = SearchQuery::fts("body", &fts_payload, 10).expect("FTS query must be valid");
    measure("fts_indexed", config, samples, |_| {
        let result = collection
            .query(black_box(&fts_query))
            .expect("FTS query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut filtered = dense_query(config, "cosine");
    filtered
        .set_filter("category == 'even'")
        .expect("filter must be valid");
    measure("dense_scalar_filter", config, samples, |sample| {
        let mut query = filtered.clone();
        query
            .set_query_vector(&vector_for(sample % config.documents, config.dimensions))
            .expect("query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("filtered query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut vector_branch = SubQuery::new().expect("sub-query must be valid");
    vector_branch
        .set_field_name("embedding")
        .expect("sub-query field must be valid");
    vector_branch
        .set_query_vector(&vector_for(17, config.dimensions))
        .expect("sub-query vector must be valid");
    vector_branch
        .set_num_candidates(16)
        .expect("candidate count must be valid");
    let mut fts_branch = SubQuery::new().expect("sub-query must be valid");
    fts_branch
        .set_field_name("body")
        .expect("sub-query field must be valid");
    fts_branch
        .set_fts(&fts_payload)
        .expect("FTS branch must be valid");
    fts_branch
        .set_num_candidates(16)
        .expect("candidate count must be valid");
    let mut multi = MultiQuery::new().expect("multi-query must be valid");
    multi.set_topk(10).expect("multi top-k must be valid");
    multi
        .add_sub_query(&vector_branch)
        .expect("vector branch must be added");
    multi
        .add_sub_query(&fts_branch)
        .expect("FTS branch must be added");
    multi.set_rerank_rrf(60).expect("RRF rerank must be valid");
    measure("multi_rrf", config, samples, |_| {
        let result = collection
            .multi_query(black_box(&multi))
            .expect("multi query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary_branch = SubQuery::new().expect("binary sub-query must be valid");
    binary_branch
        .set_field_name("bits32")
        .expect("binary sub-query field must be valid");
    binary_branch
        .set_binary_vector(&binary_for(17, 4))
        .expect("binary sub-query vector must be valid");
    binary_branch
        .set_num_candidates(16)
        .expect("binary candidate count must be valid");
    let mut binary_multi = MultiQuery::new().expect("binary multi-query must be valid");
    binary_multi
        .set_topk(10)
        .expect("binary multi top-k must be valid");
    binary_multi
        .add_sub_query(&binary_branch)
        .expect("binary sub-query must be added");
    measure("binary_multi", config, samples, |_| {
        let result = collection
            .multi_query(black_box(&binary_multi))
            .expect("binary multi-query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut binary64_branch = SubQuery::new().expect("Binary64 sub-query must be valid");
    binary64_branch
        .set_field_name("bits64")
        .expect("Binary64 sub-query field must be valid");
    binary64_branch
        .set_binary_vector(&binary_for(17, 8))
        .expect("Binary64 sub-query vector must be valid");
    binary64_branch
        .set_num_candidates(16)
        .expect("Binary64 candidate count must be valid");
    let mut binary64_multi = MultiQuery::new().expect("Binary64 multi-query must be valid");
    binary64_multi
        .set_topk(10)
        .expect("Binary64 multi top-k must be valid");
    binary64_multi
        .add_sub_query(&binary64_branch)
        .expect("Binary64 branch must be added");
    measure("binary64_multi", config, samples, |_| {
        let result = collection
            .multi_query(black_box(&binary64_multi))
            .expect("Binary64 multi-query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("Binary64 multi result count fits u64")
    });

    let grouped = GroupBySearchQuery::new(
        "embedding",
        "category",
        &vector_for(17, config.dimensions),
        2,
        3,
    )
    .expect("group query must be valid");
    measure("group_by", config, samples, |_| {
        let result = collection
            .group_by(black_box(&grouped))
            .expect("group query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.values().map(Vec::len).sum::<usize>())
            .expect("group result count fits u64")
    });

    let binary_grouped = GroupBySearchQuery::binary("bits32", "category", &binary_for(17, 4), 2, 3)
        .expect("binary group query must be valid");
    measure("binary_group_by", config, samples, |_| {
        let result = collection
            .group_by(black_box(&binary_grouped))
            .expect("binary group query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.values().map(Vec::len).sum::<usize>())
            .expect("binary group result count fits u64")
    });

    let binary64_grouped =
        GroupBySearchQuery::binary("bits64", "category", &binary_for(17, 8), 2, 3)
            .expect("Binary64 group query must be valid");
    measure("binary64_group_by", config, samples, |_| {
        let result = collection
            .group_by(black_box(&binary64_grouped))
            .expect("Binary64 group query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.values().map(Vec::len).sum::<usize>())
            .expect("Binary64 group result count fits u64")
    });

    measure("fetch_projection", config, samples, |_| {
        let result = collection
            .fetch_with_options(&["doc-00001", "doc-00017"], Some(&["body"]), false)
            .expect("fetch must succeed");
        assert_eq!(result.len(), 2);
        u64::try_from(result.len()).expect("result count fits u64")
    });

    measure("snapshot_iterator", config, config.rounds, |_| {
        let result = collection
            .iter_with_options(Some(&["category"]), false)
            .expect("iterator must be created")
            .collect::<a3s_vec::Result<Vec<_>>>()
            .expect("iterator must succeed");
        assert_eq!(result.len(), config.documents);
        u64::try_from(result.len()).expect("result count fits u64")
    });

    measure("stats_health", config, samples, |_| {
        let stats = collection.stats().expect("stats must succeed");
        assert_eq!(
            stats.doc_count,
            u64::try_from(config.documents).expect("document count fits u64")
        );
        assert!(collection
            .health()
            .expect("health must succeed")
            .is_healthy());
        1
    });

    let patch = {
        let mut doc = Doc::with_pk("doc-00017").expect("patch must be valid");
        doc.add_i32("bucket", 15).expect("patch must be valid");
        doc
    };
    measure("partial_update", config, config.rounds, |_| {
        let result = collection.update(&[&patch]).expect("update must succeed");
        assert_eq!(result.success_count, 1);
        result.success_count
    });
    measure("flush", config, config.rounds, |_| {
        collection.flush().expect("flush must succeed");
        1
    });
}

