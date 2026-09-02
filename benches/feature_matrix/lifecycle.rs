fn measure_reopen_and_sidecar(config: Config) {
    let directory = tempdir().expect("sidecar temporary directory must be available");
    let path = directory.path().join("sidecar");
    let path_string = path.to_str().expect("sidecar path must be UTF-8");
    let collection = Collection::create(path_string, &ann_schema(config.dimensions), None)
        .expect("sidecar collection must be created");
    insert_ann_fixture(&collection, config);
    collection
        .create_index(
            "embedding",
            &IndexParams::diskann(
                MetricType::L2,
                12,
                i32::try_from(config.documents).expect("document count fits i32"),
                2,
            )
            .expect("DiskANN descriptor must be valid"),
        )
        .expect("DiskANN index must build");
    collection.flush().expect("sidecar fixture must flush");
    collection.close().expect("sidecar collection must close");

    let query = {
        let mut query = SearchQuery::new("embedding", &vector_for(17, config.dimensions), 10)
            .expect("sidecar query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!("l2"));
        query
            .set_diskann_params(DiskannQueryParams::new(64))
            .expect("sidecar controls must be valid");
        query
    };
    for backend in [IoBackend::Positioned, IoBackend::Mmap] {
        let mut options = CollectionOptions::new().expect("options must be valid");
        options
            .set_read_only(true)
            .expect("read-only option must be valid");
        options
            .set_io_backend(backend)
            .expect("backend must be valid");
        let samples = config.rounds;
        measure(
            if backend == IoBackend::Mmap {
                "diskann_mmap_reopen_query"
            } else {
                "diskann_positioned_reopen_query"
            },
            config,
            samples,
            |_| {
                let reopened =
                    Collection::open(path_string, Some(&options)).expect("reopen must succeed");
                assert!(
                    reopened
                        .stats()
                        .expect("stats must succeed")
                        .index_cache_hit
                );
                let result = reopened
                    .query(black_box(&query))
                    .expect("sidecar query must succeed");
                assert!(!result.is_empty());
                let count = u64::try_from(result.len()).expect("result count fits u64");
                reopened.close().expect("reopened collection must close");
                count
            },
        );
    }
}

#[cfg(feature = "async")]
#[allow(clippy::too_many_lines)]
fn measure_async(collection: &Collection, config: Config) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Tokio runtime must be available");
    let query = dense_query(config, "cosine");
    let samples = config.rounds * config.queries;
    measure("dense_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.query_async(black_box(&query)))
            .expect("async query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let binary = SearchQuery::binary("bits32", &binary_for(17, 4), 10)
        .expect("async binary query must be valid");
    measure("binary_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.query_async(black_box(&binary)))
            .expect("async binary query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let binary64 = SearchQuery::binary("bits64", &binary_for(17, 8), 10)
        .expect("async Binary64 query must be valid");
    measure("binary64_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.query_async(black_box(&binary64)))
            .expect("async Binary64 query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("Binary64 async result count fits u64")
    });

    let mut branch = SubQuery::new().expect("async sub-query must be valid");
    branch
        .set_field_name("embedding")
        .expect("async sub-query field must be valid");
    branch
        .set_query_vector(&vector_for(17, config.dimensions))
        .expect("async sub-query vector must be valid");
    branch
        .set_num_candidates(16)
        .expect("async candidate count must be valid");
    let mut multi = MultiQuery::new().expect("async multi-query must be valid");
    multi.set_topk(10).expect("async multi top-k must be valid");
    multi
        .add_sub_query(&branch)
        .expect("async sub-query must be added");
    measure("multi_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.multi_query_async(black_box(&multi)))
            .expect("async multi-query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let grouped = GroupBySearchQuery::new(
        "embedding",
        "category",
        &vector_for(17, config.dimensions),
        2,
        3,
    )
    .expect("async group query must be valid");
    measure("group_by_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.group_by_async(black_box(&grouped)))
            .expect("async group query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.values().map(Vec::len).sum::<usize>())
            .expect("async group result count fits u64")
    });
}

fn assert_complete_matrix() {
    const BASE_OPERATIONS: &[&str] = &[
        "dense_l2",
        "dense_ip",
        "dense_cosine",
        "dense_mips_l2",
        "dense_include_doc_id",
        "dense_source_id_l2",
        "dense_source_id_ip",
        "dense_source_id_cosine",
        "dense_source_id_mips_l2",
        "sparse",
        "sparse_source_id",
        "binary32_exact",
        "binary32_radius",
        "binary32_projection",
        "binary64_exact",
        "binary64_radius",
        "binary64_projection",
        "binary_source_id",
        "binary64_source_id",
        "binary_scalar_filter",
        "binary64_scalar_filter",
        "fts_indexed",
        "dense_scalar_filter",
        "multi_rrf",
        "binary_multi",
        "binary64_multi",
        "group_by",
        "binary_group_by",
        "binary64_group_by",
        "fetch_projection",
        "snapshot_iterator",
        "stats_health",
        "partial_update",
        "flush",
        "ann_hnsw",
        "ann_ivf_soar",
        "ann_hnsw_rabitq",
        "ann_ivf_rabitq",
        "ann_vamana",
        "ann_vamana_ip",
        "ann_vamana_cosine",
        "ann_vamana_mips_l2",
        "ann_diskann_pq",
        "ann_diskann_ip_pq",
        "ann_diskann_cosine_pq",
        "ann_diskann_mips_l2_pq",
        "diskann_positioned_reopen_query",
        "diskann_mmap_reopen_query",
    ];
    let expected = BASE_OPERATIONS
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    #[cfg(feature = "async")]
    let expected = {
        let mut expected = expected;
        expected.extend(
            [
                "dense_async",
                "binary_async",
                "binary64_async",
                "multi_async",
                "group_by_async",
            ]
            .into_iter()
            .map(str::to_string),
        );
        expected
    };

    let measured = MEASURED_OPERATIONS
        .get()
        .expect("benchmark must record at least one operation")
        .lock()
        .expect("benchmark operation registry must not be poisoned")
        .clone();
    let actual = measured.iter().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        measured.len(),
        actual.len(),
        "benchmark operation names must be unique"
    );
    assert_eq!(
        actual, expected,
        "feature matrix must cover every expected operation"
    );
}

fn main() {
    let config = Config::from_environment();
    println!("# a3s-vec feature matrix; scale={config:?}; each row is an asserted operation");
    println!(
        "operation,documents,dimensions,samples,total_work,p50_us,p95_us,p99_us,work_per_second,work_per_sample"
    );

    let directory = tempdir().expect("feature matrix temporary directory must be available");
    let path = directory.path().join("routes");
    let path_string = path.to_str().expect("feature matrix path must be UTF-8");
    let collection = Collection::create(
        path_string,
        &full_schema(config.dimensions),
        Some(&manual_options()),
    )
    .expect("feature matrix collection must be created");
    insert_full_fixture(&collection, config);
    measure_full_routes(&collection, config);
    #[cfg(feature = "async")]
    measure_async(&collection, config);
    collection
        .close()
        .expect("feature matrix collection must close");

    measure_ann_modes(config);
    measure_reopen_and_sidecar(config);
    assert_complete_matrix();
}

