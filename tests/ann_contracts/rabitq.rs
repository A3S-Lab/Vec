fn set_exhaustive_rabitq_controls(query: &mut SearchQuery, index_type: IndexType, count: usize) {
    match index_type {
        IndexType::HnswRabitq => query
            .set_hnsw_params(HnswQueryParams::new(
                i32::try_from(count).expect("fixture count fits i32"),
                0.0,
                false,
                true,
            ))
            .expect("HNSW RaBitQ controls must be valid"),
        IndexType::IvfRabitq => {
            let mut params = IvfRabitqQueryParams::new(16, 0.0, false, true);
            params
                .set_scale_factor(f32::from(
                    u16::try_from(count).expect("fixture count fits u16"),
                ))
                .expect("IVF RaBitQ refinement scale must be valid");
            query
                .set_ivf_rabitq_params(params)
                .expect("IVF RaBitQ controls must be valid");
        }
        _ => panic!("fixture requires a RaBitQ index family"),
    }
}

#[test]
fn hnsw_rabitq_exhaustive_search_matches_the_exact_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("hnsw-rabitq")
            .to_str()
            .expect("UTF-8 path"),
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
            &IndexParams::hnsw_rabitq(MetricType::Cosine, 12, 64)
                .expect("HNSW RaBitQ descriptor must be valid"),
        )
        .expect("HNSW RaBitQ index creation must succeed");
    query
        .set_hnsw_params(HnswQueryParams::new(96, 0.0, false, true))
        .expect("HNSW RaBitQ controls must be accepted");
    let exhaustive = collection
        .query(&query)
        .expect("HNSW RaBitQ query must succeed");
    assert_same_ranking(&exact, &exhaustive);

    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes[0].index_type, IndexType::HnswRabitq);
    assert_eq!(stats.indexes[0].document_count, 96);
}

#[test]
fn ivf_rabitq_exhaustive_search_matches_the_exact_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("ivf-rabitq")
            .to_str()
            .expect("UTF-8 path"),
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
            &IndexParams::ivf_rabitq(MetricType::Cosine, 12, 7, 0)
                .expect("IVF RaBitQ descriptor must be valid"),
        )
        .expect("IVF RaBitQ index creation must succeed");
    let mut params = IvfRabitqQueryParams::new(12, 0.0, false, true);
    params
        .set_scale_factor(16.0)
        .expect("exhaustive refinement scale must be valid");
    query
        .set_ivf_rabitq_params(params)
        .expect("IVF RaBitQ controls must be accepted");
    let exhaustive = collection
        .query(&query)
        .expect("IVF RaBitQ query must succeed");
    assert_same_ranking(&exact, &exhaustive);

    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes[0].index_type, IndexType::IvfRabitq);
    assert_eq!(stats.indexes[0].document_count, 96);
}

#[test]
fn rabitq_indexes_track_mutations_rebuilds_and_cache_reopen() {
    for index_type in [IndexType::HnswRabitq, IndexType::IvfRabitq] {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join(format!("lifecycle-{index_type:?}"));
        let collection = Collection::create(path.to_str().expect("UTF-8 path"), &schema(), None)
            .expect("collection must be created");
        let params = match index_type {
            IndexType::HnswRabitq => IndexParams::hnsw_rabitq(MetricType::L2, 8, 32)
                .expect("HNSW RaBitQ params must be valid"),
            IndexType::IvfRabitq => IndexParams::ivf_rabitq(MetricType::L2, 16, 7, 64)
                .expect("IVF RaBitQ params must be valid"),
            _ => panic!("fixture requires a RaBitQ index family"),
        };
        collection
            .create_index("embedding", &params)
            .expect("empty RaBitQ index creation must succeed");
        insert_docs(&collection, 80);

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

        let mut exact_query =
            SearchQuery::new("embedding", &[10.0; 8], 10).expect("exact query must be valid");
        match index_type {
            IndexType::HnswRabitq => exact_query
                .set_hnsw_params(HnswQueryParams::new(79, 0.0, true, true))
                .expect("linear HNSW RaBitQ controls must be valid"),
            IndexType::IvfRabitq => exact_query
                .set_ivf_rabitq_params(IvfRabitqQueryParams::new(16, 0.0, true, true))
                .expect("linear IVF RaBitQ controls must be valid"),
            _ => panic!("fixture requires a RaBitQ index family"),
        }
        let exact = collection
            .query(&exact_query)
            .expect("exact query must succeed");
        let mut query = exact_query.clone();
        set_exhaustive_rabitq_controls(&mut query, index_type, 79);
        assert_same_ranking(
            &exact,
            &collection
                .query(&query)
                .expect("overlay RaBitQ query must succeed"),
        );

        collection
            .rebuild_index("embedding")
            .expect("targeted RaBitQ rebuild must succeed");
        collection
            .optimize()
            .expect("RaBitQ optimization must succeed");
        let stats = collection.stats().expect("stats must be available");
        assert_eq!(stats.indexes[0].index_type, index_type);
        assert_eq!(stats.indexes[0].document_count, 79);
        collection.close().expect("collection must close");

        let reopened = Collection::open(path.to_str().expect("UTF-8 path"), None)
            .expect("collection must reopen");
        let stats = reopened.stats().expect("stats must be available");
        assert!(stats.index_cache_hit);
        assert_eq!(stats.indexes[0].index_type, index_type);
        assert_same_ranking(
            &exact,
            &reopened
                .query(&query)
                .expect("cached RaBitQ query must succeed"),
        );
    }
}

#[test]
fn diskann_pq_exhaustive_search_matches_the_exact_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("diskann-pq")
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
            &IndexParams::diskann(MetricType::L2, 16, 96, 4)
                .expect("DiskANN descriptor must be valid"),
        )
        .expect("DiskANN PQ index creation must succeed");
    query
        .set_diskann_params(DiskannQueryParams::new(96))
        .expect("DiskANN controls must be accepted");
    let exhaustive = collection
        .query(&query)
        .expect("DiskANN PQ query must succeed");
    assert_same_ranking(&exact, &exhaustive);

    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes.len(), 1);
    assert_eq!(stats.indexes[0].index_type, IndexType::Diskann);
    assert_eq!(stats.indexes[0].document_count, 96);
}

#[test]
fn diskann_pq_recall_fixture_uses_adc_and_a_bounded_candidate_set() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("diskann-pq-recall")
            .to_str()
            .expect("UTF-8 path"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    insert_docs(&collection, 320);
    let mut query =
        SearchQuery::new("embedding", &vector_for(177), 10).expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    let exact = collection.query(&query).expect("exact query must succeed");
    collection
        .create_index(
            "embedding",
            &IndexParams::diskann(MetricType::L2, 24, 96, 4)
                .expect("DiskANN descriptor must be valid"),
        )
        .expect("DiskANN PQ index must build");
    query
        .set_diskann_params(DiskannQueryParams::new(64))
        .expect("DiskANN controls must be accepted");
    let before = collection
        .stats_snapshot()
        .expect("stats snapshot must be available");
    let approximate = collection
        .query(&query)
        .expect("DiskANN PQ query must succeed");
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
    assert!(after.candidates_scanned - before.candidates_scanned <= 64);
    assert!(after.candidates_scanned - before.candidates_scanned < 320);
}

