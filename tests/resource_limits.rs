use a3s_vec::{
    Collection, CollectionOptions, CollectionResourceLimits, CollectionSchema, DataType, Doc,
    ErrorCode, FieldSchema, IndexParams, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use tempfile::tempdir;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("resource-limits")
        .add_field(
            FieldSchema::new("body", DataType::String, false, 0)
                .expect("body schema must be valid"),
        )
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
                .expect("embedding schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn doc(id: &str, body: &str, vector: [f32; 2]) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_string("body", body).expect("body must be valid");
    doc.add_vector_f32("embedding", &vector)
        .expect("embedding must be valid");
    doc
}

fn options(limits: CollectionResourceLimits) -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_resource_limits(limits)
        .expect("resource limits must be accepted");
    options
}

#[test]
fn zero_resource_limits_are_rejected() {
    for result in [
        CollectionResourceLimits::new().try_with_max_documents(0),
        CollectionResourceLimits::new().try_with_max_accounted_bytes(0),
        CollectionResourceLimits::new().try_with_max_query_candidates(0),
        CollectionResourceLimits::new().try_with_max_write_batch_documents(0),
    ] {
        let error = result.expect_err("zero must not silently disable a typed limit");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
}

#[test]
fn document_and_batch_limits_fail_atomically_before_persistence() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let limits = CollectionResourceLimits::new()
        .try_with_max_documents(2)
        .expect("document limit must be valid")
        .try_with_max_write_batch_documents(1)
        .expect("batch limit must be valid");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema(),
        Some(&options(limits)),
    )
    .expect("collection must be created");

    let first = doc("first", "first", [1.0, 0.0]);
    let second = doc("second", "second", [0.0, 1.0]);
    let error = collection
        .insert(&[&first, &second])
        .expect_err("oversized batch must fail as one operation");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert_eq!(collection.count().expect("count must succeed"), 0);
    assert_eq!(collection.stats().expect("stats must succeed").revision, 0);

    collection
        .insert(&[&first])
        .expect("first insert must succeed");
    collection
        .insert(&[&second])
        .expect("second insert must succeed");
    let third = doc("third", "third", [1.0, 1.0]);
    let before = collection.stats().expect("stats must succeed");
    let error = collection
        .insert(&[&third])
        .expect_err("document quota must reject the next generation");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    let after = collection.stats().expect("stats must succeed");
    assert_eq!(after.doc_count, 2);
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.resource_limit_rejections, 2);
    assert_eq!(after.resource_limits, limits);

    collection
        .delete(&["first"])
        .expect("deletion must release document capacity");
    collection
        .insert(&[&third])
        .expect("released capacity must admit another document");
    assert_eq!(collection.count().expect("count must succeed"), 2);

    let before_filtered_delete = collection.stats().expect("stats must succeed");
    let error = collection
        .delete_by_filter("body != 'missing'")
        .expect_err("a filtered delete must use the same write batch budget");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    let after_filtered_delete = collection.stats().expect("stats must succeed");
    assert_eq!(after_filtered_delete.doc_count, 2);
    assert_eq!(
        after_filtered_delete.revision,
        before_filtered_delete.revision
    );
    assert_eq!(after_filtered_delete.resource_limit_rejections, 3);

    collection.close().expect("collection must close");
    let tighter = CollectionResourceLimits::new()
        .try_with_max_documents(1)
        .expect("document limit must be valid");
    let error = Collection::open(
        path.to_str().expect("temporary path must be UTF-8"),
        Some(&options(tighter)),
    )
    .expect_err("open must apply the caller's current resource policy");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
}

#[test]
fn accounted_byte_limit_covers_documents_and_derived_indexes() {
    let temporary = tempdir().expect("temporary directory must be available");
    let rejected_path = temporary.path().join("rejected-create");
    let impossible = CollectionResourceLimits::new()
        .try_with_max_accounted_bytes(1)
        .expect("byte limit must be valid");
    let error = Collection::create(
        rejected_path
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema(),
        Some(&options(impossible)),
    )
    .expect_err("even empty-state accounting must be admitted before creation");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert!(!rejected_path.exists());

    let document_path = temporary.path().join("documents");
    let byte_limits = CollectionResourceLimits::new()
        .try_with_max_accounted_bytes(256)
        .expect("byte limit must be valid");
    let collection = Collection::create(
        document_path
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema(),
        Some(&options(byte_limits)),
    )
    .expect("collection must be created");
    let oversized = doc("oversized", &"x".repeat(4_096), [1.0, 0.0]);
    let error = collection
        .insert(&[&oversized])
        .expect_err("accounted document bytes must be bounded");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert_eq!(collection.count().expect("count must succeed"), 0);

    let index_path = temporary.path().join("indexes");
    let index_limits = CollectionResourceLimits::new()
        .try_with_max_accounted_bytes(1_024)
        .expect("byte limit must be valid");
    let indexed = Collection::create(
        index_path.to_str().expect("temporary path must be UTF-8"),
        &schema(),
        Some(&options(index_limits)),
    )
    .expect("collection must be created");
    let docs: Vec<Doc> = (0..8)
        .map(|index| {
            let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
            doc(&format!("doc-{index}"), "bounded", [coordinate, 1.0])
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    indexed
        .insert(&refs)
        .expect("documents must fit before indexing");
    let before = indexed.stats().expect("stats must succeed");
    assert!(before.accounted_document_bytes > 0);
    assert_eq!(before.estimated_index_bytes, 0);

    let error = indexed
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("HNSW params must be valid"),
        )
        .expect_err("the derived payload must share the accounted byte budget");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    let after = indexed.stats().expect("stats must succeed");
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.estimated_index_bytes, 0);
    assert!(!indexed
        .schema()
        .expect("schema must succeed")
        .has_index("embedding"));
}

#[test]
fn query_candidate_budget_rejects_work_and_records_metadata_only_telemetry() {
    let temporary = tempdir().expect("temporary directory must be available");
    let limits = CollectionResourceLimits::new()
        .try_with_max_query_candidates(2)
        .expect("query limit must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema(),
        Some(&options(limits)),
    )
    .expect("collection must be created");
    let docs = [
        doc("first", "alpha", [1.0, 0.0]),
        doc("second", "beta", [0.9, 0.1]),
        doc("third", "gamma", [0.0, 1.0]),
    ];
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");

    let query = SearchQuery::new("embedding", &[1.0, 0.0], 1).expect("query must be constructed");
    let error = collection
        .query(&query)
        .expect_err("exact refinement candidate work must be bounded");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert!(error.message.contains('3'));
    assert!(!error.message.contains("alpha"));

    let stats = collection.stats_snapshot().expect("stats must succeed");
    assert_eq!(stats.query_count, 0);
    assert_eq!(stats.resource_limit_rejections, 1);
    assert_eq!(stats.accounted_bytes, stats.accounted_document_bytes);
    assert_eq!(stats.resource_limits, limits);
}

#[test]
fn multi_query_uses_one_cumulative_candidate_budget() {
    let temporary = tempdir().expect("temporary directory must be available");
    let limits = CollectionResourceLimits::new()
        .try_with_max_query_candidates(4)
        .expect("query limit must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema(),
        Some(&options(limits)),
    )
    .expect("collection must be created");
    let docs = [
        doc("first", "alpha", [1.0, 0.0]),
        doc("second", "beta", [0.9, 0.1]),
        doc("third", "gamma", [0.0, 1.0]),
    ];
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");

    let mut branch = SubQuery::new().expect("branch must be constructed");
    branch
        .set_field_name("embedding")
        .expect("field must be accepted");
    branch
        .set_query_vector(&[1.0, 0.0])
        .expect("vector must be accepted");
    branch
        .set_num_candidates(1)
        .expect("candidate count must be accepted");
    let mut query = MultiQuery::new().expect("multi-query must be constructed");
    query
        .add_sub_query(&branch)
        .expect("first branch must be accepted");
    query
        .add_sub_query(&branch)
        .expect("second branch must be accepted");

    let error = collection
        .multi_query(&query)
        .expect_err("branch work must share one cumulative budget");
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert!(error.message.contains('6'));
    let stats = collection.stats_snapshot().expect("stats must succeed");
    assert_eq!(stats.query_count, 0);
    assert_eq!(stats.candidates_scanned, 0);
    assert_eq!(stats.resource_limit_rejections, 1);
}
