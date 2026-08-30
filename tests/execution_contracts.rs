use a3s_vec::{
    AddColumnOption, AlterColumnOption, Collection, CollectionSchema, DataType, DiskannQueryParams,
    Doc, ErrorCode, FieldSchema, FlatQueryParams, Fts, FtsQueryParams, GroupBySearchQuery,
    HnswQueryParams, IndexParams, IndexType, IvfQueryParams, IvfRabitqQueryParams, MetricType,
    MultiQuery, QuantizeType, SearchQuery, SubQuery,
};
use serde_json::json;
use tempfile::tempdir;

fn exact_schema() -> CollectionSchema {
    CollectionSchema::builder("execution-contracts")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("title schema must be valid"),
        )
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
                .expect("vector schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn valid_doc(id: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("test primary key must be valid");
    doc.add_string("title", "alpha beta")
        .expect("title must be valid");
    doc.add_vector_f32("embedding", &[1.0, 0.0, 0.0])
        .expect("embedding must be valid");
    doc
}

#[test]
fn unsupported_index_lifecycle_does_not_mutate_collection() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &exact_schema(),
        None,
    )
    .expect("collection must be created");
    let initial = collection.stats().expect("stats must be available");
    let hnsw =
        IndexParams::hnsw(MetricType::Cosine, 16, 100).expect("HNSW descriptor must be valid");

    let error = collection
        .create_index("embedding", &hnsw)
        .expect_err("an unimplemented HNSW build must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!collection
        .schema()
        .expect("schema must be available")
        .has_index("embedding"));
    assert_eq!(
        collection.stats().expect("stats must be available"),
        initial
    );

    let fts =
        IndexParams::fts(Some("standard"), None, None).expect("FTS configuration must be valid");
    let error = collection
        .create_index("title", &fts)
        .expect_err("scan FTS configuration must not claim a physical index build");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!collection
        .schema()
        .expect("schema must be available")
        .has_index("title"));
    assert_eq!(
        collection.stats().expect("stats must be available"),
        initial
    );

    let error = collection
        .optimize()
        .expect_err("unimplemented physical optimization must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert_eq!(
        collection.stats().expect("stats must be available"),
        initial
    );
}

#[test]
fn flat_index_is_a_live_exact_path_and_never_ann_telemetry() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &exact_schema(),
        None,
    )
    .expect("collection must be created");
    let flat = IndexParams::flat(MetricType::Cosine).expect("flat descriptor must be valid");
    collection
        .create_index("embedding", &flat)
        .expect("the exact Flat execution path must be configurable");

    let doc = valid_doc("doc-1");
    collection.insert(&[&doc]).expect("insert must succeed");
    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes.len(), 1);
    assert_eq!(stats.indexes[0].index_type, IndexType::Flat);
    assert_eq!(stats.indexes[0].state, "ready");
    assert_eq!(stats.indexes[0].source_revision, stats.revision);
    assert_eq!(stats.indexes[0].document_count, 1);

    let query = SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    collection.query(&query).expect("exact query must succeed");
    let snapshot = collection
        .stats_snapshot()
        .expect("telemetry must be available");
    assert_eq!(snapshot.ann_query_count, 0);
    assert_eq!(snapshot.exact_query_count, 1);

    let mut first = SubQuery::new().expect("sub-query must be created");
    first
        .set_field_name("embedding")
        .expect("field name must be valid");
    first
        .set_query_vector(&[1.0, 0.0, 0.0])
        .expect("query vector must be valid");
    let mut second = first.clone();
    second
        .set_query_vector(&[0.0, 1.0, 0.0])
        .expect("query vector must be valid");
    let mut multi = MultiQuery::new().expect("multi-query must be created");
    multi
        .add_sub_query(&first)
        .expect("first branch must be accepted");
    multi
        .add_sub_query(&second)
        .expect("second branch must be accepted");
    collection
        .multi_query(&multi)
        .expect("exact multi-query must succeed");
    let snapshot = collection
        .stats_snapshot()
        .expect("telemetry must be available");
    assert_eq!(snapshot.ann_query_count, 0);
    assert_eq!(snapshot.exact_query_count, 2);
}

#[test]
fn future_query_controls_fail_without_mutating_the_query() {
    let mut vector =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    let error = vector
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
        .expect_err("HNSW search controls must fail until an HNSW executor exists");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(vector.params.is_empty());

    for error in [
        vector
            .set_ivf_params(IvfQueryParams::new(8, false, 1.0))
            .expect_err("IVF controls must fail explicitly"),
        vector
            .set_ivf_rabitq_params(IvfRabitqQueryParams::new(8, 0.0, false, false))
            .expect_err("IVF RaBitQ controls must fail explicitly"),
        vector
            .set_diskann_params(DiskannQueryParams::new(32))
            .expect_err("DiskANN controls must fail explicitly"),
    ] {
        assert_eq!(error.code, ErrorCode::NotSupported);
        assert!(vector.params.is_empty());
    }

    let error = vector
        .set_flat_params(FlatQueryParams::new(false, 2.0))
        .expect_err("unused Flat refinement controls must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(vector.params.is_empty());

    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("alpha beta")
        .expect("FTS expression must be valid");
    let mut lexical = SearchQuery::fts("title", &fts, 10).expect("FTS query must be valid");
    let error = lexical
        .set_fts_params(
            FtsQueryParams::new(Some("and")).expect("FTS controls must be syntactically valid"),
        )
        .expect_err("an unimplemented FTS operator must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(lexical.params.is_empty());

    let mut sub = SubQuery::new().expect("sub-query must be created");
    let error = sub
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
        .expect_err("sub-query ANN controls must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(sub.params.is_empty());

    let mut grouped = GroupBySearchQuery::new("embedding", "title", &[1.0, 0.0, 0.0], 2, 2)
        .expect("group-by query must be valid");
    let error = grouped
        .set_ivf_params(IvfQueryParams::new(8, false, 1.0))
        .expect_err("group-by ANN controls must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(grouped.params.is_empty());
}

#[test]
fn deserialized_future_and_unknown_query_controls_fail_at_execution_boundary() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &exact_schema(),
        None,
    )
    .expect("collection must be created");

    for parameter in [
        "type",
        "ef",
        "nprobe",
        "is_linear",
        "is_using_refiner",
        "scale_factor",
        "list_size",
        "operator",
    ] {
        let mut future =
            SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
        future.params.insert(parameter.into(), json!(64));
        let error = collection
            .query(&future)
            .expect_err("deserialized future controls must not be ignored");
        assert_eq!(error.code, ErrorCode::NotSupported, "parameter={parameter}");
    }

    let mut unknown =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    unknown.params.insert("mystery".into(), json!(true));
    let error = collection
        .query(&unknown)
        .expect_err("unknown query controls must not be ignored");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let snapshot = collection
        .stats_snapshot()
        .expect("telemetry must be available");
    assert_eq!(snapshot.query_count, 0);
    assert_eq!(snapshot.ann_query_count, 0);
}

#[test]
fn supported_exact_query_controls_are_executed_and_observed() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &exact_schema(),
        None,
    )
    .expect("collection must be created");
    let first = valid_doc("doc-1");
    let mut second = valid_doc("doc-2");
    second
        .add_vector_f32("embedding", &[0.0, 1.0, 0.0])
        .expect("embedding must be valid");
    collection
        .insert(&[&first, &second])
        .expect("documents must be inserted");

    let mut query =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    query.params.insert("metric".into(), json!("cosine"));
    query.set_radius(0.5).expect("radius must be valid");
    let result = collection.query(&query).expect("exact query must succeed");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get_pk(), Some("doc-1"));

    let snapshot = collection
        .stats_snapshot()
        .expect("telemetry must be available");
    assert_eq!(snapshot.query_count, 1);
    assert_eq!(snapshot.exact_query_count, 1);
    assert_eq!(snapshot.ann_query_count, 0);
    assert_eq!(snapshot.radius_query_count, 1);
}

#[test]
fn future_index_and_unused_build_parameters_fail_at_schema_boundary() {
    let future_vector_indexes = [
        IndexParams::hnsw(MetricType::Cosine, 16, 100).expect("descriptor must be valid"),
        IndexParams::ivf(MetricType::Cosine, 64, 10, false).expect("descriptor must be valid"),
        IndexParams::ivf_rabitq(MetricType::Cosine, 64, 8, 1_000)
            .expect("descriptor must be valid"),
        IndexParams::diskann(MetricType::Cosine, 32, 100, 8).expect("descriptor must be valid"),
        IndexParams::vamana(MetricType::Cosine, 32, 100, 1.2).expect("descriptor must be valid"),
        IndexParams::hnsw_rabitq(MetricType::Cosine, 16, 100).expect("descriptor must be valid"),
    ];
    for params in future_vector_indexes {
        let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
            .expect("vector schema must be valid");
        let error = field
            .set_index_params(&params)
            .expect_err("a future physical index must not enter a schema");
        assert_eq!(error.code, ErrorCode::NotSupported);
        assert!(!field.has_index());
    }

    let mut scalar =
        FieldSchema::new("title", DataType::String, false, 0).expect("title schema must be valid");
    let error = scalar
        .set_index_params(&IndexParams::invert(true, true).expect("descriptor must be valid"))
        .expect_err("an unimplemented inverted index must not enter a schema");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!scalar.has_index());

    for params in [
        IndexParams::fts(Some("standard"), Some(&["lowercase"]), None)
            .expect("descriptor must be valid"),
        IndexParams::fts(Some("standard"), None, Some("future=true"))
            .expect("descriptor must be valid"),
    ] {
        let mut field = FieldSchema::new("title", DataType::String, false, 0)
            .expect("title schema must be valid");
        let error = field
            .set_index_params(&params)
            .expect_err("unused FTS build controls must fail explicitly");
        assert_eq!(error.code, ErrorCode::NotSupported);
        assert!(!field.has_index());
    }

    let mut quantized =
        IndexParams::flat(MetricType::Cosine).expect("flat descriptor must be valid");
    quantized
        .set_quantize_type(QuantizeType::Int8)
        .expect("quantization descriptor must be syntactically valid");
    let mut vector = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    let error = vector
        .set_index_params(&quantized)
        .expect_err("unused Flat quantization must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!vector.has_index());

    let future_flat = IndexParams::flat(MetricType::Cosine)
        .expect("flat descriptor must be valid")
        .with_parameter("future", json!(true));
    let error = vector
        .set_index_params(&future_flat)
        .expect_err("unknown Flat build controls must not be persisted");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!vector.has_index());

    let mut deserialized = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    deserialized.index_params =
        Some(IndexParams::hnsw(MetricType::Cosine, 16, 100).expect("descriptor must be valid"));
    let error = CollectionSchema::builder("deserialized-index")
        .add_field(deserialized)
        .build()
        .expect_err("public or deserialized schema fields must be revalidated");
    assert_eq!(error.code, ErrorCode::NotSupported);
}

#[test]
fn unsupported_schema_tuning_does_not_mutate_schema_or_collection() {
    let error = CollectionSchema::builder("segmented")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("title schema must be valid"),
        )
        .max_doc_count_per_segment(1_000)
        .build()
        .expect_err("segment sizing must fail until segmentation exists");
    assert_eq!(error.code, ErrorCode::NotSupported);

    let reset = CollectionSchema::builder("unsegmented")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("title schema must be valid"),
        )
        .max_doc_count_per_segment(1_000)
        .max_doc_count_per_segment(0)
        .build()
        .expect("the builder's final supported value must win");
    assert_eq!(reset.max_doc_count_per_segment(), 0);

    let mut schema = exact_schema();
    let error = schema
        .set_max_doc_count_per_segment(1_000)
        .expect_err("direct segment sizing must fail until segmentation exists");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert_eq!(schema.max_doc_count_per_segment(), 0);

    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("collection must be created");
    let initial = collection.stats().expect("stats must be available");
    let category = FieldSchema::new("category", DataType::String, true, 0)
        .expect("category schema must be valid");
    let error = collection
        .add_column_with_options(&category, None, AddColumnOption { concurrency: 2 })
        .expect_err("unimplemented add-column concurrency must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!collection
        .schema()
        .expect("schema must be available")
        .has_field("category"));

    let altered_title = FieldSchema::new("title", DataType::String, true, 0)
        .expect("altered title schema must be valid");
    let error = collection
        .alter_column(&altered_title, AlterColumnOption { concurrency: 2 })
        .expect_err("unimplemented alter-column concurrency must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!collection
        .schema()
        .expect("schema must be available")
        .field("title")
        .expect("title must exist")
        .is_nullable());
    assert_eq!(
        collection.stats().expect("stats must be available"),
        initial
    );
}

#[test]
fn scanning_fts_configuration_is_not_reported_as_a_built_index() {
    let fts_params = IndexParams::fts(Some("standard"), None, None)
        .expect("standard FTS configuration must be valid");
    let mut title =
        FieldSchema::new("title", DataType::String, false, 0).expect("title schema must be valid");
    title
        .set_index_params(&fts_params)
        .expect("scan tokenizer configuration must be attachable");
    let schema = CollectionSchema::builder("fts-contract")
        .add_field(title)
        .build()
        .expect("schema must be valid");
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("collection must be created");
    assert!(collection
        .stats()
        .expect("stats must be available")
        .indexes
        .is_empty());

    let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
    doc.add_string("title", "alpha beta")
        .expect("title must be valid");
    collection.insert(&[&doc]).expect("insert must succeed");
    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("alpha")
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("title", &fts, 10).expect("FTS query must be valid");
    collection.query(&query).expect("scan FTS must succeed");
    let snapshot = collection
        .stats_snapshot()
        .expect("telemetry must be available");
    assert_eq!(snapshot.indexed_field_count, 0);
    assert_eq!(snapshot.ann_query_count, 0);
    assert_eq!(snapshot.exact_query_count, 1);
    assert_eq!(snapshot.fts_query_count, 1);
}
