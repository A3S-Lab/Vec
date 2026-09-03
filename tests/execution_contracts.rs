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
fn ann_and_fts_index_lifecycle_is_live() {
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
    let hnsw =
        IndexParams::hnsw(MetricType::Cosine, 16, 100).expect("HNSW descriptor must be valid");
    collection
        .create_index("embedding", &hnsw)
        .expect("HNSW build must succeed");
    assert!(collection
        .schema()
        .expect("schema must be available")
        .has_index("embedding"));
    let indexed = collection.stats().expect("stats must be available");
    assert_eq!(indexed.indexes[0].index_type, IndexType::Hnsw);

    let fts =
        IndexParams::fts(Some("standard"), None, None).expect("FTS configuration must be valid");
    collection
        .create_index("title", &fts)
        .expect("FTS index build must succeed");
    assert!(collection
        .schema()
        .expect("schema must be available")
        .has_index("title"));
    let indexed = collection.stats().expect("stats must be available");
    assert_eq!(indexed.indexes.len(), 2);
    assert_eq!(indexed.indexes[0].index_type, IndexType::Hnsw);
    assert_eq!(indexed.indexes[1].index_type, IndexType::Fts);

    collection
        .optimize()
        .expect("index optimization must succeed");
    assert_eq!(
        collection.stats().expect("stats must be available"),
        indexed
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
fn ann_query_controls_are_typed_while_unused_controls_remain_unsupported() {
    let mut vector =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    vector
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
        .expect("HNSW search controls must be accepted");
    assert_eq!(vector.params["type"], json!("hnsw"));
    assert_eq!(vector.params["ef"], json!(64));

    let mut ivf = SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    ivf.set_ivf_params(IvfQueryParams::new(8, false, 1.5))
        .expect("IVF controls must be accepted");
    assert_eq!(ivf.params["type"], json!("ivf"));
    assert_eq!(ivf.params["nprobe"], json!(8));

    let mut vamana =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    vamana
        .set_diskann_params(DiskannQueryParams::new(32))
        .expect("Vamana/DiskANN search controls must be accepted");
    assert_eq!(vamana.params["type"], json!("diskann"));
    assert_eq!(vamana.params["list_size"], json!(32));

    let mut ivf_rabitq =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    let mut rabitq_params = IvfRabitqQueryParams::new(8, 2.5, false, true);
    rabitq_params
        .set_scale_factor(3.0)
        .expect("RaBitQ refinement scale must be valid");
    ivf_rabitq
        .set_ivf_rabitq_params(rabitq_params)
        .expect("IVF RaBitQ controls must be accepted");
    assert_eq!(ivf_rabitq.params["type"], json!("ivf_rabitq"));
    assert_eq!(ivf_rabitq.params["nprobe"], json!(8));
    assert_eq!(ivf_rabitq.params["radius"], json!(2.5));
    assert_eq!(ivf_rabitq.params["scale_factor"], json!(3.0));
    assert_eq!(ivf_rabitq.params["is_using_refiner"], json!(true));

    let mut flat =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    let error = flat
        .set_flat_params(FlatQueryParams::new(false, 2.0))
        .expect_err("unused Flat refinement controls must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(flat.params.is_empty());

    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("alpha beta")
        .expect("FTS expression must be valid");
    let mut lexical = SearchQuery::fts("title", &fts, 10).expect("FTS query must be valid");
    lexical
        .set_fts_params(
            FtsQueryParams::new(Some("and")).expect("FTS controls must be syntactically valid"),
        )
        .expect("FTS default operator must be accepted");
    assert_eq!(lexical.params["default_operator"], json!("and"));

    let mut sub = SubQuery::new().expect("sub-query must be created");
    sub.set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
        .expect("sub-query HNSW controls must be accepted");
    assert_eq!(sub.params["ef"], json!(64));
    sub.set_diskann_params(DiskannQueryParams::new(24))
        .expect("sub-query Vamana controls must be accepted");
    assert_eq!(sub.params["type"], json!("diskann"));
    assert_eq!(sub.params["list_size"], json!(24));
    assert!(sub.params.get("ef").is_none());

    let mut grouped = GroupBySearchQuery::new("embedding", "title", &[1.0, 0.0, 0.0], 2, 2)
        .expect("group-by query must be valid");
    grouped
        .set_ivf_params(IvfQueryParams::new(8, false, 1.0))
        .expect("group-by IVF controls must be accepted");
    assert_eq!(grouped.params["nprobe"], json!(8));
    grouped
        .set_diskann_params(DiskannQueryParams::new(24))
        .expect("group-by Vamana controls must be accepted");
    assert_eq!(grouped.params["list_size"], json!(24));
    assert!(grouped.params.get("nprobe").is_none());

    let mut invalid =
        SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
    let error = invalid
        .set_hnsw_params(HnswQueryParams::new(0, 0.0, false, false))
        .expect_err("zero ef must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(invalid.params.is_empty());
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

    for (parameter, value) in [
        ("type", json!("hnsw")),
        ("ef", json!(64)),
        ("nprobe", json!(8)),
        ("is_linear", json!(true)),
        ("is_using_refiner", json!(true)),
        ("scale_factor", json!(1.5)),
        ("list_size", json!(64)),
        ("operator", json!("and")),
    ] {
        let mut future =
            SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10).expect("query must be valid");
        future.params.insert(parameter.into(), value);
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
fn live_ann_indexes_are_accepted_while_unused_build_parameters_fail_at_schema_boundary() {
    for params in [
        IndexParams::hnsw(MetricType::Cosine, 16, 100).expect("descriptor must be valid"),
        IndexParams::ivf(MetricType::Cosine, 64, 10, false).expect("descriptor must be valid"),
        IndexParams::ivf(MetricType::Cosine, 64, 10, true).expect("SOAR descriptor must be valid"),
        IndexParams::hnsw_rabitq(MetricType::Cosine, 16, 100).expect("descriptor must be valid"),
        IndexParams::ivf_rabitq(MetricType::Cosine, 64, 8, 1_000)
            .expect("descriptor must be valid"),
        IndexParams::diskann(MetricType::L2, 32, 100, 2).expect("descriptor must be valid"),
        IndexParams::vamana(MetricType::L2, 32, 100, 1.2).expect("descriptor must be valid"),
    ] {
        let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
            .expect("vector schema must be valid");
        field
            .set_index_params(&params)
            .expect("native ANN indexes must enter the schema");
        assert!(field.has_index());
    }

    let mut scalar =
        FieldSchema::new("title", DataType::String, false, 0).expect("title schema must be valid");
    scalar
        .set_index_params(&IndexParams::invert(true, true).expect("descriptor must be valid"))
        .expect("native inverted indexes must enter the schema");
    assert!(scalar.has_index());

    let mut analyzed = FieldSchema::new("analyzed", DataType::String, false, 0)
        .expect("analyzed schema must be valid");
    analyzed
        .set_index_params(
            &IndexParams::fts(Some("standard"), Some(&["lowercase"]), None)
                .expect("descriptor must be valid"),
        )
        .expect("implemented FTS filters must enter the schema");
    assert!(analyzed.has_index());

    let mut invalid_fts = FieldSchema::new("invalid", DataType::String, false, 0)
        .expect("invalid schema must be constructible");
    let error = invalid_fts
        .set_index_params(
            &IndexParams::fts(Some("standard"), None, Some("future=true"))
                .expect("descriptor must be valid"),
        )
        .expect_err("malformed FTS extra parameters must fail explicitly");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(!invalid_fts.has_index());

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
    deserialized.index_params = Some(
        IndexParams::hnsw(MetricType::Cosine, 16, 100)
            .expect("descriptor must be valid")
            .with_parameter("future", json!(true)),
    );
    let error = CollectionSchema::builder("deserialized-index")
        .add_field(deserialized)
        .build()
        .expect_err("public or deserialized schema fields must be revalidated");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

#[test]
fn diskann_accepts_numeric_metrics_and_rejects_invalid_pq_shapes() {
    let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    let cosine = IndexParams::diskann(MetricType::Cosine, 32, 100, 2)
        .expect("descriptor must be syntactically valid");
    field
        .set_index_params(&cosine)
        .expect("DiskANN full ADC path must support cosine");
    assert!(field.has_index());

    let too_many_chunks = IndexParams::diskann(MetricType::L2, 32, 100, 4)
        .expect("descriptor must be syntactically valid");
    let mut shape_field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    let error = shape_field
        .set_index_params(&too_many_chunks)
        .expect_err("PQ cannot create empty chunks");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let mut ambiguous =
        IndexParams::diskann(MetricType::L2, 32, 100, 2).expect("descriptor must be valid");
    ambiguous
        .set_quantize_type(QuantizeType::Pq)
        .expect("quantization descriptor must be syntactically valid");
    let mut ambiguous_field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    let error = ambiguous_field
        .set_index_params(&ambiguous)
        .expect_err("DiskANN PQ must have one source of truth");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!ambiguous_field.has_index());
}

#[test]
fn vamana_accepts_all_numeric_metrics_controls_and_scalar_quantization() {
    let error = IndexParams::vamana(MetricType::L2, 32, 100, 0.9)
        .expect_err("Vamana alpha below one must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    for metric in [
        MetricType::L2,
        MetricType::Ip,
        MetricType::Cosine,
        MetricType::MipsL2,
    ] {
        let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
            .expect("vector schema must be valid");
        field
            .set_index_params(
                &IndexParams::vamana(metric, 32, 100, 1.2)
                    .expect("descriptor must be syntactically valid"),
            )
            .expect("Vamana must accept every exact numeric metric");
        assert!(field.has_index());
    }

    let tuned = IndexParams::vamana(MetricType::L2, 32, 100, 1.2)
        .expect("descriptor must be valid")
        .with_parameter("max_occlusion", json!(4));
    let mut tuned_field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    tuned_field
        .set_index_params(&tuned)
        .expect("Vamana max-occlusion control must be accepted");
    assert!(tuned_field.has_index());

    for quantize in [QuantizeType::Fp16, QuantizeType::Int8, QuantizeType::Int4] {
        let mut quantized =
            IndexParams::vamana(MetricType::L2, 32, 100, 1.2).expect("descriptor must be valid");
        quantized
            .set_quantize_type(quantize)
            .expect("quantization descriptor must be syntactically valid");
        let mut quantized_field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
            .expect("vector schema must be valid");
        quantized_field
            .set_index_params(&quantized)
            .expect("scalar Vamana quantization must have an execution consumer");
        assert!(quantized_field.has_index());
    }

    let mut rabitq =
        IndexParams::vamana(MetricType::L2, 32, 100, 1.2).expect("descriptor must be valid");
    rabitq
        .set_quantize_type(QuantizeType::Rabitq)
        .expect("quantization descriptor must be syntactically valid");
    let mut rabitq_field = FieldSchema::new("embedding", DataType::VectorFp32, false, 3)
        .expect("vector schema must be valid");
    let error = rabitq_field
        .set_index_params(&rabitq)
        .expect_err("RaBitQ must use its dedicated Vamana-family index type");
    assert_eq!(error.code, ErrorCode::NotSupported);
    assert!(!rabitq_field.has_index());
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
fn fts_configuration_builds_an_observable_index() {
    let fts_params = IndexParams::fts(Some("standard"), None, None)
        .expect("standard FTS configuration must be valid");
    let mut title =
        FieldSchema::new("title", DataType::String, false, 0).expect("title schema must be valid");
    title
        .set_index_params(&fts_params)
        .expect("FTS index configuration must be attachable");
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
    let initial = collection.stats().expect("stats must be available");
    assert_eq!(initial.indexes.len(), 1);
    assert_eq!(initial.indexes[0].index_type, IndexType::Fts);
    assert_eq!(initial.indexes[0].document_count, 0);

    let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
    doc.add_string("title", "alpha beta")
        .expect("title must be valid");
    collection.insert(&[&doc]).expect("insert must succeed");
    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("alpha")
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("title", &fts, 10).expect("FTS query must be valid");
    collection.query(&query).expect("indexed FTS must succeed");
    let snapshot = collection
        .stats_snapshot()
        .expect("telemetry must be available");
    assert_eq!(snapshot.indexed_field_count, 1);
    assert_eq!(snapshot.ann_query_count, 0);
    assert_eq!(snapshot.exact_query_count, 1);
    assert_eq!(snapshot.fts_query_count, 1);
    assert_eq!(snapshot.fts_index_query_count, 1);
}
