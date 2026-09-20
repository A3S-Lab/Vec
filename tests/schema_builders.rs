#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines
)]

//! Exhaustive IndexParams / schema builder coverage (coverage + contracts).

use a3s_vec::{
    DataType, FieldSchema, IndexParams, IndexParamsBuilder, IndexType, MetricType, QuantizeType,
    VectorSchema,
};
use serde_json::json;

#[test]
fn index_params_constructors_accept_and_reject_bounds() {
    assert!(IndexParams::hnsw(MetricType::Cosine, 16, 100).is_ok());
    assert!(IndexParams::hnsw(MetricType::Cosine, 0, 100).is_err());
    assert!(IndexParams::hnsw_with_quantize(MetricType::L2, 8, 32, QuantizeType::Int8).is_ok());
    assert!(IndexParams::ivf(MetricType::Ip, 16, 10, true).is_ok());
    assert!(IndexParams::ivf(MetricType::Ip, 0, 10, false).is_err());
    assert!(IndexParams::ivf_rabitq(MetricType::Cosine, 8, 4, 100).is_ok());
    assert!(IndexParams::ivf_rabitq(MetricType::Cosine, 8, 0, 100).is_err());
    assert!(IndexParams::hnsw_rabitq(MetricType::Cosine, 8, 32).is_ok());
    assert!(IndexParams::flat(MetricType::L2).is_ok());
    assert!(IndexParams::diskann(MetricType::L2, 16, 32, 0).is_ok());
    assert!(IndexParams::diskann(MetricType::L2, 0, 32, 0).is_err());
    assert!(IndexParams::vamana(MetricType::Cosine, 16, 32, 1.2).is_ok());
    assert!(IndexParams::vamana(MetricType::Cosine, 16, 32, 0.5).is_err());
    assert!(IndexParams::vamana_with_options(MetricType::L2, 16, 32, 1.2, 8, true).is_ok());
    assert!(IndexParams::invert(true, false).is_ok());
    assert!(IndexParams::fts(Some("standard"), None, None).is_ok());
    assert!(IndexParams::fts(Some("   "), None, None).is_err());

    let mut hnsw = IndexParams::hnsw(MetricType::Cosine, 8, 16).expect("hnsw");
    assert_eq!(hnsw.index_type(), IndexType::Hnsw);
    assert_eq!(hnsw.metric_type(), MetricType::Cosine);
    hnsw.set_metric_type(MetricType::Ip).expect("metric");
    hnsw.set_quantize_type(QuantizeType::Fp16).expect("q");
    assert_eq!(hnsw.quantize_type(), QuantizeType::Fp16);
    assert!(hnsw.parameter("m").is_some());
    let with = hnsw.clone().with_parameter("custom", json!(1));
    assert_eq!(with.parameter("custom"), Some(&json!(1)));
    assert!(IndexType::Hnsw.is_vector_index());
    assert!(!IndexType::Fts.is_vector_index());
}

#[test]
fn index_params_builder_fluent_surface_covers_typed_fields() {
    let built = IndexParamsBuilder::new(IndexType::Hnsw)
        .metric(MetricType::Cosine)
        .quantize_type(QuantizeType::Undefined)
        .m(16)
        .ef_construction(64)
        .n_list(32)
        .n_iters(10)
        .use_soar(true)
        .max_degree(16)
        .list_size(64)
        .search_list_size(32)
        .pq_chunk_num(4)
        .alpha(1.2)
        .max_occlusion(8)
        .saturate(true)
        .tokenizer("standard")
        .parameter("total_bits", json!(4))
        .parameter("sample_count", json!(100))
        .build()
        .expect("builder");
    assert_eq!(built.index_type(), IndexType::Hnsw);
    assert!(built.parameter("m").is_some());
    assert!(IndexParamsBuilder::new(IndexType::Flat)
        .metric_type(MetricType::Undefined)
        .build()
        .is_err());
}

#[test]
fn field_and_vector_schema_helpers_cover_accessors() {
    let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 8).expect("field");
    assert_eq!(field.data_type(), DataType::VectorFp32);
    assert!(field.is_vector_field());
    assert!(field.is_dense_vector());
    assert!(!field.is_sparse_vector());
    assert!(!field.is_array_type());
    assert!(!field.has_index());
    field
        .set_index_params(&IndexParams::flat(MetricType::Cosine).expect("flat"))
        .expect("attach");
    assert!(field.has_index());
    assert_eq!(field.index_type(), IndexType::Flat);
    assert!(field.index_params().is_some());
    assert!(!field.is_nullable());
    assert_eq!(field.name(), "embedding");

    let mut vector = VectorSchema::new("v", DataType::VectorFp16, 4).expect("vector");
    assert_eq!(vector.data_type(), DataType::VectorFp16);
    vector
        .set_index_params(&IndexParams::hnsw(MetricType::L2, 8, 16).expect("hnsw"))
        .expect("attach");
    assert!(VectorSchema::new("bad", DataType::String, 0).is_err());
}

#[test]
fn collection_schema_builder_covers_vector_and_indexed_helpers() {
    use a3s_vec::CollectionSchema;

    let schema = CollectionSchema::builder("builder-surface")
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            4,
            IndexParams::flat(MetricType::Cosine).expect("flat"),
        )
        .add_indexed_field(
            "title",
            DataType::String,
            IndexParams::fts(Some("standard"), None, None).expect("fts"),
        )
        .build()
        .expect("schema");
    assert!(schema.has_index("embedding"));
    assert!(schema.has_index("title"));

    assert!(CollectionSchema::builder("segmented")
        .add_field(FieldSchema::new("x", DataType::Int32, false, 0).expect("field"),)
        .max_doc_count_per_segment(1_000)
        .build()
        .is_err());

    assert!(CollectionSchema::builder("bad")
        .add_vector_field(
            "embedding",
            DataType::String,
            0,
            IndexParams::flat(MetricType::L2).expect("flat"),
        )
        .build()
        .is_err());

    assert!(CollectionSchema::builder("")
        .add_field(FieldSchema::new("x", DataType::Int32, false, 0).expect("field"))
        .build()
        .is_err());

    assert!(CollectionSchema::builder("bad-name")
        .add_vector_field(
            "",
            DataType::VectorFp32,
            2,
            IndexParams::flat(MetricType::L2).expect("flat"),
        )
        .build()
        .is_err());

    assert!(CollectionSchema::builder("bad-indexed")
        .add_indexed_field(
            "",
            DataType::String,
            IndexParams::invert(false, false).expect("invert"),
        )
        .build()
        .is_err());
}

#[test]
fn collection_and_vector_schema_mutators_cover_accessors() {
    use a3s_vec::{CollectionSchema, FieldSchema, IndexParams, IndexType, VectorSchema};

    let mut schema = CollectionSchema::new("mutators").expect("schema");
    assert_eq!(schema.name(), "mutators");
    let title = FieldSchema::new("title", DataType::String, true, 0).expect("title");
    schema.add_field(&title).expect("add");
    assert!(schema.has_field("title"));
    assert_eq!(
        schema.field("title").map(a3s_vec::FieldSchema::name),
        Some("title")
    );
    assert!(title.is_nullable());
    assert!(!title.is_vector_field());
    assert!(!title.is_dense_vector());
    assert!(!title.is_sparse_vector());
    assert!(!title.is_array_type());
    assert!(!title.has_index());
    assert_eq!(title.index_type(), IndexType::Undefined);
    assert_eq!(title.data_type(), DataType::String);
    assert_eq!(title.dimension(), 0);

    let mut vector = VectorSchema::new("embedding", DataType::VectorFp32, 4).expect("vector");
    vector
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("index");
    assert_eq!(vector.name(), "embedding");
    assert_eq!(vector.dimension(), 4);
    assert_eq!(vector.data_type(), DataType::VectorFp32);
    assert!(vector.has_index());
    assert_eq!(vector.index_type(), IndexType::Flat);
    schema.add_vector_field(&vector).expect("add vector");
    assert!(schema.has_index("embedding"));
    assert!(schema.vector("embedding").is_some());

    schema
        .add_index("title", &IndexParams::invert(false, false).expect("invert"))
        .expect("add index");
    assert!(schema.has_index("title"));
    assert!(schema
        .add_index(
            "missing",
            &IndexParams::invert(false, false).expect("invert")
        )
        .is_err());
    assert!(schema.drop_field("missing").is_err());
    schema.drop_field("title").expect("drop");
    assert!(!schema.has_field("title"));

    let mut dup = CollectionSchema::new("dup").expect("dup");
    dup.add_field(&FieldSchema::new("x", DataType::Int32, false, 0).expect("x"))
        .expect("add");
    assert!(dup
        .add_field(&FieldSchema::new("x", DataType::Int32, false, 0).expect("x"))
        .is_err());
}

#[test]
fn collection_schema_validate_and_mutators_reject_unsupported_shapes() {
    use a3s_vec::{CollectionSchema, FieldSchema, IndexParams, VectorSchema};

    let mut schema = CollectionSchema::new("validate").expect("schema");
    schema
        .add_field(&FieldSchema::new("title", DataType::String, false, 0).expect("title"))
        .expect("add");
    assert!(schema.validate().is_ok());
    assert!(!schema.digest().is_empty());
    assert_eq!(schema.fields().len(), 1);
    assert!(schema.vectors().is_empty());

    assert!(schema
        .set_max_doc_count_per_segment(1)
        .unwrap_err()
        .message
        .contains("segmented"));
    schema
        .set_max_doc_count_per_segment(0)
        .expect("zero allowed");
    assert_eq!(schema.max_doc_count_per_segment(), 0);

    schema
        .add_index("title", &IndexParams::invert(false, false).expect("invert"))
        .expect("index");
    schema.drop_index("title").expect("drop index");
    assert!(!schema.has_index("title"));
    assert!(schema.drop_index("missing").is_err());

    let mut vector = VectorSchema::new("embedding", DataType::VectorFp32, 2).expect("vector");
    vector
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("params");
    schema.add_vector_field(&vector).expect("vector");
    schema.drop_index("embedding").expect("drop vector index");
    assert!(!schema.has_index("embedding"));

    // Duplicate name across scalar/vector lists must fail closed on validate.
    let mut broken = CollectionSchema::new("broken").expect("broken");
    broken
        .add_field(&FieldSchema::new("shared", DataType::Int32, false, 0).expect("scalar"))
        .expect("scalar");
    let clash = VectorSchema::new("shared", DataType::VectorFp32, 2).expect("vector");
    // Bypass builder uniqueness by mutating after construction is not possible
    // through public add_*; ensure add_vector rejects duplicate names instead.
    assert!(broken.add_vector_field(&clash).is_err());

    let empty = CollectionSchema::new("empty").expect("empty");
    assert!(empty.validate().is_err());
    assert!(empty.fields().is_empty());
    assert!(empty.vectors().is_empty());

    // Direct construction covers validate branches builders refuse to emit.
    let mut segmented = CollectionSchema::new("segmented-validate").expect("schema");
    segmented
        .add_field(&FieldSchema::new("x", DataType::Int32, false, 0).expect("x"))
        .expect("add");
    segmented.max_doc_count_per_segment = 10;
    assert!(segmented
        .validate()
        .unwrap_err()
        .message
        .contains("segmented"));

    let mut vector_in_scalar = CollectionSchema::new("vector-in-scalar").expect("schema");
    vector_in_scalar.fields.push(
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("vector-as-field"),
    );
    assert!(vector_in_scalar
        .validate()
        .unwrap_err()
        .message
        .contains("vector schema list"));

    let mut scalar_in_vector = CollectionSchema::new("scalar-in-vector").expect("schema");
    scalar_in_vector.vectors.push(VectorSchema {
        name: "title".into(),
        data_type: DataType::String,
        dimension: 0,
        index_params: None,
    });
    assert!(scalar_in_vector
        .validate()
        .unwrap_err()
        .message
        .contains("scalar schema list"));

    let mut dup_scalar = CollectionSchema::new("dup-scalar").expect("schema");
    let title = FieldSchema::new("title", DataType::String, false, 0).expect("title");
    dup_scalar.fields.push(title.clone());
    dup_scalar.fields.push(title);
    assert!(dup_scalar
        .validate()
        .unwrap_err()
        .message
        .contains("duplicate"));

    let mut dup_vector = CollectionSchema::new("dup-vector").expect("schema");
    let embedding = VectorSchema::new("embedding", DataType::VectorFp32, 2).expect("embedding");
    dup_vector.vectors.push(embedding.clone());
    dup_vector.vectors.push(embedding);
    assert!(dup_vector
        .validate()
        .unwrap_err()
        .message
        .contains("duplicate"));

    let mut cross_dup = CollectionSchema::new("cross-dup").expect("schema");
    cross_dup
        .fields
        .push(FieldSchema::new("shared", DataType::Int32, false, 0).expect("scalar"));
    cross_dup
        .vectors
        .push(VectorSchema::new("shared", DataType::VectorFp32, 2).expect("vector"));
    assert!(cross_dup
        .validate()
        .unwrap_err()
        .message
        .contains("duplicate"));
}
