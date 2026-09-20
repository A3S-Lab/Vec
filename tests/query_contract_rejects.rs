#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Query-contract rejection oracles for mismatched ANN control parameters.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    IndexParams, MetricType, SearchQuery,
};
use serde_json::json;
use tempfile::tempdir;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

fn flat_collection(name: &str) -> (tempfile::TempDir, Collection) {
    let temporary = tempdir().expect("temp");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let schema = CollectionSchema::builder(name)
        .add_field(embedding)
        .build()
        .expect("schema");
    let path = temporary.path().join(name);
    let collection = Collection::create(path.to_str().expect("utf8"), &schema, Some(&options()))
        .expect("create");
    let mut doc = Doc::with_pk("a").expect("pk");
    doc.add_vector_f32("embedding", &[1.0, 0.0]).expect("vec");
    collection.insert(&[&doc]).expect("insert");
    (temporary, collection)
}

fn hnsw_collection(name: &str) -> (tempfile::TempDir, Collection) {
    let temporary = tempdir().expect("temp");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"))
        .expect("attach");
    let schema = CollectionSchema::builder(name)
        .add_field(embedding)
        .build()
        .expect("schema");
    let path = temporary.path().join(name);
    let collection = Collection::create(path.to_str().expect("utf8"), &schema, Some(&options()))
        .expect("create");
    for index in 0..20 {
        let mut doc = Doc::with_pk(format!("d{index}")).expect("pk");
        doc.add_vector_f32("embedding", &[index as f32, 0.0])
            .expect("vec");
        collection.insert(&[&doc]).expect("insert");
    }
    (temporary, collection)
}

#[test]
fn mismatched_ann_type_and_control_params_fail_closed() {
    let (_tmp, flat) = flat_collection("qc-flat");
    let mut typed = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("query");
    typed.params.insert("type".into(), json!("hnsw"));
    assert!(
        flat.query(&typed).is_err(),
        "Flat must reject HNSW type override"
    );
    typed.params.insert("type".into(), json!("ivf"));
    assert!(flat.query(&typed).is_err());
    typed.params.insert("type".into(), json!("ivf_rabitq"));
    assert!(flat.query(&typed).is_err());
    typed.params.insert("type".into(), json!("diskann"));
    assert!(flat.query(&typed).is_err());
    typed.params.insert("type".into(), json!("vamana"));
    assert!(flat.query(&typed).is_err());
    typed.params.insert("type".into(), json!("unknown_ann"));
    assert!(flat.query(&typed).is_err());
    typed.params.insert("type".into(), json!(1));
    assert!(flat.query(&typed).is_err());

    let mut bad_ef = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("query");
    bad_ef.params.insert("ef".into(), json!(0));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("ef".into(), json!(-1));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("ef".into(), json!("x"));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("nprobe".into(), json!(0));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("list_size".into(), json!(0));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("bruteforce".into(), json!(1));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("metric".into(), json!(1));
    assert!(flat.query(&bad_ef).is_err());
    bad_ef.params.insert("metric".into(), json!("not-a-metric"));
    assert!(flat.query(&bad_ef).is_err());

    let (_tmp2, hnsw) = hnsw_collection("qc-hnsw");
    let mut ivf_on_hnsw = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("query");
    ivf_on_hnsw.params.insert("type".into(), json!("ivf"));
    assert!(hnsw.query(&ivf_on_hnsw).is_err());
    ivf_on_hnsw.params.insert("type".into(), json!("diskann"));
    assert!(hnsw.query(&ivf_on_hnsw).is_err());
    ivf_on_hnsw.params.insert("type".into(), json!("vamana"));
    assert!(hnsw.query(&ivf_on_hnsw).is_err());
    // Compatible HNSW type override must still execute.
    ivf_on_hnsw.params.insert("type".into(), json!("hnsw"));
    ivf_on_hnsw.params.insert("ef".into(), json!(16));
    assert!(hnsw.query(&ivf_on_hnsw).is_ok());

    // Positive-integer / boolean validation runs after family checks succeed.
    let mut bad_controls = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("query");
    bad_controls.params.insert("ef".into(), json!(0));
    assert!(hnsw.query(&bad_controls).is_err());
    bad_controls.params.insert("ef".into(), json!(-3));
    assert!(hnsw.query(&bad_controls).is_err());
    bad_controls.params.insert("ef".into(), json!("x"));
    assert!(hnsw.query(&bad_controls).is_err());
    bad_controls.params.insert("ef".into(), json!(32));
    bad_controls.params.insert("is_linear".into(), json!(1));
    assert!(hnsw.query(&bad_controls).is_err());
    bad_controls.params.insert("is_linear".into(), json!(true));
    bad_controls
        .params
        .insert("is_using_refiner".into(), json!("yes"));
    assert!(hnsw.query(&bad_controls).is_err());
    bad_controls
        .params
        .insert("is_using_refiner".into(), json!(false));
    assert!(hnsw.query(&bad_controls).is_ok());

    bad_controls.params.insert("operator".into(), json!("and"));
    assert!(hnsw.query(&bad_controls).is_err());
    bad_controls.params.remove("operator");
    bad_controls.params.insert("mystery".into(), json!(1));
    assert!(hnsw.query(&bad_controls).is_err());

    let mut bad_topk = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("topk builder");
    bad_topk.topk = 0;
    assert!(hnsw.query(&bad_topk).is_err());

    let mut radius = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("radius");
    radius.params.insert("radius".into(), json!("x"));
    assert!(hnsw.query(&radius).is_err());
    radius.params.insert("radius".into(), json!(f64::NAN));
    assert!(hnsw.query(&radius).is_err());
    radius.params.insert("radius".into(), json!(-1.0));
    assert!(hnsw.query(&radius).is_err());
    radius.params.insert("radius".into(), json!(1.5));
    assert!(hnsw.query(&radius).is_ok());

    let mut empty_vec = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("empty");
    empty_vec.vector = Some(vec![]);
    assert!(hnsw.query(&empty_vec).is_err());
    empty_vec.vector = Some(vec![1.0, f32::NAN]);
    assert!(hnsw.query(&empty_vec).is_err());
    empty_vec.vector = Some(vec![1.0]);
    assert!(hnsw.query(&empty_vec).is_err());
}

#[test]
fn fts_binary_and_sparse_query_contracts_fail_closed() {
    let temporary = tempdir().expect("temp");
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body");
    body.set_index_params(&IndexParams::fts(Some("whitespace"), None, None).expect("fts"))
        .expect("attach");
    let mut bits = FieldSchema::new("bits", DataType::VectorBinary32, false, 32).expect("bits");
    bits.set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let mut sparse =
        FieldSchema::new("sparse", DataType::SparseVectorFp32, false, 8).expect("sparse");
    sparse
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let schema = CollectionSchema::builder("qc-payloads")
        .add_field(body)
        .add_field(bits)
        .add_field(sparse)
        .add_field(embedding)
        .build()
        .expect("schema");
    let collection = Collection::create(
        temporary.path().join("qc-payloads").to_str().expect("utf8"),
        &schema,
        Some(&options()),
    )
    .expect("create");
    let mut doc = Doc::with_pk("a").expect("pk");
    doc.add_string("body", "rust").expect("body");
    doc.add_vector_binary32("bits", &[0; 4]).expect("bits");
    doc.add_sparse_vector_f32("sparse", &[0, 2], &[1.0, 0.5])
        .expect("sparse");
    doc.add_vector_f32("embedding", &[1.0, 0.0]).expect("vec");
    collection.insert(&[&doc]).expect("insert");

    let mut fts = a3s_vec::Fts::new().expect("fts");
    fts.set_match_string("rust").expect("match");
    let mut fts_query = SearchQuery::fts("body", &fts, 3).expect("fts query");
    fts_query.params.insert("ef".into(), json!(8));
    assert!(collection.query(&fts_query).is_err());
    fts_query.params.remove("ef");
    fts_query.params.insert("default_operator".into(), json!(1));
    assert!(collection.query(&fts_query).is_err());

    let mut binary = SearchQuery::binary("bits", &[0; 4], 3).expect("binary");
    binary.binary_vector = Some(vec![]);
    assert!(collection.query(&binary).is_err());
    binary.binary_vector = Some(vec![0; 8]);
    assert!(collection.query(&binary).is_err());
    let binary_on_dense = SearchQuery::binary("embedding", &[0; 4], 3).expect("wrong");
    assert!(collection.query(&binary_on_dense).is_err());

    let mut sparse_q = SearchQuery::sparse("sparse", &[0, 2], &[1.0, 0.5], 3).expect("sparse");
    sparse_q.sparse_vector = Some(vec![]);
    assert!(collection.query(&sparse_q).is_err());
    sparse_q.sparse_vector = Some(vec![(0, 1.0), (0, 0.5)]);
    assert!(collection.query(&sparse_q).is_err());
    sparse_q.sparse_vector = Some(vec![(0, 1.0), (9, 0.5)]);
    assert!(collection.query(&sparse_q).is_err());
    sparse_q.sparse_vector = Some(vec![(0, f32::NAN)]);
    assert!(collection.query(&sparse_q).is_err());
    let sparse_on_dense = SearchQuery::sparse("embedding", &[0], &[1.0], 3).expect("wrong");
    assert!(collection.query(&sparse_on_dense).is_err());

    let mut metric = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("metric");
    metric.params.insert("metric".into(), json!("mips_l2"));
    assert!(collection.query(&metric).is_ok());
    metric
        .params
        .insert("metric".into(), json!("inner_product"));
    assert!(collection.query(&metric).is_ok());
    metric.params.insert("metric".into(), json!("euclidean"));
    assert!(collection.query(&metric).is_ok());
    metric.params.insert("metric".into(), json!("mips-l2"));
    assert!(collection.query(&metric).is_ok());
    metric.params.insert("metric".into(), json!(true));
    assert!(collection.query(&metric).is_err());
    metric.params.insert("metric".into(), json!("bogus"));
    assert!(collection.query(&metric).is_err());

    let mut radius = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("radius");
    radius.params.insert("radius".into(), json!(f64::INFINITY));
    assert!(collection.query(&radius).is_err());
    radius.params.insert("radius".into(), json!(f64::NAN));
    assert!(collection.query(&radius).is_err());
    radius.params.insert("radius".into(), json!("1.0"));
    assert!(collection.query(&radius).is_err());

    // IVF family is required before scale_factor numeric validation runs.
    let temporary = tempdir().expect("temp");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::ivf(MetricType::L2, 4, 1, false).expect("ivf"))
        .expect("attach");
    let schema = CollectionSchema::builder("qc-ivf-scale")
        .add_field(embedding)
        .build()
        .expect("schema");
    let ivf = Collection::create(
        temporary
            .path()
            .join("qc-ivf-scale")
            .to_str()
            .expect("utf8"),
        &schema,
        Some(&options()),
    )
    .expect("create");
    for index in 0..40 {
        let mut doc = Doc::with_pk(format!("d{index}")).expect("pk");
        doc.add_vector_f32("embedding", &[index as f32, 0.0])
            .expect("vec");
        ivf.insert(&[&doc]).expect("insert");
    }
    let mut scale = SearchQuery::new("embedding", &[1.0, 0.0], 3).expect("scale");
    scale.params.insert("nprobe".into(), json!(2));
    scale.params.insert("scale_factor".into(), json!(0.0));
    assert!(ivf.query(&scale).is_err());
    scale.params.insert("scale_factor".into(), json!(f64::NAN));
    assert!(ivf.query(&scale).is_err());
    scale.params.insert("scale_factor".into(), json!("x"));
    assert!(ivf.query(&scale).is_err());
    scale.params.insert("scale_factor".into(), json!(1.5));
    assert!(ivf.query(&scale).is_ok());
}
