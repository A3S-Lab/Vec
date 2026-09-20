#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Schema index-contract validation error surface (I2).

use a3s_vec::{DataType, FieldSchema, IndexParams, IndexType, MetricType, QuantizeType};
use serde_json::json;

fn attach(data_type: DataType, dimension: u32, params: &IndexParams) -> Result<(), a3s_vec::Error> {
    let mut field = FieldSchema::new("f", data_type, false, dimension)?;
    field.set_index_params(params)
}

#[test]
fn index_contract_rejects_incompatible_and_invalid_configs() {
    assert!(attach(
        DataType::String,
        0,
        &IndexParams::flat(MetricType::L2).expect("flat")
    )
    .is_err());

    assert!(attach(
        DataType::VectorBinary32,
        4,
        &IndexParams::flat(MetricType::Cosine).expect("flat")
    )
    .is_err());

    assert!(attach(
        DataType::VectorFp32,
        8,
        &IndexParams::flat(MetricType::Undefined).expect("flat")
    )
    .is_err());

    let mut flat = IndexParams::flat(MetricType::L2).expect("flat");
    flat.set_quantize_type(QuantizeType::Int8)
        .expect("quantize");
    assert!(attach(DataType::VectorFp32, 8, &flat).is_err());

    let mut flat_extra = IndexParams::flat(MetricType::L2).expect("flat");
    flat_extra.params.insert("mystery".into(), json!(1));
    assert!(attach(DataType::VectorFp32, 8, &flat_extra).is_err());

    assert!(IndexParams::hnsw(MetricType::L2, 0, 16).is_err());
    let mut hnsw = IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw");
    hnsw.quantize_type = QuantizeType::Rabitq;
    hnsw.params
        .insert("quantize_type".into(), json!(QuantizeType::Rabitq));
    assert!(attach(DataType::VectorFp32, 8, &hnsw).is_err());

    let mut hnsw_unknown = IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw");
    hnsw_unknown.params.insert("nope".into(), json!(1));
    assert!(attach(DataType::VectorFp32, 8, &hnsw_unknown).is_err());

    let mut ivf = IndexParams::ivf(MetricType::L2, 4, 1, false).expect("ivf");
    ivf.params.insert("use_soar".into(), json!("yes"));
    assert!(attach(DataType::VectorFp32, 8, &ivf).is_err());

    let mut ivf_rabitq_wrong = IndexParams::ivf(MetricType::L2, 4, 1, false).expect("ivf");
    ivf_rabitq_wrong.quantize_type = QuantizeType::Rabitq;
    ivf_rabitq_wrong
        .params
        .insert("quantize_type".into(), json!(QuantizeType::Rabitq));
    assert!(attach(DataType::VectorFp32, 8, &ivf_rabitq_wrong).is_err());

    assert!(IndexParams::ivf_rabitq(MetricType::L2, 4, 10, 0).is_err());
    assert!(IndexParams::ivf_rabitq(MetricType::MipsL2, 4, 4, 0).is_ok());
    let mut rabitq = IndexParams::ivf_rabitq(MetricType::L2, 4, 4, 0).expect("rabitq");
    rabitq.quantize_type = QuantizeType::Undefined;
    assert!(attach(DataType::VectorFp32, 8, &rabitq).is_err());

    assert!(IndexParams::diskann(MetricType::L2, 8, 16, -1).is_err());
    let mut diskann = IndexParams::diskann(MetricType::L2, 8, 16, 4).expect("diskann");
    diskann.params.insert("alpha".into(), json!(0.5));
    assert!(attach(DataType::VectorFp32, 8, &diskann).is_err());
    diskann.params.insert("alpha".into(), json!(1.2));
    diskann.params.insert("pq_chunk_num".into(), json!(64));
    assert!(attach(DataType::VectorFp32, 8, &diskann).is_err());

    let mut vamana = IndexParams::vamana(MetricType::L2, 8, 16, 1.2).expect("vamana");
    vamana.params.insert("alpha".into(), json!(0.5));
    assert!(attach(DataType::VectorFp32, 8, &vamana).is_err());
    vamana.quantize_type = QuantizeType::Rabitq;
    vamana
        .params
        .insert("quantize_type".into(), json!(QuantizeType::Rabitq));
    vamana.params.insert("alpha".into(), json!(1.2));
    assert!(attach(DataType::VectorFp32, 8, &vamana).is_err());

    assert!(attach(
        DataType::Bool,
        0,
        &IndexParams::invert(true, false).expect("invert")
    )
    .is_err());
    assert!(attach(
        DataType::Int32,
        0,
        &IndexParams::invert(false, true).expect("invert")
    )
    .is_err());
    let mut invert = IndexParams::invert(false, false).expect("invert");
    invert.metric_type = MetricType::L2;
    assert!(attach(DataType::String, 0, &invert).is_err());

    let mut fts = IndexParams::fts(Some("standard"), None, None).expect("fts");
    fts.metric_type = MetricType::L2;
    assert!(attach(DataType::String, 0, &fts).is_err());
    let mut fts_empty = IndexParams::fts(Some("standard"), None, None).expect("fts");
    fts_empty.params.insert("tokenizer_name".into(), json!(""));
    assert!(attach(DataType::String, 0, &fts_empty).is_err());
    let mut fts_unknown = IndexParams::fts(Some("standard"), None, None).expect("fts");
    fts_unknown.params.insert("mystery".into(), json!(1));
    assert!(attach(DataType::String, 0, &fts_unknown).is_err());

    let mut undefined = IndexParams::flat(MetricType::L2).expect("flat");
    undefined.index_type = IndexType::Undefined;
    assert!(attach(DataType::VectorFp32, 8, &undefined).is_err());
}

#[test]
fn remaining_index_contract_error_branches() {
    // FTS on non-string
    assert!(attach(
        DataType::Int32,
        0,
        &IndexParams::fts(Some("standard"), None, None).expect("fts")
    )
    .is_err());

    // HNSW RaBitQ bits out of range via mutated params
    let mut hnsw_rq = IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw rabitq");
    hnsw_rq.params.insert("total_bits".into(), json!(10));
    assert!(attach(DataType::VectorFp32, 8, &hnsw_rq).is_err());

    let mut hnsw_rq = IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw rabitq");
    hnsw_rq.quantize_type = QuantizeType::Undefined;
    assert!(attach(DataType::VectorFp32, 8, &hnsw_rq).is_err());

    // IVF missing use_soar already covered; n_list zero via mutation
    let mut ivf = IndexParams::ivf(MetricType::L2, 4, 1, false).expect("ivf");
    ivf.params.insert("n_list".into(), json!(0));
    assert!(attach(DataType::VectorFp32, 8, &ivf).is_err());

    // Invert on vector field
    assert!(attach(
        DataType::VectorFp32,
        8,
        &IndexParams::invert(false, false).expect("invert")
    )
    .is_err());

    // Flat with extra param already covered; FTS non-string tokenizer type
    let mut fts = IndexParams::fts(Some("standard"), None, None).expect("fts");
    fts.params.insert("tokenizer_name".into(), json!(1));
    assert!(attach(DataType::String, 0, &fts).is_err());

    // DiskANN with quantize_type set
    let mut diskann = IndexParams::diskann(MetricType::L2, 8, 16, 2).expect("diskann");
    diskann.quantize_type = QuantizeType::Int8;
    assert!(attach(DataType::VectorFp32, 8, &diskann).is_err());

    // Vamana unsupported metric already construction-time; mutate metric after
    let mut vamana = IndexParams::vamana(MetricType::L2, 8, 16, 1.2).expect("vamana");
    vamana.metric_type = MetricType::Undefined;
    assert!(attach(DataType::VectorFp32, 8, &vamana).is_err());

    // ANN on binary vector
    assert!(attach(
        DataType::VectorBinary32,
        4,
        &IndexParams::hnsw(MetricType::L2, 8, 16).expect("hnsw")
    )
    .is_err());

    // Redundant quantize_type disagreement
    let mut hnsw = IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw");
    hnsw.params
        .insert("quantize_type".into(), json!(QuantizeType::Int8));
    assert!(attach(DataType::VectorFp32, 8, &hnsw).is_err());

    // DiskANN pq_chunk_num exceeding dimension
    let mut diskann_chunks = IndexParams::diskann(MetricType::L2, 8, 16, 2).expect("diskann");
    diskann_chunks
        .params
        .insert("pq_chunk_num".into(), json!(64));
    assert!(attach(DataType::VectorFp32, 8, &diskann_chunks).is_err());

    // Invert on non-scalar array types fails closed.
    assert!(attach(
        DataType::ArrayString,
        0,
        &IndexParams::invert(false, false).expect("invert")
    )
    .is_err());
    assert!(attach(
        DataType::Binary,
        0,
        &IndexParams::invert(false, false).expect("invert")
    )
    .is_ok());

    // IVF RaBitQ missing total_bits after mutation
    let mut ivf_rq = IndexParams::ivf_rabitq(MetricType::L2, 4, 4, 0).expect("ivf rq");
    ivf_rq.params.remove("total_bits");
    assert!(attach(DataType::VectorFp32, 8, &ivf_rq).is_err());
}
