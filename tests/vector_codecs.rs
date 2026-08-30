use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, SearchQuery, VectorValue,
};
use serde_json::json;
use tempfile::tempdir;

#[test]
fn dense_schema_rejects_a_different_physical_vector_type() {
    let temporary = tempdir().expect("temporary directory must be available");
    for (data_type, dimension) in [
        (DataType::VectorFp16, 3),
        (DataType::VectorFp64, 3),
        (DataType::VectorInt4, 3),
        (DataType::VectorInt8, 3),
        (DataType::VectorInt16, 3),
        (DataType::VectorBinary32, 32),
        (DataType::VectorBinary64, 64),
    ] {
        let schema = CollectionSchema::builder("strict-vector-type")
            .add_field(
                FieldSchema::new("embedding", data_type, false, dimension)
                    .expect("vector schema must be valid"),
            )
            .build()
            .expect("collection schema must be valid");
        let path = temporary.path().join(format!("{data_type:?}"));
        let collection = Collection::create(
            path.to_str().expect("temporary path must be UTF-8"),
            &schema,
            None,
        )
        .expect("collection must be created");
        let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
        doc.add_vector_f32("embedding", &vec![0.0; dimension as usize])
            .expect("f32 payload must be valid in isolation");

        let result = collection
            .insert(&[&doc])
            .expect("typed rejection must be a per-document result");
        assert_eq!(result.success_count, 0, "data_type={data_type:?}");
        assert_eq!(result.error_count, 1, "data_type={data_type:?}");
        assert_eq!(
            result.results[0].code,
            ErrorCode::InvalidArgument,
            "data_type={data_type:?}"
        );
    }
}

#[test]
fn native_int4_and_fp16_payloads_reject_unrepresentable_values() {
    for values in [vec![-9], vec![8], vec![-8, 0, 7, 8]] {
        let mut doc = Doc::with_pk("int4").expect("primary key must be valid");
        let error = doc
            .add_vector_i4("embedding", &values)
            .expect_err("signed INT4 values must stay in -8..=7");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(!doc.has_field("embedding"));
    }

    for bits in [0x7c00, 0xfc00, 0x7e00, 0xfe00] {
        let mut doc = Doc::with_pk("fp16").expect("primary key must be valid");
        let error = doc
            .add_vector_fp16("embedding", &[bits])
            .expect_err("FP16 infinity and NaN must fail at the document boundary");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(!doc.has_field("embedding"));
    }
}

#[test]
fn binary_schema_and_payload_respect_the_declared_chunk_width() {
    for (data_type, invalid_dimension) in [
        (DataType::VectorBinary32, 31),
        (DataType::VectorBinary32, 33),
        (DataType::VectorBinary64, 32),
        (DataType::VectorBinary64, 65),
    ] {
        let error = FieldSchema::new("fingerprint", data_type, false, invalid_dimension)
            .expect_err("binary schema dimensions must align to their chunk width");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }

    let mut doc = Doc::with_pk("binary").expect("primary key must be valid");
    let error = doc
        .add_vector_binary32("fingerprint", &[0; 3])
        .expect_err("Binary32 payloads must contain complete 32-bit chunks");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = doc
        .add_vector_binary64("fingerprint", &[0; 4])
        .expect_err("Binary64 payloads must contain complete 64-bit chunks");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(!doc.has_field("fingerprint"));
}

fn native_vector_schema() -> CollectionSchema {
    CollectionSchema::builder("native-vector-roundtrip")
        .add_field(
            FieldSchema::new("fp16", DataType::VectorFp16, false, 3)
                .expect("FP16 schema must be valid"),
        )
        .add_field(
            FieldSchema::new("int4", DataType::VectorInt4, false, 3)
                .expect("INT4 schema must be valid"),
        )
        .add_field(
            FieldSchema::new("int8", DataType::VectorInt8, false, 3)
                .expect("INT8 schema must be valid"),
        )
        .add_field(
            FieldSchema::new("int16", DataType::VectorInt16, false, 3)
                .expect("INT16 schema must be valid"),
        )
        .add_field(
            FieldSchema::new("binary32", DataType::VectorBinary32, false, 32)
                .expect("Binary32 schema must be valid"),
        )
        .add_field(
            FieldSchema::new("binary64", DataType::VectorBinary64, false, 64)
                .expect("Binary64 schema must be valid"),
        )
        .add_field(
            FieldSchema::new("sparse_fp16", DataType::SparseVectorFp16, false, 4)
                .expect("sparse FP16 schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn native_vector_doc() -> Doc {
    let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
    doc.add_vector_fp16("fp16", &[0x0001, 0x3c00, 0x7bff])
        .expect("FP16 bits must be valid");
    doc.add_vector_i4("int4", &[-8, 0, 7])
        .expect("INT4 values must be valid");
    doc.add_vector_i8("int8", &[i8::MIN, 0, i8::MAX])
        .expect("INT8 values must be valid");
    doc.add_vector_i16("int16", &[i16::MIN, 0, i16::MAX])
        .expect("INT16 values must be valid");
    doc.add_vector_binary32("binary32", &[0xde, 0xad, 0xbe, 0xef])
        .expect("Binary32 bytes must be valid");
    doc.add_vector_binary64(
        "binary64",
        &[0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef],
    )
    .expect("Binary64 bytes must be valid");
    doc.add_sparse_vector_fp16("sparse_fp16", &[0, 3], &[0x0001, 0x7bff])
        .expect("sparse FP16 bits must be valid");
    doc
}

fn assert_native_vector_values(doc: &Doc) {
    assert_eq!(
        doc.vector("fp16"),
        Some(&VectorValue::Fp16(vec![0x0001, 0x3c00, 0x7bff]))
    );
    assert_eq!(doc.vector("int4"), Some(&VectorValue::Int4(vec![-8, 0, 7])));
    assert_eq!(
        doc.vector("int8"),
        Some(&VectorValue::Int8(vec![i8::MIN, 0, i8::MAX]))
    );
    assert_eq!(
        doc.vector("int16"),
        Some(&VectorValue::Int16(vec![i16::MIN, 0, i16::MAX]))
    );
    assert_eq!(
        doc.get_vector_fp16("fp16")
            .expect("typed getter must succeed"),
        Some(vec![0x0001, 0x3c00, 0x7bff])
    );
    assert_eq!(
        doc.get_vector_i4("int4")
            .expect("typed getter must succeed"),
        Some(vec![-8, 0, 7])
    );
    assert_eq!(
        doc.get_vector_binary32("binary32")
            .expect("typed getter must succeed"),
        Some(vec![0xde, 0xad, 0xbe, 0xef])
    );
    assert_eq!(
        doc.get_vector_binary64("binary64")
            .expect("typed getter must succeed"),
        Some(vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef])
    );
    assert_eq!(
        doc.vector("sparse_fp16"),
        Some(&VectorValue::SparseFp16 {
            indices: vec![0, 3],
            values: vec![0x0001, 0x7bff],
        })
    );
    assert_eq!(
        doc.get_sparse_vector_fp16("sparse_fp16")
            .expect("typed sparse getter must succeed"),
        Some((vec![0, 3], vec![0x0001, 0x7bff]))
    );
}

#[test]
fn sparse_payloads_reject_duplicates_and_invalid_fp16_bits() {
    let mut doc = Doc::with_pk("sparse").expect("primary key must be valid");
    let error = doc
        .add_sparse_vector_f32("fp32", &[1, 1], &[1.0, 2.0])
        .expect_err("duplicate sparse FP32 indices must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = doc
        .add_sparse_vector_fp16("fp16", &[1, 2], &[0x3c00, 0x7c00])
        .expect_err("non-finite sparse FP16 values must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = doc
        .add_sparse_vector_fp16("fp16", &[1, 1], &[0x3c00, 0x4000])
        .expect_err("duplicate sparse FP16 indices must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(!doc.has_field("fp32"));
    assert!(!doc.has_field("fp16"));
}

#[test]
fn native_vector_encodings_round_trip_exactly_through_storage() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &native_vector_schema(),
        None,
    )
    .expect("collection must be created");
    let doc = native_vector_doc();
    collection.insert(&[&doc]).expect("insert must succeed");
    collection.close().expect("close must checkpoint");

    let collection = Collection::open(path.to_str().expect("temporary path must be UTF-8"), None)
        .expect("collection must reopen");
    let stored = collection.fetch(&["doc-1"]).expect("document must load");
    assert_native_vector_values(&stored[0]);
}

#[test]
fn every_native_numeric_vector_type_matches_the_exact_metric_reference() {
    let temporary = tempdir().expect("temporary directory must be available");
    let vectors = [
        (
            DataType::VectorFp16,
            VectorValue::Fp16(vec![0x3c00, 0x4000]),
        ),
        (DataType::VectorFp32, VectorValue::Fp32(vec![1.0, 2.0])),
        (DataType::VectorFp64, VectorValue::Fp64(vec![1.0, 2.0])),
        (DataType::VectorInt4, VectorValue::Int4(vec![1, 2])),
        (DataType::VectorInt8, VectorValue::Int8(vec![1, 2])),
        (DataType::VectorInt16, VectorValue::Int16(vec![1, 2])),
    ];
    for (data_type, vector) in vectors {
        for (metric, expected) in [
            ("l2", -2.0_f32),
            ("ip", 4.0),
            ("cosine", 0.8),
            ("mips_l2", 4.0),
        ] {
            let schema = CollectionSchema::builder("exact-native-vector")
                .add_field(
                    FieldSchema::new("embedding", data_type, false, 2)
                        .expect("vector schema must be valid"),
                )
                .build()
                .expect("collection schema must be valid");
            let path = temporary.path().join(format!("{data_type:?}-{metric}"));
            let collection = Collection::create(
                path.to_str().expect("temporary path must be UTF-8"),
                &schema,
                None,
            )
            .expect("collection must be created");
            let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
            doc.set_vector_value("embedding", vector.clone())
                .expect("native vector must be valid");
            collection.insert(&[&doc]).expect("insert must succeed");
            let mut query =
                SearchQuery::new("embedding", &[2.0, 1.0], 1).expect("query must be valid");
            query.params.insert("metric".into(), json!(metric));
            let result = collection.query(&query).expect("query must succeed");
            assert_eq!(result.len(), 1);
            assert!(
                (result[0].get_score() - expected).abs() <= 1.0e-6,
                "data_type={data_type:?}, metric={metric}, expected={expected}, actual={}",
                result[0].get_score()
            );
        }
    }
}

#[test]
fn fp16_encoder_is_round_to_even_and_has_a_bounded_error() {
    let source = [
        0.0_f32,
        -0.0,
        1.0,
        -2.0,
        0.333_25,
        f32::from_bits(0x3300_0000),
        65_504.0,
    ];
    let encoded = VectorValue::encode_fp16(&source).expect("finite FP16 values must encode");
    let VectorValue::Fp16(bits) = &encoded else {
        panic!("FP16 encoder must return an FP16 vector");
    };
    assert_eq!(bits[0], 0x0000);
    assert_eq!(bits[1], 0x8000);
    assert_eq!(bits[2], 0x3c00);
    assert_eq!(bits[3], 0xc000);
    assert_eq!(bits[6], 0x7bff);

    let decoded = encoded
        .to_dense_f32()
        .expect("FP16 values must decode to f32");
    for (original, decoded) in source.into_iter().zip(decoded) {
        let bound = (original.abs() * 2.0_f32.powi(-10)).max(2.0_f32.powi(-24));
        assert!(
            (original - decoded).abs() <= bound,
            "original={original}, decoded={decoded}, bound={bound}"
        );
    }

    let halfway_to_odd =
        VectorValue::encode_fp16(&[1.000_488_3]).expect("halfway value must be representable");
    assert_eq!(halfway_to_odd, VectorValue::Fp16(vec![0x3c00]));
    let halfway_to_even =
        VectorValue::encode_fp16(&[1.001_464_8]).expect("halfway value must be representable");
    assert_eq!(halfway_to_even, VectorValue::Fp16(vec![0x3c02]));

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 65_505.0] {
        let error = VectorValue::encode_fp16(&[invalid])
            .expect_err("non-finite or overflowing FP16 values must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
    assert_eq!(
        VectorValue::encode_fp16(&[])
            .expect_err("an empty FP16 vector must fail")
            .code,
        ErrorCode::InvalidArgument
    );
}

#[test]
fn sparse_fp16_and_fp32_match_every_exact_metric() {
    let temporary = tempdir().expect("temporary directory must be available");
    for (data_type, vector) in [
        (
            DataType::SparseVectorFp16,
            VectorValue::SparseFp16 {
                indices: vec![0, 1],
                values: vec![0x3c00, 0x4000],
            },
        ),
        (
            DataType::SparseVectorFp32,
            VectorValue::SparseFp32 {
                indices: vec![0, 1],
                values: vec![1.0, 2.0],
            },
        ),
    ] {
        for (metric, expected) in [
            ("l2", -6.0_f32),
            ("ip", 2.0),
            ("cosine", 0.4),
            ("mips_l2", 2.0),
        ] {
            let schema = CollectionSchema::builder("exact-sparse-vector")
                .add_field(
                    FieldSchema::new("embedding", data_type, false, 3)
                        .expect("sparse schema must be valid"),
                )
                .build()
                .expect("collection schema must be valid");
            let path = temporary.path().join(format!("{data_type:?}-{metric}"));
            let collection = Collection::create(
                path.to_str().expect("temporary path must be UTF-8"),
                &schema,
                None,
            )
            .expect("collection must be created");
            let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
            doc.set_vector_value("embedding", vector.clone())
                .expect("sparse vector must be valid");
            collection.insert(&[&doc]).expect("insert must succeed");
            let mut query = SearchQuery::sparse("embedding", &[0, 2], &[2.0, 1.0], 1)
                .expect("sparse query must be valid");
            query.params.insert("metric".into(), json!(metric));
            let result = collection.query(&query).expect("query must succeed");
            assert_eq!(result.len(), 1);
            assert!(
                (result[0].get_score() - expected).abs() <= 1.0e-6,
                "data_type={data_type:?}, metric={metric}, expected={expected}, actual={}",
                result[0].get_score()
            );
        }
    }
}

#[test]
fn fp64_exact_scan_does_not_narrow_coordinates_before_scoring() {
    let schema = CollectionSchema::builder("fp64-precision")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp64, false, 2)
                .expect("FP64 schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid");
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
    let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
    doc.add_vector_f64("embedding", &[1.000_000_001, 1.0])
        .expect("FP64 vector must be valid");
    collection.insert(&[&doc]).expect("insert must succeed");

    let mut query = SearchQuery::new("embedding", &[100_000_000.0, -100_000_000.0], 1)
        .expect("query must be valid");
    query.params.insert("metric".into(), json!("ip"));
    let result = collection.query(&query).expect("query must succeed");
    assert_eq!(result.len(), 1);
    assert!((result[0].get_score() - 0.1).abs() <= 1.0e-5);
}
