use a3s_vec::{DataType, Doc, ErrorCode, FieldValue, VectorValue};
use std::collections::BTreeMap;

#[test]
#[allow(clippy::too_many_lines)]
fn scalar_and_array_document_accessors_round_trip_and_project() {
    let mut doc = Doc::new().expect("document must be constructible");
    assert!(doc.is_empty());
    assert_eq!(doc.field_count(), 0);
    assert_eq!(doc.get_pk(), None);
    assert!(doc.get_score().abs() < f32::EPSILON);
    assert!(doc.fields().is_empty());
    assert!(doc.vectors().is_empty());

    doc.set_pk("doc-1");
    doc.set_score(1.5).expect("finite score must be accepted");
    assert!((doc.score() - 1.5).abs() < f32::EPSILON);
    doc.add_string("string", "value")
        .expect("string must be accepted");
    doc.add_bool("bool", true).expect("bool must be accepted");
    doc.add_i32("i32", -32).expect("i32 must be accepted");
    doc.add_i64("i64", -64).expect("i64 must be accepted");
    doc.add_u32("u32", 32).expect("u32 must be accepted");
    doc.add_u64("u64", 64).expect("u64 must be accepted");
    doc.add_f32("f32", 1.25).expect("f32 must be accepted");
    doc.add_f64("f64", 2.5).expect("f64 must be accepted");
    doc.add_binary("binary", &[0, 255])
        .expect("binary must be accepted");
    doc.add_array_binary("binary_array", &[vec![1], vec![2, 3]])
        .expect("binary arrays must be accepted");
    doc.add_array_string("strings", &["a", "b"])
        .expect("string arrays must be accepted");
    doc.add_array_bool("bools", &[true, false])
        .expect("bool arrays must be accepted");
    doc.add_array_i32("i32s", &[-1, 2])
        .expect("i32 arrays must be accepted");
    doc.add_array_i64("i64s", &[-3, 4])
        .expect("i64 arrays must be accepted");
    doc.add_array_u32("u32s", &[5, 6])
        .expect("u32 arrays must be accepted");
    doc.add_array_u64("u64s", &[7, 8])
        .expect("u64 arrays must be accepted");
    doc.add_array_f32("f32s", &[1.25, 2.5])
        .expect("f32 arrays must be accepted");
    doc.add_array_f64("f64s", &[3.5, 4.75])
        .expect("f64 arrays must be accepted");

    assert_eq!(doc.field_count(), 18);
    assert_eq!(
        doc.get_string("string").expect("string getter"),
        Some("value".to_string())
    );
    assert_eq!(doc.get_bool("bool").expect("bool getter"), Some(true));
    assert_eq!(doc.get_i32("i32").expect("i32 getter"), Some(-32));
    assert_eq!(doc.get_i64("i64").expect("i64 getter"), Some(-64));
    assert_eq!(doc.get_u32("u32").expect("u32 getter"), Some(32));
    assert_eq!(doc.get_u64("u64").expect("u64 getter"), Some(64));
    assert_eq!(doc.get_f32("f32").expect("f32 getter"), Some(1.25));
    assert_eq!(doc.get_f64("f64").expect("f64 getter"), Some(2.5));
    assert_eq!(
        doc.get_binary("binary").expect("binary getter"),
        Some(vec![0, 255])
    );
    assert_eq!(
        doc.get_array_bool("bools").expect("bool array getter"),
        Some(vec![true, false])
    );
    assert_eq!(
        doc.get_array_i32("i32s").expect("i32 array getter"),
        Some(vec![-1, 2])
    );
    assert_eq!(
        doc.get_array_i64("i64s").expect("i64 array getter"),
        Some(vec![-3, 4])
    );
    assert_eq!(
        doc.get_array_u32("u32s").expect("u32 array getter"),
        Some(vec![5, 6])
    );
    assert_eq!(
        doc.get_array_u64("u64s").expect("u64 array getter"),
        Some(vec![7, 8])
    );
    assert_eq!(
        doc.get_array_f32("f32s").expect("f32 array getter"),
        Some(vec![1.25, 2.5])
    );
    assert_eq!(
        doc.get_array_f64("f64s").expect("f64 array getter"),
        Some(vec![3.5, 4.75])
    );
    assert_eq!(
        doc.field("binary_array"),
        Some(&FieldValue::ArrayBinary(vec![vec![1], vec![2, 3]]))
    );
    assert_eq!(
        doc.field("strings"),
        Some(&FieldValue::ArrayString(vec!["a".into(), "b".into()]))
    );
    assert_eq!(doc.field("missing"), None);
    assert_eq!(doc.get_i32("missing").expect("missing getter"), None);
    assert_eq!(
        doc.get_i32("string")
            .expect_err("wrong getter type must fail")
            .code,
        ErrorCode::InvalidArgument
    );

    doc.set_field_null("nullable")
        .expect("null must be accepted");
    assert!(doc.is_field_null("nullable"));
    assert_eq!(doc.get_string("nullable").expect("null getter"), None);
    doc.remove_field("nullable")
        .expect("removing a field must succeed");
    assert!(!doc.has_field("nullable"));

    let output_fields = vec!["string".to_string(), "f32".to_string()];
    let projected = doc.project(Some(&output_fields), false);
    assert_eq!(projected.get_pk(), Some("doc-1"));
    assert_eq!(projected.field_count(), 2);
    assert!(projected.has_field("string"));
    assert!(projected.has_field("f32"));
    assert!(!projected.has_field("i32"));
    assert!(projected.vectors().is_empty());

    doc.set_score(f32::NAN)
        .expect_err("non-finite scores must be rejected");
    doc.clear();
    assert_eq!(doc.get_pk(), Some("doc-1"));
    assert_eq!(doc.field_count(), 0);
    assert!(!doc.is_empty(), "the primary key remains after clear");
    assert!(doc.get_score().abs() < f32::EPSILON);
}

#[test]
#[allow(clippy::too_many_lines)]
fn native_vector_accessors_and_conversions_cover_every_variant() {
    let dense_cases = [
        (
            VectorValue::Fp16(vec![0x3c00, 0x4000]),
            DataType::VectorFp16,
            2,
            vec![1.0, 2.0],
        ),
        (
            VectorValue::Fp32(vec![1.0, 2.0]),
            DataType::VectorFp32,
            2,
            vec![1.0, 2.0],
        ),
        (
            VectorValue::Fp64(vec![1.0, 2.0]),
            DataType::VectorFp64,
            2,
            vec![1.0, 2.0],
        ),
        (
            VectorValue::Int4(vec![-1, 2]),
            DataType::VectorInt4,
            2,
            vec![-1.0, 2.0],
        ),
        (
            VectorValue::Int8(vec![-3, 4]),
            DataType::VectorInt8,
            2,
            vec![-3.0, 4.0],
        ),
        (
            VectorValue::Int16(vec![-5, 6]),
            DataType::VectorInt16,
            2,
            vec![-5.0, 6.0],
        ),
    ];
    for (value, data_type, dimension, expected) in dense_cases {
        assert_eq!(value.data_type(), data_type);
        assert_eq!(value.dimension(), dimension);
        assert!(!value.is_sparse());
        assert_eq!(value.to_dense_f32(), Some(expected.clone()));
        assert_eq!(
            value.to_dense_f64(),
            Some(
                expected
                    .iter()
                    .map(|coordinate| f64::from(*coordinate))
                    .collect()
            )
        );
        assert_eq!(value.to_sparse_f64(), None);
    }

    let binary32 = VectorValue::Binary32(vec![0xaa, 0x55, 0, 0xff]);
    assert_eq!(binary32.data_type(), DataType::VectorBinary32);
    assert_eq!(binary32.dimension(), 32);
    assert!(!binary32.is_sparse());
    assert_eq!(binary32.to_dense_f32(), None);
    assert_eq!(binary32.to_dense_f64(), None);

    let binary64 = VectorValue::Binary64(vec![0xaa, 0x55, 0, 0xff, 1, 2, 3, 4]);
    assert_eq!(binary64.data_type(), DataType::VectorBinary64);
    assert_eq!(binary64.dimension(), 64);

    let sparse16 = VectorValue::SparseFp16 {
        indices: vec![1, 4],
        values: vec![0x3c00, 0x4000],
    };
    let sparse32 = VectorValue::SparseFp32 {
        indices: vec![1, 4],
        values: vec![1.0, 2.0],
    };
    for (value, data_type) in [
        (&sparse16, DataType::SparseVectorFp16),
        (&sparse32, DataType::SparseVectorFp32),
    ] {
        assert_eq!(value.data_type(), data_type);
        assert_eq!(value.dimension(), 5);
        assert!(value.is_sparse());
        assert_eq!(value.to_dense_f32(), None);
        assert_eq!(value.to_dense_f64(), None);
        let expected = BTreeMap::from([(1, 1.0), (4, 2.0)]);
        assert_eq!(value.to_sparse_f64(), Some(expected));
    }

    let mut doc = Doc::with_pk("vectors").expect("document must be valid");
    doc.add_vector_f32("f32", &[1.0, 2.0])
        .expect("f32 vector must be accepted");
    doc.add_vector_f64("f64", &[3.0, 4.0])
        .expect("f64 vector must be accepted");
    doc.add_vector_fp16("fp16", &[0x3c00, 0x4000])
        .expect("FP16 vector must be accepted");
    doc.add_vector_fp16_f32("fp16_from_f32", &[1.0, 2.0])
        .expect("FP16 conversion helper must be accepted");
    doc.add_vector_i4("i4", &[-1, 2])
        .expect("INT4 vector must be accepted");
    doc.add_vector_i8("i8", &[-3, 4])
        .expect("INT8 vector must be accepted");
    doc.add_vector_i16("i16", &[-5, 6])
        .expect("INT16 vector must be accepted");
    doc.add_vector_binary32("binary32", &[0; 4])
        .expect("Binary32 vector must be accepted");
    doc.add_vector_binary64("binary64", &[0; 8])
        .expect("Binary64 vector must be accepted");
    doc.add_sparse_vector("sparse32", &[1, 4], &[1.0, 2.0])
        .expect("sparse FP32 vector must be accepted");
    doc.add_sparse_vector_f32("sparse32_alias", &[1, 4], &[1.0, 2.0])
        .expect("sparse FP32 alias must be accepted");
    doc.add_sparse_vector_fp16("sparse16", &[1, 4], &[0x3c00, 0x4000])
        .expect("sparse FP16 vector must be accepted");
    doc.add_sparse_vector_fp16_f32("sparse16_from_f32", &[1, 4], &[1.0, 2.0])
        .expect("sparse FP16 conversion helper must be accepted");

    assert_eq!(
        doc.get_vector_f32("f32").expect("f32 getter"),
        Some(vec![1.0, 2.0])
    );
    assert_eq!(
        doc.get_vector_f64("f64").expect("f64 getter"),
        Some(vec![3.0, 4.0])
    );
    assert_eq!(
        doc.get_vector_fp16("fp16").expect("FP16 getter"),
        Some(vec![0x3c00, 0x4000])
    );
    assert_eq!(
        doc.get_vector_i4("i4").expect("INT4 getter"),
        Some(vec![-1, 2])
    );
    assert_eq!(
        doc.get_vector_i8("i8").expect("INT8 getter"),
        Some(vec![-3, 4])
    );
    assert_eq!(
        doc.get_vector_i16("i16").expect("INT16 getter"),
        Some(vec![-5, 6])
    );
    assert_eq!(
        doc.get_vector_binary32("binary32")
            .expect("Binary32 getter"),
        Some(vec![0; 4])
    );
    assert_eq!(
        doc.get_vector_binary64("binary64")
            .expect("Binary64 getter"),
        Some(vec![0; 8])
    );
    assert_eq!(
        doc.get_sparse_vector_f32("sparse32")
            .expect("sparse FP32 getter"),
        Some((vec![1, 4], vec![1.0, 2.0]))
    );
    assert_eq!(
        doc.get_sparse_vector_fp16("sparse16")
            .expect("sparse FP16 getter"),
        Some((vec![1, 4], vec![0x3c00, 0x4000]))
    );
    assert_eq!(
        doc.get_vector_f32("missing")
            .expect("missing vector getter"),
        None
    );
    assert_eq!(
        doc.get_vector_f32("f64")
            .expect_err("wrong vector getter type must fail")
            .code,
        ErrorCode::InvalidArgument
    );

    let projected = doc.project(Some(&["f32".to_string(), "binary32".to_string()]), true);
    assert_eq!(projected.vectors().len(), 2);
    assert!(projected.vector("f32").is_some());
    assert!(projected.vector("binary32").is_some());
    assert!(projected.vector("f64").is_none());
}
