use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, FieldValue, Fts,
    IndexParams, SearchQuery, VectorSchema,
};
use serde_json::{json, Value};
use tempfile::tempdir;

fn contract_schema() -> CollectionSchema {
    CollectionSchema::builder("contracts")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("title schema must be valid"),
        )
        .add_field(
            FieldSchema::new("age", DataType::Int32, false, 0).expect("age schema must be valid"),
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
    doc.add_string("title", "reference")
        .expect("title must be valid");
    doc.add_i32("age", 42).expect("age must be valid");
    doc.add_vector_f32("embedding", &[1.0, 0.0, 0.0])
        .expect("embedding must be valid");
    doc
}

#[test]
fn dense_query_dimension_is_rejected_for_every_metric_and_numeric_vector_type() {
    let temporary = tempdir().expect("temporary directory must be available");
    for data_type in [
        DataType::VectorFp16,
        DataType::VectorFp32,
        DataType::VectorFp64,
        DataType::VectorInt4,
        DataType::VectorInt8,
        DataType::VectorInt16,
    ] {
        let path = temporary.path().join(format!("{data_type:?}"));
        let schema = CollectionSchema::builder("dimension-contract")
            .add_field(
                FieldSchema::new("embedding", data_type, false, 3)
                    .expect("vector schema must be valid"),
            )
            .build()
            .expect("collection schema must be valid");
        let collection = Collection::create(
            path.to_str().expect("temporary path must be UTF-8"),
            &schema,
            None,
        )
        .expect("collection must be created");

        for metric in ["l2", "ip", "cosine", "mips_l2"] {
            let mut query =
                SearchQuery::new("embedding", &[1.0, 0.0], 10).expect("query must be created");
            query.params.insert("metric".into(), json!(metric));

            let error = collection
                .query(&query)
                .expect_err("dimension mismatch must be a typed query error");
            assert_eq!(
                error.code,
                ErrorCode::InvalidArgument,
                "data_type={data_type:?}, metric={metric}"
            );
            assert!(error.message.contains("expected 3, got 2"));
        }
    }
}

#[test]
fn externally_mutated_schema_descriptors_are_revalidated_before_execution() {
    let mut malformed_scalar = FieldSchema::new("scalar", DataType::String, false, 0)
        .expect("initial scalar descriptor must be valid");
    malformed_scalar.dimension = 1;
    let mut schema = CollectionSchema::new("shape-contract").expect("schema must be valid");
    let error = schema
        .add_field(&malformed_scalar)
        .expect_err("a scalar dimension changed through a public field must be rejected");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let mut malformed_dense = FieldSchema::new("dense", DataType::VectorFp32, false, 3)
        .expect("initial dense descriptor must be valid");
    malformed_dense.dimension = 0;
    let error = CollectionSchema {
        name: "shape-contract".to_string(),
        fields: Vec::new(),
        vectors: vec![VectorSchema {
            name: malformed_dense.name.clone(),
            data_type: malformed_dense.data_type,
            dimension: malformed_dense.dimension,
            index_params: None,
        }],
        max_doc_count_per_segment: 0,
    }
    .validate()
    .expect_err("a zero-dimensional dense vector must be rejected on validation");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let mut malformed_binary = VectorSchema::new("bits", DataType::VectorBinary32, 32)
        .expect("initial binary descriptor must be valid");
    malformed_binary.dimension = 40;
    let error = CollectionSchema::new("shape-contract")
        .expect("schema must be valid")
        .add_vector_field(&malformed_binary)
        .expect_err("binary dimensions must remain chunk-aligned after mutation");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let malformed_layout = CollectionSchema {
        name: "shape-contract".to_string(),
        fields: vec![
            FieldSchema::new("wrong-list", DataType::VectorFp32, false, 3)
                .expect("vector descriptor must be valid"),
        ],
        vectors: Vec::new(),
        max_doc_count_per_segment: 0,
    };
    let error = malformed_layout
        .validate()
        .expect_err("vectors must not be smuggled into the scalar field list");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let malformed_name = CollectionSchema {
        name: String::new(),
        fields: vec![FieldSchema::new("value", DataType::String, false, 0)
            .expect("field descriptor must be valid")],
        vectors: Vec::new(),
        max_doc_count_per_segment: 0,
    };
    let error = malformed_name
        .validate()
        .expect_err("a directly mutated collection name must be rejected");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

#[test]
fn mutation_and_schema_rename_boundaries_reject_invalid_names() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("name-boundaries");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &contract_schema(),
        None,
    )
    .expect("collection must be created");
    let doc = valid_doc("doc-1");
    collection
        .insert(&[&doc])
        .expect("document must be inserted");

    let result = collection
        .delete(&["doc-1\0suffix"])
        .expect("invalid delete names must return per-document results");
    assert_eq!(result.success_count, 0);
    assert_eq!(result.error_count, 1);
    assert_eq!(result.results[0].code, ErrorCode::InvalidArgument);
    assert_eq!(collection.count().expect("count must succeed"), 1);

    let error = collection
        .rename_column("\0old", "new")
        .expect_err("invalid old field names must fail before mutation");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = collection
        .rename_column("title", "")
        .expect_err("invalid new field names must fail before mutation");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    let error = collection
        .rename_column("", "")
        .expect_err("invalid no-op field names must not be silently accepted");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(collection
        .schema()
        .expect("schema must remain readable")
        .has_field("title"));
}

#[test]
fn sparse_query_contracts_fail_explicitly() {
    let temporary = tempdir().expect("temporary directory must be available");
    for data_type in [DataType::SparseVectorFp16, DataType::SparseVectorFp32] {
        let path = temporary.path().join(format!("{data_type:?}"));
        let schema = CollectionSchema::builder("sparse-contract")
            .add_field(
                FieldSchema::new("embedding", data_type, false, 3)
                    .expect("sparse schema must be valid"),
            )
            .build()
            .expect("collection schema must be valid");
        let collection = Collection::create(
            path.to_str().expect("temporary path must be UTF-8"),
            &schema,
            None,
        )
        .expect("collection must be created");

        let dense = SearchQuery::new("embedding", &[1.0, 0.0, 0.0], 10)
            .expect("dense query must be created");
        let error = collection
            .query(&dense)
            .expect_err("dense payload on a sparse field must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument);

        let sparse = SearchQuery::sparse("embedding", &[0, 3], &[1.0, 2.0], 10)
            .expect("sparse query must be created");
        let error = collection
            .query(&sparse)
            .expect_err("out-of-range sparse index must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
}

#[test]
fn query_route_must_match_the_schema_field_type() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &contract_schema(),
        None,
    )
    .expect("collection must be created");

    let vector_query = SearchQuery::new("age", &[1.0], 10).expect("query must be created");
    let error = collection
        .query(&vector_query)
        .expect_err("a scalar field cannot execute a vector query");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("forty two")
        .expect("FTS expression must be valid");
    let fts_query = SearchQuery::fts("age", &fts, 10).expect("query must be created");
    let error = collection
        .query(&fts_query)
        .expect_err("a non-string field cannot execute an FTS query");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

#[test]
fn json_adapter_values_are_typed_or_rejected_at_the_write_boundary() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &contract_schema(),
        None,
    )
    .expect("collection must be created");

    let mut invalid = valid_doc("invalid");
    invalid
        .set_field_value("age", FieldValue::Json(json!("not an integer")))
        .expect("adapter JSON must be accepted by the document container");
    let mut invalid_null = valid_doc("invalid-null");
    invalid_null
        .set_field_value("age", FieldValue::Json(json!(null)))
        .expect("adapter JSON null must be accepted by the document container");
    let mut compatible = valid_doc("compatible");
    compatible
        .set_field_value("age", FieldValue::Json(json!(43)))
        .expect("adapter JSON must be accepted by the document container");

    let result = collection
        .insert(&[&invalid, &invalid_null, &compatible])
        .expect("batch result must be returned");
    assert_eq!(result.success_count, 1);
    assert_eq!(result.error_count, 2);
    assert_eq!(result.results[0].code, ErrorCode::InvalidArgument);
    assert_eq!(result.results[1].code, ErrorCode::InvalidArgument);

    let stored = collection
        .fetch(&["compatible"])
        .expect("compatible document must be readable");
    assert_eq!(stored.len(), 1);
    assert_eq!(
        stored[0]
            .get_i32("age")
            .expect("JSON adapter value must be canonicalized"),
        Some(43)
    );
}

type JsonAdapterCase = (&'static str, DataType, Value, FieldValue);

fn json_adapter_cases() -> Vec<JsonAdapterCase> {
    vec![
        (
            "string",
            DataType::String,
            json!("value"),
            FieldValue::String("value".to_string()),
        ),
        ("bool", DataType::Bool, json!(true), FieldValue::Bool(true)),
        ("i32", DataType::Int32, json!(-32), FieldValue::Int32(-32)),
        ("i64", DataType::Int64, json!(-64), FieldValue::Int64(-64)),
        ("u32", DataType::Uint32, json!(32), FieldValue::Uint32(32)),
        ("u64", DataType::Uint64, json!(64), FieldValue::Uint64(64)),
        ("f32", DataType::Float, json!(1.25), FieldValue::Float(1.25)),
        ("f64", DataType::Double, json!(2.5), FieldValue::Double(2.5)),
        (
            "strings",
            DataType::ArrayString,
            json!(["a", "b"]),
            FieldValue::ArrayString(vec!["a".to_string(), "b".to_string()]),
        ),
        (
            "bools",
            DataType::ArrayBool,
            json!([true, false]),
            FieldValue::ArrayBool(vec![true, false]),
        ),
        (
            "i32s",
            DataType::ArrayInt32,
            json!([-1, 2]),
            FieldValue::ArrayInt32(vec![-1, 2]),
        ),
        (
            "i64s",
            DataType::ArrayInt64,
            json!([-3, 4]),
            FieldValue::ArrayInt64(vec![-3, 4]),
        ),
        (
            "u32s",
            DataType::ArrayUint32,
            json!([5, 6]),
            FieldValue::ArrayUint32(vec![5, 6]),
        ),
        (
            "u64s",
            DataType::ArrayUint64,
            json!([7, 8]),
            FieldValue::ArrayUint64(vec![7, 8]),
        ),
        (
            "f32s",
            DataType::ArrayFloat,
            json!([1.25, 2.5]),
            FieldValue::ArrayFloat(vec![1.25, 2.5]),
        ),
        (
            "f64s",
            DataType::ArrayDouble,
            json!([3.5, 4.75]),
            FieldValue::ArrayDouble(vec![3.5, 4.75]),
        ),
    ]
}

type TypedFieldCase = (&'static str, DataType, FieldValue);

fn typed_scalar_and_array_cases() -> Vec<TypedFieldCase> {
    let mut cases: Vec<_> = json_adapter_cases()
        .into_iter()
        .map(|(name, data_type, _, value)| (name, data_type, value))
        .collect();
    cases.extend([
        (
            "binary",
            DataType::Binary,
            FieldValue::Binary(vec![0x00, 0xff]),
        ),
        (
            "binary-array",
            DataType::ArrayBinary,
            FieldValue::ArrayBinary(vec![vec![], vec![0x00, 0xff]]),
        ),
    ]);
    cases
}

#[test]
fn nullability_is_enforced_for_every_scalar_and_array_type() {
    let temporary = tempdir().expect("temporary directory must be available");

    for (name, data_type, typed_value) in typed_scalar_and_array_cases() {
        let nullable_path = temporary.path().join(format!("{name}-nullable"));
        let nullable_schema = CollectionSchema::builder("nullable-contract")
            .add_field(
                FieldSchema::new("value", data_type, true, 0)
                    .expect("nullable field schema must be valid"),
            )
            .build()
            .expect("nullable collection schema must be valid");
        let nullable = Collection::create(
            nullable_path
                .to_str()
                .expect("temporary path must be UTF-8"),
            &nullable_schema,
            None,
        )
        .expect("nullable collection must be created");

        let missing = Doc::with_pk("missing").expect("primary key must be valid");
        let mut typed_null = Doc::with_pk("typed-null").expect("primary key must be valid");
        typed_null
            .set_field_null("value")
            .expect("typed null must fit the document container");
        let mut json_null = Doc::with_pk("json-null").expect("primary key must be valid");
        json_null
            .set_field_value("value", FieldValue::Json(Value::Null))
            .expect("JSON null must fit the document container");
        let mut populated = Doc::with_pk("populated").expect("primary key must be valid");
        populated
            .set_field_value("value", typed_value.clone())
            .expect("typed value must fit the document container");

        let result = nullable
            .insert(&[&missing, &typed_null, &json_null, &populated])
            .expect("nullable batch must return per-document results");
        assert_eq!(result.success_count, 4, "data_type={data_type:?}");

        let mut null_patch = Doc::with_pk("populated").expect("primary key must be valid");
        null_patch
            .set_field_value("value", FieldValue::Json(Value::Null))
            .expect("JSON null patch must fit the document container");
        let result = nullable
            .update(&[&null_patch])
            .expect("nullable update must return a per-document result");
        assert_eq!(result.success_count, 1, "data_type={data_type:?}");
        drop(nullable);

        let nullable = Collection::open(
            nullable_path
                .to_str()
                .expect("temporary path must be UTF-8"),
            None,
        )
        .expect("nullable collection must reopen");
        let stored = nullable
            .fetch(&["missing", "typed-null", "json-null", "populated"])
            .expect("nullable documents must be readable");
        let find = |id: &str| {
            stored
                .iter()
                .find(|doc| doc.get_pk() == Some(id))
                .expect("requested document must exist")
        };
        assert!(
            !find("missing").has_field("value"),
            "data_type={data_type:?}"
        );
        for id in ["typed-null", "json-null", "populated"] {
            assert!(
                find(id).is_field_null("value"),
                "data_type={data_type:?}, id={id}"
            );
            assert_eq!(find(id).field("value"), Some(&FieldValue::Null));
        }

        let required_path = temporary.path().join(format!("{name}-required"));
        let required_schema = CollectionSchema::builder("required-contract")
            .add_field(
                FieldSchema::new("value", data_type, false, 0)
                    .expect("required field schema must be valid"),
            )
            .build()
            .expect("required collection schema must be valid");
        let required = Collection::create(
            required_path
                .to_str()
                .expect("temporary path must be UTF-8"),
            &required_schema,
            None,
        )
        .expect("required collection must be created");
        let mut valid = Doc::with_pk("valid").expect("primary key must be valid");
        valid
            .set_field_value("value", typed_value)
            .expect("typed value must fit the document container");
        let result = required
            .insert(&[&missing, &typed_null, &json_null, &valid])
            .expect("required batch must return per-document results");
        assert_eq!(result.success_count, 1, "data_type={data_type:?}");
        assert_eq!(result.error_count, 3, "data_type={data_type:?}");
        for outcome in &result.results[..3] {
            assert_eq!(outcome.code, ErrorCode::InvalidArgument);
        }
    }
}

#[test]
fn json_adapter_canonicalizes_every_supported_scalar_and_array_type() {
    let cases = json_adapter_cases();
    let mut schema = CollectionSchema::new("json-contract").expect("schema name must be valid");
    for (name, data_type, _, _) in &cases {
        schema
            .add_field(
                &FieldSchema::new(name, *data_type, true, 0)
                    .expect("JSON-adapted field schema must be valid"),
            )
            .expect("field names must be unique");
    }
    for (name, data_type) in [
        ("binary", DataType::Binary),
        ("binary_array", DataType::ArrayBinary),
    ] {
        schema
            .add_field(
                &FieldSchema::new(name, data_type, true, 0)
                    .expect("binary field schema must be valid"),
            )
            .expect("field names must be unique");
    }

    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("collection must be created");
    let mut compatible = Doc::with_pk("compatible").expect("primary key must be valid");
    for (name, _, value, _) in &cases {
        compatible
            .set_field_value(name, FieldValue::Json(value.clone()))
            .expect("JSON adapter value must fit the document container");
    }
    let result = collection
        .insert(&[&compatible])
        .expect("compatible JSON values must be accepted");
    assert_eq!(result.success_count, 1);
    let stored = collection
        .fetch(&["compatible"])
        .expect("canonicalized document must be readable");
    for (name, _, _, expected) in &cases {
        assert_eq!(stored[0].field(name), Some(expected), "field={name}");
    }

    for (id, field, value) in [
        ("binary", "binary", json!("AQI=")),
        ("binary-array", "binary_array", json!(["AQI="])),
        ("i32-overflow", "i32", json!(i64::from(i32::MAX) + 1)),
        ("u32-negative", "u32", json!(-1)),
    ] {
        let mut incompatible = Doc::with_pk(id).expect("primary key must be valid");
        incompatible
            .set_field_value(field, FieldValue::Json(value))
            .expect("JSON adapter value must fit the document container");
        let result = collection
            .insert(&[&incompatible])
            .expect("binary adapter rejection must be a per-document result");
        assert_eq!(result.success_count, 0);
        assert_eq!(result.results[0].code, ErrorCode::InvalidArgument);
    }
}

fn json_number(literal: &str) -> Value {
    serde_json::from_str(literal).expect("boundary JSON number must parse")
}

type AcceptedNumericCase = (&'static str, &'static str, Value, FieldValue);
type RejectedNumericCase = (&'static str, &'static str, Value);

fn numeric_boundary_schema() -> CollectionSchema {
    let mut schema = CollectionSchema::new("numeric-boundaries").expect("schema must be valid");
    for (name, data_type) in [
        ("i32", DataType::Int32),
        ("i64", DataType::Int64),
        ("u32", DataType::Uint32),
        ("u64", DataType::Uint64),
        ("f32", DataType::Float),
        ("f64", DataType::Double),
        ("i32s", DataType::ArrayInt32),
        ("i64s", DataType::ArrayInt64),
        ("u32s", DataType::ArrayUint32),
        ("u64s", DataType::ArrayUint64),
        ("f32s", DataType::ArrayFloat),
        ("f64s", DataType::ArrayDouble),
    ] {
        schema
            .add_field(
                &FieldSchema::new(name, data_type, true, 0)
                    .expect("numeric field schema must be valid"),
            )
            .expect("numeric field names must be unique");
    }
    schema
}

fn accepted_numeric_scalar_boundaries() -> Vec<AcceptedNumericCase> {
    vec![
        (
            "i32-min",
            "i32",
            json!(i32::MIN),
            FieldValue::Int32(i32::MIN),
        ),
        (
            "i32-max",
            "i32",
            json!(i32::MAX),
            FieldValue::Int32(i32::MAX),
        ),
        (
            "i64-min",
            "i64",
            json!(i64::MIN),
            FieldValue::Int64(i64::MIN),
        ),
        (
            "i64-max",
            "i64",
            json!(i64::MAX),
            FieldValue::Int64(i64::MAX),
        ),
        ("u32-min", "u32", json!(0), FieldValue::Uint32(0)),
        (
            "u32-max",
            "u32",
            json!(u32::MAX),
            FieldValue::Uint32(u32::MAX),
        ),
        ("u64-min", "u64", json!(0), FieldValue::Uint64(0)),
        (
            "u64-max",
            "u64",
            json!(u64::MAX),
            FieldValue::Uint64(u64::MAX),
        ),
        (
            "f32-min",
            "f32",
            json!(f32::MIN),
            FieldValue::Float(f32::MIN),
        ),
        (
            "f32-max",
            "f32",
            json!(f32::MAX),
            FieldValue::Float(f32::MAX),
        ),
        (
            "f64-min",
            "f64",
            json!(f64::MIN),
            FieldValue::Double(f64::MIN),
        ),
        (
            "f64-max",
            "f64",
            json!(f64::MAX),
            FieldValue::Double(f64::MAX),
        ),
    ]
}

fn accepted_numeric_array_boundaries() -> Vec<AcceptedNumericCase> {
    vec![
        (
            "i32-array-extrema",
            "i32s",
            json!([i32::MIN, i32::MAX]),
            FieldValue::ArrayInt32(vec![i32::MIN, i32::MAX]),
        ),
        (
            "i64-array-extrema",
            "i64s",
            json!([i64::MIN, i64::MAX]),
            FieldValue::ArrayInt64(vec![i64::MIN, i64::MAX]),
        ),
        (
            "u32-array-extrema",
            "u32s",
            json!([0, u32::MAX]),
            FieldValue::ArrayUint32(vec![0, u32::MAX]),
        ),
        (
            "u64-array-extrema",
            "u64s",
            json!([0, u64::MAX]),
            FieldValue::ArrayUint64(vec![0, u64::MAX]),
        ),
        (
            "f32-array-extrema",
            "f32s",
            json!([f32::MIN, f32::MAX]),
            FieldValue::ArrayFloat(vec![f32::MIN, f32::MAX]),
        ),
        (
            "f64-array-extrema",
            "f64s",
            json!([f64::MIN, f64::MAX]),
            FieldValue::ArrayDouble(vec![f64::MIN, f64::MAX]),
        ),
    ]
}

fn rejected_numeric_scalar_boundaries() -> Vec<RejectedNumericCase> {
    vec![
        ("i32-below", "i32", json!(i64::from(i32::MIN) - 1)),
        ("i32-above", "i32", json!(i64::from(i32::MAX) + 1)),
        ("i32-fraction", "i32", json!(1.5)),
        ("i64-below", "i64", json_number("-9223372036854775809")),
        ("i64-above", "i64", json!(i64::MAX as u64 + 1)),
        ("i64-fraction", "i64", json!(1.5)),
        ("u32-negative", "u32", json!(-1)),
        ("u32-above", "u32", json!(u64::from(u32::MAX) + 1)),
        ("u32-fraction", "u32", json!(1.5)),
        ("u64-negative", "u64", json!(-1)),
        ("u64-above", "u64", json_number("18446744073709551616")),
        ("u64-fraction", "u64", json!(1.5)),
        ("f32-below", "f32", json!(-f64::from(f32::MAX) * 2.0)),
        ("f32-above", "f32", json!(f64::from(f32::MAX) * 2.0)),
        ("f64-wrong-type", "f64", json!("1.0")),
    ]
}

fn rejected_numeric_array_boundaries() -> Vec<RejectedNumericCase> {
    vec![
        (
            "i32-array-overflow",
            "i32s",
            json!([0, i64::from(i32::MAX) + 1]),
        ),
        (
            "i64-array-overflow",
            "i64s",
            Value::Array(vec![json!(0), json_number("9223372036854775808")]),
        ),
        ("u32-array-negative", "u32s", json!([0, -1])),
        (
            "u64-array-overflow",
            "u64s",
            Value::Array(vec![json!(0), json_number("18446744073709551616")]),
        ),
        (
            "f32-array-overflow",
            "f32s",
            json!([0.0, f64::from(f32::MAX) * 2.0]),
        ),
        ("f64-array-wrong-type", "f64s", json!([0.0, "1.0"])),
    ]
}

#[test]
fn json_numeric_boundaries_are_canonicalized_without_wrapping() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &numeric_boundary_schema(),
        None,
    )
    .expect("collection must be created");

    for (id, field, value, expected) in accepted_numeric_scalar_boundaries()
        .into_iter()
        .chain(accepted_numeric_array_boundaries())
    {
        let mut doc = Doc::with_pk(id).expect("primary key must be valid");
        doc.set_field_value(field, FieldValue::Json(value))
            .expect("JSON boundary value must fit the document container");
        let result = collection
            .insert(&[&doc])
            .expect("accepted boundary must return a per-document result");
        assert_eq!(result.success_count, 1, "id={id}");
        let stored = collection
            .fetch(&[id])
            .expect("accepted boundary document must be readable");
        assert_eq!(stored[0].field(field), Some(&expected), "id={id}");
    }

    for (id, field, value) in rejected_numeric_scalar_boundaries()
        .into_iter()
        .chain(rejected_numeric_array_boundaries())
    {
        let mut doc = Doc::with_pk(id).expect("primary key must be valid");
        doc.set_field_value(field, FieldValue::Json(value))
            .expect("JSON boundary value must fit the document container");
        let result = collection
            .insert(&[&doc])
            .expect("rejected boundary must return a per-document result");
        assert_eq!(result.success_count, 0, "id={id}");
        assert_eq!(
            result.results[0].code,
            ErrorCode::InvalidArgument,
            "id={id}"
        );
    }
}

#[test]
fn typed_floating_point_fields_reject_non_finite_scalars_and_array_members() {
    for value in [
        FieldValue::Float(f32::NAN),
        FieldValue::Float(f32::INFINITY),
        FieldValue::Float(f32::NEG_INFINITY),
        FieldValue::Double(f64::NAN),
        FieldValue::Double(f64::INFINITY),
        FieldValue::Double(f64::NEG_INFINITY),
        FieldValue::ArrayFloat(vec![0.0, f32::NAN]),
        FieldValue::ArrayFloat(vec![0.0, f32::INFINITY]),
        FieldValue::ArrayDouble(vec![0.0, f64::NAN]),
        FieldValue::ArrayDouble(vec![0.0, f64::NEG_INFINITY]),
    ] {
        let mut doc = Doc::with_pk("non-finite").expect("primary key must be valid");
        let error = doc
            .set_field_value("value", value)
            .expect_err("non-finite field values must fail at the document boundary");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
}

#[test]
fn replacement_upserts_cannot_remove_required_fields() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &contract_schema(),
        None,
    )
    .expect("collection must be created");
    let original = valid_doc("doc-1");
    collection
        .insert(&[&original])
        .expect("original document must be inserted");

    let mut incomplete = Doc::with_pk("doc-1").expect("primary key must be valid");
    incomplete
        .add_i32("age", 43)
        .expect("partial replacement field must be valid");
    let result = collection
        .upsert(&[&incomplete])
        .expect("invalid replacement must be a per-document result");
    assert_eq!(result.success_count, 0);
    assert_eq!(result.results[0].code, ErrorCode::InvalidArgument);

    let stored = collection
        .fetch(&["doc-1"])
        .expect("original document must remain readable");
    assert_eq!(stored, vec![original]);
}

#[test]
fn schema_backfill_defaults_are_canonicalized_and_required() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let path = path.to_str().expect("temporary path must be UTF-8");
    let collection =
        Collection::create(path, &contract_schema(), None).expect("collection must be created");
    let doc = valid_doc("doc-1");
    collection.insert(&[&doc]).expect("insert must succeed");

    let rank =
        FieldSchema::new("rank", DataType::Int32, false, 0).expect("rank schema must be valid");
    collection
        .add_column(&rank, Some("7"))
        .expect("typed numeric backfill must succeed");
    let missing = FieldSchema::new("missing", DataType::String, false, 0)
        .expect("missing schema must be valid");
    let error = collection
        .add_column(&missing, None)
        .expect_err("non-null column on existing documents requires a default");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    drop(collection);

    let collection = Collection::open(path, None).expect("collection must reopen");
    let stored = collection.fetch(&["doc-1"]).expect("document must load");
    assert_eq!(
        stored[0]
            .get_i32("rank")
            .expect("backfill must use the declared field type"),
        Some(7)
    );
    assert!(!collection
        .schema()
        .expect("schema must load")
        .has_field("missing"));
}

fn jieba_schema() -> CollectionSchema {
    let mut title =
        FieldSchema::new("title", DataType::String, false, 0).expect("title schema must be valid");
    title
        .set_index_params(
            &IndexParams::fts(Some("jieba"), None, None)
                .expect("Jieba index parameters must be valid"),
        )
        .expect("Jieba parameters must fit a string field");
    CollectionSchema::builder("jieba-contract")
        .add_field(title)
        .build()
        .expect("Jieba collection schema must be valid")
}

#[cfg(feature = "jieba")]
fn jieba_query() -> SearchQuery {
    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("北京大学")
        .expect("FTS expression must be valid");
    SearchQuery::fts("title", &fts, 10).expect("FTS query must be valid")
}

#[cfg(not(feature = "jieba"))]
#[test]
fn disabled_jieba_feature_fails_instead_of_changing_tokenizer_semantics() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let error = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &jieba_schema(),
        None,
    )
    .expect_err("disabled Jieba index construction must fail explicitly");
    assert_eq!(error.code, ErrorCode::NotSupported);
}

#[cfg(feature = "jieba")]
#[test]
fn enabled_jieba_feature_executes_the_requested_tokenizer() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &jieba_schema(),
        None,
    )
    .expect("collection must be created");
    let mut doc = Doc::with_pk("doc-1").expect("primary key must be valid");
    doc.add_string("title", "北京大学")
        .expect("title must be valid");
    collection.insert(&[&doc]).expect("insert must succeed");

    let result = collection
        .query(&jieba_query())
        .expect("enabled Jieba query must execute");
    assert_eq!(result.len(), 1);
}
