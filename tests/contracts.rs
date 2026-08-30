use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, FieldValue, Fts,
    IndexParams, SearchQuery,
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
fn sparse_and_binary_query_contracts_fail_explicitly() {
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

    for data_type in [DataType::VectorBinary32, DataType::VectorBinary64] {
        let path = temporary.path().join(format!("{data_type:?}"));
        let dimension = if data_type == DataType::VectorBinary32 {
            32
        } else {
            64
        };
        let schema = CollectionSchema::builder("binary-contract")
            .add_field(
                FieldSchema::new("embedding", data_type, false, dimension)
                    .expect("binary schema must be valid"),
            )
            .build()
            .expect("collection schema must be valid");
        let collection = Collection::create(
            path.to_str().expect("temporary path must be UTF-8"),
            &schema,
            None,
        )
        .expect("collection must be created");
        let query = SearchQuery::new("embedding", &vec![0.0; dimension as usize], 10)
            .expect("binary-shaped query must be created");
        let error = collection
            .query(&query)
            .expect_err("unsupported binary scoring must fail explicitly");
        assert_eq!(error.code, ErrorCode::NotSupported);
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
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &jieba_schema(),
        None,
    )
    .expect("collection must be created");

    let error = collection
        .query(&jieba_query())
        .expect_err("disabled Jieba support must be explicit");
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
