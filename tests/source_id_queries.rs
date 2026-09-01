use a3s_vec::{Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, SearchQuery};
use serde_json::json;
use tempfile::tempdir;

fn sparse_schema(data_type: DataType) -> CollectionSchema {
    CollectionSchema::builder("sparse-source-id")
        .add_field(
            FieldSchema::new("category", DataType::String, false, 0)
                .expect("category field must be valid"),
        )
        .add_field(
            FieldSchema::new("embedding", data_type, false, 4)
                .expect("sparse vector field must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn sparse_doc(
    id: &str,
    category: &str,
    data_type: DataType,
    indices: &[u32],
    values: &[f32],
) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_string("category", category)
        .expect("category must be valid");
    match data_type {
        DataType::SparseVectorFp16 => doc
            .add_sparse_vector_fp16_f32("embedding", indices, values)
            .expect("FP16 sparse vector must be valid"),
        DataType::SparseVectorFp32 => doc
            .add_sparse_vector_f32("embedding", indices, values)
            .expect("FP32 sparse vector must be valid"),
        _ => panic!("fixture requires a sparse vector type"),
    }
    doc
}

fn queries(metric: &str) -> (SearchQuery, SearchQuery) {
    let mut explicit = SearchQuery::sparse("embedding", &[0, 2], &[1.0, 2.0], 3)
        .expect("explicit sparse query must be valid");
    explicit
        .set_filter("category == 'keep'")
        .expect("filter must be valid");
    explicit
        .set_include_vector(true)
        .expect("vector projection must be valid");
    explicit
        .set_radius(if metric == "l2" { 10.0 } else { -100.0 })
        .expect("radius must be valid");
    explicit.params.insert("metric".into(), json!(metric));

    let mut by_id =
        SearchQuery::by_id("embedding", "source", 3).expect("source-id query must be valid");
    by_id
        .set_filter("category == 'keep'")
        .expect("filter must be valid");
    by_id
        .set_include_vector(true)
        .expect("vector projection must be valid");
    by_id
        .set_radius(if metric == "l2" { 10.0 } else { -100.0 })
        .expect("radius must be valid");
    by_id.params.insert("metric".into(), json!(metric));
    (explicit, by_id)
}

fn assert_source_id_matches_explicit(collection: &Collection) {
    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let (explicit, by_id) = queries(metric);
        let expected = collection
            .query(&explicit)
            .expect("explicit sparse query must succeed");
        assert_eq!(
            expected.iter().filter_map(Doc::get_pk).collect::<Vec<_>>(),
            ["same", "partial"],
            "metric={metric}"
        );
        let actual = collection
            .query(&by_id)
            .expect("source-id sparse query must succeed");
        assert_eq!(actual, expected, "metric={metric}");
    }
}

#[test]
fn sparse_source_id_matches_explicit_fp16_and_fp32_before_and_after_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    for data_type in [DataType::SparseVectorFp16, DataType::SparseVectorFp32] {
        let path = temporary.path().join(format!("{data_type:?}"));
        let collection = Collection::create(
            path.to_str().expect("temporary path must be UTF-8"),
            &sparse_schema(data_type),
            None,
        )
        .expect("collection must be created");
        let docs = [
            sparse_doc("source", "source", data_type, &[0, 2], &[1.0, 2.0]),
            sparse_doc("same", "keep", data_type, &[0, 2], &[1.0, 2.0]),
            sparse_doc("partial", "keep", data_type, &[0, 1], &[1.0, 1.0]),
            sparse_doc("filtered", "drop", data_type, &[1, 3], &[3.0, 1.0]),
        ];
        collection
            .insert(&docs.iter().collect::<Vec<_>>())
            .expect("documents must be inserted");

        assert_source_id_matches_explicit(&collection);
        collection.flush().expect("collection must flush");
        collection.close().expect("collection must close");

        let reopened = Collection::open(path.to_str().expect("temporary path must be UTF-8"), None)
            .expect("collection must reopen");
        assert_source_id_matches_explicit(&reopened);

        #[cfg(feature = "async")]
        {
            let (explicit, by_id) = queries("cosine");
            let expected = reopened
                .query(&explicit)
                .expect("synchronous sparse query must succeed");
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("Tokio runtime must build");
            let actual = runtime
                .block_on(reopened.query_async(&by_id))
                .expect("asynchronous source-id query must succeed");
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn source_id_errors_distinguish_missing_documents_and_vectors() {
    let temporary = tempdir().expect("temporary directory must be available");
    let schema = CollectionSchema::builder("source-id-errors")
        .add_field(
            FieldSchema::new("category", DataType::String, false, 0)
                .expect("category field must be valid"),
        )
        .add_field(
            FieldSchema::new("dense", DataType::VectorFp32, true, 2)
                .expect("dense field must be valid"),
        )
        .add_field(
            FieldSchema::new("sparse", DataType::SparseVectorFp32, true, 4)
                .expect("sparse field must be valid"),
        )
        .build()
        .expect("collection schema must be valid");
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
    let mut sparse_vectorless =
        Doc::with_pk("sparse-vectorless").expect("primary key must be valid");
    sparse_vectorless
        .add_string("category", "empty")
        .expect("category must be valid");
    sparse_vectorless
        .add_vector_f32("dense", &[1.0, 0.0])
        .expect("required dense vector must be valid");
    let write = collection
        .insert(&[&sparse_vectorless])
        .expect("write operation must complete");
    assert_eq!(write.success_count, 1);
    assert_eq!(write.error_count, 0);

    for field in ["dense", "sparse"] {
        let missing =
            SearchQuery::by_id(field, "missing", 2).expect("source-id query must be constructed");
        let error = collection
            .query(&missing)
            .expect_err("a missing source document must fail");
        assert_eq!(error.code, ErrorCode::NotFound, "field={field}");
        assert!(error.message.contains("missing"));
    }

    let vectorless = SearchQuery::by_id("sparse", "sparse-vectorless", 2)
        .expect("source-id query must be constructed");
    let error = collection
        .query(&vectorless)
        .expect_err("an absent source vector must fail");
    assert_eq!(error.code, ErrorCode::FailedPrecondition);
    assert!(error.message.contains("sparse"));

    let dense = SearchQuery::by_id("dense", "sparse-vectorless", 2)
        .expect("dense source-id query must be constructed");
    let result = collection
        .query(&dense)
        .expect("the present dense source vector must be searchable");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get_pk(), Some("sparse-vectorless"));

    let mut empty_id =
        SearchQuery::by_id("sparse", "source", 2).expect("query must be constructed");
    empty_id.id = Some(String::new());
    let error = collection
        .query(&empty_id)
        .expect_err("an empty public source id must fail validation");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let error = SearchQuery::by_id("sparse", "bad\0id", 2)
        .expect_err("a source id containing NUL must fail construction");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}
