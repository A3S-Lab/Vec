use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, Fts,
    FtsQueryParams, IndexParams, SearchQuery,
};
use tempfile::tempdir;

const NGRAM_CONFIG: &str = r#"{"ngram_min":2,"ngram_max":3,"token_chars":["letter","digit"]}"#;

fn ngram_params(extra_params: Option<&str>) -> IndexParams {
    IndexParams::fts(Some("ngram"), None, extra_params).expect("descriptor must be valid")
}

fn ngram_schema(name: &str) -> CollectionSchema {
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("body field must be valid");
    body.set_index_params(&ngram_params(Some(NGRAM_CONFIG)))
        .expect("n-gram FTS must be attachable");
    CollectionSchema::builder(name)
        .add_field(body)
        .build()
        .expect("n-gram schema must be valid")
}

fn document(id: &str, body: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_string("body", body).expect("body must be valid");
    doc
}

fn ranking(collection: &Collection, expression: &str) -> Vec<(String, u32)> {
    ranking_with_operator(collection, expression, None)
}

fn ranking_with_operator(
    collection: &Collection,
    expression: &str,
    operator: Option<&str>,
) -> Vec<(String, u32)> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string(expression)
        .expect("FTS expression must be valid");
    let mut query = SearchQuery::fts("body", &fts, 32).expect("FTS query must be valid");
    if let Some(operator) = operator {
        query
            .set_fts_params(
                FtsQueryParams::new(Some(operator)).expect("operator must be syntactically valid"),
            )
            .expect("FTS operator must be accepted");
    }
    collection
        .query(&query)
        .expect("n-gram query must succeed")
        .into_iter()
        .map(|doc| {
            (
                doc.get_pk()
                    .expect("result must have a primary key")
                    .to_string(),
                doc.get_score().to_bits(),
            )
        })
        .collect()
}

#[test]
fn ngram_configuration_is_validated_at_the_schema_boundary() {
    for extra_params in [None, Some(""), Some("{}"), Some(NGRAM_CONFIG)] {
        let mut field =
            FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
        field
            .set_index_params(&ngram_params(extra_params))
            .expect("supported n-gram configuration must be accepted");
    }

    for extra_params in [
        "[]",
        "not-json",
        r#"{"ngram_min":"2"}"#,
        r#"{"ngram_min":0}"#,
        r#"{"ngram_max":0}"#,
        r#"{"ngram_min":3,"ngram_max":2}"#,
        r#"{"ngram_min":1,"ngram_max":3}"#,
        r#"{"token_chars":"letter"}"#,
        r#"{"token_chars":[1]}"#,
        r#"{"token_chars":["custom"]}"#,
        r#"{"unused":true}"#,
    ] {
        let mut field =
            FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
        let error = field
            .set_index_params(&ngram_params(Some(extra_params)))
            .expect_err("invalid n-gram configuration must be rejected");
        assert_eq!(error.code, ErrorCode::InvalidArgument, "{extra_params}");
        assert!(!field.has_index());
    }

    let mut unknown =
        FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
    let error = unknown
        .set_index_params(
            &IndexParams::fts(Some("unknown"), None, None).expect("descriptor must be valid"),
        )
        .expect_err("an unknown tokenizer must be rejected");
    assert_eq!(error.code, ErrorCode::InvalidArgument);

    let mut standard =
        FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
    standard
        .set_index_params(
            &IndexParams::fts(Some("standard"), None, Some("")).expect("descriptor must be valid"),
        )
        .expect("an empty compatibility payload must be accepted");
}

#[test]
fn ngram_index_tracks_workspace_text_mutations_and_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("ngram-lifecycle");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &ngram_schema("ngram-lifecycle"),
        None,
    )
    .expect("collection must be created");
    let chinese = document("chinese", "中文分词 WorkspaceIndex42");
    let code = document("code", "fn workspace_index42() { vectorSearch(); }");
    let unrelated = document("unrelated", "spurious database tokenizer");
    collection
        .insert(&[&chinese, &code, &unrelated])
        .expect("documents must be inserted");

    assert_eq!(
        ranking(&collection, "中文")
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        ["chinese"]
    );
    let workspace = ranking(&collection, "space");
    assert_eq!(workspace.len(), 3);
    assert!(workspace.iter().any(|(id, _)| id == "chinese"));
    assert!(workspace.iter().any(|(id, _)| id == "code"));
    assert_eq!(
        ranking_with_operator(&collection, "space", Some("AND"))
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        ["chinese", "code"]
    );

    let replacement = document("code", "removed lexical content");
    collection
        .update(&[&replacement])
        .expect("indexed text must update");
    collection
        .delete(&["chinese"])
        .expect("indexed text must delete");
    let japanese = document("japanese", "高速検索エンジン workspace");
    collection
        .insert(&[&japanese])
        .expect("new indexed text must insert");
    assert_eq!(
        ranking_with_operator(&collection, "space", Some("and"))
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        ["japanese"]
    );
    let expected = ranking(&collection, "検索");
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].0, "japanese");
    collection.close().expect("collection must close");

    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only option must be valid");
    let reopened = Collection::open(
        path.to_str().expect("temporary path must be UTF-8"),
        Some(&options),
    )
    .expect("collection must reopen");
    assert!(
        reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(ranking(&reopened, "検索"), expected);
    assert_eq!(
        reopened
            .stats_snapshot()
            .expect("telemetry must be readable")
            .fts_index_query_count,
        1
    );
}
