use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, Fts,
    IndexParams, SearchQuery,
};
use tempfile::tempdir;

fn schema(name: &str, filters: Option<&[&str]>, extra_params: Option<&str>) -> CollectionSchema {
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("body field must be valid");
    body.set_index_params(
        &IndexParams::fts(Some("whitespace"), filters, extra_params)
            .expect("FTS descriptor must be valid"),
    )
    .expect("FTS configuration must be executable");
    CollectionSchema::builder(name)
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

fn document(id: &str, body: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_string("body", body).expect("body must be valid");
    doc
}

fn ids(collection: &Collection, expression: &str) -> Vec<String> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string(expression)
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("body", &fts, 32).expect("FTS query must be valid");
    collection
        .query(&query)
        .expect("FTS query must succeed")
        .into_iter()
        .map(|doc| {
            doc.get_pk()
                .expect("result must have a primary key")
                .to_string()
        })
        .collect()
}

#[test]
fn default_lowercase_filter_and_explicit_empty_pipeline_have_distinct_semantics() {
    let temporary = tempdir().expect("temporary directory must be available");
    let default = Collection::create(
        temporary
            .path()
            .join("default-lowercase")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("default-lowercase", None, None),
        None,
    )
    .expect("default-filter collection must be created");
    let case_sensitive = Collection::create(
        temporary
            .path()
            .join("case-sensitive")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("case-sensitive", Some(&[]), None),
        None,
    )
    .expect("empty-filter collection must be created");
    let doc = document("symbol", "ResolveWorkspaceIndex");
    default.insert(&[&doc]).expect("insert must succeed");
    case_sensitive.insert(&[&doc]).expect("insert must succeed");

    assert_eq!(ids(&default, "resolveworkspaceindex"), ["symbol"]);
    assert!(ids(&case_sensitive, "resolveworkspaceindex").is_empty());
    assert_eq!(ids(&case_sensitive, "ResolveWorkspaceIndex"), ["symbol"]);
}

#[test]
fn ordered_ascii_folding_and_stemming_apply_to_documents_and_queries() {
    let temporary = tempdir().expect("temporary directory must be available");
    let folded = Collection::create(
        temporary
            .path()
            .join("folded")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("folded", Some(&["lowercase", "ascii_folding"]), None),
        None,
    )
    .expect("folding collection must be created");
    let stemmed = Collection::create(
        temporary
            .path()
            .join("stemmed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema(
            "stemmed",
            Some(&["lowercase", "stemmer"]),
            Some(r#"{"stemmer_lang":"english"}"#),
        ),
        None,
    )
    .expect("stemming collection must be created");
    let accents = document("accents", "CAFÉ Straße smørrebrød");
    folded.insert(&[&accents]).expect("insert must succeed");
    let inflections = document("inflections", "running connected repositories");
    stemmed
        .insert(&[&inflections])
        .expect("insert must succeed");

    assert_eq!(ids(&folded, "cafe strasse smorrebrod"), ["accents"]);
    assert_eq!(ids(&stemmed, "runs connecting repository"), ["inflections"]);
}

#[test]
fn filter_pipeline_survives_mutation_and_cache_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("filter-reopen");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema("filter-reopen", Some(&["lowercase", "ascii_folding"]), None),
        None,
    )
    .expect("collection must be created");
    let original = document("doc", "RÉSUMÉ Parser");
    collection
        .insert(&[&original])
        .expect("insert must succeed");
    assert_eq!(ids(&collection, "resume"), ["doc"]);
    let replacement = document("doc", "CAFÉ Index");
    collection
        .update(&[&replacement])
        .expect("update must succeed");
    assert!(ids(&collection, "resume").is_empty());
    assert_eq!(ids(&collection, "cafe"), ["doc"]);
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
    assert_eq!(ids(&reopened, "CAFE"), ["doc"]);
}

#[test]
fn invalid_filter_configuration_fails_at_the_schema_boundary() {
    for (filters, extra_params) in [
        (vec!["unknown"], None),
        (vec!["stemmer"], Some(r#"{"stemmer_lang":"klingon"}"#)),
        (vec!["lowercase"], Some(r#"{"stemmer_lang":"english"}"#)),
    ] {
        let mut body =
            FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
        let error = body
            .set_index_params(
                &IndexParams::fts(Some("whitespace"), Some(&filters), extra_params)
                    .expect("descriptor construction must succeed"),
            )
            .expect_err("invalid filter configuration must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(!body.has_index());
    }
}
