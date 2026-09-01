use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, Fts,
    FtsQueryParams, IndexParams, SearchQuery,
};
use tempfile::tempdir;

fn schema(name: &str, indexed: bool) -> CollectionSchema {
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("body field must be valid");
    if indexed {
        body.set_index_params(
            &IndexParams::fts(Some("standard"), None, None).expect("FTS descriptor must be valid"),
        )
        .expect("FTS index must be valid");
    }
    CollectionSchema::builder(name)
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

fn docs() -> Vec<Doc> {
    [
        ("a", "Rust vector database engine"),
        ("b", "Rust database vector engine"),
        ("c", "Python vector database"),
        ("d", "Rust vector search database"),
        ("e", "legacy Rust index"),
    ]
    .into_iter()
    .map(|(id, body)| {
        let mut doc = Doc::with_pk(id).expect("primary key must be valid");
        doc.add_string("body", body).expect("body must be valid");
        doc
    })
    .collect()
}

fn insert_fixture(collection: &Collection) {
    let docs = docs();
    let refs: Vec<_> = docs.iter().collect();
    collection.insert(&refs).expect("insert must succeed");
}

fn query(collection: &Collection, expression: &str, operator: Option<&str>) -> Vec<Doc> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_query_string(expression)
        .expect("query string must be valid");
    let mut query = SearchQuery::fts("body", &fts, 32).expect("FTS query must be valid");
    if let Some(operator) = operator {
        query
            .set_fts_params(FtsQueryParams::new(Some(operator)).expect("operator must be valid"))
            .expect("operator must be accepted");
    }
    collection
        .query(&query)
        .expect("structured FTS query must succeed")
}

fn ids(docs: &[Doc]) -> Vec<&str> {
    docs.iter().filter_map(Doc::get_pk).collect()
}

fn comparable(docs: &[Doc]) -> Vec<(&str, u32)> {
    docs.iter()
        .map(|doc| (doc.get_pk().unwrap_or_default(), doc.get_score().to_bits()))
        .collect()
}

fn generated_docs() -> Vec<Doc> {
    (0..256)
        .map(|index| {
            let mut terms = vec!["workspace"];
            if index % 2 == 0 {
                terms.extend(["alpha", "beta"]);
            } else {
                terms.extend(["beta", "alpha"]);
            }
            if index % 3 == 0 {
                terms.push("gamma");
            }
            if index % 5 == 0 {
                terms.extend(["gamma", "delta"]);
            }
            if index % 7 == 0 {
                terms.extend(["alpha", "alpha"]);
            }
            let mut doc =
                Doc::with_pk(format!("generated-{index:04}")).expect("primary key must be valid");
            doc.add_string("body", &terms.join(" "))
                .expect("body must be valid");
            doc
        })
        .collect()
}

#[test]
fn indexed_boolean_phrase_queries_match_the_scan_oracle_exactly() {
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed = Collection::create(
        temporary
            .path()
            .join("indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("indexed", true),
        None,
    )
    .expect("indexed collection must be created");
    let scan = Collection::create(
        temporary
            .path()
            .join("scan")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("scan", false),
        None,
    )
    .expect("scan collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&scan);

    for (expression, operator, expected) in [
        ("rust AND \"vector database\"", None, vec!["a"]),
        (
            "(rust OR python) AND \"vector database\"",
            None,
            vec!["c", "a"],
        ),
        ("rust NOT legacy", None, vec!["a", "b", "d"]),
        ("+rust database", None, vec!["a", "b", "d", "e"]),
        (
            "rust OR python AND database",
            None,
            vec!["c", "e", "a", "b", "d"],
        ),
        ("rust OR python", Some("AND"), vec!["c", "e", "a", "b", "d"]),
        ("rust database", Some("AND"), vec!["a", "b", "d"]),
    ] {
        let indexed_result = query(&indexed, expression, operator);
        let scan_result = query(&scan, expression, operator);
        assert_eq!(ids(&indexed_result), expected, "{expression}");
        assert_eq!(comparable(&indexed_result), comparable(&scan_result));
    }
}

#[test]
fn structured_phrase_queries_track_mutation_and_cache_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("phrase-reopen");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema("phrase-reopen", true),
        None,
    )
    .expect("collection must be created");
    insert_fixture(&collection);
    assert_eq!(
        ids(&query(&collection, "\"vector database\"", None)),
        ["c", "a"]
    );

    let mut replacement = Doc::with_pk("b").expect("primary key must be valid");
    replacement
        .add_string("body", "Rust vector database engine")
        .expect("body must be valid");
    collection
        .update(&[&replacement])
        .expect("update must succeed");
    assert_eq!(
        ids(&query(&collection, "rust AND \"vector database\"", None)),
        ["a", "b"]
    );
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
    assert_eq!(
        ids(&query(&reopened, "rust AND \"vector database\"", None)),
        ["a", "b"]
    );
}

#[test]
fn generated_boolean_and_phrase_queries_match_index_and_scan_bit_for_bit() {
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed = Collection::create(
        temporary
            .path()
            .join("generated-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("generated-indexed", true),
        None,
    )
    .expect("indexed collection must be created");
    let scan = Collection::create(
        temporary
            .path()
            .join("generated-scan")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("generated-scan", false),
        None,
    )
    .expect("scan collection must be created");
    let docs = generated_docs();
    let refs: Vec<_> = docs.iter().collect();
    indexed.insert(&refs).expect("insert must succeed");
    scan.insert(&refs).expect("insert must succeed");

    for (expression, operator) in [
        ("alpha AND beta", None),
        ("alpha OR beta AND gamma", None),
        ("(alpha OR beta) AND gamma", None),
        ("alpha NOT beta", None),
        ("+alpha beta -gamma", None),
        ("\"alpha beta\"", None),
        ("\"alpha beta\" OR gamma", None),
        ("alpha AND (\"beta gamma\" OR delta)", None),
        ("(alpha beta) NOT \"gamma delta\"", None),
        ("alpha OR alpha", None),
        ("\"alpha alpha\"", None),
        ("alpha beta gamma", Some("AND")),
    ] {
        let indexed_result = query(&indexed, expression, operator);
        let scan_result = query(&scan, expression, operator);
        assert_eq!(
            comparable(&indexed_result),
            comparable(&scan_result),
            "expression={expression}, operator={operator:?}"
        );
    }
}

#[test]
fn structured_planner_indexes_selective_queries_and_scans_broad_queries() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("phrase-planner")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("phrase-planner", true),
        None,
    )
    .expect("collection must be created");
    insert_fixture(&collection);

    let before = collection
        .stats_snapshot()
        .expect("statistics must be readable");
    assert_eq!(
        ids(&query(&collection, "engine AND \"vector database\"", None)),
        ["a"]
    );
    let after_selective = collection
        .stats_snapshot()
        .expect("statistics must be readable");
    assert_eq!(
        after_selective.fts_index_query_count - before.fts_index_query_count,
        1
    );
    assert_eq!(
        after_selective.candidates_scanned - before.candidates_scanned,
        1
    );

    assert_eq!(
        ids(&query(&collection, "\"vector database\"", None)),
        ["c", "a"]
    );
    let after_broad = collection
        .stats_snapshot()
        .expect("statistics must be readable");
    assert_eq!(
        after_broad.fts_index_query_count,
        after_selective.fts_index_query_count
    );
    assert_eq!(
        after_broad.candidates_scanned - after_selective.candidates_scanned,
        u64::try_from(docs().len()).expect("fixture size must fit u64")
    );

    assert_eq!(
        ids(&query(
            &collection,
            "(rust OR python) AND database NOT legacy",
            None
        )),
        ["c", "a", "b", "d"]
    );
    let after_boolean = collection
        .stats_snapshot()
        .expect("statistics must be readable");
    assert_eq!(
        after_boolean.fts_index_query_count,
        after_broad.fts_index_query_count
    );
    assert_eq!(
        after_boolean.candidates_scanned - after_broad.candidates_scanned,
        u64::try_from(docs().len()).expect("fixture size must fit u64")
    );
}

#[test]
fn unsupported_or_malformed_structured_syntax_fails_explicitly() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("errors")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("errors", true),
        None,
    )
    .expect("collection must be created");
    insert_fixture(&collection);

    for expression in ["rust && vector", "rust || vector"] {
        let mut fts = Fts::new().expect("FTS payload must be valid");
        fts.set_query_string(expression)
            .expect("query string setter must accept opaque text");
        let query = SearchQuery::fts("body", &fts, 10).expect("query must be valid");
        let error = collection
            .query(&query)
            .expect_err("unsupported syntax must fail");
        assert_eq!(error.code, ErrorCode::NotSupported, "{expression}");
    }

    for expression in [
        "NOT rust",
        "rust AND",
        "(rust OR vector",
        "\"vector database",
        "other:rust",
        "body:",
        "rust^0",
        "rust^-1",
        "rust^nan",
        "rust~3",
        "rust~0",
        "rust*~1",
        "\"vector engine\"~-1",
        "[alpha omega]",
        "[alpha TO omega",
    ] {
        let mut fts = Fts::new().expect("FTS payload must be valid");
        fts.set_query_string(expression)
            .expect("query string setter must accept opaque text");
        let query = SearchQuery::fts("body", &fts, 10).expect("query must be valid");
        let error = collection
            .query(&query)
            .expect_err("malformed syntax must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument, "{expression}");
    }
}
