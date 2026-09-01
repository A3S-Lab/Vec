use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, Fts, IndexParams,
    SearchQuery,
};
use std::cmp::Ordering;
use tempfile::tempdir;

#[derive(Clone, Copy)]
struct TextCase {
    id: &'static str,
    body: Option<&'static str>,
}

fn count_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test corpus count must fit u32"))
}

fn fts_schema() -> CollectionSchema {
    let params =
        IndexParams::fts(Some("whitespace"), None, None).expect("FTS parameters must be valid");
    let mut body =
        FieldSchema::new("body", DataType::String, true, 0).expect("body schema must be valid");
    body.set_index_params(&params)
        .expect("scan tokenizer must be attachable");
    CollectionSchema::builder("differential-fts")
        .add_field(body)
        .build()
        .expect("collection schema must be valid")
}

fn insert_text(collection: &Collection, cases: &[TextCase]) {
    let docs: Vec<Doc> = cases
        .iter()
        .map(|case| {
            let mut doc = Doc::with_pk(case.id).expect("primary key must be valid");
            if let Some(body) = case.body {
                doc.add_string("body", body).expect("body must be valid");
            }
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection.insert(&refs).expect("insert must succeed");
    assert_eq!(
        result.success_count,
        u64::try_from(cases.len()).expect("case count must fit u64")
    );
}

fn sort_reference(values: &mut [(String, f64)]) {
    values.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
}

fn assert_ranked_result(actual: &[Doc], expected: &[(String, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, (expected_id, expected_score)) in actual.iter().zip(expected) {
        assert_eq!(actual.get_pk(), Some(expected_id.as_str()));
        let tolerance = (expected_score.abs() * 1.0e-6).max(1.0e-6);
        assert!(
            (f64::from(actual.get_score()) - expected_score).abs() <= tolerance,
            "id={expected_id}, expected={expected_score}, actual={}",
            actual.get_score()
        );
    }
}

fn reference_bm25(cases: &[TextCase], terms: &[&str]) -> Vec<(String, f64)> {
    let corpus: Vec<Vec<&str>> = cases
        .iter()
        .filter_map(|case| case.body.map(|body| body.split_whitespace().collect()))
        .collect();
    let document_count = count_to_f64(corpus.len());
    let average_length = count_to_f64(corpus.iter().map(Vec::len).sum()) / document_count;
    let mut scored = Vec::new();
    for case in cases {
        let Some(body) = case.body else { continue };
        let tokens: Vec<&str> = body.split_whitespace().collect();
        let mut score = 0.0;
        for term in terms {
            let frequency = count_to_f64(tokens.iter().filter(|token| **token == *term).count());
            if frequency == 0.0 {
                continue;
            }
            let document_frequency =
                count_to_f64(corpus.iter().filter(|tokens| tokens.contains(term)).count());
            let idf = ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                + 1.0)
                .ln();
            let denominator =
                frequency + 1.2 * (1.0 - 0.75 + 0.75 * count_to_f64(tokens.len()) / average_length);
            score += idf * (frequency * 2.2 / denominator);
        }
        if score > 0.0 {
            scored.push((case.id.to_string(), score));
        }
    }
    sort_reference(&mut scored);
    scored
}

#[test]
fn scan_bm25_matches_a_text_bearing_corpus_reference() {
    let cases = [
        TextCase {
            id: "a",
            body: Some("rust rust vector"),
        },
        TextCase {
            id: "b",
            body: Some("rust database"),
        },
        TextCase {
            id: "c",
            body: None,
        },
        TextCase {
            id: "d",
            body: Some("database vector search search"),
        },
    ];
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("fts")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &fts_schema(),
        None,
    )
    .expect("collection must be created");
    insert_text(&collection, &cases);
    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("rust vector")
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("body", &fts, 10).expect("FTS query must be valid");
    let actual = collection.query(&query).expect("FTS query must succeed");
    let expected = reference_bm25(&cases, &["rust", "vector"]);
    assert_ranked_result(&actual, &expected);
}

#[test]
fn scan_bm25_preserves_a_subunit_average_document_length() {
    let cases = [
        TextCase {
            id: "a",
            body: Some("rust"),
        },
        TextCase {
            id: "b",
            body: Some(""),
        },
        TextCase {
            id: "c",
            body: Some(""),
        },
    ];
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("short-fts")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &fts_schema(),
        None,
    )
    .expect("collection must be created");
    insert_text(&collection, &cases);
    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("rust")
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("body", &fts, 10).expect("FTS query must be valid");
    let actual = collection.query(&query).expect("FTS query must succeed");
    let expected = reference_bm25(&cases, &["rust"]);
    assert_ranked_result(&actual, &expected);
}

#[test]
fn structured_fts_executes_supported_syntax_and_rejects_symbolic_boolean_aliases() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("advanced-fts")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &fts_schema(),
        None,
    )
    .expect("collection must be created");
    insert_text(
        &collection,
        &[TextCase {
            id: "doc",
            body: Some("rust vector database"),
        }],
    );
    let mut simple = Fts::new().expect("FTS payload must be created");
    simple
        .set_query_string("rust vector")
        .expect("simple query string must be valid");
    let simple_query = SearchQuery::fts("body", &simple, 10).expect("FTS query must be valid");
    let simple_result = collection
        .query(&simple_query)
        .expect("a whitespace-separated query string must execute");
    assert_eq!(simple_result[0].get_pk(), Some("doc"));

    for expression in [
        "rust AND vector",
        "\"rust vector\"",
        "+rust",
        "rust*",
        "body:rust",
        "rust^2",
        "rust~1",
        "[rust TO vector]",
        "\"rust database\"~1",
    ] {
        let mut fts = Fts::new().expect("FTS payload must be created");
        fts.set_query_string(expression)
            .expect("syntactic query string must be accepted");
        let query = SearchQuery::fts("body", &fts, 10).expect("FTS query must be valid");
        let result = collection
            .query(&query)
            .expect("supported structured syntax must execute");
        assert_eq!(result[0].get_pk(), Some("doc"), "{expression}");
    }

    for expression in ["rust && vector", "rust || vector"] {
        let mut fts = Fts::new().expect("FTS payload must be created");
        fts.set_query_string(expression)
            .expect("syntactic query string must be accepted");
        let query = SearchQuery::fts("body", &fts, 10).expect("FTS query must be valid");
        let error = collection
            .query(&query)
            .expect_err("unsupported symbolic boolean syntax must fail");
        assert_eq!(error.code, ErrorCode::NotSupported, "{expression}");
    }
}

#[test]
fn ambiguous_fts_expression_forms_are_rejected() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("ambiguous-fts")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &fts_schema(),
        None,
    )
    .expect("collection must be created");
    let mut fts = Fts::new().expect("FTS payload must be created");
    fts.set_match_string("rust")
        .expect("match expression must be valid");
    fts.set_query_string("vector")
        .expect("query expression must be valid");
    let query = SearchQuery::fts("body", &fts, 10).expect("FTS query must be valid");
    let error = collection
        .query(&query)
        .expect_err("two FTS expression forms must be ambiguous");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}
