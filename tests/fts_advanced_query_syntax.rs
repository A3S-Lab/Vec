use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams,
    SearchQuery,
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

fn query(collection: &Collection, expression: &str) -> Vec<Doc> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_query_string(expression)
        .expect("query string must be valid");
    collection
        .query(&SearchQuery::fts("body", &fts, 32).expect("FTS query must be valid"))
        .expect("structured FTS query must succeed")
}

fn advanced_docs() -> Vec<Doc> {
    [
        ("exact", "rust database"),
        ("prefix", "rustacean database"),
        ("single", "rusty database"),
        ("fuzzy", "trust database"),
        ("literal", "rust* database"),
        ("phrase-0", "vector engine"),
        ("phrase-1", "vector fast engine"),
        ("phrase-2", "vector very fast engine"),
        ("range-mango", "mango database"),
        ("python", "python python python database"),
        ("omega", "omega database"),
    ]
    .into_iter()
    .map(|(id, body)| {
        let mut doc = Doc::with_pk(id).expect("primary key must be valid");
        doc.add_string("body", body).expect("body must be valid");
        doc
    })
    .collect()
}

fn ids(docs: &[Doc]) -> Vec<&str> {
    docs.iter().filter_map(Doc::get_pk).collect()
}

fn sorted_ids(docs: &[Doc]) -> Vec<&str> {
    let mut ids = ids(docs);
    ids.sort_unstable();
    ids
}

fn comparable(docs: &[Doc]) -> Vec<(&str, u32)> {
    docs.iter()
        .map(|doc| (doc.get_pk().unwrap_or_default(), doc.get_score().to_bits()))
        .collect()
}

#[test]
fn advanced_term_phrase_field_boost_and_range_queries_match_the_scan_oracle() {
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed = Collection::create(
        temporary
            .path()
            .join("advanced-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("advanced-indexed", true),
        None,
    )
    .expect("indexed collection must be created");
    let scan = Collection::create(
        temporary
            .path()
            .join("advanced-scan")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("advanced-scan", false),
        None,
    )
    .expect("scan collection must be created");
    let docs = advanced_docs();
    let refs: Vec<_> = docs.iter().collect();
    indexed.insert(&refs).expect("insert must succeed");
    scan.insert(&refs).expect("insert must succeed");

    for (expression, expected) in [
        (
            "*",
            vec![
                "exact",
                "fuzzy",
                "literal",
                "omega",
                "phrase-0",
                "phrase-1",
                "phrase-2",
                "prefix",
                "python",
                "range-mango",
                "single",
            ],
        ),
        ("rust*", vec!["exact", "literal", "prefix", "single"]),
        ("r?sty", vec!["single"]),
        (
            "*base",
            vec![
                "exact",
                "fuzzy",
                "literal",
                "omega",
                "prefix",
                "python",
                "range-mango",
                "single",
            ],
        ),
        (r"rust\*", vec!["exact", "literal"]),
        ("rust~1", vec!["exact", "fuzzy", "literal", "single"]),
        ("rsut~1", vec!["exact", "literal"]),
        (
            "[mango TO rust]",
            vec!["exact", "literal", "omega", "python", "range-mango"],
        ),
        ("{mango TO rust}", vec!["omega", "python"]),
        ("[mango TO rust}", vec!["omega", "python", "range-mango"]),
        ("\"vector engine\"", vec!["phrase-0"]),
        ("\"vector engine\"~0", vec!["phrase-0"]),
        ("\"vector engine\"~1", vec!["phrase-0", "phrase-1"]),
        (
            "\"vector engine\"~2",
            vec!["phrase-0", "phrase-1", "phrase-2"],
        ),
        ("body:rust", vec!["exact", "literal"]),
        ("body:(rust OR python)", vec!["exact", "literal", "python"]),
        ("body:\"vector engine\"~1", vec!["phrase-0", "phrase-1"]),
        (
            "body:[mango TO rust]",
            vec!["exact", "literal", "omega", "python", "range-mango"],
        ),
    ] {
        let indexed_result = query(&indexed, expression);
        let scan_result = query(&scan, expression);
        assert_eq!(sorted_ids(&indexed_result), expected, "{expression}");
        assert_eq!(comparable(&indexed_result), comparable(&scan_result));
    }

    let unboosted = query(&indexed, "rust OR python");
    assert_eq!(ids(&unboosted).first().copied(), Some("python"));
    let boosted_indexed = query(&indexed, "rust^10 OR python");
    let boosted_scan = query(&scan, "rust^10 OR python");
    assert_eq!(ids(&boosted_indexed).first().copied(), Some("exact"));
    assert_eq!(comparable(&boosted_indexed), comparable(&boosted_scan));
}

#[test]
fn advanced_queries_track_mutation_and_cache_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("advanced-reopen");
    let collection = Collection::create(
        path.to_str().expect("temporary path must be UTF-8"),
        &schema("advanced-reopen", true),
        None,
    )
    .expect("collection must be created");
    let docs = advanced_docs();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("insert must succeed");
    assert_eq!(
        sorted_ids(&query(&collection, "rustac* OR \"vector engine\"~1")),
        ["phrase-0", "phrase-1", "prefix"]
    );

    let mut replacement = Doc::with_pk("prefix").expect("primary key must be valid");
    replacement
        .add_string("body", "python database")
        .expect("body must be valid");
    collection
        .update(&[&replacement])
        .expect("update must succeed");
    assert_eq!(
        sorted_ids(&query(&collection, "rustac* OR \"vector engine\"~1")),
        ["phrase-0", "phrase-1"]
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
        sorted_ids(&query(&reopened, "rustac* OR \"vector engine\"~1")),
        ["phrase-0", "phrase-1"]
    );
}
