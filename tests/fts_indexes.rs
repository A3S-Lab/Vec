use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    FieldValue, Fts, FtsQueryParams, IndexParams, IndexType, SearchQuery,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::tempdir;

const DOCUMENTS: usize = 256;

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn schema(name: &str, indexed_fts: bool) -> CollectionSchema {
    let mut bucket =
        FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket must be valid");
    bucket
        .set_index_params(
            &IndexParams::invert(true, false).expect("scalar descriptor must be valid"),
        )
        .expect("bucket must support an inverted index");
    let mut body = FieldSchema::new("body", DataType::String, true, 0).expect("body must be valid");
    if indexed_fts {
        body.set_index_params(
            &IndexParams::fts(Some("standard"), None, None).expect("FTS descriptor must be valid"),
        )
        .expect("body must support an FTS index");
    }
    CollectionSchema::builder(name)
        .add_field(bucket)
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

fn document(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:04}")).expect("document must be valid");
    doc.add_i32(
        "bucket",
        i32::try_from(index % 16).expect("bucket fits i32"),
    )
    .expect("bucket must be valid");
    if index % 19 == 0 {
        doc.set_field_value("body", FieldValue::Null)
            .expect("null body must be valid");
    } else if index % 17 != 0 {
        let language = if index % 2 == 0 { "rust" } else { "python" };
        doc.add_string(
            "body",
            &format!(
                "workspace {language} vector search token{} group{}",
                index % 13,
                index % 7
            ),
        )
        .expect("body must be valid");
    }
    doc
}

fn insert_fixture(collection: &Collection) {
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection
        .insert(&refs)
        .expect("fixture insert must succeed");
    assert_eq!(
        result.success_count,
        u64::try_from(DOCUMENTS).expect("document count fits u64")
    );
}

fn query(collection: &Collection, expression: &str, filter: Option<&str>) -> Vec<Doc> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string(expression)
        .expect("FTS expression must be valid");
    let mut query = SearchQuery::fts(
        "body",
        &fts,
        i32::try_from(DOCUMENTS).expect("document count fits i32"),
    )
    .expect("FTS query must be valid");
    if let Some(filter) = filter {
        query.set_filter(filter).expect("filter must be valid");
    }
    collection.query(&query).expect("FTS query must succeed")
}

fn comparable_results(docs: &[Doc]) -> Vec<(&str, f32)> {
    docs.iter()
        .map(|doc| (doc.get_pk().unwrap_or_default(), doc.get_score()))
        .collect()
}

fn assert_same_results(
    indexed: &Collection,
    fallback: &Collection,
    expression: &str,
    filter: Option<&str>,
) {
    let indexed = query(indexed, expression, filter);
    let fallback = query(fallback, expression, filter);
    assert_eq!(
        comparable_results(&indexed),
        comparable_results(&fallback),
        "expression={expression}, filter={filter:?}"
    );
}

fn query_with_operator(collection: &Collection, expression: &str, operator: &str) -> Vec<Doc> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string(expression)
        .expect("FTS expression must be valid");
    let mut query = SearchQuery::fts(
        "body",
        &fts,
        i32::try_from(DOCUMENTS).expect("document count fits i32"),
    )
    .expect("FTS query must be valid");
    query
        .set_fts_params(
            FtsQueryParams::new(Some(operator)).expect("operator must be syntactically valid"),
        )
        .expect("FTS operator must be accepted");
    collection.query(&query).expect("FTS query must succeed")
}

#[test]
fn indexed_bm25_matches_scan_for_terms_duplicates_nulls_and_scalar_filters() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("indexed", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("fallback", false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let cases = [
        ("rust vector", None),
        ("token3", None),
        ("workspace missingterm", Some("bucket >= 12")),
        ("rust rust token7", Some("bucket == 7")),
        ("absent", Some("bucket < 4")),
    ];
    for (expression, filter) in cases {
        assert_same_results(&indexed, &fallback, expression, filter);
    }

    let indexed_stats = indexed
        .stats_snapshot()
        .expect("indexed statistics must be readable");
    assert_eq!(indexed_stats.fts_query_count, 5);
    assert_eq!(indexed_stats.fts_index_query_count, 5);
    assert_eq!(indexed_stats.scalar_index_query_count, 3);
    let fallback_stats = fallback
        .stats_snapshot()
        .expect("fallback statistics must be readable");
    assert_eq!(fallback_stats.fts_query_count, 5);
    assert_eq!(fallback_stats.fts_index_query_count, 0);
}

#[test]
fn indexed_and_operator_matches_scan_intersection_and_scores() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("and-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("and-indexed", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("and-fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("and-fallback", false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let indexed = query_with_operator(&indexed, "rust token7", "AND");
    let fallback = query_with_operator(&fallback, "rust token7", "and");
    assert!(!indexed.is_empty());
    assert_eq!(comparable_results(&indexed), comparable_results(&fallback));
}

#[test]
fn conservative_scalar_prefilter_does_not_truncate_before_final_filter() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("conservative-filter")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("conservative-filter", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    insert_fixture(&indexed);
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string("workspace")
        .expect("FTS expression must be valid");
    let mut query = SearchQuery::fts("body", &fts, 1).expect("FTS query must be valid");
    query
        .set_filter("bucket >= 0 and body == 'workspace rust vector search token7 group2'")
        .expect("filter must be valid");
    let before = indexed
        .stats_snapshot()
        .expect("statistics must be readable");

    let result = indexed.query(&query).expect("FTS query must succeed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get_pk(), Some("doc-0072"));
    let after = indexed
        .stats_snapshot()
        .expect("statistics must be readable");
    assert!(after.candidates_scanned - before.candidates_scanned > 1);
}

#[test]
fn fts_index_lifecycle_tracks_mutations_reopen_drop_recreate_and_rebuild() {
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed_path = temporary.path().join("lifecycle-indexed");
    let options = manual_options();
    let indexed = Collection::create(
        indexed_path.to_str().expect("temporary path must be UTF-8"),
        &schema("lifecycle-indexed", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("lifecycle-fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("lifecycle-fallback", false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let mut patch = Doc::with_pk("doc-0003").expect("patch must be valid");
    patch
        .add_string("body", "replacement lexical token9")
        .expect("replacement body must be valid");
    indexed.update(&[&patch]).expect("update must succeed");
    fallback.update(&[&patch]).expect("update must succeed");

    let mut replacement = Doc::with_pk("doc-0019").expect("replacement must be valid");
    replacement
        .add_i32("bucket", 3)
        .expect("bucket must be valid");
    replacement
        .set_field_value("body", FieldValue::Null)
        .expect("null body must be valid");
    indexed
        .upsert(&[&replacement])
        .expect("replacement must succeed");
    fallback
        .upsert(&[&replacement])
        .expect("replacement must succeed");
    indexed.delete(&["doc-0021"]).expect("delete must succeed");
    fallback.delete(&["doc-0021"]).expect("delete must succeed");

    let mut inserted = document(999);
    inserted.set_pk("doc-0999");
    indexed.insert(&[&inserted]).expect("insert must succeed");
    fallback.insert(&[&inserted]).expect("insert must succeed");

    for expression in ["replacement", "workspace vector", "token9", "python"] {
        assert_same_results(&indexed, &fallback, expression, Some("bucket >= 3"));
    }
    indexed.flush().expect("collection must flush");
    indexed.close().expect("collection must close");
    let reopened = Collection::open(
        indexed_path.to_str().expect("temporary path must be UTF-8"),
        Some(&options),
    )
    .expect("collection must reopen");
    assert_same_results(&reopened, &fallback, "replacement token9", None);

    reopened
        .drop_index("body")
        .expect("FTS index must be dropped");
    let before_scan = reopened
        .stats_snapshot()
        .expect("statistics must be readable");
    assert_same_results(&reopened, &fallback, "workspace vector", None);
    let after_scan = reopened
        .stats_snapshot()
        .expect("statistics must be readable");
    assert_eq!(
        after_scan.fts_index_query_count,
        before_scan.fts_index_query_count
    );

    reopened
        .create_index(
            "body",
            &IndexParams::fts(Some("standard"), None, None).expect("descriptor must be valid"),
        )
        .expect("FTS index must be recreated");
    reopened
        .rebuild_index("body")
        .expect("FTS index must rebuild");
    assert_same_results(&reopened, &fallback, "workspace vector", None);
    let rebuilt = reopened.stats().expect("statistics must be readable");
    let fts = rebuilt
        .indexes
        .iter()
        .find(|index| index.index_type == IndexType::Fts)
        .expect("FTS index stats must exist");
    assert_eq!(fts.source_revision, rebuilt.revision);
}

#[test]
fn retired_fts_ordinals_compact_without_changing_bm25_results() {
    const RETIRED: usize = 80;

    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let indexed = Collection::create(
        temporary
            .path()
            .join("compaction-indexed")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("compaction-indexed", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let fallback = Collection::create(
        temporary
            .path()
            .join("compaction-fallback")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("compaction-fallback", false),
        Some(&options),
    )
    .expect("fallback collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&fallback);

    let retired: Vec<String> = (0..RETIRED)
        .map(|index| format!("doc-{index:04}"))
        .collect();
    let retired_refs: Vec<&str> = retired.iter().map(String::as_str).collect();
    indexed
        .delete(&retired_refs)
        .expect("indexed documents must be retired");
    fallback
        .delete(&retired_refs)
        .expect("fallback documents must be retired");
    let replacements: Vec<Doc> = (0..RETIRED).map(|index| document(1_000 + index)).collect();
    let replacement_refs: Vec<&Doc> = replacements.iter().collect();
    indexed
        .insert(&replacement_refs)
        .expect("indexed replacements must be inserted");
    fallback
        .insert(&replacement_refs)
        .expect("fallback replacements must be inserted");

    for expression in ["workspace vector", "rust token11", "group4"] {
        assert_same_results(&indexed, &fallback, expression, Some("bucket >= 4"));
    }
    let stats = indexed.stats().expect("statistics must be readable");
    let fts = stats
        .indexes
        .iter()
        .find(|index| index.index_type == IndexType::Fts)
        .expect("FTS index stats must exist");
    assert_eq!(fts.source_revision, stats.revision);
}

fn concurrent_schema() -> CollectionSchema {
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("body must be valid");
    body.set_index_params(
        &IndexParams::fts(Some("standard"), None, None).expect("descriptor must be valid"),
    )
    .expect("body must support an FTS index");
    CollectionSchema::builder("fts-concurrency")
        .add_field(
            FieldSchema::new("epoch", DataType::Int32, false, 0).expect("epoch must be valid"),
        )
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

fn concurrent_doc(index: usize, epoch: i32) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:04}")).expect("document primary key must be valid");
    doc.add_i32("epoch", epoch).expect("epoch must be valid");
    let parity = i32::try_from(index % 2).expect("parity fits i32");
    let term = if (parity + epoch).rem_euclid(2) == 0 {
        "alpha"
    } else {
        "beta"
    };
    doc.add_string("body", &format!("{term} workspace"))
        .expect("body must be valid");
    doc
}

fn alpha_query() -> SearchQuery {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string("alpha")
        .expect("FTS expression must be valid");
    SearchQuery::fts(
        "body",
        &fts,
        i32::try_from(DOCUMENTS).expect("document count fits i32"),
    )
    .expect("query must be valid")
}

fn validate_fts_generation(collection: &Collection, query: &SearchQuery) {
    let docs = collection
        .query(query)
        .expect("concurrent FTS query must succeed");
    assert_eq!(docs.len(), DOCUMENTS / 2);
    let epoch = docs[0]
        .get_i32("epoch")
        .expect("epoch must have the declared type")
        .expect("epoch must be present");
    assert!(docs.iter().all(|doc| {
        doc.get_i32("epoch")
            .expect("epoch must have the declared type")
            == Some(epoch)
            && doc
                .get_string("body")
                .expect("body must have the declared type")
                .is_some_and(|body| body.starts_with("alpha"))
    }));
}

#[test]
fn fts_readers_observe_only_complete_posting_generations() {
    const ROUNDS: i32 = 32;
    const READERS: usize = 2;

    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("concurrent")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &concurrent_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let initial: Vec<Doc> = (0..DOCUMENTS)
        .map(|index| concurrent_doc(index, 0))
        .collect();
    collection
        .insert(&initial.iter().collect::<Vec<_>>())
        .expect("initial generation must be inserted");

    let started = Arc::new(Barrier::new(READERS + 1));
    let finished = Arc::new(AtomicBool::new(false));
    let query = alpha_query();
    thread::scope(|scope| {
        let writer_collection = collection.clone();
        let writer_started = Arc::clone(&started);
        let writer_finished = Arc::clone(&finished);
        let writer = scope.spawn(move || {
            writer_started.wait();
            for epoch in 1..=ROUNDS {
                let docs: Vec<Doc> = (0..DOCUMENTS)
                    .map(|index| concurrent_doc(index, epoch))
                    .collect();
                let result = writer_collection
                    .upsert(&docs.iter().collect::<Vec<_>>())
                    .expect("replacement generation must succeed");
                assert_eq!(
                    result.success_count,
                    u64::try_from(DOCUMENTS).expect("document count fits u64")
                );
            }
            writer_finished.store(true, Ordering::Release);
        });

        let readers = (0..READERS)
            .map(|_| {
                let reader_collection = collection.clone();
                let reader_started = Arc::clone(&started);
                let reader_finished = Arc::clone(&finished);
                let reader_query = query.clone();
                scope.spawn(move || {
                    reader_started.wait();
                    while !reader_finished.load(Ordering::Acquire) {
                        validate_fts_generation(&reader_collection, &reader_query);
                    }
                    validate_fts_generation(&reader_collection, &reader_query);
                })
            })
            .collect::<Vec<_>>();

        writer.join().expect("writer must not panic");
        for reader in readers {
            reader.join().expect("reader must not panic");
        }
    });

    let stats = collection.stats().expect("statistics must be readable");
    let fts = stats
        .indexes
        .iter()
        .find(|index| index.index_type == IndexType::Fts)
        .expect("FTS index stats must exist");
    assert_eq!(fts.source_revision, stats.revision);
}
