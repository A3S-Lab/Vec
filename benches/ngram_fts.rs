//! Reproducible workspace-shaped n-gram FTS build, query, and reopen fixture.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    FtsQueryParams, IndexParams, SearchQuery,
};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DEFAULT_DOCUMENTS: usize = 10_000;
const DEFAULT_ROUNDS: usize = 5;
const QUERIES_PER_ROUND: usize = 16;
const TOPK: i32 = 20;
const WRITES: usize = 64;
const CACHE_PATH: &str = "indexes/index-cache.bin";
const NGRAM_CONFIG: &str = r#"{"ngram_min":2,"ngram_max":3,"token_chars":["letter","digit"]}"#;

fn positive_env(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn read_only_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only option must be valid");
    options
}

fn schema() -> CollectionSchema {
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("body field must be valid");
    body.set_index_params(
        &IndexParams::fts(Some("ngram"), None, Some(NGRAM_CONFIG))
            .expect("FTS params must be valid"),
    )
    .expect("n-gram index must be valid");
    CollectionSchema::builder("ngram-workspace-benchmark")
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

fn document(index: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:06}")).expect("document primary key must be valid");
    let language = if index % 2 == 0 { "rust" } else { "typescript" };
    doc.add_string(
        "body",
        &format!(
            "src/component{}/workspace_index_{index}.{language} resolveSymbol{index} vector database 高速检索 编码助手",
            index % 128
        ),
    )
    .expect("body must be valid");
    doc
}

fn replacement(epoch: usize) -> Doc {
    let mut doc = Doc::with_pk("doc-000000").expect("document primary key must be valid");
    doc.add_string(
        "body",
        &format!("src/live/workspace_delta.rs resolveReplacement{epoch} vector database 增量检索"),
    )
    .expect("body must be valid");
    doc
}

fn query(expression: &str, operator: Option<&str>) -> SearchQuery {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string(expression)
        .expect("FTS expression must be valid");
    let mut query = SearchQuery::fts("body", &fts, TOPK).expect("query must be valid");
    if let Some(operator) = operator {
        query
            .set_fts_params(
                FtsQueryParams::new(Some(operator)).expect("operator must be syntactically valid"),
            )
            .expect("FTS operator must be accepted");
    }
    query
}

fn run_queries(
    collection: &Collection,
    query: &SearchQuery,
    rounds: usize,
) -> (Duration, Vec<Doc>, f64) {
    let before = collection
        .stats_snapshot()
        .expect("statistics must be readable")
        .candidates_scanned;
    let mut last = collection
        .query(black_box(query))
        .expect("warmup query must succeed");
    black_box(&last);
    let mut durations = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let started = Instant::now();
        for _ in 0..QUERIES_PER_ROUND {
            last = collection
                .query(black_box(query))
                .expect("benchmark query must succeed");
            black_box(&last);
        }
        durations.push(started.elapsed());
    }
    durations.sort_unstable();
    let after = collection
        .stats_snapshot()
        .expect("statistics must be readable")
        .candidates_scanned;
    let operations = rounds.saturating_mul(QUERIES_PER_ROUND).saturating_add(1);
    let candidates = after.saturating_sub(before);
    (
        durations[rounds / 2],
        last,
        count_to_f64(usize::try_from(candidates).unwrap_or(usize::MAX)) / count_to_f64(operations),
    )
}

fn timed_updates(collection: &Collection) -> Duration {
    let warmup = replacement(usize::MAX);
    collection
        .update(&[&warmup])
        .expect("warmup update must succeed");
    let started = Instant::now();
    for epoch in 0..WRITES {
        let doc = replacement(epoch);
        black_box(
            collection
                .update(&[black_box(&doc)])
                .expect("indexed update must succeed"),
        );
    }
    let duration = started.elapsed();
    let result = collection
        .query(&query("Replacement63", Some("and")))
        .expect("updated text must be searchable");
    assert_eq!(result[0].get_pk(), Some("doc-000000"));
    duration
}

#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_query(duration: Duration) -> f64 {
    duration.as_micros() as f64 / count_to_f64(QUERIES_PER_ROUND)
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_operation(duration: Duration, operations: usize) -> f64 {
    duration.as_micros() as f64 / count_to_f64(operations)
}

#[allow(clippy::cast_precision_loss)]
fn milliseconds(duration: Duration) -> f64 {
    duration.as_micros() as f64 / 1_000.0
}

fn file_bytes(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |metadata| metadata.len())
}

fn main() {
    let documents = positive_env("A3S_VEC_NGRAM_DOCUMENTS", DEFAULT_DOCUMENTS);
    let rounds = positive_env("A3S_VEC_NGRAM_ROUNDS", DEFAULT_ROUNDS);
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("ngram");
    let collection = Collection::create(
        path.to_str().expect("benchmark path must be UTF-8"),
        &schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let docs: Vec<_> = (0..documents).map(document).collect();
    let refs: Vec<_> = docs.iter().collect();
    let started = Instant::now();
    collection
        .insert(&refs)
        .expect("workspace fixture must be inserted");
    let insert = started.elapsed();
    println!("fixture,documents,milliseconds");
    println!(
        "ngram_initial_insert,{documents},{:.2}",
        milliseconds(insert)
    );

    println!("case,documents,queries,rounds,candidates_per_query,micros_per_query");
    let unique_index = documents - 1;
    for (name, expression, operator, expected_id) in [
        (
            "identifier_suffix_or",
            format!("resolveSymbol{unique_index}"),
            None,
            Some(format!("doc-{unique_index:06}")),
        ),
        (
            "identifier_suffix_and",
            format!("resolveSymbol{unique_index}"),
            Some("and"),
            Some(format!("doc-{unique_index:06}")),
        ),
        (
            "path_fragment_and",
            "workspace_index".to_string(),
            Some("and"),
            None,
        ),
        ("cjk_phrase_and", "高速检索".to_string(), Some("and"), None),
    ] {
        let (duration, results, candidates) =
            run_queries(&collection, &query(&expression, operator), rounds);
        assert!(!results.is_empty(), "{name} must return a result");
        if let Some(expected_id) = expected_id {
            assert_eq!(results[0].get_pk(), Some(expected_id.as_str()));
        }
        println!(
            "{name},{documents},{QUERIES_PER_ROUND},{rounds},{candidates:.2},{:.2}",
            micros_per_query(duration)
        );
    }

    let updates = timed_updates(&collection);
    let started = Instant::now();
    collection
        .rebuild_index("body")
        .expect("n-gram index must rebuild");
    let rebuild = started.elapsed();
    println!("operation,documents,samples,micros_per_operation");
    println!(
        "ngram_update,{documents},{WRITES},{:.2}",
        micros_per_operation(updates, WRITES)
    );
    println!(
        "ngram_rebuild,{documents},1,{:.2}",
        micros_per_operation(rebuild, 1)
    );

    let started = Instant::now();
    collection.close().expect("collection must close");
    let close = started.elapsed();
    let cache_bytes = file_bytes(&path.join(CACHE_PATH));
    let started = Instant::now();
    let reopened = Collection::open(
        path.to_str().expect("benchmark path must be UTF-8"),
        Some(&read_only_options()),
    )
    .expect("collection must reopen");
    let reopen = started.elapsed();
    assert!(
        reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    println!("persistence,documents,milliseconds,cache_bytes");
    println!(
        "ngram_close,{documents},{:.2},{cache_bytes}",
        milliseconds(close)
    );
    println!(
        "ngram_cache_reopen,{documents},{:.2},{cache_bytes}",
        milliseconds(reopen)
    );
}
