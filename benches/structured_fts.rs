//! Reproducible indexed-versus-scan benchmark for Workspace FTS expressions.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    IndexParams, SearchQuery,
};
use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DEFAULT_DOCUMENTS: usize = 25_000;
const DEFAULT_ROUNDS: usize = 5;
const QUERIES_PER_ROUND: usize = 8;
const TOPK: i32 = 20;

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

fn document(index: usize) -> Doc {
    let language = if index % 2 == 0 { "rust" } else { "typescript" };
    let phrase = if index % 5 == 0 {
        "vector fast database"
    } else {
        "vector database"
    };
    let lifecycle = if index % 11 == 0 { "legacy" } else { "current" };
    let mut doc = Doc::with_pk(format!("doc-{index:06}")).expect("primary key must be valid");
    doc.add_string(
        "body",
        &format!(
            "Workspace src component{} ResolveSymbol{index} {language} {phrase} {lifecycle}",
            index % 128
        ),
    )
    .expect("body must be valid");
    doc
}

fn query(expression: &str) -> SearchQuery {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_query_string(expression)
        .expect("query string must be valid");
    SearchQuery::fts("body", &fts, TOPK).expect("FTS query must be valid")
}

fn run_queries(
    collection: &Collection,
    query: &SearchQuery,
    rounds: usize,
) -> (Duration, f64, bool, Vec<(String, u32)>) {
    let before = collection
        .stats_snapshot()
        .expect("statistics must be readable");
    let mut result = collection
        .query(black_box(query))
        .expect("warmup query must succeed");
    let mut durations = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let started = Instant::now();
        for _ in 0..QUERIES_PER_ROUND {
            result = collection
                .query(black_box(query))
                .expect("benchmark query must succeed");
            black_box(&result);
        }
        durations.push(started.elapsed());
    }
    durations.sort_unstable();
    let after = collection
        .stats_snapshot()
        .expect("statistics must be readable");
    let operations = rounds.saturating_mul(QUERIES_PER_ROUND).saturating_add(1);
    let candidates = after
        .candidates_scanned
        .saturating_sub(before.candidates_scanned);
    let indexed_queries = after
        .fts_index_query_count
        .saturating_sub(before.fts_index_query_count);
    assert!(
        indexed_queries == 0 || indexed_queries == u64::try_from(operations).unwrap_or(u64::MAX),
        "a stable query plan must not switch execution paths during measurement"
    );
    let comparable = result
        .into_iter()
        .map(|doc| {
            (
                doc.get_pk()
                    .expect("result must have a primary key")
                    .to_string(),
                doc.get_score().to_bits(),
            )
        })
        .collect();
    (
        durations[rounds / 2],
        count_to_f64(usize::try_from(candidates).unwrap_or(usize::MAX)) / count_to_f64(operations),
        indexed_queries > 0,
        comparable,
    )
}

#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_query(duration: Duration) -> f64 {
    duration.as_micros() as f64 / count_to_f64(QUERIES_PER_ROUND)
}

fn main() {
    let documents = positive_env("A3S_VEC_STRUCTURED_FTS_DOCUMENTS", DEFAULT_DOCUMENTS).max(12);
    let rounds = positive_env("A3S_VEC_STRUCTURED_FTS_ROUNDS", DEFAULT_ROUNDS);
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed = Collection::create(
        temporary
            .path()
            .join("indexed")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema("structured-fts-indexed", true),
        Some(&manual_options()),
    )
    .expect("indexed collection must be created");
    let scan = Collection::create(
        temporary
            .path()
            .join("scan")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema("structured-fts-scan", false),
        Some(&manual_options()),
    )
    .expect("scan collection must be created");
    let docs: Vec<_> = (0..documents).map(document).collect();
    let refs: Vec<_> = docs.iter().collect();
    indexed.insert(&refs).expect("indexed insert must succeed");
    scan.insert(&refs).expect("scan insert must succeed");
    let unique = (0..documents)
        .rev()
        .find(|index| index % 5 != 0 && index % 11 != 0)
        .expect("positive document count must provide a selective fixture");
    let proximity_unique = (0..documents)
        .rev()
        .find(|index| index % 5 == 0 && index % 11 != 0)
        .expect("positive document count must provide a proximity fixture");

    println!("case,mode,documents,candidates_per_query,micros_per_query");
    for (name, expression) in [
        (
            "selective_phrase",
            format!("resolvesymbol{unique} AND \"vector database\""),
        ),
        (
            "selective_required_optional",
            format!("+resolvesymbol{unique} workspace"),
        ),
        ("selective_wildcard", format!("resolvesymbol{unique}*")),
        ("selective_fuzzy", format!("resolvesymbol{unique}~1")),
        (
            "selective_range",
            format!("[resolvesymbol{unique} TO resolvesymbol{unique}]"),
        ),
        (
            "selective_proximity",
            format!("resolvesymbol{proximity_unique} AND \"vector database\"~1"),
        ),
        ("common_phrase", "\"vector database\"".to_string()),
        (
            "broad_boolean_not",
            "(rust OR typescript) AND vector NOT legacy".to_string(),
        ),
    ] {
        let query = query(&expression);
        let (indexed_duration, indexed_candidates, used_index, indexed_result) =
            run_queries(&indexed, &query, rounds);
        let (scan_duration, scan_candidates, _, scan_result) = run_queries(&scan, &query, rounds);
        assert!(!indexed_result.is_empty(), "{name} must return results");
        assert_eq!(indexed_result, scan_result, "{name} must remain exact");
        let planner_mode = if used_index {
            "indexed"
        } else {
            "planner_scan"
        };
        println!(
            "{name},{planner_mode},{documents},{indexed_candidates:.2},{:.2}",
            micros_per_query(indexed_duration)
        );
        println!(
            "{name},scan,{documents},{scan_candidates:.2},{:.2}",
            micros_per_query(scan_duration)
        );
    }
}
