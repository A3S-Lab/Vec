//! Reproducible indexed-BM25 versus scan-BM25 latency fixture.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    IndexParams, SearchQuery,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DOCUMENTS: usize = 50_000;
const QUERIES: usize = 16;
const ROUNDS: usize = 5;
const TOPK: usize = 20;
const WRITES: usize = 64;
const CASES: [(&str, &str); 5] = [
    ("unique_symbol", "symbol49999"),
    ("component_scope", "component17"),
    ("sparse_multi", "component17 symbol49999"),
    ("language_common", "rust"),
    ("mixed_terms", "rust component17 symbol49999"),
];

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn schema(name: &str) -> CollectionSchema {
    CollectionSchema::builder(name)
        .add_field(
            FieldSchema::new("body", DataType::String, false, 0).expect("body field must be valid"),
        )
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
            "workspace coding agent vector database {language} component{} symbol{index} search ranking",
            index % 64
        ),
    )
    .expect("body must be valid");
    doc
}

fn query(expression: &str) -> SearchQuery {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string(expression)
        .expect("FTS expression must be valid");
    SearchQuery::fts("body", &fts, i32::try_from(TOPK).expect("TOPK fits i32"))
        .expect("query must be valid")
}

fn replacement_body(suffix: &str) -> Doc {
    let mut doc = Doc::with_pk("doc-000000").expect("document primary key must be valid");
    doc.add_string(
        "body",
        &format!("workspace coding agent vector database rust component0 {suffix} search ranking"),
    )
    .expect("replacement body must be valid");
    doc
}

fn comparable_results(docs: &[Doc]) -> Vec<(&str, f32)> {
    docs.iter()
        .map(|doc| {
            (
                doc.get_pk().expect("result must have an id"),
                doc.get_score(),
            )
        })
        .collect()
}

fn run_queries(collection: &Collection, query: &SearchQuery) -> (Duration, Vec<Doc>, f64) {
    let before = collection
        .stats_snapshot()
        .expect("statistics must be readable")
        .candidates_scanned;
    let mut last = collection
        .query(black_box(query))
        .expect("warmup query must succeed");
    black_box(&last);
    let mut durations = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let started = Instant::now();
        for _ in 0..QUERIES {
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
    let operations = ROUNDS.saturating_mul(QUERIES).saturating_add(1);
    let candidates = after.saturating_sub(before);
    (
        durations[ROUNDS / 2],
        last,
        count_to_f64(usize::try_from(candidates).unwrap_or(usize::MAX)) / count_to_f64(operations),
    )
}

#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_query(duration: Duration) -> f64 {
    duration.as_micros() as f64 / count_to_f64(QUERIES)
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_operation(duration: Duration, operations: usize) -> f64 {
    duration.as_micros() as f64 / operations as f64
}

fn benchmark_updates(scan: &Collection, indexed: &Collection) {
    let warmup = replacement_body("warmup");
    black_box(scan.update(&[&warmup]).expect("scan warmup must succeed"));
    black_box(
        indexed
            .update(&[&warmup])
            .expect("indexed warmup must succeed"),
    );
    let started = Instant::now();
    for write in 0..WRITES {
        let doc = replacement_body(&format!("epoch{write}"));
        black_box(
            scan.update(&[black_box(&doc)])
                .expect("scan update must succeed"),
        );
    }
    let scan_update = started.elapsed();
    let started = Instant::now();
    for write in 0..WRITES {
        let doc = replacement_body(&format!("epoch{write}"));
        black_box(
            indexed
                .update(&[black_box(&doc)])
                .expect("indexed update must succeed"),
        );
    }
    let indexed_update = started.elapsed();
    assert_eq!(
        indexed
            .query(&query("epoch63"))
            .expect("updated term must be searchable")[0]
            .get_pk(),
        Some("doc-000000")
    );
    let started = Instant::now();
    indexed
        .rebuild_index("body")
        .expect("FTS rebuild must succeed");
    let rebuild = started.elapsed();

    println!("operation,documents,samples,micros_per_operation");
    println!(
        "scan_body_update,{DOCUMENTS},{WRITES},{:.2}",
        micros_per_operation(scan_update, WRITES)
    );
    println!(
        "indexed_fts_update,{DOCUMENTS},{WRITES},{:.2}",
        micros_per_operation(indexed_update, WRITES)
    );
    println!(
        "full_fts_rebuild,{DOCUMENTS},1,{:.2}",
        micros_per_operation(rebuild, 1)
    );
}

fn main() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let scan = Collection::create(
        temporary
            .path()
            .join("scan")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema("scan"),
        Some(&options),
    )
    .expect("scan collection must be created");
    let indexed = Collection::create(
        temporary
            .path()
            .join("indexed")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema("indexed"),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    scan.insert(&refs).expect("scan fixture must be inserted");
    indexed
        .insert(&refs)
        .expect("indexed fixture must be inserted");
    indexed
        .create_index(
            "body",
            &IndexParams::fts(Some("standard"), None, None).expect("FTS descriptor must be valid"),
        )
        .expect("FTS index must build");

    println!("case,mode,documents,queries,rounds,candidates_per_query,micros_per_query");
    for (name, expression) in CASES {
        let query = query(expression);
        let (scan_time, scan_results, scan_candidates) = run_queries(&scan, &query);
        let (indexed_time, indexed_results, indexed_candidates) = run_queries(&indexed, &query);
        assert_eq!(
            comparable_results(&indexed_results),
            comparable_results(&scan_results),
            "indexed BM25 must match scan BM25 for {name}"
        );
        println!(
            "{name},scan,{DOCUMENTS},{QUERIES},{ROUNDS},{scan_candidates:.2},{:.2}",
            micros_per_query(scan_time)
        );
        println!(
            "{name},indexed,{DOCUMENTS},{QUERIES},{ROUNDS},{indexed_candidates:.2},{:.2}",
            micros_per_query(indexed_time)
        );
    }

    benchmark_updates(&scan, &indexed);
}
