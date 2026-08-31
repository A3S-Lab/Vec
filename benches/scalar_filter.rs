//! Reproducible scalar bitmap pre-filter latency fixture.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    IndexParams, SearchQuery,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DOCUMENTS: usize = 100_000;
const DIMENSIONS: usize = 4;
const QUERIES: usize = 32;
const ROUNDS: usize = 5;
const TOPK: usize = 10;
const LANGUAGES: [&str; 8] = [
    "rust",
    "typescript",
    "python",
    "go",
    "java",
    "kotlin",
    "swift",
    "ruby",
];
const CASES: [(&str, &str); 4] = [
    ("language_equality", "language == 'rust'"),
    ("recent_range", "modified_at >= 99900"),
    ("path_prefix", "path has_prefix 'src/pkg-099/'"),
    (
        "workspace_conjunction",
        "language == 'go' and path has_prefix 'src/pkg-099/' and modified_at >= 90000",
    ),
];

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn scalar_field(
    name: &str,
    data_type: DataType,
    indexed: bool,
    range: bool,
    wildcard: bool,
) -> FieldSchema {
    let mut field = FieldSchema::new(name, data_type, false, 0).expect("field must be valid");
    if indexed {
        field
            .set_index_params(
                &IndexParams::invert(range, wildcard).expect("descriptor must be valid"),
            )
            .expect("field must support an inverted index");
    }
    field
}

fn schema(name: &str, indexed: bool) -> CollectionSchema {
    CollectionSchema::builder(name)
        .add_field(scalar_field(
            "language",
            DataType::String,
            indexed,
            false,
            false,
        ))
        .add_field(scalar_field(
            "modified_at",
            DataType::Int64,
            indexed,
            true,
            false,
        ))
        .add_field(scalar_field("path", DataType::String, indexed, true, true))
        .add_field(
            FieldSchema::new(
                "embedding",
                DataType::VectorFp32,
                false,
                u32::try_from(DIMENSIONS).expect("dimension fits u32"),
            )
            .expect("vector field must be valid"),
        )
        .build()
        .expect("schema must be valid")
}

fn document(index: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:06}")).expect("document primary key must be valid");
    doc.add_string("language", LANGUAGES[index % LANGUAGES.len()])
        .expect("language must be valid");
    doc.add_i64(
        "modified_at",
        i64::try_from(index).expect("fixture index fits i64"),
    )
    .expect("timestamp must be valid");
    doc.add_string(
        "path",
        &format!("src/pkg-{:03}/file-{index:06}.rs", index % 1_000),
    )
    .expect("path must be valid");
    let coordinate = f32::from(u16::try_from(index % 997).expect("coordinate fits u16")) / 997.0;
    doc.add_vector_f32(
        "embedding",
        &[coordinate, coordinate * 0.5, 1.0 - coordinate, 1.0],
    )
    .expect("embedding must be valid");
    doc
}

fn query(filter: &str) -> SearchQuery {
    let mut query = SearchQuery::new(
        "embedding",
        &[0.25, 0.125, 0.75, 1.0],
        i32::try_from(TOPK).expect("TOPK fits i32"),
    )
    .expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query.set_filter(filter).expect("filter must be valid");
    query
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

fn main() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let fallback = Collection::create(
        temporary
            .path()
            .join("scan")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema("scan", false),
        Some(&options),
    )
    .expect("scan collection must be created");
    let indexed = Collection::create(
        temporary
            .path()
            .join("bitmap")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema("bitmap", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    fallback
        .insert(&refs)
        .expect("scan fixture must be inserted");
    indexed
        .insert(&refs)
        .expect("indexed fixture must be inserted");

    println!("case,mode,documents,dimensions,queries,rounds,candidates_per_query,micros_per_query");
    for (name, filter) in CASES {
        let query = query(filter);
        let (scan_time, scan_results, scan_candidates) = run_queries(&fallback, &query);
        let (bitmap_time, bitmap_results, bitmap_candidates) = run_queries(&indexed, &query);
        assert_eq!(
            comparable_results(&bitmap_results),
            comparable_results(&scan_results),
            "bitmap execution must match scan execution for {name}"
        );
        println!(
            "{name},scan,{DOCUMENTS},{DIMENSIONS},{QUERIES},{ROUNDS},{scan_candidates:.2},{:.2}",
            micros_per_query(scan_time)
        );
        println!(
            "{name},bitmap,{DOCUMENTS},{DIMENSIONS},{QUERIES},{ROUNDS},{bitmap_candidates:.2},{:.2}",
            micros_per_query(bitmap_time)
        );
    }
    let started = Instant::now();
    indexed
        .rebuild_index("language")
        .expect("scalar indexes must rebuild");
    let rebuild = started.elapsed();
    println!("operation,documents,indexes,micros_per_operation");
    println!(
        "full_scalar_rebuild,{DOCUMENTS},3,{:.2}",
        rebuild.as_secs_f64() * 1_000_000.0
    );
}
