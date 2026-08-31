//! Reproducible ANN recall/latency fixture.

use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, HnswQueryParams, IndexParams,
    IvfQueryParams, MetricType, SearchQuery,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DOCUMENTS: usize = 2_000;
const DIMENSIONS: usize = 32;
const QUERIES: usize = 48;
const ROUNDS: usize = 5;
const TOPK: usize = 10;

struct Measurement {
    median_round: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    rankings: Vec<Vec<String>>,
}

fn vector_for(index: usize) -> Vec<f32> {
    (0..DIMENSIONS)
        .map(|dimension| {
            let raw = (index * 31 + dimension * 17 + index * dimension * 3) % 1_009;
            let raw = u16::try_from(raw).expect("fixture value fits u16");
            (f32::from(raw) - 504.0) / 504.0
        })
        .collect()
}

fn query_vectors() -> Vec<Vec<f32>> {
    (0..QUERIES)
        .map(|index| vector_for((index * 37 + 11) % DOCUMENTS))
        .collect()
}

fn run_queries(
    collection: &Collection,
    vectors: &[Vec<f32>],
    configure: impl Fn(&mut SearchQuery),
) -> Measurement {
    if let Some(vector) = vectors.first() {
        let mut warmup = SearchQuery::new(
            "embedding",
            vector,
            i32::try_from(TOPK).expect("TOPK fits i32"),
        )
        .expect("warmup query must be valid");
        configure(&mut warmup);
        black_box(
            collection
                .query(&warmup)
                .expect("warmup query must succeed"),
        );
    }
    let mut round_durations = Vec::with_capacity(ROUNDS);
    let mut query_durations = Vec::with_capacity(ROUNDS.saturating_mul(vectors.len()));
    let mut rankings = Vec::new();
    for _ in 0..ROUNDS {
        let started = Instant::now();
        rankings = vectors
            .iter()
            .map(|vector| {
                let mut query = SearchQuery::new(
                    "embedding",
                    vector,
                    i32::try_from(TOPK).expect("TOPK fits i32"),
                )
                .expect("benchmark query must be valid");
                configure(&mut query);
                let query_started = Instant::now();
                let result = collection
                    .query(black_box(&query))
                    .expect("benchmark query must succeed");
                query_durations.push(query_started.elapsed());
                result
                    .into_iter()
                    .map(|doc| doc.get_pk().expect("result must have an id").to_string())
                    .collect()
            })
            .collect();
        round_durations.push(started.elapsed());
    }
    round_durations.sort_unstable();
    query_durations.sort_unstable();
    Measurement {
        median_round: round_durations[ROUNDS / 2],
        p50: percentile(&query_durations, 50),
        p95: percentile(&query_durations, 95),
        p99: percentile(&query_durations, 99),
        rankings,
    }
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let rank = samples.len().saturating_mul(percentile).saturating_add(99) / 100;
    samples[rank.saturating_sub(1).min(samples.len().saturating_sub(1))]
}

fn recall(reference: &[Vec<String>], candidate: &[Vec<String>]) -> f64 {
    let hits = reference
        .iter()
        .zip(candidate)
        .map(|(expected, actual)| actual.iter().filter(|id| expected.contains(id)).count())
        .sum::<usize>();
    let total = reference.len().saturating_mul(TOPK);
    count_to_f64(hits) / count_to_f64(total)
}

#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_query(duration: Duration) -> f64 {
    micros(duration) / count_to_f64(QUERIES)
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn estimated_payload_bytes(collection: &Collection) -> u64 {
    collection
        .stats_snapshot()
        .expect("statistics must be available")
        .indexes
        .into_iter()
        .find(|index| index.name == "embedding")
        .and_then(|index| index.estimated_payload_bytes)
        .unwrap_or_default()
}

fn main() {
    let temporary = tempdir().expect("temporary directory must be available");
    let dimension = u32::try_from(DIMENSIONS).expect("dimension fits u32");
    let schema = CollectionSchema::builder("ann-benchmark")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, dimension)
                .expect("field must be valid"),
        )
        .build()
        .expect("schema must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema,
        None,
    )
    .expect("collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("document must be valid");
            doc.add_vector_f32("embedding", &vector_for(index))
                .expect("vector must be valid");
            doc
        })
        .collect();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    let queries = query_vectors();

    let exact = run_queries(&collection, &queries, |_| {});
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index must build");
    let hnsw = run_queries(&collection, &queries, |query| {
        query
            .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
            .expect("HNSW controls must be valid");
    });
    let hnsw_payload = estimated_payload_bytes(&collection);

    collection
        .create_index(
            "embedding",
            &IndexParams::ivf(MetricType::Cosine, 64, 8, false)
                .expect("IVF descriptor must be valid"),
        )
        .expect("IVF index must build");
    let ivf = run_queries(&collection, &queries, |query| {
        query
            .set_ivf_params(IvfQueryParams::new(8, true, 8.0))
            .expect("IVF controls must be valid");
    });
    let ivf_payload = estimated_payload_bytes(&collection);

    println!(
        "mode,documents,dimensions,queries,rounds,recall_at_10,median_round_us,p50_us,p95_us,p99_us,estimated_payload_bytes"
    );
    println!(
        "exact,{DOCUMENTS},{DIMENSIONS},{QUERIES},{ROUNDS},1.0000,{:.2},{:.2},{:.2},{:.2},0",
        micros_per_query(exact.median_round),
        micros(exact.p50),
        micros(exact.p95),
        micros(exact.p99),
    );
    println!(
        "hnsw_ef64,{DOCUMENTS},{DIMENSIONS},{QUERIES},{ROUNDS},{:.4},{:.2},{:.2},{:.2},{:.2},{hnsw_payload}",
        recall(&exact.rankings, &hnsw.rankings),
        micros_per_query(hnsw.median_round),
        micros(hnsw.p50),
        micros(hnsw.p95),
        micros(hnsw.p99),
    );
    println!(
        "ivf_nprobe8,{DOCUMENTS},{DIMENSIONS},{QUERIES},{ROUNDS},{:.4},{:.2},{:.2},{:.2},{:.2},{ivf_payload}",
        recall(&exact.rankings, &ivf.rankings),
        micros_per_query(ivf.median_round),
        micros(ivf.p50),
        micros(ivf.p95),
        micros(ivf.p99),
    );
}
