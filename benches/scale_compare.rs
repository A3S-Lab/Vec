//! Configurable larger-corpus vector benchmark for cross-project comparisons.
//!
//! The release CI intentionally uses the bounded `ann_recall` and feature
//! fixtures. This benchmark is a separate, reproducible measurement tool for
//! comparing a3s-vec with another engine at a chosen corpus size. It emits
//! machine-readable rows and never treats a timing value as a correctness gate.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, HnswQueryParams,
    IndexParams, MetricType, SearchQuery,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const TOPK: i32 = 10;
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

#[derive(Clone, Copy, Debug)]
struct Config {
    documents: usize,
    dimensions: usize,
    queries: usize,
    rounds: usize,
    batch_size: usize,
    ef_search: i32,
    hnsw_m: i32,
    ef_construction: i32,
}

impl Config {
    fn from_environment() -> Self {
        let smoke = std::env::var("A3S_VEC_BENCH_SCALE").as_deref() == Ok("smoke");
        let config = if smoke {
            Self {
                documents: 96,
                dimensions: 8,
                queries: 8,
                rounds: 2,
                batch_size: 32,
                ef_search: 64,
                hnsw_m: 16,
                ef_construction: 96,
            }
        } else {
            Self {
                documents: env_usize("A3S_VEC_SCALE_DOCUMENTS", 10_000),
                dimensions: env_usize("A3S_VEC_SCALE_DIMENSIONS", 128),
                queries: env_usize("A3S_VEC_SCALE_QUERIES", 32),
                rounds: env_usize("A3S_VEC_SCALE_ROUNDS", 3),
                batch_size: env_usize("A3S_VEC_SCALE_BATCH_SIZE", 512),
                ef_search: env_i32("A3S_VEC_SCALE_EF_SEARCH", 64),
                hnsw_m: env_i32("A3S_VEC_SCALE_HNSW_M", 16),
                ef_construction: env_i32("A3S_VEC_SCALE_EF_CONSTRUCTION", 96),
            }
        };
        assert!(
            config.documents >= TOPK as usize,
            "documents must be at least 10"
        );
        assert!(config.dimensions > 0, "dimensions must be positive");
        assert!(config.queries > 0, "queries must be positive");
        assert!(config.rounds > 0, "rounds must be positive");
        assert!(config.batch_size > 0, "batch size must be positive");
        assert!(config.ef_search > 0, "ef search must be positive");
        assert!(config.hnsw_m > 0, "HNSW m must be positive");
        assert!(
            config.ef_construction > 0,
            "HNSW ef construction must be positive"
        );
        config
    }

    fn modes() -> Vec<Mode> {
        match std::env::var("A3S_VEC_SCALE_MODE")
            .unwrap_or_else(|_| "both".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "flat" => vec![Mode::Flat],
            "hnsw" => vec![Mode::Hnsw],
            "both" => vec![Mode::Flat, Mode::Hnsw],
            value => panic!("A3S_VEC_SCALE_MODE must be flat, hnsw, or both; got {value}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Flat,
    Hnsw,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Hnsw => "hnsw",
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn splitmix64(value: u64) -> u64 {
    let mut state = value.wrapping_add(SPLITMIX_INCREMENT);
    state = (state ^ (state >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_1);
    state = (state ^ (state >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_2);
    state ^ (state >> 31)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn vector_for(index: usize, dimensions: usize) -> Vec<f32> {
    let index = u64::try_from(index).expect("document index must fit u64");
    (0..dimensions)
        .map(|dimension| {
            let dimension = u64::try_from(dimension).expect("dimension must fit u64");
            let seed = index
                .wrapping_mul(0xD6E8_FEB8_6659_FD93)
                .wrapping_add(dimension.wrapping_mul(0xA5A3_564D_9C4F_1B27));
            // Keep the conversion identical to the companion Python harness:
            // 53 random bits become a [0, 1) double and then an f32 value.
            // Use separate multiply/subtract operations rather than a fused
            // multiply-add so the two language runtimes round at the same
            // f64 boundary before narrowing to f32.
            let unit = (splitmix64(seed) >> 11) as f64 / 9_007_199_254_740_992.0;
            (unit * 2.0 - 1.0) as f32
        })
        .collect()
}

fn query_vectors(config: Config) -> Vec<Vec<f32>> {
    (0..config.queries)
        .map(|query| {
            let query = u64::try_from(query).expect("query index must fit u64");
            let documents = u64::try_from(config.documents).expect("documents must fit u64");
            let index = query
                .wrapping_mul(7_919)
                .wrapping_add(17)
                .wrapping_rem(documents);
            vector_for(
                usize::try_from(index).expect("query document index must fit usize"),
                config.dimensions,
            )
        })
        .collect()
}

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("benchmark options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn schema(config: Config) -> CollectionSchema {
    CollectionSchema::builder("scale-compare")
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            u32::try_from(config.dimensions).expect("dimensions must fit u32"),
            IndexParams::flat(MetricType::Cosine).expect("flat parameters must be valid"),
        )
        .build()
        .expect("benchmark schema must be valid")
}

fn build_collection(path: &str, config: Config) -> (Collection, Duration) {
    let collection = Collection::create(path, &schema(config), Some(&options()))
        .expect("benchmark collection must be created");
    let started = Instant::now();
    let mut first = 0usize;
    while first < config.documents {
        let last = first
            .saturating_add(config.batch_size)
            .min(config.documents);
        let documents: Vec<Doc> = (first..last)
            .map(|index| {
                let mut document = Doc::with_pk(format!("doc-{index:08}"))
                    .expect("benchmark document id must be valid");
                document
                    .add_vector_f32("embedding", &vector_for(index, config.dimensions))
                    .expect("benchmark vector must be valid");
                document
            })
            .collect();
        let references: Vec<&Doc> = documents.iter().collect();
        collection
            .insert(&references)
            .expect("benchmark batch must be inserted");
        first = last;
    }
    // Keep the initial checkpoint inside the load measurement. The companion
    // zvec harness applies the same insert-plus-flush boundary, while the
    // index-build column remains a separate post-load lifecycle measure.
    collection.flush().expect("benchmark collection must flush");
    (collection, started.elapsed())
}

fn ids(results: &[Doc]) -> Vec<String> {
    results
        .iter()
        .map(|document| {
            document
                .get_pk()
                .expect("benchmark result must have a primary key")
                .to_string()
        })
        .collect()
}

#[derive(Debug)]
struct Measurement {
    elapsed: Duration,
    samples: Vec<Duration>,
    rankings: Vec<Vec<String>>,
}

fn run_queries(
    collection: &Collection,
    queries: &[Vec<f32>],
    config: Config,
    mode: Mode,
) -> Measurement {
    let mut warmup =
        SearchQuery::new("embedding", &queries[0], TOPK).expect("warmup query must be valid");
    if matches!(mode, Mode::Hnsw) {
        warmup
            .set_hnsw_params(HnswQueryParams::new(config.ef_search, 0.0, false, false))
            .expect("HNSW controls must be valid");
    }
    black_box(
        collection
            .query(&warmup)
            .expect("warmup query must succeed"),
    );

    let started = Instant::now();
    let mut samples = Vec::with_capacity(config.queries.saturating_mul(config.rounds));
    let mut rankings = Vec::with_capacity(config.queries);
    for _ in 0..config.rounds {
        rankings = queries
            .iter()
            .map(|vector| {
                let mut query = SearchQuery::new("embedding", vector, TOPK)
                    .expect("benchmark query must be valid");
                if matches!(mode, Mode::Hnsw) {
                    query
                        .set_hnsw_params(HnswQueryParams::new(config.ef_search, 0.0, false, false))
                        .expect("HNSW controls must be valid");
                }
                let query_started = Instant::now();
                let result = collection
                    .query(black_box(&query))
                    .expect("benchmark query must succeed");
                samples.push(query_started.elapsed());
                ids(&result)
            })
            .collect();
    }
    Measurement {
        elapsed: started.elapsed(),
        samples,
        rankings,
    }
}

fn percentile(samples: &[Duration], percentage: usize) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = sorted.len().saturating_mul(percentage).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len().saturating_sub(1))]
}

fn recall_at_topk(expected: &[Vec<String>], actual: &[Vec<String>]) -> f64 {
    let hits = expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| actual.iter().filter(|id| expected.contains(id)).count())
        .sum::<usize>();
    let total = expected.len().saturating_mul(TOPK as usize);
    #[allow(clippy::cast_precision_loss)]
    {
        hits as f64 / total as f64
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn microseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn print_row(
    mode: Mode,
    config: Config,
    insert: Duration,
    index_build: Duration,
    measurement: &Measurement,
    recall: f64,
) {
    let p50 = percentile(&measurement.samples, 50);
    let p95 = percentile(&measurement.samples, 95);
    let p99 = percentile(&measurement.samples, 99);
    let total_queries = config.queries.saturating_mul(config.rounds).max(1);
    #[allow(clippy::cast_precision_loss)]
    let qps = total_queries as f64 / measurement.elapsed.as_secs_f64();
    assert!(
        p50 <= p95 && p95 <= p99,
        "latency percentiles must be ordered"
    );
    assert!(recall.is_finite() && (0.0..=1.0).contains(&recall));
    assert!(qps.is_finite() && qps > 0.0);
    println!(
        "a3s-vec,{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},0.000,{:.3},{:.4},{:.3},{:.3},{:.3},{:.2}",
        env!("CARGO_PKG_VERSION"),
        mode.name(),
        config.documents,
        config.dimensions,
        config.queries,
        config.rounds,
        config.batch_size,
        config.ef_search,
        config.hnsw_m,
        config.ef_construction,
        milliseconds(insert),
        milliseconds(index_build),
        milliseconds(insert) + milliseconds(index_build),
        recall,
        microseconds(p50),
        microseconds(p95),
        microseconds(p99),
        qps,
    );
}

fn main() {
    let config = Config::from_environment();
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary
        .path()
        .join("collection")
        .to_str()
        .expect("benchmark path must be UTF-8")
        .to_owned();
    let (collection, insert) = build_collection(&path, config);
    let queries = query_vectors(config);
    let exact = run_queries(&collection, &queries, config, Mode::Flat);
    let expected = exact.rankings.clone();

    println!(
        "engine,version,mode,documents,dimensions,queries,rounds,batch_size,ef_search,hnsw_m,ef_construction,insert_ms,index_build_ms,optimize_ms,total_build_ms,recall_at_10,p50_us,p95_us,p99_us,qps"
    );
    for mode in Config::modes() {
        match mode {
            Mode::Flat => print_row(mode, config, insert, Duration::ZERO, &exact, 1.0),
            Mode::Hnsw => {
                let started = Instant::now();
                collection
                    .create_index(
                        "embedding",
                        &IndexParams::hnsw(
                            MetricType::Cosine,
                            config.hnsw_m,
                            config.ef_construction,
                        )
                        .expect("HNSW parameters must be valid"),
                    )
                    .expect("HNSW index must build");
                let index_build = started.elapsed();
                let measurement = run_queries(&collection, &queries, config, mode);
                let recall = recall_at_topk(&expected, &measurement.rankings);
                print_row(mode, config, insert, index_build, &measurement, recall);
            }
        }
    }
}
