//! Deterministic concurrent-query latency and throughput evidence.
//!
//! This benchmark complements the single-thread feature matrix. It keeps the
//! fixture small enough for CI, but starts all workers together and reports
//! nearest-rank p50/p95/p99 over every query sample plus wall-clock throughput.
//! It is an observation, not a hardware-independent SLO.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, HnswQueryParams,
    IndexParams, MetricType, SearchQuery,
};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempdir;

#[derive(Clone, Copy, Debug)]
struct Config {
    documents: usize,
    dimensions: usize,
    queries: usize,
    rounds: usize,
}

impl Config {
    fn from_environment() -> Self {
        if std::env::var("A3S_VEC_BENCH_SCALE").as_deref() == Ok("smoke") {
            return Self {
                documents: 96,
                dimensions: 8,
                queries: 8,
                rounds: 2,
            };
        }
        Self {
            documents: env_usize("A3S_VEC_CONCURRENT_DOCUMENTS", 2_000),
            dimensions: env_usize("A3S_VEC_CONCURRENT_DIMENSIONS", 32),
            queries: env_usize("A3S_VEC_CONCURRENT_QUERIES", 48),
            rounds: env_usize("A3S_VEC_CONCURRENT_ROUNDS", 5),
        }
    }

    fn worker_counts() -> Vec<usize> {
        let raw = std::env::var("A3S_VEC_CONCURRENCY").unwrap_or_else(|_| "1,2,4,8".into());
        let mut counts: Vec<usize> = raw
            .split(',')
            .filter_map(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .collect();
        counts.sort_unstable();
        counts.dedup();
        if counts.is_empty() {
            vec![1]
        } else {
            counts
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

#[derive(Debug)]
struct Measurement {
    elapsed: Duration,
    samples: Vec<Duration>,
    hits: usize,
}

fn vector_for(index: usize, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|dimension| {
            let raw = (index * 31 + dimension * 17 + index * dimension * 3) % 1_009;
            let raw = u16::try_from(raw).expect("fixture value fits u16");
            (f32::from(raw) - 504.0) / 504.0
        })
        .collect()
}

fn query_vectors(config: Config) -> Vec<Vec<f32>> {
    (0..config.queries)
        .map(|index| vector_for((index * 37 + 11) % config.documents, config.dimensions))
        .collect()
}

fn schema(config: Config) -> CollectionSchema {
    CollectionSchema::builder("concurrent-query-bench")
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            u32::try_from(config.dimensions).expect("dimension fits u32"),
            IndexParams::flat(MetricType::Cosine).expect("flat index parameters must be valid"),
        )
        .build()
        .expect("schema must be valid")
}

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn build_collection(path: &str, config: Config) -> Collection {
    let collection = Collection::create(path, &schema(config), Some(&options()))
        .expect("collection must be created");
    let docs: Vec<Doc> = (0..config.documents)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}"))
                .expect("document primary key must be valid");
            doc.add_vector_f32("embedding", &vector_for(index, config.dimensions))
                .expect("vector must be valid");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("fixture must be inserted");
    collection
}

fn ids(results: &[Doc]) -> Vec<String> {
    results
        .iter()
        .map(|doc| {
            doc.get_pk()
                .expect("result must have a primary key")
                .to_string()
        })
        .collect()
}

fn exact_rankings(collection: &Collection, queries: &[Vec<f32>]) -> Vec<Vec<String>> {
    queries
        .iter()
        .map(|vector| {
            let query = SearchQuery::new("embedding", vector, 10).expect("query must be valid");
            ids(&collection.query(&query).expect("exact query must succeed"))
        })
        .collect()
}

fn percentile(samples: &[Duration], percentage: usize) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * percentage).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len().saturating_sub(1))]
}

fn count_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark count fits u32"))
}

fn run_concurrent(
    collection: &Collection,
    queries: &[Vec<f32>],
    expected: &[Vec<String>],
    workers: usize,
    rounds: usize,
) -> Measurement {
    let collection = Arc::new(collection.clone());
    let barrier = Arc::new(Barrier::new(workers));
    let started = Instant::now();
    thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let collection = Arc::clone(&collection);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    let mut samples = Vec::with_capacity(queries.len() * rounds);
                    let mut hits = 0usize;
                    for _ in 0..rounds {
                        for (index, vector) in queries.iter().enumerate() {
                            let mut query = SearchQuery::new("embedding", vector, 10)
                                .expect("query must be valid");
                            query
                                .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
                                .expect("HNSW controls must be valid");
                            let query_started = Instant::now();
                            let results = collection
                                .query(black_box(&query))
                                .expect("concurrent query must succeed");
                            samples.push(query_started.elapsed());
                            hits += ids(&results)
                                .iter()
                                .filter(|id| expected[index].contains(id))
                                .count();
                        }
                    }
                    (samples, hits)
                })
            })
            .collect();
        let mut samples = Vec::with_capacity(workers * queries.len() * rounds);
        let mut hits = 0usize;
        for handle in handles {
            let (worker_samples, worker_hits) = handle.join().expect("worker must not panic");
            samples.extend(worker_samples);
            hits += worker_hits;
        }
        Measurement {
            elapsed: started.elapsed(),
            samples,
            hits,
        }
    })
}

fn main() {
    let config = Config::from_environment();
    assert!(config.documents >= 10);
    assert!(config.queries > 0 && config.rounds > 0);
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary
        .path()
        .join("collection")
        .to_str()
        .expect("benchmark path must be UTF-8")
        .to_owned();
    let collection = build_collection(&path, config);
    let queries = query_vectors(config);
    let expected = exact_rankings(&collection, &queries);
    let index_started = Instant::now();
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW parameters must be valid"),
        )
        .expect("HNSW index must build");
    let index_build_ms = index_started.elapsed().as_secs_f64() * 1_000.0;
    let _ = collection
        .query(&SearchQuery::new("embedding", &queries[0], 10).expect("warmup query must be valid"))
        .expect("warmup query must succeed");

    println!(
        "mode,workers,documents,dimensions,queries,rounds,index_build_ms,recall_at_10,p50_us,p95_us,p99_us,qps"
    );
    for workers in Config::worker_counts() {
        let measurement = run_concurrent(&collection, &queries, &expected, workers, config.rounds);
        let total = config.queries * config.rounds * workers;
        assert_eq!(measurement.samples.len(), total);
        let p50 = percentile(&measurement.samples, 50);
        let p95 = percentile(&measurement.samples, 95);
        let p99 = percentile(&measurement.samples, 99);
        assert!(p50 <= p95 && p95 <= p99);
        let recall = count_to_f64(measurement.hits) / count_to_f64(total * 10);
        let qps = count_to_f64(total) / measurement.elapsed.as_secs_f64();
        assert!(recall.is_finite() && (0.0..=1.0).contains(&recall));
        assert!(qps.is_finite() && qps > 0.0);
        println!(
            "hnsw,{workers},{},{},{},{},{index_build_ms:.3},{recall:.4},{:.2},{:.2},{:.2},{qps:.2}",
            config.documents,
            config.dimensions,
            config.queries,
            config.rounds,
            p50.as_secs_f64() * 1_000_000.0,
            p95.as_secs_f64() * 1_000_000.0,
            p99.as_secs_f64() * 1_000_000.0,
        );
    }
}
