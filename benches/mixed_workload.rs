//! Deterministic mixed read/write contention evidence.
//!
//! The fixture keeps vectors immutable while a writer updates a scalar field.
//! That makes the flat-oracle recall comparison stable and isolates collection
//! publication/locking cost from changing nearest-neighbour ground truth.
//! Vector-index mutation is covered separately by `incremental_write`.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    HnswQueryParams, IndexParams, MetricType, SearchQuery,
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
    writes: usize,
}

impl Config {
    fn from_environment() -> Self {
        if std::env::var("A3S_VEC_BENCH_SCALE").as_deref() == Ok("smoke") {
            return Self {
                documents: 96,
                dimensions: 8,
                queries: 8,
                rounds: 2,
                writes: 24,
            };
        }
        Self {
            documents: env_usize("A3S_VEC_MIXED_DOCUMENTS", 2_000),
            dimensions: env_usize("A3S_VEC_MIXED_DIMENSIONS", 32),
            queries: env_usize("A3S_VEC_MIXED_QUERIES", 48),
            rounds: env_usize("A3S_VEC_MIXED_ROUNDS", 5),
            writes: env_usize("A3S_VEC_MIXED_WRITES", 192),
        }
    }

    fn reader_counts() -> Vec<usize> {
        let raw = std::env::var("A3S_VEC_MIXED_READERS").unwrap_or_else(|_| "1,2,4,8".into());
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
    read_samples: Vec<Duration>,
    write_samples: Vec<Duration>,
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
    let epoch =
        FieldSchema::new("epoch", DataType::Int32, false, 0).expect("epoch field must be valid");
    CollectionSchema::builder("mixed-workload-bench")
        .add_field(epoch)
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            u32::try_from(config.dimensions).expect("dimension fits u32"),
            IndexParams::flat(MetricType::Cosine).expect("flat index parameters must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("collection options must be valid");
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
            doc.add_i32("epoch", 0).expect("epoch must be valid");
            doc.add_vector_f32("embedding", &vector_for(index, config.dimensions))
                .expect("vector must be valid");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection
        .insert(&refs)
        .expect("fixture documents must be inserted");
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

#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: usize) -> f64 {
    value as f64
}

fn run_mixed(
    collection: &Collection,
    queries: &[Vec<f32>],
    expected: &[Vec<String>],
    documents: usize,
    readers: usize,
    rounds: usize,
    writes: usize,
) -> Measurement {
    let collection = Arc::new(collection.clone());
    let barrier = Arc::new(Barrier::new(readers + 1));
    let started = Instant::now();
    thread::scope(|scope| {
        let writer_collection = Arc::clone(&collection);
        let writer_barrier = Arc::clone(&barrier);
        let writer_handle = scope.spawn(move || {
            writer_barrier.wait();
            let mut samples = Vec::with_capacity(writes);
            for operation in 0..writes {
                let index = operation % documents;
                let mut patch = Doc::with_pk(format!("doc-{index:05}"))
                    .expect("writer primary key must be valid");
                patch
                    .add_i32(
                        "epoch",
                        i32::try_from(operation + 1).expect("operation fits i32"),
                    )
                    .expect("writer field must be valid");
                let batch = [&patch];
                let operation_started = Instant::now();
                let result = writer_collection
                    .update(black_box(&batch))
                    .expect("mixed-workload update must succeed");
                samples.push(operation_started.elapsed());
                assert_eq!(result.success_count, 1);
            }
            samples
        });

        let query_handles: Vec<_> = (0..readers)
            .map(|_| {
                let reader_collection = Arc::clone(&collection);
                let reader_barrier = Arc::clone(&barrier);
                let query_ref = queries;
                let expected_ref = expected;
                scope.spawn(move || {
                    reader_barrier.wait();
                    let mut samples = Vec::with_capacity(query_ref.len() * rounds);
                    let mut hits = 0usize;
                    for _ in 0..rounds {
                        for (index, vector) in query_ref.iter().enumerate() {
                            let mut query = SearchQuery::new("embedding", vector, 10)
                                .expect("query must be valid");
                            query
                                .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
                                .expect("HNSW controls must be valid");
                            let query_started = Instant::now();
                            let results = reader_collection
                                .query(black_box(&query))
                                .expect("mixed-workload query must succeed");
                            samples.push(query_started.elapsed());
                            let result_ids = ids(&results);
                            assert!(!result_ids.is_empty());
                            hits += result_ids
                                .iter()
                                .filter(|id| expected_ref[index].contains(id))
                                .count();
                        }
                    }
                    (samples, hits)
                })
            })
            .collect();

        let write_samples = writer_handle.join().expect("writer must not panic");
        let mut read_samples = Vec::new();
        let mut hits = 0usize;
        for handle in query_handles {
            let (samples, worker_hits) = handle.join().expect("reader must not panic");
            read_samples.extend(samples);
            hits += worker_hits;
        }
        Measurement {
            elapsed: started.elapsed(),
            read_samples,
            write_samples,
            hits,
        }
    })
}

fn main() {
    let config = Config::from_environment();
    assert!(config.documents >= 10);
    assert!(config.queries > 0 && config.rounds > 0 && config.writes > 0);

    let temporary = tempdir().expect("temporary directory must be available");
    let queries = query_vectors(config);

    println!(
        "mode,readers,documents,dimensions,queries,rounds,writes,index_build_ms,recall_at_10,read_p50_us,read_p95_us,read_p99_us,write_p50_us,write_p95_us,write_p99_us,read_qps,write_qps,final_revision,accounted_bytes"
    );
    for readers in Config::reader_counts() {
        let collection_path = temporary
            .path()
            .join(format!("collection-{readers}"))
            .to_str()
            .expect("benchmark path must be UTF-8")
            .to_owned();
        let collection = build_collection(&collection_path, config);
        let expected = exact_rankings(&collection, &queries);
        let index_started = Instant::now();
        collection
            .create_index(
                "embedding",
                &IndexParams::hnsw(MetricType::Cosine, 16, 96)
                    .expect("HNSW parameters must be valid"),
            )
            .expect("HNSW index must build");
        let index_build_ms = index_started.elapsed().as_secs_f64() * 1_000.0;
        let _ = collection
            .query(
                &SearchQuery::new("embedding", &queries[0], 10)
                    .expect("warmup query must be valid"),
            )
            .expect("warmup query must succeed");
        let measurement = run_mixed(
            &collection,
            &queries,
            &expected,
            config.documents,
            readers,
            config.rounds,
            config.writes,
        );
        let read_total = config.queries * config.rounds * readers;
        assert_eq!(measurement.read_samples.len(), read_total);
        assert_eq!(measurement.write_samples.len(), config.writes);
        let read_p50 = percentile(&measurement.read_samples, 50);
        let read_p95 = percentile(&measurement.read_samples, 95);
        let read_p99 = percentile(&measurement.read_samples, 99);
        let write_p50 = percentile(&measurement.write_samples, 50);
        let write_p95 = percentile(&measurement.write_samples, 95);
        let write_p99 = percentile(&measurement.write_samples, 99);
        assert!(read_p50 <= read_p95 && read_p95 <= read_p99);
        assert!(write_p50 <= write_p95 && write_p95 <= write_p99);
        let recall = count_to_f64(measurement.hits) / count_to_f64(read_total * 10);
        let elapsed = measurement.elapsed.as_secs_f64();
        let read_qps = count_to_f64(read_total) / elapsed;
        let write_qps = count_to_f64(config.writes) / elapsed;
        assert!(recall.is_finite() && (0.0..=1.0).contains(&recall));
        assert!(read_qps.is_finite() && read_qps > 0.0);
        assert!(write_qps.is_finite() && write_qps > 0.0);
        let stats = collection
            .stats()
            .expect("mixed-workload stats must succeed");
        assert_eq!(stats.doc_count, config.documents as u64);
        assert!(stats.accounted_bytes > 0);
        println!(
            "mixed,{readers},{},{},{},{},{},{index_build_ms:.3},{recall:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{read_qps:.2},{write_qps:.2},{},{}",
            config.documents,
            config.dimensions,
            config.queries,
            config.rounds,
            config.writes,
            read_p50.as_secs_f64() * 1_000_000.0,
            read_p95.as_secs_f64() * 1_000_000.0,
            read_p99.as_secs_f64() * 1_000_000.0,
            write_p50.as_secs_f64() * 1_000_000.0,
            write_p95.as_secs_f64() * 1_000_000.0,
            write_p99.as_secs_f64() * 1_000_000.0,
            stats.revision,
            stats.accounted_bytes,
        );
    }
}
