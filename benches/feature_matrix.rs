//! Small, deterministic latency matrix for every public retrieval route.
//!
//! This is deliberately a standalone benchmark instead of a Criterion suite:
//! it keeps the crate's dependency graph unchanged and emits a CSV artifact
//! that can be archived by CI.  Every measured operation is also an assertion
//! gate; a disconnected route makes the benchmark fail rather than producing
//! a plausible-looking number.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc, Durability,
    FieldSchema, Fts, GroupBySearchQuery, HnswQueryParams, IndexParams, IoBackend, IvfQueryParams,
    IvfRabitqQueryParams, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tempfile::tempdir;

static MEASURED_OPERATIONS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn record_operation(name: &str) {
    MEASURED_OPERATIONS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("benchmark operation registry must not be poisoned")
        .push(name.to_string());
}

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
            Self {
                documents: 96,
                dimensions: 8,
                queries: 6,
                rounds: 2,
            }
        } else {
            Self {
                documents: 512,
                dimensions: 16,
                queries: 16,
                rounds: 3,
            }
        }
    }
}

#[derive(Debug)]
struct Measurement {
    samples: usize,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    work: u64,
}

impl Measurement {
    #[allow(clippy::cast_precision_loss)]
    fn work_per_second(&self) -> f64 {
        let seconds = self.p50.as_secs_f64();
        if seconds == 0.0 {
            f64::INFINITY
        } else {
            self.work as f64 / (seconds * self.samples as f64)
        }
    }
}

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("benchmark options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
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

fn binary_for(index: usize, byte_length: usize) -> Vec<u8> {
    (0..byte_length)
        .map(|offset| {
            let value = (index * 37 + offset * 101 + index * offset * 13) % 256;
            u8::try_from(value).expect("binary fixture value fits u8")
        })
        .collect()
}

fn full_document(index: usize, dimensions: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("benchmark id must be valid");
    doc.add_string("category", if index % 2 == 0 { "even" } else { "odd" })
        .expect("category must be valid");
    doc.add_i32(
        "bucket",
        i32::try_from(index % 16).expect("bucket fits i32"),
    )
    .expect("bucket must be valid");
    doc.add_string(
        "body",
        &format!(
            "rust workspace vector retrieval document group{} token{}",
            index % 8,
            index % 17
        ),
    )
    .expect("body must be valid");
    doc.add_vector_f32("embedding", &vector_for(index, dimensions))
        .expect("dense vector must be valid");
    let first = u32::try_from(index % 32).expect("sparse index fits u32");
    let second = (first + 7) % 32;
    doc.add_sparse_vector("sparse", &[first, second], &[1.0, 0.25])
        .expect("sparse vector must be valid");
    doc.add_vector_binary32("bits32", &binary_for(index, 4))
        .expect("Binary32 vector must be valid");
    doc.add_vector_binary64("bits64", &binary_for(index, 8))
        .expect("Binary64 vector must be valid");
    doc
}

fn ann_document(index: usize, dimensions: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("benchmark id must be valid");
    doc.add_vector_f32("embedding", &vector_for(index, dimensions))
        .expect("dense vector must be valid");
    doc
}

fn full_schema(dimensions: usize) -> CollectionSchema {
    let mut category =
        FieldSchema::new("category", DataType::String, false, 0).expect("category schema");
    category
        .set_index_params(&IndexParams::invert(true, true).expect("scalar index"))
        .expect("category index");
    let mut bucket = FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket schema");
    bucket
        .set_index_params(&IndexParams::invert(true, false).expect("scalar index"))
        .expect("bucket index");
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body schema");
    body.set_index_params(
        &IndexParams::fts(Some("standard"), Some(&["lowercase"]), None).expect("FTS index"),
    )
    .expect("body index");
    let mut bits32 =
        FieldSchema::new("bits32", DataType::VectorBinary32, false, 32).expect("Binary32 schema");
    bits32
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("binary Flat index"))
        .expect("Binary32 Flat index");
    let mut bits64 =
        FieldSchema::new("bits64", DataType::VectorBinary64, false, 64).expect("Binary64 schema");
    bits64
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("binary Flat index"))
        .expect("Binary64 Flat index");
    CollectionSchema::builder("feature-matrix-bench")
        .add_field(category)
        .add_field(bucket)
        .add_field(body)
        .add_field(
            FieldSchema::new(
                "embedding",
                DataType::VectorFp32,
                false,
                u32::try_from(dimensions).expect("dimension fits u32"),
            )
            .expect("embedding schema"),
        )
        .add_field(
            FieldSchema::new("sparse", DataType::SparseVectorFp32, false, 32)
                .expect("sparse schema"),
        )
        .add_field(bits32)
        .add_field(bits64)
        .build()
        .expect("full schema must be valid")
}

fn ann_schema(dimensions: usize) -> CollectionSchema {
    CollectionSchema::builder("ann-feature-matrix-bench")
        .add_field(
            FieldSchema::new(
                "embedding",
                DataType::VectorFp32,
                false,
                u32::try_from(dimensions).expect("dimension fits u32"),
            )
            .expect("embedding schema"),
        )
        .build()
        .expect("ANN schema must be valid")
}

fn insert_full_fixture(collection: &Collection, config: Config) {
    let docs: Vec<Doc> = (0..config.documents)
        .map(|index| full_document(index, config.dimensions))
        .collect();
    let inserted = collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("full fixture insert must succeed");
    assert_eq!(
        inserted.success_count,
        u64::try_from(config.documents).expect("document count fits u64")
    );
}

fn insert_ann_fixture(collection: &Collection, config: Config) {
    let docs: Vec<Doc> = (0..config.documents)
        .map(|index| ann_document(index, config.dimensions))
        .collect();
    let inserted = collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("ANN fixture insert must succeed");
    assert_eq!(
        inserted.success_count,
        u64::try_from(config.documents).expect("document count fits u64")
    );
}

fn percentile(samples: &[Duration], percentage: usize) -> Duration {
    let rank = samples.len().saturating_mul(percentage).saturating_add(99) / 100;
    samples[rank.saturating_sub(1).min(samples.len().saturating_sub(1))]
}

fn measure<F>(name: &str, config: Config, samples: usize, mut operation: F) -> Measurement
where
    F: FnMut(usize) -> u64,
{
    record_operation(name);
    assert!(samples > 0, "{name} must collect at least one sample");
    let _ = operation(usize::MAX);
    let mut durations = Vec::with_capacity(samples);
    let mut work = 0_u64;
    for sample in 0..samples {
        let started = Instant::now();
        let completed = operation(sample);
        let elapsed = started.elapsed();
        assert!(completed > 0, "{name} completed no work");
        work = work.saturating_add(completed);
        durations.push(elapsed);
    }
    durations.sort_unstable();
    let measurement = Measurement {
        samples,
        p50: percentile(&durations, 50),
        p95: percentile(&durations, 95),
        p99: percentile(&durations, 99),
        work,
    };
    assert!(!measurement.p50.is_zero(), "{name} p50 must be positive");
    assert!(
        measurement.p50 <= measurement.p95 && measurement.p95 <= measurement.p99,
        "{name} percentiles must be monotonic"
    );
    assert!(
        measurement.work_per_second().is_finite()
            && measurement.work_per_second().is_sign_positive(),
        "{name} throughput must be finite and positive"
    );
    println!(
        "{name},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
        config.documents,
        config.dimensions,
        samples,
        measurement.work,
        micros(measurement.p50),
        micros(measurement.p95),
        micros(measurement.p99),
        measurement.work_per_second(),
        f64::from(u32::try_from(measurement.work).expect("benchmark work fits u32"))
            / f64::from(u32::try_from(samples).expect("benchmark samples fit u32")),
    );
    measurement
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

include!("feature_matrix/routes.rs");
include!("feature_matrix/ann.rs");
include!("feature_matrix/lifecycle.rs");
