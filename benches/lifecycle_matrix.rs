//! Deterministic lifecycle, mutation, schema, resource, and maintenance metrics.
//!
//! The public feature matrix focuses on retrieval routes.  This companion
//! matrix covers the management surface that surrounds those routes and emits
//! the same percentile/throughput shape, so a successful query benchmark
//! cannot hide a slow or disconnected lifecycle operation.

use a3s_vec::{
    AlterColumnOption, Collection, CollectionMaintenanceOptions, CollectionOptions,
    CollectionResourceLimits, CollectionSchema, DataType, Doc, Durability, ErrorCode, FieldSchema,
    IndexParams, MetricType,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

#[derive(Clone, Copy, Debug)]
struct Config {
    documents: usize,
    dimensions: usize,
    samples: usize,
}

impl Config {
    fn from_environment() -> Self {
        let smoke = std::env::var("A3S_VEC_BENCH_SCALE").as_deref() == Ok("smoke");
        let defaults = if smoke {
            Self {
                documents: 96,
                dimensions: 8,
                samples: 2,
            }
        } else {
            Self {
                documents: env_usize("A3S_VEC_LIFECYCLE_DOCUMENTS", 512),
                dimensions: env_usize("A3S_VEC_LIFECYCLE_DIMENSIONS", 16),
                samples: env_usize("A3S_VEC_LIFECYCLE_SAMPLES", 3),
            }
        };
        assert!(defaults.documents > 0, "documents must be positive");
        assert!(defaults.dimensions > 0, "dimensions must be positive");
        assert!(defaults.samples > 0, "samples must be positive");
        defaults
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn slot(value: usize) -> usize {
    if value == usize::MAX {
        0
    } else {
        value
    }
}

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn schema(name: &str, vector_index: bool, scalar_indexes: bool) -> CollectionSchema {
    let mut category =
        FieldSchema::new("category", DataType::String, false, 0).expect("category schema");
    let mut bucket = FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket schema");
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body schema");
    if scalar_indexes {
        category
            .set_index_params(&IndexParams::invert(true, true).expect("category index"))
            .expect("category index must be valid");
        bucket
            .set_index_params(&IndexParams::invert(true, false).expect("bucket index"))
            .expect("bucket index must be valid");
        body.set_index_params(&IndexParams::fts(Some("standard"), None, None).expect("body index"))
            .expect("body index must be valid");
    }
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 8).expect("embedding schema");
    // The dimension is replaced by the caller through `schema_with_dimension`.
    if vector_index {
        embedding
            .set_index_params(&IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW index"))
            .expect("HNSW index must be valid");
    }
    CollectionSchema::builder(name)
        .add_field(category)
        .add_field(bucket)
        .add_field(body)
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn schema_with_dimension(
    name: &str,
    dimensions: usize,
    vector_index: bool,
    scalar_indexes: bool,
) -> CollectionSchema {
    let mut schema = schema(name, vector_index, scalar_indexes);
    let embedding = schema
        .vectors
        .iter_mut()
        .find(|field| field.name == "embedding")
        .expect("embedding field must exist");
    embedding.dimension = u32::try_from(dimensions).expect("dimension fits u32");
    schema
        .validate()
        .expect("dimension-adjusted schema must be valid");
    schema
}

fn vector_for(index: usize, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|dimension| {
            let raw = (index.wrapping_mul(31)
                + dimension.wrapping_mul(17)
                + index.wrapping_mul(dimension).wrapping_mul(3))
                % 1_009;
            let raw = u16::try_from(raw).expect("fixture value fits u16");
            (f32::from(raw) - 504.0) / 504.0
        })
        .collect()
}

fn document(index: usize, dimensions: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:06}")).expect("document primary key must be valid");
    doc.add_string("category", if index % 2 == 0 { "even" } else { "odd" })
        .expect("category must be valid");
    doc.add_i32(
        "bucket",
        i32::try_from(index % 16).expect("bucket fits i32"),
    )
    .expect("bucket must be valid");
    doc.add_string(
        "body",
        &format!("workspace retrieval document token{}", index % 17),
    )
    .expect("body must be valid");
    doc.add_vector_f32("embedding", &vector_for(index, dimensions))
        .expect("embedding must be valid");
    doc
}

fn insert_fixture(collection: &Collection, config: Config, count: usize) -> Vec<Doc> {
    let documents: Vec<Doc> = (0..count)
        .map(|index| document(index, config.dimensions))
        .collect();
    let references: Vec<&Doc> = documents.iter().collect();
    let result = collection
        .insert(&references)
        .expect("fixture insert must succeed");
    assert_eq!(
        result.success_count,
        u64::try_from(count).expect("document count fits u64")
    );
    documents
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    elapsed: Duration,
    work: u64,
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
        self.work as f64 / (seconds * self.samples as f64)
    }
}

fn percentile(samples: &[Duration], percentage: usize) -> Duration {
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let rank = ordered.len().saturating_mul(percentage).div_ceil(100);
    ordered[rank.saturating_sub(1).min(ordered.len().saturating_sub(1))]
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).expect("fixture count must fit u64")
}

#[allow(clippy::cast_precision_loss)]
fn measure<F>(name: &str, config: Config, mut operation: F) -> Measurement
where
    F: FnMut(usize) -> Sample,
{
    let _ = operation(usize::MAX);
    let mut durations = Vec::with_capacity(config.samples);
    let mut work = 0_u64;
    for sample in 0..config.samples {
        let result = operation(sample);
        assert!(result.work > 0, "{name} must report positive work");
        assert!(!result.elapsed.is_zero(), "{name} must report elapsed time");
        durations.push(result.elapsed);
        work = work.saturating_add(result.work);
    }
    let measurement = Measurement {
        samples: config.samples,
        p50: percentile(&durations, 50),
        p95: percentile(&durations, 95),
        p99: percentile(&durations, 99),
        work,
    };
    assert!(
        measurement.p50 <= measurement.p95 && measurement.p95 <= measurement.p99,
        "{name} percentiles must be monotonic"
    );
    let throughput = measurement.work_per_second();
    assert!(
        throughput.is_finite() && throughput > 0.0,
        "{name} throughput must be finite and positive"
    );
    println!(
        "{name},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
        config.documents,
        config.dimensions,
        measurement.samples,
        measurement.work,
        micros(measurement.p50),
        micros(measurement.p95),
        micros(measurement.p99),
        throughput,
        measurement.work as f64 / measurement.samples as f64,
    );
    measurement
}

fn create_path(root: &std::path::Path, name: &str) -> String {
    root.join(name)
        .to_str()
        .expect("benchmark path must be UTF-8")
        .to_owned()
}

fn one_document() -> usize {
    1
}

fn small_fixture_count(config: Config) -> usize {
    config.documents.clamp(1, 32)
}

include!("lifecycle_matrix/operations.rs");

fn main() {
    run_lifecycle_matrix();
}
