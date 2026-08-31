//! Reproducible Workspace-shaped indexed collection reopen fixture.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, IndexParams,
    MetricType,
};
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DEFAULT_DOCUMENTS: usize = 5_000;
const DIMENSIONS: usize = 32;
const DEFAULT_ROUNDS: usize = 5;

#[derive(Clone, Copy)]
enum IndexMode {
    None,
    ScalarFts,
    All,
}

#[derive(Clone, Copy)]
struct FixtureMeasurement {
    insert: Duration,
    close: Duration,
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

fn schema(name: &str, mode: IndexMode) -> CollectionSchema {
    let mut language =
        FieldSchema::new("language", DataType::String, false, 0).expect("field must be valid");
    if matches!(mode, IndexMode::ScalarFts | IndexMode::All) {
        language
            .set_index_params(
                &IndexParams::invert(false, false).expect("index params must be valid"),
            )
            .expect("scalar index must be valid");
    }
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
    if matches!(mode, IndexMode::ScalarFts | IndexMode::All) {
        body.set_index_params(
            &IndexParams::fts(Some("standard"), None, None).expect("index params must be valid"),
        )
        .expect("FTS index must be valid");
    }
    let mut embedding = FieldSchema::new(
        "embedding",
        DataType::VectorFp32,
        false,
        u32::try_from(DIMENSIONS).expect("dimension fits u32"),
    )
    .expect("field must be valid");
    if matches!(mode, IndexMode::All) {
        embedding
            .set_index_params(
                &IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW params must be valid"),
            )
            .expect("HNSW index must be valid");
    }
    CollectionSchema::builder(name)
        .add_field(language)
        .add_field(body)
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn document(index: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:05}")).expect("document primary key must be valid");
    doc.add_string(
        "language",
        ["rust", "typescript", "python", "go"][index % 4],
    )
    .expect("language must be valid");
    let body = format!(
        "workspace component{} symbol{index} retrieval agent",
        index % 64
    );
    doc.add_string("body", &body).expect("body must be valid");
    doc.add_vector_f32("embedding", &vector_for(index))
        .expect("embedding must be valid");
    doc
}

fn read_only_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only option must be valid");
    options
}

fn measure_reopen(path: &Path, documents: usize, expected_cache_hit: bool) -> Duration {
    let options = read_only_options();
    let started = Instant::now();
    let collection = Collection::open(
        path.to_str().expect("benchmark path must be UTF-8"),
        Some(&options),
    )
    .expect("indexed collection must reopen");
    assert_eq!(collection.count().expect("count must succeed"), documents);
    let stats = collection.stats().expect("stats must succeed");
    assert_eq!(stats.index_cache_hit, expected_cache_hit);
    black_box(stats);
    collection.close().expect("read-only close must succeed");
    started.elapsed()
}

fn median_reopen(
    path: &Path,
    documents: usize,
    rounds: usize,
    expected_cache_hit: bool,
) -> Duration {
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        samples.push(measure_reopen(path, documents, expected_cache_hit));
    }
    samples.sort_unstable();
    samples[rounds / 2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn create_fixture(path: &Path, name: &str, mode: IndexMode, docs: &[Doc]) -> FixtureMeasurement {
    let collection = Collection::create(
        path.to_str().expect("benchmark path must be UTF-8"),
        &schema(name, mode),
        None,
    )
    .expect("collection must be created");
    let started = Instant::now();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    let insert = started.elapsed();
    let started = Instant::now();
    collection.close().expect("collection must close");
    FixtureMeasurement {
        insert,
        close: started.elapsed(),
    }
}

fn print_fixture_measurement(
    mode: &str,
    path: &Path,
    documents: usize,
    measurement: FixtureMeasurement,
) {
    let cache = path.join("indexes/index-cache.bin");
    let cache_bytes = if cache.exists() {
        cache_bytes(&cache)
    } else {
        0
    };
    println!(
        "{mode},{documents},{DIMENSIONS},{:.2},{:.2},{cache_bytes},{}",
        milliseconds(measurement.insert),
        milliseconds(measurement.close),
        snapshot_bytes(path),
    );
}

fn print_measurement(
    mode: &str,
    path: &Path,
    documents: usize,
    rounds: usize,
    expected_cache_hit: bool,
    cache_bytes: u64,
) {
    println!(
        "{mode},{documents},{DIMENSIONS},{rounds},{:.2},{cache_bytes},{}",
        milliseconds(median_reopen(path, documents, rounds, expected_cache_hit)),
        snapshot_bytes(path),
    );
}

fn cache_bytes(path: &Path) -> u64 {
    path.metadata()
        .expect("cache metadata must be available")
        .len()
}

fn snapshot_bytes(path: &Path) -> u64 {
    std::fs::read_dir(path.join("segments"))
        .expect("snapshot directory must be readable")
        .map(|entry| {
            entry
                .expect("snapshot entry must be readable")
                .metadata()
                .expect("snapshot metadata must be readable")
                .len()
        })
        .sum()
}

fn print_rebuild_measurements(path: &Path, documents: usize, fields: &[&str]) {
    let collection = Collection::open(path.to_str().expect("benchmark path must be UTF-8"), None)
        .expect("indexed collection must open for rebuilds");
    println!("operation,documents,dimensions,milliseconds");
    for field in fields {
        let started = Instant::now();
        collection
            .rebuild_index(field)
            .expect("targeted index rebuild must succeed");
        println!(
            "rebuild_{field},{documents},{DIMENSIONS},{:.2}",
            milliseconds(started.elapsed())
        );
    }
    let started = Instant::now();
    collection.optimize().expect("full rebuild must succeed");
    println!(
        "optimize_all,{documents},{DIMENSIONS},{:.2}",
        milliseconds(started.elapsed())
    );
    collection.close().expect("collection must close");
}

fn positive_env(name: &str, default: usize) -> usize {
    let Ok(value) = std::env::var(name) else {
        return default;
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| panic!("{name} must be a positive integer"))
}

fn print_scalar_fts_reopen_measurements(path: &Path, documents: usize, rounds: usize) {
    let cache = path.join("indexes/index-cache.bin");
    if cache.exists() {
        print_measurement(
            "scalar_fts_cache_hit",
            path,
            documents,
            rounds,
            true,
            cache_bytes(&cache),
        );
        std::fs::remove_file(&cache)
            .expect("scalar/FTS cache must be removable for the rebuild row");
    }
    print_measurement("scalar_fts_rebuild", path, documents, rounds, false, 0);
}

fn print_all_index_reopen_measurements(path: &Path, documents: usize, rounds: usize) {
    let cache = path.join("indexes/index-cache.bin");
    if cache.exists() {
        print_measurement(
            "cache_hit",
            path,
            documents,
            rounds,
            true,
            cache_bytes(&cache),
        );
        std::fs::remove_file(&cache).expect("benchmark cache must be removable");
    }
    print_measurement("ann_rebuild", path, documents, rounds, false, 0);
}

fn main() {
    let documents = positive_env("A3S_VEC_REOPEN_DOCUMENTS", DEFAULT_DOCUMENTS);
    let rounds = positive_env("A3S_VEC_REOPEN_ROUNDS", DEFAULT_ROUNDS);
    let scalar_fts_only = std::env::var_os("A3S_VEC_REOPEN_SCALAR_FTS_ONLY").is_some();
    let temporary = tempdir().expect("temporary directory must be available");
    let docs: Vec<Doc> = (0..documents).map(document).collect();
    let documents_path = temporary.path().join("documents");
    let scalar_fts_path = temporary.path().join("scalar-fts");
    let all_indexes_path = temporary.path().join("all-indexes");
    let documents_fixture = create_fixture(
        &documents_path,
        "workspace-reopen-documents",
        IndexMode::None,
        &docs,
    );
    let scalar_fts_fixture = create_fixture(
        &scalar_fts_path,
        "workspace-reopen-scalar-fts",
        IndexMode::ScalarFts,
        &docs,
    );
    let all_indexes_fixture = if scalar_fts_only {
        None
    } else {
        Some(create_fixture(
            &all_indexes_path,
            "workspace-reopen-all-indexes",
            IndexMode::All,
            &docs,
        ))
    };

    println!(
        "fixture_mode,documents,dimensions,insert_milliseconds,close_milliseconds,cache_bytes,snapshot_bytes"
    );
    print_fixture_measurement(
        "documents_only",
        &documents_path,
        documents,
        documents_fixture,
    );
    print_fixture_measurement(
        "scalar_fts",
        &scalar_fts_path,
        documents,
        scalar_fts_fixture,
    );
    if let Some(measurement) = all_indexes_fixture {
        print_fixture_measurement("all_indexes", &all_indexes_path, documents, measurement);
    }

    println!("mode,documents,dimensions,rounds,median_milliseconds,cache_bytes,snapshot_bytes");
    print_measurement(
        "documents_only",
        &documents_path,
        documents,
        rounds,
        false,
        0,
    );
    print_scalar_fts_reopen_measurements(&scalar_fts_path, documents, rounds);
    if scalar_fts_only {
        print_rebuild_measurements(&scalar_fts_path, documents, &["language", "body"]);
        return;
    }
    print_all_index_reopen_measurements(&all_indexes_path, documents, rounds);
    print_rebuild_measurements(
        &all_indexes_path,
        documents,
        &["language", "body", "embedding"],
    );
}
