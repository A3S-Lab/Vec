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
use std::hint::black_box;
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

fn dense_query(config: Config, metric: &str) -> SearchQuery {
    let mut query = SearchQuery::new("embedding", &vector_for(17, config.dimensions), 10)
        .expect("dense query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!(metric));
    query
}

#[allow(clippy::too_many_lines)]
fn measure_full_routes(collection: &Collection, config: Config) {
    let samples = config.rounds * config.queries;
    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let query = dense_query(config, metric);
        measure(&format!("dense_{metric}"), config, samples, |sample| {
            let mut query = query.clone();
            query
                .set_query_vector(&vector_for(sample % config.documents, config.dimensions))
                .expect("query vector must be valid");
            let result = collection
                .query(black_box(&query))
                .expect("dense query must succeed");
            assert_eq!(result.len(), 10.min(config.documents));
            u64::try_from(result.len()).expect("result count fits u64")
        });
    }

    let mut include_doc_id = dense_query(config, "cosine");
    include_doc_id
        .set_include_doc_id(true)
        .expect("include-doc-id control must be valid");
    measure("dense_include_doc_id", config, samples, |sample| {
        let mut query = include_doc_id.clone();
        query
            .set_query_vector(&vector_for(sample % config.documents, config.dimensions))
            .expect("query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("include-doc-id query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        assert!(result.iter().all(|doc| doc.doc_id().is_some()));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    for metric in ["l2", "ip", "cosine", "mips_l2"] {
        let mut query = SearchQuery::by_id("embedding", "doc-00017", 10)
            .expect("dense source-ID query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!(metric));
        measure(
            &format!("dense_source_id_{metric}"),
            config,
            samples,
            |_| {
                let result = collection
                    .query(black_box(&query))
                    .expect("dense source-ID query must succeed");
                assert_eq!(result.len(), 10.min(config.documents));
                u64::try_from(result.len()).expect("result count fits u64")
            },
        );
    }

    let sparse = SearchQuery::sparse("sparse", &[1, 8], &[1.0, 0.25], 10)
        .expect("sparse query must be valid");
    measure("sparse", config, samples, |sample| {
        let mut query = sparse.clone();
        let first = u32::try_from(sample % 16).expect("sparse index fits u32");
        query.sparse_vector = Some(vec![(first, 1.0), ((first + 7) % 32, 0.25)]);
        let result = collection
            .query(black_box(&query))
            .expect("sparse query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let sparse_source =
        SearchQuery::by_id("sparse", "doc-00017", 10).expect("sparse source-ID query");
    measure("sparse_source_id", config, samples, |_| {
        let result = collection
            .query(black_box(&sparse_source))
            .expect("sparse source-ID query must succeed");
        assert_eq!(result.len(), 10.min(config.documents));
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut fts_payload = Fts::new().expect("FTS payload must be valid");
    fts_payload
        .set_query_string("rust AND vector")
        .expect("FTS expression must be valid");
    let fts_query = SearchQuery::fts("body", &fts_payload, 10).expect("FTS query must be valid");
    measure("fts_indexed", config, samples, |_| {
        let result = collection
            .query(black_box(&fts_query))
            .expect("FTS query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut filtered = dense_query(config, "cosine");
    filtered
        .set_filter("category == 'even'")
        .expect("filter must be valid");
    measure("dense_scalar_filter", config, samples, |sample| {
        let mut query = filtered.clone();
        query
            .set_query_vector(&vector_for(sample % config.documents, config.dimensions))
            .expect("query vector must be valid");
        let result = collection
            .query(black_box(&query))
            .expect("filtered query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut vector_branch = SubQuery::new().expect("sub-query must be valid");
    vector_branch
        .set_field_name("embedding")
        .expect("sub-query field must be valid");
    vector_branch
        .set_query_vector(&vector_for(17, config.dimensions))
        .expect("sub-query vector must be valid");
    vector_branch
        .set_num_candidates(16)
        .expect("candidate count must be valid");
    let mut fts_branch = SubQuery::new().expect("sub-query must be valid");
    fts_branch
        .set_field_name("body")
        .expect("sub-query field must be valid");
    fts_branch
        .set_fts(&fts_payload)
        .expect("FTS branch must be valid");
    fts_branch
        .set_num_candidates(16)
        .expect("candidate count must be valid");
    let mut multi = MultiQuery::new().expect("multi-query must be valid");
    multi.set_topk(10).expect("multi top-k must be valid");
    multi
        .add_sub_query(&vector_branch)
        .expect("vector branch must be added");
    multi
        .add_sub_query(&fts_branch)
        .expect("FTS branch must be added");
    multi.set_rerank_rrf(60).expect("RRF rerank must be valid");
    measure("multi_rrf", config, samples, |_| {
        let result = collection
            .multi_query(black_box(&multi))
            .expect("multi query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let grouped = GroupBySearchQuery::new(
        "embedding",
        "category",
        &vector_for(17, config.dimensions),
        2,
        3,
    )
    .expect("group query must be valid");
    measure("group_by", config, samples, |_| {
        let result = collection
            .group_by(black_box(&grouped))
            .expect("group query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.values().map(Vec::len).sum::<usize>())
            .expect("group result count fits u64")
    });

    measure("fetch_projection", config, samples, |_| {
        let result = collection
            .fetch_with_options(&["doc-00001", "doc-00017"], Some(&["body"]), false)
            .expect("fetch must succeed");
        assert_eq!(result.len(), 2);
        u64::try_from(result.len()).expect("result count fits u64")
    });

    measure("snapshot_iterator", config, config.rounds, |_| {
        let result = collection
            .iter_with_options(Some(&["category"]), false)
            .expect("iterator must be created")
            .collect::<a3s_vec::Result<Vec<_>>>()
            .expect("iterator must succeed");
        assert_eq!(result.len(), config.documents);
        u64::try_from(result.len()).expect("result count fits u64")
    });

    measure("stats_health", config, samples, |_| {
        let stats = collection.stats().expect("stats must succeed");
        assert_eq!(
            stats.doc_count,
            u64::try_from(config.documents).expect("document count fits u64")
        );
        assert!(collection
            .health()
            .expect("health must succeed")
            .is_healthy());
        1
    });

    let patch = {
        let mut doc = Doc::with_pk("doc-00017").expect("patch must be valid");
        doc.add_i32("bucket", 15).expect("patch must be valid");
        doc
    };
    measure("partial_update", config, config.rounds, |_| {
        let result = collection.update(&[&patch]).expect("update must succeed");
        assert_eq!(result.success_count, 1);
        result.success_count
    });
    measure("flush", config, config.rounds, |_| {
        collection.flush().expect("flush must succeed");
        1
    });
}

#[derive(Clone, Copy)]
enum AnnMode {
    Hnsw,
    IvfSoar,
    HnswRabitq,
    IvfRabitq,
    Vamana,
    VamanaIp,
    VamanaCosine,
    VamanaMipsL2,
    Diskann,
    DiskannIp,
    DiskannCosine,
    DiskannMipsL2,
}

impl AnnMode {
    fn name(self) -> &'static str {
        match self {
            Self::Hnsw => "ann_hnsw",
            Self::IvfSoar => "ann_ivf_soar",
            Self::HnswRabitq => "ann_hnsw_rabitq",
            Self::IvfRabitq => "ann_ivf_rabitq",
            Self::Vamana => "ann_vamana",
            Self::VamanaIp => "ann_vamana_ip",
            Self::VamanaCosine => "ann_vamana_cosine",
            Self::VamanaMipsL2 => "ann_vamana_mips_l2",
            Self::Diskann => "ann_diskann_pq",
            Self::DiskannIp => "ann_diskann_ip_pq",
            Self::DiskannCosine => "ann_diskann_cosine_pq",
            Self::DiskannMipsL2 => "ann_diskann_mips_l2_pq",
        }
    }

    fn metric(self) -> MetricType {
        match self {
            Self::Hnsw
            | Self::IvfSoar
            | Self::HnswRabitq
            | Self::IvfRabitq
            | Self::VamanaCosine
            | Self::DiskannCosine => MetricType::Cosine,
            Self::Vamana | Self::Diskann => MetricType::L2,
            Self::VamanaIp | Self::DiskannIp => MetricType::Ip,
            Self::VamanaMipsL2 | Self::DiskannMipsL2 => MetricType::MipsL2,
        }
    }

    fn params(self, documents: usize) -> IndexParams {
        match self {
            Self::Hnsw => IndexParams::hnsw(MetricType::Cosine, 8, 32),
            Self::IvfSoar => IndexParams::ivf(MetricType::Cosine, 8, 4, true),
            Self::HnswRabitq => {
                IndexParams::hnsw_rabitq_with_options(MetricType::Cosine, 8, 32, 5, 8, 0)
            }
            Self::IvfRabitq => IndexParams::ivf_rabitq(MetricType::Cosine, 8, 5, 0),
            Self::Vamana | Self::VamanaIp | Self::VamanaCosine | Self::VamanaMipsL2 => {
                IndexParams::vamana(
                    self.metric(),
                    12,
                    i32::try_from(documents).expect("document count fits i32"),
                    1.2,
                )
            }
            Self::Diskann | Self::DiskannIp | Self::DiskannCosine | Self::DiskannMipsL2 => {
                IndexParams::diskann(
                    self.metric(),
                    12,
                    i32::try_from(documents).expect("document count fits i32"),
                    2,
                )
            }
        }
        .expect("ANN descriptor must be valid")
    }

    fn query(self, config: Config) -> SearchQuery {
        let mut query = SearchQuery::new("embedding", &vector_for(17, config.dimensions), 10)
            .expect("ANN query must be valid");
        match self {
            Self::Hnsw | Self::HnswRabitq => query
                .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
                .expect("HNSW controls must be valid"),
            Self::IvfSoar => query
                .set_ivf_params(IvfQueryParams::new(8, true, 1.0))
                .expect("IVF controls must be valid"),
            Self::IvfRabitq => {
                let mut controls = IvfRabitqQueryParams::new(8, 0.0, false, true);
                controls
                    .set_scale_factor(4.0)
                    .expect("IVF RaBitQ scale must be valid");
                query
                    .set_ivf_rabitq_params(controls)
                    .expect("IVF RaBitQ controls must be valid");
            }
            Self::Vamana
            | Self::VamanaIp
            | Self::VamanaCosine
            | Self::VamanaMipsL2
            | Self::Diskann
            | Self::DiskannIp
            | Self::DiskannCosine
            | Self::DiskannMipsL2 => {
                query.params.insert(
                    "metric".into(),
                    serde_json::json!(match self.metric() {
                        MetricType::L2 => "l2",
                        MetricType::Ip => "ip",
                        MetricType::Cosine => "cosine",
                        MetricType::MipsL2 => "mips_l2",
                        MetricType::Undefined => "undefined",
                    }),
                );
                query
                    .set_diskann_params(DiskannQueryParams::new(64))
                    .expect("DiskANN controls must be valid");
            }
        }
        query
    }
}

fn measure_ann_modes(config: Config) {
    for mode in [
        AnnMode::Hnsw,
        AnnMode::IvfSoar,
        AnnMode::HnswRabitq,
        AnnMode::IvfRabitq,
        AnnMode::Vamana,
        AnnMode::VamanaIp,
        AnnMode::VamanaCosine,
        AnnMode::VamanaMipsL2,
        AnnMode::Diskann,
        AnnMode::DiskannIp,
        AnnMode::DiskannCosine,
        AnnMode::DiskannMipsL2,
    ] {
        let directory = tempdir().expect("ANN temporary directory must be available");
        let path = directory.path().join(mode.name());
        let path_string = path.to_str().expect("ANN path must be UTF-8");
        let collection = Collection::create(path_string, &ann_schema(config.dimensions), None)
            .expect("ANN collection must be created");
        insert_ann_fixture(&collection, config);
        collection
            .create_index("embedding", &mode.params(config.documents))
            .expect("ANN index must build");
        let query = mode.query(config);
        let samples = config.rounds * config.queries;
        measure(mode.name(), config, samples, |_| {
            let result = collection
                .query(black_box(&query))
                .expect("ANN query must succeed");
            assert!(!result.is_empty());
            u64::try_from(result.len()).expect("result count fits u64")
        });
        collection.close().expect("ANN collection must close");
    }
}

fn measure_reopen_and_sidecar(config: Config) {
    let directory = tempdir().expect("sidecar temporary directory must be available");
    let path = directory.path().join("sidecar");
    let path_string = path.to_str().expect("sidecar path must be UTF-8");
    let collection = Collection::create(path_string, &ann_schema(config.dimensions), None)
        .expect("sidecar collection must be created");
    insert_ann_fixture(&collection, config);
    collection
        .create_index(
            "embedding",
            &IndexParams::diskann(
                MetricType::L2,
                12,
                i32::try_from(config.documents).expect("document count fits i32"),
                2,
            )
            .expect("DiskANN descriptor must be valid"),
        )
        .expect("DiskANN index must build");
    collection.flush().expect("sidecar fixture must flush");
    collection.close().expect("sidecar collection must close");

    let query = {
        let mut query = SearchQuery::new("embedding", &vector_for(17, config.dimensions), 10)
            .expect("sidecar query must be valid");
        query
            .params
            .insert("metric".into(), serde_json::json!("l2"));
        query
            .set_diskann_params(DiskannQueryParams::new(64))
            .expect("sidecar controls must be valid");
        query
    };
    for backend in [IoBackend::Positioned, IoBackend::Mmap] {
        let mut options = CollectionOptions::new().expect("options must be valid");
        options
            .set_read_only(true)
            .expect("read-only option must be valid");
        options
            .set_io_backend(backend)
            .expect("backend must be valid");
        let samples = config.rounds;
        measure(
            if backend == IoBackend::Mmap {
                "diskann_mmap_reopen_query"
            } else {
                "diskann_positioned_reopen_query"
            },
            config,
            samples,
            |_| {
                let reopened =
                    Collection::open(path_string, Some(&options)).expect("reopen must succeed");
                assert!(
                    reopened
                        .stats()
                        .expect("stats must succeed")
                        .index_cache_hit
                );
                let result = reopened
                    .query(black_box(&query))
                    .expect("sidecar query must succeed");
                assert!(!result.is_empty());
                let count = u64::try_from(result.len()).expect("result count fits u64");
                reopened.close().expect("reopened collection must close");
                count
            },
        );
    }
}

#[cfg(feature = "async")]
#[allow(clippy::too_many_lines)]
fn measure_async(collection: &Collection, config: Config) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Tokio runtime must be available");
    let query = dense_query(config, "cosine");
    let samples = config.rounds * config.queries;
    measure("dense_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.query_async(black_box(&query)))
            .expect("async query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let mut branch = SubQuery::new().expect("async sub-query must be valid");
    branch
        .set_field_name("embedding")
        .expect("async sub-query field must be valid");
    branch
        .set_query_vector(&vector_for(17, config.dimensions))
        .expect("async sub-query vector must be valid");
    branch
        .set_num_candidates(16)
        .expect("async candidate count must be valid");
    let mut multi = MultiQuery::new().expect("async multi-query must be valid");
    multi.set_topk(10).expect("async multi top-k must be valid");
    multi
        .add_sub_query(&branch)
        .expect("async sub-query must be added");
    measure("multi_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.multi_query_async(black_box(&multi)))
            .expect("async multi-query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.len()).expect("result count fits u64")
    });

    let grouped = GroupBySearchQuery::new(
        "embedding",
        "category",
        &vector_for(17, config.dimensions),
        2,
        3,
    )
    .expect("async group query must be valid");
    measure("group_by_async", config, samples, |_| {
        let result = runtime
            .block_on(collection.group_by_async(black_box(&grouped)))
            .expect("async group query must succeed");
        assert!(!result.is_empty());
        u64::try_from(result.values().map(Vec::len).sum::<usize>())
            .expect("async group result count fits u64")
    });
}

fn main() {
    let config = Config::from_environment();
    println!("# a3s-vec feature matrix; scale={config:?}; each row is an asserted operation");
    println!(
        "operation,documents,dimensions,samples,total_work,p50_us,p95_us,p99_us,work_per_second,work_per_sample"
    );

    let directory = tempdir().expect("feature matrix temporary directory must be available");
    let path = directory.path().join("routes");
    let path_string = path.to_str().expect("feature matrix path must be UTF-8");
    let collection = Collection::create(
        path_string,
        &full_schema(config.dimensions),
        Some(&manual_options()),
    )
    .expect("feature matrix collection must be created");
    insert_full_fixture(&collection, config);
    measure_full_routes(&collection, config);
    #[cfg(feature = "async")]
    measure_async(&collection, config);
    collection
        .close()
        .expect("feature matrix collection must close");

    measure_ann_modes(config);
    measure_reopen_and_sidecar(config);
}
