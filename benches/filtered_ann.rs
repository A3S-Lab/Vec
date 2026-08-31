//! Reproducible filter-aware ANN latency and recall fixture.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    HnswQueryParams, IndexParams, IvfQueryParams, MetricType, SearchQuery,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DOCUMENTS: usize = 8_400;
const ALLOWED_START: usize = DOCUMENTS / 2;
const QUERIES: usize = 32;
const ROUNDS: usize = 5;
const TOPK: usize = 10;

struct Measurement {
    duration: Duration,
    rankings: Vec<Vec<String>>,
    rerank_candidates: f64,
    results: f64,
}

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("collection options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn schema(name: &str, indexed: bool) -> CollectionSchema {
    let mut scope =
        FieldSchema::new("scope", DataType::String, false, 0).expect("scope must be valid");
    let mut shard =
        FieldSchema::new("shard", DataType::Int32, false, 0).expect("shard must be valid");
    let embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
        .expect("embedding must be valid");
    if indexed {
        scope
            .set_index_params(&IndexParams::invert(false, false).expect("descriptor must be valid"))
            .expect("scope index must be valid");
        shard
            .set_index_params(&IndexParams::invert(false, false).expect("descriptor must be valid"))
            .expect("shard index must be valid");
    }
    CollectionSchema::builder(name)
        .add_field(scope)
        .add_field(shard)
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn vector_for(index: usize) -> [f32; 2] {
    let local = index % ALLOWED_START;
    let local = f32::from(u16::try_from(local).expect("fixture coordinate fits u16"));
    let offset = if index >= ALLOWED_START {
        10_000.0
    } else {
        0.0
    };
    [offset + local, local % 17.0]
}

fn document(index: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:05}")).expect("document primary key must be valid");
    doc.add_string(
        "scope",
        if index >= ALLOWED_START {
            "allowed"
        } else {
            "excluded"
        },
    )
    .expect("scope must be valid");
    doc.add_i32(
        "shard",
        i32::try_from(index % 2).expect("shard value fits i32"),
    )
    .expect("shard must be valid");
    doc.add_vector_f32("embedding", &vector_for(index))
        .expect("embedding must be valid");
    doc
}

fn query_vectors() -> Vec<[f32; 2]> {
    (0..QUERIES)
        .map(|index| vector_for((index * 97 + 11) % ALLOWED_START))
        .collect()
}

fn query(vector: &[f32], topk: usize, filter: Option<&str>) -> SearchQuery {
    let mut query = SearchQuery::new(
        "embedding",
        vector,
        i32::try_from(topk).expect("top-k fits i32"),
    )
    .expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    if let Some(filter) = filter {
        query.set_filter(filter).expect("filter must be valid");
    }
    query
}

fn run_queries(
    collection: &Collection,
    vectors: &[[f32; 2]],
    make_query: impl Fn(&[f32]) -> SearchQuery,
    postprocess: impl Fn(Vec<Doc>) -> Vec<Doc>,
) -> Measurement {
    if let Some(vector) = vectors.first() {
        let result = collection
            .query(&make_query(vector))
            .expect("warmup query must succeed");
        black_box(postprocess(result));
    }
    let before = collection
        .stats_snapshot()
        .expect("statistics must be available")
        .candidates_scanned;
    let mut durations = Vec::with_capacity(ROUNDS);
    let mut rankings = Vec::new();
    let mut result_count = 0_usize;
    for _ in 0..ROUNDS {
        let started = Instant::now();
        rankings = vectors
            .iter()
            .map(|vector| {
                let result = collection
                    .query(black_box(&make_query(vector)))
                    .expect("benchmark query must succeed");
                let result = postprocess(result);
                result_count = result_count.saturating_add(result.len());
                result
                    .into_iter()
                    .map(|doc| doc.get_pk().expect("result must have an id").to_string())
                    .collect()
            })
            .collect();
        durations.push(started.elapsed());
    }
    let after = collection
        .stats_snapshot()
        .expect("statistics must be available")
        .candidates_scanned;
    durations.sort_unstable();
    let operations = ROUNDS.saturating_mul(QUERIES);
    Measurement {
        duration: durations[ROUNDS / 2],
        rankings,
        rerank_candidates: count_to_f64(
            usize::try_from(after.saturating_sub(before)).unwrap_or(usize::MAX),
        ) / count_to_f64(operations),
        results: count_to_f64(result_count) / count_to_f64(operations),
    }
}

fn recall(reference: &[Vec<String>], candidate: &[Vec<String>]) -> f64 {
    let hits = reference
        .iter()
        .zip(candidate)
        .map(|(expected, actual)| actual.iter().filter(|id| expected.contains(id)).count())
        .sum::<usize>();
    count_to_f64(hits) / count_to_f64(reference.len().saturating_mul(TOPK))
}

fn print_measurement(name: &str, measurement: &Measurement, reference: &[Vec<String>]) {
    println!(
        "{name},{DOCUMENTS},2,{QUERIES},{ROUNDS},{:.2},{:.2},{:.4},{:.2}",
        measurement.results,
        measurement.rerank_candidates,
        recall(reference, &measurement.rankings),
        micros_per_query(measurement.duration),
    );
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
    let options = options();
    let exact = Collection::create(
        temporary.path().join("exact").to_str().expect("UTF-8 path"),
        &schema("exact", false),
        Some(&options),
    )
    .expect("exact collection must be created");
    let indexed = Collection::create(
        temporary
            .path()
            .join("indexed")
            .to_str()
            .expect("UTF-8 path"),
        &schema("indexed", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    exact.insert(&refs).expect("exact fixture must insert");
    indexed.insert(&refs).expect("indexed fixture must insert");
    let vectors = query_vectors();

    let exact_scope = run_queries(
        &exact,
        &vectors,
        |vector| query(vector, TOPK, Some("scope == 'allowed'")),
        |result| result,
    );
    indexed
        .create_index(
            "embedding",
            &IndexParams::ivf(MetricType::L2, 32, 5, false).expect("IVF descriptor must be valid"),
        )
        .expect("IVF index must build");
    let ivf_postfilter = run_queries(
        &indexed,
        &vectors,
        |vector| {
            let mut query = query(vector, 80, None);
            query
                .set_ivf_params(IvfQueryParams::new(2, true, 1.0))
                .expect("IVF controls must be valid");
            query
        },
        |result| {
            result
                .into_iter()
                .filter(|doc| doc.get_pk().is_some_and(|id| id >= "doc-04200"))
                .take(TOPK)
                .collect()
        },
    );
    let ivf_prefilter = run_queries(
        &indexed,
        &vectors,
        |vector| {
            let mut query = query(vector, TOPK, Some("scope == 'allowed'"));
            query
                .set_ivf_params(IvfQueryParams::new(2, true, 8.0))
                .expect("IVF controls must be valid");
            query
        },
        |result| result,
    );

    let exact_shard = run_queries(
        &exact,
        &vectors,
        |vector| query(vector, TOPK, Some("shard == 0")),
        |result| result,
    );
    indexed
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::L2, 12, 64).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index must build");
    let hnsw_prefilter = run_queries(
        &indexed,
        &vectors,
        |vector| {
            let mut query = query(vector, TOPK, Some("shard == 0"));
            query
                .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
                .expect("HNSW controls must be valid");
            query
        },
        |result| result,
    );

    println!(
        "mode,documents,dimensions,queries,rounds,results_per_query,rerank_candidates,recall_at_10,micros_per_query"
    );
    print_measurement("exact_scope", &exact_scope, &exact_scope.rankings);
    print_measurement("ivf_postfilter", &ivf_postfilter, &exact_scope.rankings);
    print_measurement("ivf_prefilter", &ivf_prefilter, &exact_scope.rankings);
    print_measurement("exact_shard", &exact_shard, &exact_shard.rankings);
    print_measurement("hnsw_prefilter", &hnsw_prefilter, &exact_shard.rankings);
}
