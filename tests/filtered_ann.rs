#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    HnswQueryParams, IndexParams, IvfQueryParams, MetricType, SearchQuery,
};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::tempdir;

const DOCUMENTS: usize = 8_400;
const ALLOWED_START: usize = DOCUMENTS / 2;
const TOPK: usize = 10;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("collection options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn schema(name: &str, indexed: bool) -> CollectionSchema {
    let mut scope =
        FieldSchema::new("scope", DataType::String, false, 0).expect("scope field must be valid");
    let mut shard =
        FieldSchema::new("shard", DataType::Int32, false, 0).expect("shard field must be valid");
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
        .expect("embedding field must be valid");
    if indexed {
        scope
            .set_index_params(
                &IndexParams::invert(false, false).expect("inverted descriptor must be valid"),
            )
            .expect("scope must support an inverted index");
        shard
            .set_index_params(
                &IndexParams::invert(false, false).expect("inverted descriptor must be valid"),
            )
            .expect("shard must support an inverted index");
        embedding
            .set_index_params(
                &IndexParams::ivf(MetricType::L2, 32, 5, false)
                    .expect("IVF descriptor must be valid"),
            )
            .expect("embedding must support IVF");
    }
    CollectionSchema::builder(name)
        .add_field(scope)
        .add_field(shard)
        .add_field(
            FieldSchema::new("tags", DataType::ArrayString, false, 0)
                .expect("tags field must be valid"),
        )
        .add_field(embedding)
        .build()
        .expect("collection schema must be valid")
}

fn document(index: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:05}")).expect("document primary key must be valid");
    let allowed = index >= ALLOWED_START;
    doc.add_string("scope", if allowed { "allowed" } else { "excluded" })
        .expect("scope must be valid");
    doc.add_i32(
        "shard",
        i32::try_from(index % 2).expect("shard value fits i32"),
    )
    .expect("shard must be valid");
    let tags: &[&str] = if index % 100 == 0 {
        &["workspace"]
    } else {
        &["workspace", "target"]
    };
    doc.add_array_string("tags", tags)
        .expect("tags must be valid");
    let local = index % ALLOWED_START;
    let local = f32::from(u16::try_from(local).expect("fixture coordinate fits u16"));
    let offset = if allowed { 10_000.0 } else { 0.0 };
    doc.add_vector_f32("embedding", &[offset + local, local % 17.0])
        .expect("embedding must be valid");
    doc
}

fn insert_fixture(collection: &Collection) {
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection
        .insert(&refs)
        .expect("fixture insert must succeed");
    assert_eq!(result.success_count, DOCUMENTS as u64);
}

fn exact_query(filter: &str) -> SearchQuery {
    let mut query = SearchQuery::new(
        "embedding",
        &[0.0, 0.0],
        i32::try_from(TOPK).expect("top-k fits i32"),
    )
    .expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query.set_filter(filter).expect("filter must be valid");
    query
}

fn indexed_query() -> SearchQuery {
    let mut query = exact_query("scope == 'allowed'");
    query
        .set_ivf_params(IvfQueryParams::new(2, true, 8.0))
        .expect("IVF controls must be valid");
    query
}

fn hnsw_query(filter: &str) -> SearchQuery {
    let mut query = exact_query(filter);
    query
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
        .expect("HNSW controls must be valid");
    query
}

fn comparable(docs: &[Doc]) -> Vec<(&str, f32)> {
    docs.iter()
        .map(|doc| {
            (
                doc.get_pk().expect("result must have a primary key"),
                doc.get_score(),
            )
        })
        .collect()
}

fn assert_filtered_query_matches_exact(indexed: &Collection, exact: &Collection) {
    let expected = exact
        .query(&exact_query("scope == 'allowed'"))
        .expect("exact filtered query must succeed");
    let before = indexed
        .stats_snapshot()
        .expect("statistics must be available");
    let actual = indexed
        .query(&indexed_query())
        .expect("filtered IVF query must succeed");
    let after = indexed
        .stats_snapshot()
        .expect("statistics must be available");

    assert_eq!(actual.len(), TOPK);
    assert_eq!(comparable(&actual), comparable(&expected));
    assert_eq!(after.ann_query_count - before.ann_query_count, 1);
    assert_eq!(
        after.scalar_index_query_count - before.scalar_index_query_count,
        1
    );
    assert!(after.candidates_scanned - before.candidates_scanned <= 80);
}

fn assert_filtered_hnsw_matches_exact(indexed: &Collection, exact: &Collection, filter: &str) {
    let expected = exact
        .query(&exact_query(filter))
        .expect("exact filtered query must succeed");
    let before = indexed
        .stats_snapshot()
        .expect("statistics must be available");
    let actual = indexed
        .query(&hnsw_query(filter))
        .expect("filtered HNSW query must succeed");
    let after = indexed
        .stats_snapshot()
        .expect("statistics must be available");

    assert_eq!(actual.len(), TOPK);
    assert_eq!(comparable(&actual), comparable(&expected));
    assert_eq!(after.ann_query_count - before.ann_query_count, 1);
    assert_eq!(
        after.scalar_index_query_count - before.scalar_index_query_count,
        1
    );
    assert!(after.candidates_scanned - before.candidates_scanned <= 64);
}

fn exercise_concurrent_filtered_generations(collection: &Collection) {
    const WRITES: usize = 32;
    const READS: usize = 64;
    const READERS: usize = 2;

    let started = Arc::new(Barrier::new(READERS + 1));
    thread::scope(|scope| {
        let writer_collection = collection.clone();
        let writer_started = Arc::clone(&started);
        let writer = scope.spawn(move || {
            writer_started.wait();
            for revision in 0..WRITES {
                let mut patch = Doc::with_pk("doc-00000").expect("patch must be valid");
                patch
                    .add_string(
                        "scope",
                        if revision % 2 == 0 {
                            "excluded"
                        } else {
                            "allowed"
                        },
                    )
                    .expect("scope patch must be valid");
                let result = writer_collection
                    .update(&[&patch])
                    .expect("concurrent scope update must succeed");
                assert_eq!(result.success_count, 1);
            }
        });

        let readers = (0..READERS)
            .map(|_| {
                let reader_collection = collection.clone();
                let reader_started = Arc::clone(&started);
                scope.spawn(move || {
                    reader_started.wait();
                    for _ in 0..READS {
                        let result = reader_collection
                            .query(&indexed_query())
                            .expect("concurrent filtered query must succeed");
                        assert_eq!(result.len(), TOPK);
                        assert!(result.iter().all(|doc| {
                            doc.get_string("scope").expect("scope must be readable")
                                == Some("allowed".into())
                        }));
                        assert!(matches!(
                            result[0].get_pk(),
                            Some("doc-00000" | "doc-04200")
                        ));
                    }
                })
            })
            .collect::<Vec<_>>();

        writer.join().expect("writer must not panic");
        for reader in readers {
            reader.join().expect("reader must not panic");
        }
    });
}

#[test]
fn filtered_ann_is_complete_generation_safe_and_durable() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = options();
    let indexed_path = temporary.path().join("indexed");
    let indexed = Collection::create(
        indexed_path.to_str().expect("temporary path must be UTF-8"),
        &schema("indexed", true),
        Some(&options),
    )
    .expect("indexed collection must be created");
    let exact = Collection::create(
        temporary
            .path()
            .join("exact")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema("exact", false),
        Some(&options),
    )
    .expect("exact collection must be created");
    insert_fixture(&indexed);
    insert_fixture(&exact);

    assert_filtered_query_matches_exact(&indexed, &exact);

    let mut patch = Doc::with_pk("doc-00000").expect("patch must be valid");
    patch
        .add_string("scope", "allowed")
        .expect("scope patch must be valid");
    indexed
        .update(&[&patch])
        .expect("indexed scope update must succeed");
    exact
        .update(&[&patch])
        .expect("exact scope update must succeed");
    assert_filtered_query_matches_exact(&indexed, &exact);
    exercise_concurrent_filtered_generations(&indexed);
    assert_filtered_query_matches_exact(&indexed, &exact);

    indexed.flush().expect("indexed collection must flush");
    indexed.close().expect("indexed collection must close");
    let reopened = Collection::open(
        indexed_path.to_str().expect("temporary path must be UTF-8"),
        Some(&options),
    )
    .expect("indexed collection must reopen");
    assert_filtered_query_matches_exact(&reopened, &exact);

    reopened
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::L2, 12, 64).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index must build");
    assert_filtered_hnsw_matches_exact(&reopened, &exact, "shard == 0");
    assert_filtered_hnsw_matches_exact(
        &reopened,
        &exact,
        "shard == 0 and tags contain_all ['target']",
    );
}

fn vamana_schema(name: &str) -> CollectionSchema {
    let mut scope =
        FieldSchema::new("scope", DataType::String, false, 0).expect("scope field must be valid");
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
        .expect("embedding field must be valid");
    scope
        .set_index_params(
            &IndexParams::invert(false, false).expect("inverted descriptor must be valid"),
        )
        .expect("scope must support an inverted index");
    embedding
        .set_index_params(
            &IndexParams::vamana(MetricType::L2, 16, 64, 1.2)
                .expect("Vamana descriptor must be valid"),
        )
        .expect("embedding must support Vamana");
    CollectionSchema::builder(name)
        .add_field(scope)
        .add_field(embedding)
        .build()
        .expect("collection schema must be valid")
}

/// Large scalar prefilter + Vamana must keep allow-list semantics vs exact Flat.
#[test]
fn filtered_vamana_matches_exact_on_large_allow_list() {
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed = Collection::create(
        temporary
            .path()
            .join("filtered-vamana")
            .to_str()
            .expect("utf8"),
        &vamana_schema("filtered-vamana"),
        Some(&options()),
    )
    .expect("indexed");
    let mut exact_embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    exact_embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("flat");
    let mut scope = FieldSchema::new("scope", DataType::String, false, 0).expect("scope");
    scope
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("invert");
    let exact_schema = CollectionSchema::builder("filtered-vamana-exact")
        .add_field(scope)
        .add_field(exact_embedding)
        .build()
        .expect("schema");
    let exact = Collection::create(
        temporary
            .path()
            .join("filtered-vamana-exact")
            .to_str()
            .expect("utf8"),
        &exact_schema,
        Some(&options()),
    )
    .expect("exact");

    const N: usize = 8_400;
    let docs: Vec<Doc> = (0..N)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("pk");
            let allowed = index >= N / 2;
            doc.add_string("scope", if allowed { "allowed" } else { "excluded" })
                .expect("scope");
            let local = (index % (N / 2)) as f32;
            doc.add_vector_f32(
                "embedding",
                &[if allowed { 1_000.0 + local } else { local }, local % 13.0],
            )
            .expect("vector");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    indexed.insert(&refs).expect("insert");
    exact.insert(&refs).expect("insert");
    indexed.optimize().expect("optimize");
    exact.optimize().expect("optimize");

    let mut query = SearchQuery::new("embedding", &[1_050.0, 1.0], TOPK as i32).expect("query");
    query.set_filter("scope == \"allowed\"").expect("filter");
    let mut approx_query = query.clone();
    approx_query
        .set_diskann_params(a3s_vec::DiskannQueryParams::new(64))
        .expect("list_size");
    let approximate = indexed.query(&approx_query).expect("filtered Vamana");
    let truth = exact.query(&query).expect("filtered exact");
    assert_eq!(
        approximate
            .iter()
            .map(|doc| doc.get_pk().unwrap().to_string())
            .collect::<Vec<_>>(),
        truth
            .iter()
            .map(|doc| doc.get_pk().unwrap().to_string())
            .collect::<Vec<_>>()
    );
}

fn diskann_schema(name: &str) -> CollectionSchema {
    let mut scope =
        FieldSchema::new("scope", DataType::String, false, 0).expect("scope field must be valid");
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
        .expect("embedding field must be valid");
    scope
        .set_index_params(
            &IndexParams::invert(false, false).expect("inverted descriptor must be valid"),
        )
        .expect("scope must support an inverted index");
    embedding
        .set_index_params(
            &IndexParams::diskann(MetricType::L2, 16, 64, 0)
                .expect("DiskANN descriptor must be valid"),
        )
        .expect("embedding must support DiskANN");
    CollectionSchema::builder(name)
        .add_field(scope)
        .add_field(embedding)
        .build()
        .expect("collection schema must be valid")
}

#[test]
fn filtered_diskann_matches_exact_on_large_allow_list() {
    let temporary = tempdir().expect("temporary directory must be available");
    let indexed = Collection::create(
        temporary
            .path()
            .join("filtered-diskann")
            .to_str()
            .expect("utf8"),
        &diskann_schema("filtered-diskann"),
        Some(&options()),
    )
    .expect("indexed");
    let mut exact_embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    exact_embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("flat");
    let mut scope = FieldSchema::new("scope", DataType::String, false, 0).expect("scope");
    scope
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("invert");
    let exact_schema = CollectionSchema::builder("filtered-diskann-exact")
        .add_field(scope)
        .add_field(exact_embedding)
        .build()
        .expect("schema");
    let exact = Collection::create(
        temporary
            .path()
            .join("filtered-diskann-exact")
            .to_str()
            .expect("utf8"),
        &exact_schema,
        Some(&options()),
    )
    .expect("exact");

    const N: usize = 8_400;
    let docs: Vec<Doc> = (0..N)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("pk");
            let allowed = index >= N / 2;
            doc.add_string("scope", if allowed { "allowed" } else { "excluded" })
                .expect("scope");
            let local = (index % (N / 2)) as f32;
            doc.add_vector_f32(
                "embedding",
                &[if allowed { 1_000.0 + local } else { local }, local % 13.0],
            )
            .expect("vector");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    indexed.insert(&refs).expect("insert");
    exact.insert(&refs).expect("insert");
    indexed.optimize().expect("optimize");
    exact.optimize().expect("optimize");

    let mut query = SearchQuery::new("embedding", &[1_050.0, 1.0], TOPK as i32).expect("query");
    query.set_filter("scope == \"allowed\"").expect("filter");
    let mut approx_query = query.clone();
    approx_query
        .set_diskann_params(a3s_vec::DiskannQueryParams::new(64))
        .expect("list_size");
    let approximate = indexed.query(&approx_query).expect("filtered DiskANN");
    let truth = exact.query(&query).expect("filtered exact");
    assert_eq!(
        approximate
            .iter()
            .map(|doc| doc.get_pk().unwrap().to_string())
            .collect::<Vec<_>>(),
        truth
            .iter()
            .map(|doc| doc.get_pk().unwrap().to_string())
            .collect::<Vec<_>>()
    );
}

#[test]
fn filtered_diskann_with_large_list_size_exercises_ann_candidate_path() {
    let temporary = tempdir().expect("temp");
    let indexed = Collection::create(
        temporary
            .path()
            .join("filtered-diskann-wide")
            .to_str()
            .expect("utf8"),
        &diskann_schema("filtered-diskann-wide"),
        Some(&options()),
    )
    .expect("indexed");

    const N: usize = 8_400;
    let docs: Vec<Doc> = (0..N)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("pk");
            let allowed = index >= N / 2;
            doc.add_string("scope", if allowed { "allowed" } else { "excluded" })
                .expect("scope");
            let local = (index % (N / 2)) as f32;
            doc.add_vector_f32(
                "embedding",
                &[if allowed { 1_000.0 + local } else { local }, local % 13.0],
            )
            .expect("vector");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    indexed.insert(&refs).expect("insert");
    indexed.optimize().expect("optimize");

    let mut query = SearchQuery::new("embedding", &[1_050.0, 1.0], TOPK as i32).expect("query");
    query.set_filter("scope == \"allowed\"").expect("filter");
    // Large list_size keeps traversal below the full allow-list so DiskANN
    // filtered candidate planning runs instead of exact fallback.
    query
        .set_diskann_params(a3s_vec::DiskannQueryParams::new(512))
        .expect("list_size");
    let hits = indexed.query(&query).expect("filtered DiskANN wide");
    assert_eq!(hits.len(), TOPK);
    for hit in &hits {
        match hit.field("scope") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "allowed"),
            other => panic!("expected allowed scope, got {other:?}"),
        }
    }
}

#[test]
fn large_scalar_and_conjunction_intersects_two_nonselective_bitmaps() {
    // Both sides exceed CONJUNCTION_EARLY_STOP (4096), so evaluation must
    // intersect rather than early-stop on a selective branch.
    let temporary = tempdir().expect("temp");
    let exact = Collection::create(
        temporary
            .path()
            .join("scalar-and-exact")
            .to_str()
            .expect("utf8"),
        &schema("scalar-and-exact", false),
        Some(&options()),
    )
    .expect("create exact");
    let indexed = Collection::create(
        temporary.path().join("scalar-and").to_str().expect("utf8"),
        &schema("scalar-and", true),
        Some(&options()),
    )
    .expect("create indexed");
    insert_fixture(&exact);
    insert_fixture(&indexed);

    // Query near the allowed cluster so filtered top-k is non-empty and stable.
    let filter = "scope == 'allowed' AND shard == 0";
    let mut query = SearchQuery::new("embedding", &[10_000.0, 0.0], TOPK as i32).expect("query");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query.set_filter(filter).expect("filter");
    let expected = exact.query(&query).expect("exact AND");
    // Exact scan on the indexed collection still evaluates inverted AND bitmaps.
    let actual = indexed.query(&query).expect("indexed AND");
    assert_eq!(comparable(&actual), comparable(&expected));
    assert_eq!(actual.len(), TOPK);
    for hit in &actual {
        match hit.field("scope") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "allowed"),
            other => panic!("expected allowed scope, got {other:?}"),
        }
        match hit.field("shard") {
            Some(a3s_vec::FieldValue::Int32(0)) => {}
            other => panic!("expected shard 0, got {other:?}"),
        }
    }
}

#[test]
fn empty_scalar_and_branch_short_circuits_without_evaluating_right() {
    // Empty left bitmap must return immediately (CONJUNCTION early empty path).
    let temporary = tempdir().expect("temp");
    let indexed = Collection::create(
        temporary
            .path()
            .join("scalar-and-empty")
            .to_str()
            .expect("utf8"),
        &schema("scalar-and-empty", true),
        Some(&options()),
    )
    .expect("create");
    insert_fixture(&indexed);

    let mut empty_left =
        SearchQuery::new("embedding", &[10_000.0, 0.0], TOPK as i32).expect("query");
    empty_left
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    empty_left
        .set_filter("scope == 'never-matches' AND shard == 0")
        .expect("filter");
    assert!(indexed.query(&empty_left).expect("empty left").is_empty());

    // Empty right after a non-selective left.
    let mut empty_right =
        SearchQuery::new("embedding", &[10_000.0, 0.0], TOPK as i32).expect("query");
    empty_right
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    empty_right
        .set_filter("scope == 'allowed' AND scope == 'never-matches'")
        .expect("filter");
    assert!(indexed.query(&empty_right).expect("empty right").is_empty());

    // Indexed left with unindexed contain_all right falls back conservatively.
    let mut mixed = SearchQuery::new("embedding", &[10_000.0, 0.0], TOPK as i32).expect("query");
    mixed
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    mixed
        .set_filter("scope == 'allowed' AND tags contain_all ['workspace']")
        .expect("filter");
    let hits = indexed.query(&mixed).expect("mixed AND");
    assert!(!hits.is_empty());
    for hit in &hits {
        match hit.field("scope") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "allowed"),
            other => panic!("expected allowed scope, got {other:?}"),
        }
    }

    // Inexact large scalar prefilter is refined; a non-matching contain_all
    // shrinks the candidate set below the exact-scan threshold.
    let mut refined = SearchQuery::new("embedding", &[10_000.0, 0.0], TOPK as i32).expect("query");
    refined
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    refined
        .set_filter("scope == 'allowed' AND tags contain_all ['no-such-tag']")
        .expect("filter");
    assert!(indexed.query(&refined).expect("refined empty").is_empty());
}
