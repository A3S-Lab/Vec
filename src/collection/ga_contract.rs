//! Shipped-path checks for the durability, cache, and exact-score contract.

use super::Collection;
use crate::{
    CollectionSchema, DataType, Doc, Durability, FieldSchema, FieldValue, HnswQueryParams,
    IndexParams, MetricType, SearchQuery,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

fn path_of(directory: &tempfile::TempDir, name: &str) -> String {
    directory
        .path()
        .join(name)
        .to_str()
        .expect("temporary path must be UTF-8")
        .to_string()
}

fn vector_schema(name: &str, metric: MetricType, hnsw: bool) -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("vector field");
    let params = if hnsw {
        IndexParams::hnsw(metric, 8, 64).expect("hnsw params")
    } else {
        IndexParams::flat(metric).expect("flat params")
    };
    embedding
        .set_index_params(&params)
        .expect("index params must attach");
    CollectionSchema::builder(name)
        .add_field(embedding)
        .build()
        .expect("schema")
}

fn vector_doc(id: &str, vector: &[f32]) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key");
    doc.add_vector_f32("embedding", vector)
        .expect("vector value");
    doc
}

fn ranked(docs: &[Doc]) -> Vec<(String, u32)> {
    docs.iter()
        .map(|doc| {
            (
                doc.get_pk()
                    .expect("query hit must have a primary key")
                    .to_string(),
                doc.get_score().to_bits(),
            )
        })
        .collect()
}

#[test]
fn enterprise_ga_defaults_stay_exact() {
    assert_eq!(Durability::default(), Durability::Always);
    let ivf = IndexParams::ivf(MetricType::L2, 4, 1, false).expect("ivf params");
    assert!(
        ivf.params.get("scale_factor").is_none(),
        "plain IVF must not gain a default scale_factor"
    );
    let query = SearchQuery::new("embedding", &[1.0, 0.0, 0.0, 0.0], 3).expect("query");
    assert_ne!(
        query
            .params
            .get("is_using_refiner")
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );
}

#[test]
fn enterprise_ga_public_scores_match_exact_oracle_and_radius() {
    let temporary = tempdir().expect("temporary directory");
    let docs = [
        vector_doc("doc-same", &[1.0, 0.0, 0.0, 0.0]),
        vector_doc("doc-mid", &[1.0, 1.0, 0.0, 0.0]),
        vector_doc("doc-far", &[0.0, 1.0, 0.0, 0.0]),
        vector_doc("doc-back", &[-1.0, 0.0, 0.0, 0.0]),
    ];
    let query_vector = [1.0_f32, 0.0, 0.0, 0.0];
    let mut flat_hits = None;
    let mut hnsw_hits = None;
    for (name, hnsw, slot) in [
        ("flat-oracle", false, &mut flat_hits),
        ("hnsw-oracle", true, &mut hnsw_hits),
    ] {
        let collection = Collection::create(
            &path_of(&temporary, name),
            &vector_schema(name, MetricType::Cosine, hnsw),
            None,
        )
        .expect("collection");
        let refs: Vec<&Doc> = docs.iter().collect();
        collection.insert(&refs).expect("insert");
        let query = SearchQuery::new("embedding", &query_vector, 4).expect("query");
        *slot = Some(collection.query(&query).expect("query must return"));
    }
    let flat_hits = flat_hits.expect("flat hits");
    let hnsw_hits = hnsw_hits.expect("hnsw hits");
    assert_eq!(ranked(&hnsw_hits), ranked(&flat_hits));
    assert_eq!(
        flat_hits
            .iter()
            .filter_map(|doc| doc.get_pk())
            .collect::<Vec<_>>(),
        vec!["doc-same", "doc-mid", "doc-far", "doc-back"]
    );

    let collection = Collection::open(&path_of(&temporary, "hnsw-oracle"), None).expect("reopen");
    let mut excluding = SearchQuery::new("embedding", &query_vector, 4).expect("query");
    excluding.set_radius(2.0).expect("radius");
    assert!(collection
        .query(&excluding)
        .expect("high radius query")
        .is_empty());
    let mut open_radius = SearchQuery::new("embedding", &query_vector, 4).expect("query");
    open_radius.set_radius(-1.0).expect("radius");
    assert_eq!(
        ranked(&collection.query(&open_radius).expect("negative radius")),
        ranked(&hnsw_hits)
    );
}

#[test]
fn enterprise_ga_default_hnsw_ef_matches_explicit_64() {
    let temporary = tempdir().expect("temporary directory");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("vector field");
    embedding
        .set_index_params(&IndexParams::hnsw(MetricType::L2, 8, 64).expect("hnsw"))
        .expect("attach");
    let schema = CollectionSchema::builder("ef-64")
        .add_field(embedding)
        .build()
        .expect("schema");
    let collection =
        Collection::create(&path_of(&temporary, "ef"), &schema, None).expect("collection");
    let docs: Vec<Doc> = (0..96)
        .map(|index| {
            let id = format!("doc-{index:03}");
            let base = f32::from(u16::try_from(index).expect("index fits"));
            vector_doc(&id, &[base, base * 0.5, base.sin(), (base + 1.0).cos()])
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    let query_vector = [3.0_f32, 1.5, 0.1, 0.2];
    let implicit = SearchQuery::new("embedding", &query_vector, 5).expect("query");
    let mut explicit = SearchQuery::new("embedding", &query_vector, 5).expect("query");
    explicit
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
        .expect("ef 64");
    assert_eq!(
        ranked(&collection.query(&implicit).expect("default ef")),
        ranked(&collection.query(&explicit).expect("explicit ef"))
    );
}

#[test]
fn enterprise_ga_always_sync_lets_prior_revision_query_return() {
    let temporary = tempdir().expect("temporary directory");
    let collection = Collection::create(
        &path_of(&temporary, "sync"),
        &vector_schema("sync", MetricType::Cosine, false),
        None,
    )
    .expect("collection");
    let published = vector_doc("published", &[1.0, 0.0, 0.0, 0.0]);
    collection.insert(&[&published]).expect("first insert");

    let gate = collection.test_arm_wal_sync_stall();
    let acked = Arc::new(AtomicBool::new(false));
    let pending = vector_doc("pending", &[0.0, 1.0, 0.0, 0.0]);
    let writer = collection.clone();
    let acked_flag = Arc::clone(&acked);
    let insert_thread = thread::spawn(move || {
        let result = writer.insert(&[&pending]);
        acked_flag.store(true, Ordering::Release);
        result
    });

    assert!(
        gate.wait_entered(Duration::from_secs(5)),
        "Always sync did not reach the durability stall"
    );
    assert!(
        !acked.load(Ordering::Acquire),
        "commit was acknowledged before the durability sync finished"
    );

    let reader = collection.clone();
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let query = SearchQuery::new("embedding", &[1.0, 0.0, 0.0, 0.0], 4).expect("query");
        let _ = sender.send(reader.query(&query));
    });
    let query_result = receiver.recv_timeout(Duration::from_secs(2));
    gate.release();
    let hits = query_result
        .expect("published revision query blocked during durability sync")
        .expect("query");
    let ids: Vec<_> = hits.iter().filter_map(|doc| doc.get_pk()).collect();
    assert_eq!(ids, vec!["published"]);
    assert!(collection.fetch(&["pending"]).expect("fetch").is_empty());

    insert_thread
        .join()
        .expect("insert thread")
        .expect("stalled insert must succeed after sync");
    assert!(acked.load(Ordering::Acquire));
    assert_eq!(collection.fetch(&["pending"]).expect("fetch").len(), 1);

    let path = path_of(&temporary, "sync");
    drop(collection);
    let reopened = Collection::open(&path, None).expect("reopen");
    assert_eq!(reopened.count().expect("count"), 2);
    assert_eq!(reopened.fetch(&["published"]).expect("fetch").len(), 1);
    assert_eq!(reopened.fetch(&["pending"]).expect("fetch").len(), 1);
}

#[test]
fn enterprise_ga_diskann_sidecar_failure_keeps_index_cache() {
    let temporary = tempdir().expect("temporary directory");
    let mut tag = FieldSchema::new("tag", DataType::String, false, 0).expect("tag field");
    tag.set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("invert attaches");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 8).expect("vector field");
    embedding
        .set_index_params(&IndexParams::diskann(MetricType::L2, 8, 16, 1).expect("diskann"))
        .expect("diskann attaches");
    let schema = CollectionSchema::builder("sidecar")
        .add_field(tag)
        .add_field(embedding)
        .build()
        .expect("schema");
    let path = path_of(&temporary, "sidecar");
    let collection = Collection::create(&path, &schema, None).expect("collection");
    let mut docs = Vec::new();
    for index in 0..8 {
        let id = if index == 0 {
            "alpha-doc".to_string()
        } else {
            format!("doc-{index}")
        };
        let mut doc = Doc::with_pk(&id).expect("primary key");
        let tag_value = if index == 0 { "alpha" } else { "beta" };
        doc.add_string("tag", tag_value).expect("tag");
        let vector: Vec<f32> = (0..8)
            .map(|coordinate| {
                f32::from(u16::try_from(index + coordinate).expect("coordinate fits"))
            })
            .collect();
        doc.add_vector_f32("embedding", &vector).expect("vector");
        docs.push(doc);
    }
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.test_arm_diskann_write_fault();
    collection.flush().expect("flush");
    assert!(
        collection.test_diskann_write_fault_fired(),
        "DiskANN sidecar write was not attempted"
    );
    drop(collection);

    let reopened = Collection::open(&path, None).expect("reopen");
    assert!(
        reopened.stats().expect("stats").index_cache_hit,
        "sidecar failure rebuilt the index registry instead of restoring the cache"
    );
    let mut query =
        SearchQuery::new("embedding", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 4).expect("query");
    query.set_filter("tag == \"alpha\"").expect("filter");
    let hits = reopened.query(&query).expect("filtered query");
    assert_eq!(
        hits.iter()
            .filter_map(|doc| doc.get_pk())
            .collect::<Vec<_>>(),
        vec!["alpha-doc"]
    );
    assert_eq!(
        reopened
            .fetch(&["alpha-doc"])
            .expect("fetch")
            .first()
            .and_then(|doc| doc.field("tag")),
        Some(&FieldValue::String("alpha".to_string()))
    );
}
