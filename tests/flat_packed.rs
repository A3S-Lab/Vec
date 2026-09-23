#![allow(
    clippy::approx_constant,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

//! Packed dense Flat vs DocumentMap oracle (TESTING.md Wave A / F-P0-1..4).
//!
//! Public scores remain the DocumentMap Cosine/`f64` contract. Packed Flat is
//! only an acceleration cache rebuilt by `optimize` / `rebuild_index`.

use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, IndexParams, IndexType, MetricType,
    SearchQuery,
};
use serde_json::json;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use tempfile::tempdir;

fn flat_schema(dimension: u32) -> CollectionSchema {
    flat_schema_metric(dimension, MetricType::Cosine)
}

fn flat_schema_metric(dimension: u32, metric: MetricType) -> CollectionSchema {
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, dimension)
        .expect("embedding schema");
    embedding
        .set_index_params(&IndexParams::flat(metric).expect("Flat"))
        .expect("attach Flat");
    CollectionSchema::builder("flat-packed-metric")
        .add_field(embedding)
        .build()
        .expect("schema")
}

fn l2_oracle(query: &[f32], vector: &[f32]) -> f64 {
    -query
        .iter()
        .zip(vector)
        .map(|(a, b)| {
            let d = f64::from(*a) - f64::from(*b);
            d * d
        })
        .sum::<f64>()
}

fn ip_oracle(query: &[f32], vector: &[f32]) -> f64 {
    query
        .iter()
        .zip(vector)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn ranked_metric_oracle(
    corpus: &[(String, Vec<f32>)],
    query: &[f32],
    topk: usize,
    score: fn(&[f32], &[f32]) -> f64,
) -> Vec<(String, f64)> {
    let mut ranked: Vec<(String, f64)> = corpus
        .iter()
        .map(|(id, vector)| (id.clone(), score(query, vector)))
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(topk);
    ranked
}

#[test]
fn dense_flat_l2_and_ip_after_optimize_match_oracles() {
    let temporary = tempdir().expect("tempdir");
    for (metric, score_fn, name) in [
        (MetricType::L2, l2_oracle as fn(&[f32], &[f32]) -> f64, "l2"),
        (MetricType::Ip, ip_oracle, "ip"),
    ] {
        let path = temporary
            .path()
            .join(name)
            .to_str()
            .expect("utf8")
            .to_string();
        let collection =
            Collection::create(&path, &flat_schema_metric(3, metric), None).expect("create");
        let corpus: Vec<(String, Vec<f32>)> = vec![
            ("p".into(), vec![1.0, 0.0, 0.0]),
            ("q".into(), vec![0.0, 1.0, 0.0]),
            ("r".into(), vec![0.5, 0.5, 0.0]),
            ("s".into(), vec![-0.2, 0.1, 0.8]),
        ];
        let docs: Vec<Doc> = corpus
            .iter()
            .map(|(id, vector)| dense_doc(id, vector))
            .collect();
        let refs: Vec<&Doc> = docs.iter().collect();
        collection.insert(&refs).expect("insert");
        collection.optimize().expect("optimize");
        let query = [0.8_f32, 0.2, 0.1];
        let hits = collection
            .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
            .expect("query");
        assert_matches_oracle(&hits, &ranked_metric_oracle(&corpus, &query, 3, score_fn));
    }
}

fn binary_flat_schema() -> CollectionSchema {
    let mut bits =
        FieldSchema::new("bits", DataType::VectorBinary32, false, 32).expect("binary schema");
    bits.set_index_params(&IndexParams::flat(MetricType::L2).expect("Flat L2"))
        .expect("attach Flat");
    CollectionSchema::builder("binary-flat-packed")
        .add_field(bits)
        .build()
        .expect("schema")
}

fn dense_doc(id: &str, vector: &[f32]) -> Doc {
    let mut doc = Doc::with_pk(id).expect("pk");
    doc.add_vector_f32("embedding", vector).expect("vector");
    doc
}

fn cosine_oracle(query: &[f32], vector: &[f32]) -> f64 {
    let query: Vec<f64> = query.iter().map(|v| f64::from(*v)).collect();
    let vector: Vec<f64> = vector.iter().map(|v| f64::from(*v)).collect();
    let dot: f64 = query.iter().zip(&vector).map(|(a, b)| a * b).sum();
    let qn: f64 = query.iter().map(|v| v * v).sum::<f64>().sqrt();
    let vn: f64 = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
    if qn == 0.0 || vn == 0.0 {
        0.0
    } else {
        dot / (qn * vn)
    }
}

fn ranked_oracle(corpus: &[(String, Vec<f32>)], query: &[f32], topk: usize) -> Vec<(String, f64)> {
    let mut ranked: Vec<(String, f64)> = corpus
        .iter()
        .map(|(id, vector)| (id.clone(), cosine_oracle(query, vector)))
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(topk);
    ranked
}

fn assert_matches_oracle(hits: &[Doc], expected: &[(String, f64)]) {
    assert_eq!(hits.len(), expected.len(), "hit count");
    for (hit, (id, score)) in hits.iter().zip(expected) {
        assert_eq!(hit.get_pk(), Some(id.as_str()), "id order");
        let actual = f64::from(hit.get_score());
        let tolerance = (score.abs() * 1.0e-5).max(1.0e-5);
        assert!(
            (actual - score).abs() <= tolerance,
            "id={id} expected={score} actual={actual}"
        );
    }
}

fn open_flat(path: &str, dimension: u32) -> Collection {
    Collection::create(path, &flat_schema(dimension), None).expect("create")
}

/// F-P0-2: after insert, before optimize, Flat stays ready and matches oracle.
#[test]
fn dense_flat_without_pack_matches_document_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("pre-optimize")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 4);
    let corpus: Vec<(String, Vec<f32>)> = vec![
        ("a".into(), vec![1.0_f32, 0.0, 0.0, 0.0]),
        ("b".into(), vec![0.0, 1.0, 0.0, 0.0]),
        ("c".into(), vec![0.8, 0.2, 0.0, 0.0]),
        ("d".into(), vec![-1.0, 0.0, 0.0, 0.0]),
        ("e".into(), vec![0.6, 0.4, 0.1, 0.0]),
    ];
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");

    let stats = collection.stats().expect("stats");
    let flat = stats
        .indexes
        .iter()
        .find(|index| index.name == "embedding")
        .expect("Flat stats");
    assert_eq!(flat.index_type, IndexType::Flat);
    assert_eq!(flat.state, "ready");
    assert_eq!(flat.document_count, 5);

    let query = [0.9_f32, 0.1, 0.0, 0.0];
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
        .expect("query");
    assert_matches_oracle(&hits, &ranked_oracle(&corpus, &query, 3));
}

/// F-P0-1: after optimize, packed Flat still matches DocumentMap Cosine oracle.
#[test]
fn dense_flat_after_optimize_matches_document_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("post-optimize")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 4);
    let corpus: Vec<(String, Vec<f32>)> = (0..64)
        .map(|index| {
            let angle = index as f32 * 0.11;
            (
                format!("doc-{index:03}"),
                vec![angle.cos(), angle.sin(), 0.25, -0.1],
            )
        })
        .collect();
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("optimize rebuilds Flat pack");

    let stats = collection.stats().expect("stats");
    let flat = stats
        .indexes
        .iter()
        .find(|index| index.name == "embedding")
        .expect("Flat stats");
    assert_eq!(flat.state, "ready");
    assert_eq!(flat.document_count, 64);
    assert!(
        flat.estimated_payload_bytes.is_some_and(|bytes| bytes > 0),
        "packed Flat should expose payload bytes after rebuild"
    );

    let query = [0.7_f32, 0.3, 0.2, -0.05];
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 10).expect("query"))
        .expect("query");
    assert_matches_oracle(&hits, &ranked_oracle(&corpus, &query, 10));
}

/// F-P0-3: binary Flat never enters the f32 pack path; reopen stays healthy.
#[test]
fn binary_flat_never_builds_packed_f32_index_and_reopens() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("binary-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection =
        Collection::create(&path, &binary_flat_schema(), None).expect("create binary Flat");
    let mut doc = Doc::with_pk("bits-0").expect("pk");
    doc.add_vector_binary32("bits", &[0b1010_1010; 4])
        .expect("bits");
    collection.insert(&[&doc]).expect("insert");
    collection
        .optimize()
        .expect("optimize must not encode binary");

    let stats = collection.stats().expect("stats");
    let flat = stats
        .indexes
        .iter()
        .find(|index| index.name == "bits")
        .expect("binary Flat stats");
    assert_eq!(flat.index_type, IndexType::Flat);
    assert_eq!(flat.state, "ready");
    assert_eq!(flat.document_count, 1);
    assert!(
        flat.estimated_payload_bytes.is_none(),
        "binary Flat must not report a packed f32 payload"
    );

    let hits = collection
        .query(&SearchQuery::binary("bits", &[0b1010_1010; 4], 1).expect("query"))
        .expect("binary query");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].get_pk(), Some("bits-0"));

    collection.flush().expect("flush");
    collection.close().expect("close");
    let reopened = Collection::open(&path, None).expect("reopen without encode_vector failure");
    let again = reopened
        .query(&SearchQuery::binary("bits", &[0b1010_1010; 4], 1).expect("query"))
        .expect("reopened query");
    assert_eq!(again[0].get_pk(), Some("bits-0"));
}

/// F-P0-4: large Flat corpus (≥4096) matches oracle under the process Rayon pool.
///
/// Serial (`RAYON_NUM_THREADS=1`) and multi-worker paths are both proven by
/// matching the same independent DocumentMap Cosine oracle—not by lowering
/// `ef` or weakening public scores.
#[test]
fn dense_flat_large_corpus_matches_oracle_under_rayon_pool() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("large-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let dimension = 8_u32;
    let collection = open_flat(&path, dimension);
    let n = 4_200_usize;
    let corpus: Vec<(String, Vec<f32>)> = (0..n)
        .map(|index| {
            let mut vector = vec![0.0_f32; usize::try_from(dimension).expect("dim")];
            for (offset, slot) in vector.iter_mut().enumerate() {
                let raw = ((index * 17 + offset * 13) % 97) as i32 - 48;
                *slot = f32::from(raw as i16) / 24.0;
            }
            (format!("row-{index:05}"), vector)
        })
        .collect();

    const BATCH: usize = 256;
    for chunk in corpus.chunks(BATCH) {
        let docs: Vec<Doc> = chunk
            .iter()
            .map(|(id, vector)| dense_doc(id, vector))
            .collect();
        let refs: Vec<&Doc> = docs.iter().collect();
        collection.insert(&refs).expect("batch insert");
    }
    collection.optimize().expect("pack Flat");

    let query = {
        let mut vector = vec![0.0_f32; usize::try_from(dimension).expect("dim")];
        for (offset, slot) in vector.iter_mut().enumerate() {
            *slot = (offset as f32 + 1.0) * 0.07;
        }
        vector
    };
    let topk = 15;
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, topk as i32).expect("query"))
        .expect("large Flat query");
    assert_matches_oracle(&hits, &ranked_oracle(&corpus, &query, topk));
}

/// Deletes and upserts between rebuilds still match the DocumentMap oracle.
#[test]
fn dense_flat_mutations_then_rebuild_match_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("mutate-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 3);
    let mut corpus: BTreeMap<String, Vec<f32>> = [
        ("keep".into(), vec![1.0_f32, 0.0, 0.0]),
        ("drop".into(), vec![0.0, 1.0, 0.0]),
        ("change".into(), vec![0.0, 0.0, 1.0]),
    ]
    .into_iter()
    .collect();
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("initial pack");

    collection.delete(&["drop"]).expect("delete");
    corpus.remove("drop");
    let updated = dense_doc("change", &[0.5, 0.5, 0.0]);
    collection.upsert(&[&updated]).expect("upsert");
    corpus.insert("change".into(), vec![0.5, 0.5, 0.0]);
    collection.optimize().expect("rebuild after mutations");

    let query = [0.6_f32, 0.4, 0.0];
    let ordered: Vec<(String, Vec<f32>)> = corpus.into_iter().collect();
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
        .expect("query");
    assert_matches_oracle(&hits, &ranked_oracle(&ordered, &query, 3));
}

/// Zero-norm Cosine stays finite and keeps primary-key top-k (F-P0-6).
#[test]
fn dense_flat_zero_norm_query_matches_oracle_topk() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("zero-norm")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 2);
    // Insert order is not primary-key order, and top-k is smaller than the corpus.
    let corpus = vec![
        ("c".to_string(), vec![0.0_f32, 1.0]),
        ("a".to_string(), vec![1.0, 0.0]),
        ("b".to_string(), vec![0.0, 1.0]),
    ];
    for (id, vector) in &corpus {
        let doc = dense_doc(id, vector);
        collection.insert(&[&doc]).expect("insert");
    }
    let query = [0.0_f32, 0.0];
    let expected = ranked_oracle(&corpus, &query, 2);
    let before = collection
        .query(&SearchQuery::new("embedding", &query, 2).expect("query"))
        .expect("query before pack");
    assert_matches_oracle(&before, &expected);
    assert!(before.iter().all(|hit| hit.get_score().is_finite()));
    collection.optimize().expect("pack Flat");
    let after = collection
        .query(&SearchQuery::new("embedding", &query, 2).expect("query"))
        .expect("query after pack");
    assert_matches_oracle(&after, &expected);
    assert!(after.iter().all(|hit| hit.get_score().is_finite()));
}

/// Sparse live set (delete without rebuild) still matches the oracle.
#[test]
fn dense_flat_sparse_live_set_after_delete_matches_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("sparse-live")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 2);
    let mut corpus: Vec<(String, Vec<f32>)> = vec![
        ("a".into(), vec![1.0, 0.0]),
        ("b".into(), vec![0.0, 1.0]),
        ("c".into(), vec![0.7, 0.3]),
        ("d".into(), vec![0.2, 0.8]),
    ];
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("pack");
    collection.delete(&["b"]).expect("delete without rebuild");
    corpus.retain(|(id, _)| id != "b");
    let query = [0.9_f32, 0.1];
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
        .expect("query");
    assert_matches_oracle(&hits, &ranked_oracle(&corpus, &query, 3));
}

#[test]
fn flat_cosine_metric_override_still_matches_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("metric-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 2);
    let corpus: Vec<(String, Vec<f32>)> = vec![
        ("x".into(), vec![1.0_f32, 0.0]),
        ("y".into(), vec![0.0, 1.0]),
        ("z".into(), vec![0.7071, 0.7071]),
    ];
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("optimize");

    let query = [1.0_f32, 0.0];
    let mut search = SearchQuery::new("embedding", &query, 2).expect("query");
    search.params.insert("metric".into(), json!("cosine"));
    let hits = collection.query(&search).expect("query");
    assert_matches_oracle(&hits, &ranked_oracle(&corpus, &query, 2));
}

/// Insert without optimize still matches DocumentMap (packed cache absent).
#[test]
fn dense_flat_without_optimize_matches_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("no-pack")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 2);
    let corpus: Vec<(String, Vec<f32>)> = vec![
        ("a".into(), vec![1.0, 0.0]),
        ("b".into(), vec![0.0, 1.0]),
        ("c".into(), vec![0.6, 0.8]),
    ];
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert without optimize");
    let query = [0.9_f32, 0.1];
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
        .expect("unpacked Flat query");
    assert_matches_oracle(&hits, &ranked_oracle(&corpus, &query, 3));
}

/// Filtered Flat query matches the filtered DocumentMap oracle.
#[test]
fn dense_flat_filtered_query_matches_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("filtered-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::Cosine).expect("Flat"))
        .expect("attach");
    let mut bucket = FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket");
    bucket
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("invert");
    let schema = CollectionSchema::builder("filtered-flat")
        .add_field(embedding)
        .add_field(bucket)
        .build()
        .expect("schema");
    let collection = Collection::create(&path, &schema, None).expect("create");
    let mut docs = Vec::new();
    for (id, vector, bucket_id) in [
        ("a", vec![1.0_f32, 0.0], 1),
        ("b", vec![0.0, 1.0], 2),
        ("c", vec![0.7, 0.3], 1),
        ("d", vec![0.2, 0.8], 2),
    ] {
        let mut doc = Doc::with_pk(id).expect("pk");
        doc.add_vector_f32("embedding", &vector).expect("vector");
        doc.add_i32("bucket", bucket_id).expect("bucket");
        docs.push(doc);
    }
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("optimize");

    let query = [1.0_f32, 0.0];
    let mut search = SearchQuery::new("embedding", &query, 3).expect("query");
    search.set_filter("bucket == 1").expect("filter");
    let hits = collection.query(&search).expect("filtered query");
    let filtered_corpus = vec![
        ("a".into(), vec![1.0_f32, 0.0]),
        ("c".into(), vec![0.7, 0.3]),
    ];
    assert_matches_oracle(&hits, &ranked_oracle(&filtered_corpus, &query, 3));
}

/// Force Rayon pool >1 over ≥4096 slots; results still match the oracle (I6).
#[test]
fn dense_flat_forced_multi_worker_rayon_matches_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("rayon-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let dimension = 4_u32;
    let collection = open_flat(&path, dimension);
    let n = 4_200_usize;
    let corpus: Vec<(String, Vec<f32>)> = (0..n)
        .map(|index| {
            let mut vector = vec![0.0_f32; usize::try_from(dimension).expect("dim")];
            for (offset, slot) in vector.iter_mut().enumerate() {
                let raw = ((index * 19 + offset * 11) % 89) as i32 - 44;
                *slot = f32::from(raw as i16) / 22.0;
            }
            (format!("r-{index:05}"), vector)
        })
        .collect();
    for chunk in corpus.chunks(256) {
        let docs: Vec<Doc> = chunk
            .iter()
            .map(|(id, vector)| dense_doc(id, vector))
            .collect();
        let refs: Vec<&Doc> = docs.iter().collect();
        collection.insert(&refs).expect("batch");
    }
    collection.optimize().expect("pack");

    let query = {
        let mut vector = vec![0.0_f32; usize::try_from(dimension).expect("dim")];
        for (offset, slot) in vector.iter_mut().enumerate() {
            *slot = (offset as f32 + 1.0) * 0.11;
        }
        vector
    };
    let topk = 12;
    let expected = ranked_oracle(&corpus, &query, topk);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("rayon pool");
    pool.install(|| {
        let hits = collection
            .query(&SearchQuery::new("embedding", &query, topk as i32).expect("query"))
            .expect("multi-worker Flat");
        assert_matches_oracle(&hits, &expected);
    });
}

/// Upsert without rebuild keeps Flat revision current with overlay; results
/// still match DocumentMap (exercises live-scan / delta path, not packed-only).
#[test]
fn dense_flat_upsert_without_rebuild_matches_oracle() {
    let temporary = tempdir().expect("tempdir");
    let path = temporary
        .path()
        .join("upsert-flat")
        .to_str()
        .expect("utf8")
        .to_string();
    let collection = open_flat(&path, 2);
    let mut corpus: BTreeMap<String, Vec<f32>> = [
        ("a".into(), vec![1.0_f32, 0.0]),
        ("b".into(), vec![0.0, 1.0]),
        ("c".into(), vec![0.7, 0.3]),
    ]
    .into_iter()
    .collect();
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, vector)| dense_doc(id, vector))
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.optimize().expect("pack");

    let updated = dense_doc("b", &[0.9, 0.1]);
    collection
        .upsert(&[&updated])
        .expect("upsert without rebuild");
    corpus.insert("b".into(), vec![0.9, 0.1]);

    let query = [1.0_f32, 0.0];
    let ordered: Vec<(String, Vec<f32>)> = corpus.into_iter().collect();
    let hits = collection
        .query(&SearchQuery::new("embedding", &query, 3).expect("query"))
        .expect("overlay Flat query");
    assert_matches_oracle(&hits, &ranked_oracle(&ordered, &query, 3));
}
