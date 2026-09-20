//! Engine contract for A3S Code workspace retrieval.
//!
//! Code's shipped lexical path uses `zvec-rust` FTS with a whitespace tokenizer
//! and one temporary collection per file partition. Semantic retrieval today
//! uses `a3s-memory`. The platform target replaces both with `a3s-vec` while
//! keeping Code as the catalog and authority for chunk identity. This fixture
//! locks the engine surface that replacement needs: session-scoped temporary
//! collections, whitespace FTS, dense HNSW cosine, atomic partition replace,
//! hybrid RRF, and a clean close.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema, Fts,
    HnswQueryParams, IndexParams, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use std::collections::BTreeSet;
use tempfile::tempdir;

const DIMENSION: usize = 8;
const INSERT_BATCH_SIZE: usize = 256;

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn read_only_options() -> CollectionOptions {
    let mut options = manual_options();
    options
        .set_read_only(true)
        .expect("read-only options must be valid");
    options
}

fn lexical_schema() -> CollectionSchema {
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body field");
    body.set_index_params(
        &IndexParams::fts(Some("whitespace"), None, None).expect("whitespace FTS"),
    )
    .expect("body accepts FTS");
    CollectionSchema::builder("workspace_lexical")
        .add_field(body)
        .build()
        .expect("lexical schema")
}

fn semantic_schema() -> CollectionSchema {
    let mut partition =
        FieldSchema::new("partition", DataType::String, false, 0).expect("partition field");
    partition
        .set_index_params(&IndexParams::invert(true, true).expect("partition invert"))
        .expect("partition accepts invert");
    let mut embedding = FieldSchema::new(
        "embedding",
        DataType::VectorFp32,
        false,
        u32::try_from(DIMENSION).expect("dimension fits"),
    )
    .expect("embedding field");
    embedding
        .set_index_params(&IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW params"))
        .expect("embedding accepts HNSW");
    CollectionSchema::builder("workspace_semantic")
        .add_field(partition)
        .add_field(embedding)
        .build()
        .expect("semantic schema")
}

fn dense_vector(seed: usize) -> Vec<f32> {
    (0..DIMENSION)
        .map(|axis| {
            let raw =
                u16::try_from((seed.wrapping_mul(31).wrapping_add(axis.wrapping_mul(17))) % 101)
                    .expect("fixture value fits u16");
            (f32::from(raw) - 50.0) / 50.0
        })
        .collect()
}

fn publish_lexical_partition(
    root: &std::path::Path,
    documents: &[(&str, &str)],
) -> std::path::PathBuf {
    let path = root.join("partition");
    let path_str = path.to_str().expect("UTF-8 path");
    let collection =
        Collection::create_and_open(path_str, &lexical_schema(), Some(&manual_options()))
            .expect("lexical collection must open");
    for batch in documents.chunks(INSERT_BATCH_SIZE) {
        let docs: Vec<Doc> = batch
            .iter()
            .map(|(id, body)| {
                let mut doc = Doc::with_pk(*id).expect("primary key");
                doc.add_string("body", body).expect("body");
                doc
            })
            .collect();
        let refs: Vec<&Doc> = docs.iter().collect();
        collection.insert(&refs).expect("lexical insert");
    }
    collection.flush().expect("lexical flush");
    collection.close().expect("lexical close");
    path
}

fn query_lexical(path: &std::path::Path, terms: &str, topk: i32) -> Vec<String> {
    let path_str = path.to_str().expect("UTF-8 path");
    let collection =
        Collection::open(path_str, Some(&read_only_options())).expect("read-only reopen");
    let mut fts = Fts::new().expect("FTS handle");
    fts.set_match_string(terms)
        .expect("FTS expression must be valid");
    let query = SearchQuery::fts("body", &fts, topk).expect("FTS query");
    let results = collection.query(&query).expect("FTS results");
    let ids = results
        .into_iter()
        .filter_map(|doc| doc.get_pk().map(str::to_owned))
        .collect();
    collection.close().expect("read-only close");
    ids
}

#[test]
fn whitespace_fts_partition_matches_code_lexical_lifecycle() {
    let root = tempdir().expect("temp root");
    let path = publish_lexical_partition(
        root.path(),
        &[
            ("d0", "workspace retrieval contract"),
            ("d1", "vector index partition"),
            ("d2", "workspace retrieval engine"),
        ],
    );
    let hits = query_lexical(&path, "workspace retrieval", 10);
    assert_eq!(hits, vec!["d0".to_owned(), "d2".to_owned()]);
}

#[test]
fn dense_hnsw_partition_replace_is_atomic_for_one_file() {
    let root = tempdir().expect("temp root");
    let path = root.path().join("semantic");
    let path_str = path.to_str().expect("UTF-8 path");
    let collection =
        Collection::create_and_open(path_str, &semantic_schema(), Some(&manual_options()))
            .expect("semantic collection");

    let publish = |partition: &str, seeds: &[usize]| {
        let docs: Vec<Doc> = seeds
            .iter()
            .map(|seed| {
                let mut doc =
                    Doc::with_pk(format!("{partition}-{seed}")).expect("semantic primary key");
                doc.add_string("partition", partition).expect("partition");
                doc.add_vector_f32("embedding", &dense_vector(*seed))
                    .expect("embedding");
                doc
            })
            .collect();
        let refs: Vec<&Doc> = docs.iter().collect();
        collection.upsert(&refs).expect("partition upsert");
    };

    publish("file-a", &[1, 2, 3]);
    publish("file-b", &[10, 11]);
    collection.flush().expect("flush after first publish");

    // Atomic file replace: delete the old partition keys, then publish the new
    // generation. Code fences this with catalog revision ownership.
    collection
        .delete(&["file-a-1", "file-a-2", "file-a-3"])
        .expect("tombstone old partition");
    publish("file-a", &[4, 5]);
    collection.flush().expect("flush after replace");

    let mut query = SearchQuery::new("embedding", &dense_vector(4), 5).expect("dense query");
    query
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
        .expect("HNSW params");
    query
        .set_filter("partition == \"file-a\"")
        .expect("partition filter");
    let results = collection.query(&query).expect("semantic query");
    let ids: BTreeSet<_> = results
        .into_iter()
        .filter_map(|doc| doc.get_pk().map(str::to_owned))
        .collect();
    assert_eq!(
        ids,
        BTreeSet::from(["file-a-4".to_owned(), "file-a-5".to_owned()])
    );
    assert!(!ids.contains("file-a-1"));
    collection.close().expect("semantic close");
}

#[test]
fn hybrid_rrf_fuses_whitespace_fts_and_dense_hnsw() {
    let root = tempdir().expect("temp root");
    let path = root.path().join("hybrid");
    let path_str = path.to_str().expect("UTF-8 path");

    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body");
    body.set_index_params(
        &IndexParams::fts(Some("whitespace"), None, None).expect("whitespace FTS"),
    )
    .expect("body FTS");
    let mut embedding = FieldSchema::new(
        "embedding",
        DataType::VectorFp32,
        false,
        u32::try_from(DIMENSION).expect("dimension"),
    )
    .expect("embedding");
    embedding
        .set_index_params(&IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW"))
        .expect("embedding HNSW");
    let schema = CollectionSchema::builder("workspace_hybrid")
        .add_field(body)
        .add_field(embedding)
        .build()
        .expect("hybrid schema");

    let collection =
        Collection::create_and_open(path_str, &schema, Some(&manual_options())).expect("open");
    let corpus = [
        ("d0", "workspace retrieval contract", 1usize),
        ("d1", "unrelated tokens", 50usize),
        ("d2", "workspace retrieval engine", 2usize),
    ];
    let docs: Vec<Doc> = corpus
        .iter()
        .map(|(id, body, seed)| {
            let mut doc = Doc::with_pk(*id).expect("pk");
            doc.add_string("body", body).expect("body");
            doc.add_vector_f32("embedding", &dense_vector(*seed))
                .expect("embedding");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.flush().expect("flush");

    let mut fts = Fts::new().expect("FTS");
    fts.set_match_string("workspace retrieval")
        .expect("lexical text");
    let mut lexical = SubQuery::new().expect("lexical sub");
    lexical.set_field_name("body").expect("lexical field");
    lexical.set_fts(&fts).expect("lexical FTS");
    lexical.set_num_candidates(10).expect("lexical candidates");

    let mut dense = SubQuery::new().expect("dense sub");
    dense.set_field_name("embedding").expect("dense field");
    dense
        .set_query_vector(&dense_vector(1))
        .expect("dense vector");
    dense.set_num_candidates(10).expect("dense candidates");
    dense
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, false))
        .expect("hnsw");

    let mut multi = MultiQuery::new().expect("multi");
    multi.set_topk(10).expect("topk");
    multi.add_sub_query(&lexical).expect("add lexical");
    multi.add_sub_query(&dense).expect("add dense");
    multi.set_rerank_rrf(60).expect("RRF");
    let fused = collection.multi_query(&multi).expect("hybrid");
    let top = fused.first().and_then(Doc::get_pk).expect("top hit");
    assert_eq!(top, "d0");
    collection.close().expect("close");
}
