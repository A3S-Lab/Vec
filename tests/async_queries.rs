#![cfg(feature = "async")]

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc,
    FieldSchema, GroupBySearchQuery, IndexParams, MetricType, MultiQuery, SearchQuery, SubQuery,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const DISKANN_PATH: &str = "indexes/diskann-graph.bin";
const DISKANN_SECTOR_BYTES: u64 = 4_096;

fn schema() -> CollectionSchema {
    let language =
        FieldSchema::new("language", DataType::String, false, 0).expect("field must be valid");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field must be valid");
    embedding
        .set_index_params(
            &IndexParams::diskann(MetricType::L2, 16, 64, 1)
                .expect("DiskANN parameters must be valid"),
        )
        .expect("DiskANN index must be valid");
    CollectionSchema::builder("async-diskann-contract")
        .add_field(language)
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn documents(count: usize) -> Vec<Doc> {
    (0..count)
        .map(|index| {
            let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
            let mut doc = Doc::with_pk(format!("doc-{index:03}")).expect("document must be valid");
            doc.add_string("language", if index % 2 == 0 { "rust" } else { "python" })
                .expect("language must be valid");
            doc.add_vector_f32("embedding", &[coordinate, coordinate % 7.0])
                .expect("embedding must be valid");
            doc
        })
        .collect()
}

fn vector_query(coordinate: f32, topk: i32) -> SearchQuery {
    let mut query = SearchQuery::new("embedding", &[coordinate, coordinate % 7.0], topk)
        .expect("query must be valid");
    query
        .set_diskann_params(DiskannQueryParams::new(48))
        .expect("DiskANN query controls must be valid");
    query
}

fn multi_query(coordinate: f32) -> MultiQuery {
    let mut branch = SubQuery::new().expect("sub-query must be valid");
    branch
        .set_field_name("embedding")
        .expect("field name must be valid");
    branch
        .set_query_vector(&[coordinate, coordinate % 7.0])
        .expect("query vector must be valid");
    branch
        .set_num_candidates(10)
        .expect("candidate limit must be valid");
    branch
        .set_diskann_params(DiskannQueryParams::new(48))
        .expect("DiskANN query controls must be valid");
    let mut query = MultiQuery::new().expect("multi-query must be valid");
    query
        .add_sub_query(&branch)
        .expect("sub-query must be accepted");
    query.set_topk(10).expect("top-k must be valid");
    query
}

fn group_query(coordinate: f32) -> GroupBySearchQuery {
    let mut query = GroupBySearchQuery::new(
        "embedding",
        "language",
        &[coordinate, coordinate % 7.0],
        2,
        2,
    )
    .expect("group-by query must be valid");
    query
        .set_diskann_params(DiskannQueryParams::new(48))
        .expect("DiskANN query controls must be valid");
    query
}

fn ranking(docs: Vec<Doc>) -> Vec<(String, u32)> {
    docs.into_iter()
        .map(|doc| {
            (
                doc.get_pk().expect("result must have an id").to_string(),
                doc.get_score().to_bits(),
            )
        })
        .collect()
}

fn grouped_ranking(groups: HashMap<String, Vec<Doc>>) -> Vec<(String, Vec<(String, u32)>)> {
    let mut groups: Vec<_> = groups
        .into_iter()
        .map(|(key, docs)| (key, ranking(docs)))
        .collect();
    groups.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    groups
}

fn read_only_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only mode must be valid");
    options
}

fn sidecar_path(root: &Path) -> PathBuf {
    root.join(DISKANN_PATH)
}

#[test]
fn async_queries_match_sync_diskann_and_preserve_corruption_fallback() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("async-diskann");
    let collection = Collection::create(
        path.to_str().expect("collection path must be UTF-8"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    let docs = documents(192);
    let inserted = collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    assert_eq!(inserted.success_count, 192);
    collection
        .rebuild_index("embedding")
        .expect("DiskANN index must rebuild");

    let query = vector_query(191.5, 10);
    let multi = multi_query(191.5);
    let group = group_query(191.5);
    let expected = ranking(collection.query(&query).expect("sync query must succeed"));
    let expected_multi = ranking(
        collection
            .multi_query(&multi)
            .expect("sync multi-query must succeed"),
    );
    let expected_groups = grouped_ranking(
        collection
            .group_by(&group)
            .expect("sync group-by query must succeed"),
    );
    collection.close().expect("collection must close");

    let reopened = Collection::open(
        path.to_str().expect("collection path must be UTF-8"),
        Some(&read_only_options()),
    )
    .expect("collection must reopen");
    assert!(
        reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("Tokio runtime must build");
    assert_send(reopened.query_async(&query));
    let before = reopened.stats_snapshot().expect("stats must succeed");
    let (actual, actual_multi, actual_groups) = runtime.block_on(async {
        let actual = reopened
            .query_async(&query)
            .await
            .expect("async query must succeed");
        let actual_multi = reopened
            .multi_query_async(&multi)
            .await
            .expect("async multi-query must succeed");
        let actual_groups = reopened
            .group_by_async(&group)
            .await
            .expect("async group-by query must succeed");
        (actual, actual_multi, actual_groups)
    });
    assert_eq!(ranking(actual), expected);
    assert_eq!(ranking(actual_multi), expected_multi);
    assert_eq!(grouped_ranking(actual_groups), expected_groups);
    let after = reopened.stats_snapshot().expect("stats must succeed");
    assert_eq!(after.diskann_query_count - before.diskann_query_count, 3);
    assert!(after.diskann_sector_read_count > before.diskann_sector_read_count);

    let sidecar = fs::OpenOptions::new()
        .write(true)
        .open(sidecar_path(&path))
        .expect("sidecar must open for fault injection");
    sidecar
        .set_len(DISKANN_SECTOR_BYTES)
        .expect("sidecar must truncate");
    let before_fallback = reopened.stats_snapshot().expect("stats must succeed");
    let fallback = runtime
        .block_on(reopened.query_async(&query))
        .expect("async query must fall back to memory");
    assert_eq!(ranking(fallback), expected);
    let after_fallback = reopened.stats_snapshot().expect("stats must succeed");
    assert_eq!(
        after_fallback.diskann_query_count,
        before_fallback.diskann_query_count
    );
    assert_eq!(
        after_fallback.diskann_sector_read_count,
        before_fallback.diskann_sector_read_count
    );
}

fn assert_send<T: Send>(_: T) {}
