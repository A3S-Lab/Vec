use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc,
    FieldSchema, IndexParams, IoBackend, MetricType, SearchQuery,
};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const DISKANN_PATH: &str = "indexes/diskann-graph.bin";
const DISKANN_SECTOR_BYTES: u64 = 4_096;

fn schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field must be valid");
    embedding
        .set_index_params(
            &IndexParams::diskann(MetricType::L2, 16, 64, 1)
                .expect("DiskANN parameters must be valid"),
        )
        .expect("DiskANN index must be valid");
    CollectionSchema::builder("mmap-diskann-contract")
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn documents(count: usize) -> Vec<Doc> {
    (0..count)
        .map(|index| {
            let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
            let mut doc = Doc::with_pk(format!("doc-{index:03}")).expect("document must be valid");
            doc.add_vector_f32("embedding", &[coordinate, coordinate % 7.0])
                .expect("embedding must be valid");
            doc
        })
        .collect()
}

fn query(coordinate: f32) -> SearchQuery {
    let mut query = SearchQuery::new("embedding", &[coordinate, coordinate % 7.0], 10)
        .expect("query must be valid");
    query
        .set_diskann_params(DiskannQueryParams::new(48))
        .expect("DiskANN query controls must be valid");
    query
}

fn ranking(collection: &Collection, query: &SearchQuery) -> Vec<(String, u32)> {
    collection
        .query(query)
        .expect("query must succeed")
        .into_iter()
        .map(|doc| {
            (
                doc.get_pk().expect("result must have an id").to_string(),
                doc.get_score().to_bits(),
            )
        })
        .collect()
}

fn mmap_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only mode must be valid");
    options
        .set_io_backend(IoBackend::Mmap)
        .expect("mmap backend must be valid");
    assert_eq!(options.io_backend(), Some(IoBackend::Mmap));
    options
}

fn sidecar_path(root: &Path) -> PathBuf {
    root.join(DISKANN_PATH)
}

#[test]
fn mmap_snapshot_matches_positioned_results_and_isolated_from_late_truncation() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("mmap-diskann");
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
    let query = query(191.5);
    let in_memory = ranking(&collection, &query);
    collection.close().expect("collection must close");

    let mut positioned_options = CollectionOptions::new().expect("options must be valid");
    positioned_options
        .set_read_only(true)
        .expect("read-only mode must be valid");
    let positioned = Collection::open(
        path.to_str().expect("collection path must be UTF-8"),
        Some(&positioned_options),
    )
    .expect("collection must reopen with positioned I/O");
    let positioned_stats = positioned.stats().expect("stats must succeed");
    assert!(positioned_stats.index_cache_hit);
    assert_eq!(positioned_stats.io_backend, IoBackend::Positioned);
    let before_positioned = positioned.stats_snapshot().expect("stats must succeed");
    let expected = ranking(&positioned, &query);
    assert_eq!(expected, in_memory);
    let after_positioned = positioned.stats_snapshot().expect("stats must succeed");
    assert_eq!(
        after_positioned.diskann_query_count - before_positioned.diskann_query_count,
        1
    );
    assert_eq!(
        after_positioned.diskann_mmap_query_count - before_positioned.diskann_mmap_query_count,
        0
    );
    positioned
        .close()
        .expect("positioned collection must close");

    let options = mmap_options();
    let mapped = Collection::open(
        path.to_str().expect("collection path must be UTF-8"),
        Some(&options),
    )
    .expect("collection must reopen with mmap");
    let stats = mapped.stats().expect("stats must succeed");
    assert!(stats.index_cache_hit);
    assert_eq!(stats.io_backend, IoBackend::Mmap);
    let before = mapped.stats_snapshot().expect("stats must succeed");
    assert_eq!(before.io_backend, IoBackend::Mmap);
    assert_eq!(ranking(&mapped, &query), expected);
    let after = mapped.stats_snapshot().expect("stats must succeed");
    assert_eq!(after.diskann_query_count - before.diskann_query_count, 1);
    assert_eq!(
        after.diskann_mmap_query_count - before.diskann_mmap_query_count,
        1
    );
    assert!(after.diskann_sector_read_count > before.diskann_sector_read_count);

    let sidecar = fs::OpenOptions::new()
        .write(true)
        .open(sidecar_path(&path))
        .expect("sidecar must open for fault injection");
    sidecar
        .set_len(DISKANN_SECTOR_BYTES)
        .expect("sidecar must truncate");
    let before_truncated = mapped.stats_snapshot().expect("stats must succeed");
    assert_eq!(ranking(&mapped, &query), expected);
    let after_truncated = mapped.stats_snapshot().expect("stats must succeed");
    assert_eq!(
        after_truncated.diskann_mmap_query_count - before_truncated.diskann_mmap_query_count,
        1
    );
    mapped.close().expect("mapped collection must close");

    let rebuilt = Collection::open(
        path.to_str().expect("collection path must be UTF-8"),
        Some(&options),
    )
    .expect("corrupt sidecar must fall back to an in-memory rebuild");
    assert!(!rebuilt.stats().expect("stats must succeed").index_cache_hit);
    assert_eq!(ranking(&rebuilt, &query), expected);
    let rebuilt_stats = rebuilt.stats_snapshot().expect("stats must succeed");
    assert_eq!(rebuilt_stats.diskann_query_count, 0);
    assert_eq!(rebuilt_stats.diskann_mmap_query_count, 0);
}
