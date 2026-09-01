use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc,
    FieldSchema, Fts, HnswQueryParams, IndexParams, IndexType, MetricType, SearchQuery,
};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const CACHE_PATH: &str = "indexes/index-cache.bin";
const DISKANN_PATH: &str = "indexes/diskann-graph.bin";
const DISKANN_SECTOR_BYTES: usize = 4_096;

fn schema() -> CollectionSchema {
    let mut language =
        FieldSchema::new("language", DataType::String, false, 0).expect("field must be valid");
    language
        .set_index_params(&IndexParams::invert(false, false).expect("params must be valid"))
        .expect("scalar index must be valid");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field must be valid");
    embedding
        .set_index_params(&IndexParams::hnsw(MetricType::L2, 8, 32).expect("params must be valid"))
        .expect("HNSW index must be valid");
    CollectionSchema::builder("index-cache-contract")
        .add_field(language)
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn vamana_schema() -> CollectionSchema {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field must be valid");
    embedding
        .set_index_params(
            &IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("Vamana params must be valid"),
        )
        .expect("Vamana index must be valid");
    CollectionSchema::builder("vamana-cache-contract")
        .add_field(embedding)
        .build()
        .expect("schema must be valid")
}

fn document(id: &str, coordinate: f32, language: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("document must be valid");
    doc.add_string("language", language)
        .expect("language must be valid");
    doc.add_vector_f32("embedding", &[coordinate, coordinate % 7.0])
        .expect("embedding must be valid");
    doc
}

fn documents(count: usize) -> Vec<Doc> {
    (0..count)
        .map(|index| {
            let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
            document(
                &format!("doc-{index:03}"),
                coordinate,
                if index % 2 == 0 { "rust" } else { "python" },
            )
        })
        .collect()
}

fn read_only_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only option must be valid");
    options
}

fn open_read_only(path: &Path) -> Collection {
    Collection::open(
        path.to_str().expect("collection path must be UTF-8"),
        Some(&read_only_options()),
    )
    .expect("collection must open read-only")
}

fn exhaustive_query(coordinate: f32, topk: i32, count: usize) -> SearchQuery {
    let mut query = SearchQuery::new("embedding", &[coordinate, coordinate % 7.0], topk)
        .expect("query must be valid");
    query
        .set_hnsw_params(HnswQueryParams::new(
            i32::try_from(count).expect("fixture count fits i32"),
            0.0,
            false,
            true,
        ))
        .expect("HNSW controls must be valid");
    query
}

fn exhaustive_vamana_query(coordinate: f32, topk: i32, count: usize) -> SearchQuery {
    let mut query = SearchQuery::new("embedding", &[coordinate, coordinate % 7.0], topk)
        .expect("query must be valid");
    query
        .set_diskann_params(DiskannQueryParams::new(
            i32::try_from(count).expect("fixture count fits i32"),
        ))
        .expect("Vamana controls must be valid");
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

fn create_fixture(path: &Path, count: usize) -> Vec<(String, u32)> {
    let collection = Collection::create(
        path.to_str().expect("collection path must be UTF-8"),
        &schema(),
        None,
    )
    .expect("collection must be created");
    let docs = documents(count);
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    let expected = ranking(
        &collection,
        &exhaustive_query(
            f32::from(u16::try_from(count - 1).expect("fixture count fits u16")),
            8,
            count,
        ),
    );
    collection.close().expect("collection must close");
    expected
}

fn cache_path(root: &Path) -> PathBuf {
    root.join(CACHE_PATH)
}

fn diskann_path(root: &Path) -> PathBuf {
    root.join(DISKANN_PATH)
}

fn scalar_fts_schema() -> CollectionSchema {
    let mut language =
        FieldSchema::new("language", DataType::String, false, 0).expect("field must be valid");
    language
        .set_index_params(&IndexParams::invert(false, false).expect("params must be valid"))
        .expect("scalar index must be valid");
    let mut body =
        FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
    body.set_index_params(
        &IndexParams::fts(Some("standard"), None, None).expect("params must be valid"),
    )
    .expect("FTS index must be valid");
    CollectionSchema::builder("scalar-fts-cache-contract")
        .add_field(language)
        .add_field(body)
        .build()
        .expect("schema must be valid")
}

fn lexical_document(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:03}")).expect("document must be valid");
    let language = if index % 2 == 0 { "rust" } else { "python" };
    doc.add_string("language", language)
        .expect("language must be valid");
    doc.add_string(
        "body",
        &format!("workspace {language} symbol{} retrieval", index % 11),
    )
    .expect("body must be valid");
    doc
}

fn lexical_ranking(collection: &Collection) -> Vec<(String, u32)> {
    let mut fts = Fts::new().expect("FTS payload must be valid");
    fts.set_match_string("workspace rust")
        .expect("FTS expression must be valid");
    let mut query = SearchQuery::fts("body", &fts, 128).expect("FTS query must be valid");
    query
        .set_filter("language == 'rust'")
        .expect("filter must be valid");
    ranking(collection, &query)
}

fn create_lexical_fixture(path: &Path) -> Vec<(String, u32)> {
    let collection = Collection::create(
        path.to_str().expect("collection path must be UTF-8"),
        &scalar_fts_schema(),
        None,
    )
    .expect("collection must be created");
    let docs: Vec<_> = (0..128).map(lexical_document).collect();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    let expected = lexical_ranking(&collection);
    collection.close().expect("collection must close");
    expected
}

#[test]
fn scalar_and_fts_generations_round_trip_without_an_ann_index() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("scalar-fts");
    let expected = create_lexical_fixture(&path);
    assert!(cache_path(&path).exists());

    let reopened = open_read_only(&path);
    assert!(
        reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(lexical_ranking(&reopened), expected);
}

#[test]
fn cached_scalar_and_fts_deltas_match_an_authoritative_rebuild() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("scalar-fts-deltas");
    create_lexical_fixture(&path);

    let collection = Collection::open(path.to_str().expect("collection path must be UTF-8"), None)
        .expect("collection must reopen writable");
    assert!(
        collection
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    let mut replacement = Doc::with_pk("doc-000").expect("document must be valid");
    replacement
        .add_string("language", "python")
        .expect("language must be valid");
    replacement
        .add_string("body", "workspace python replacement lexical")
        .expect("body must be valid");
    collection
        .update(&[&replacement])
        .expect("replacement must update");
    collection
        .delete(&["doc-002"])
        .expect("document must delete");
    let mut late = Doc::with_pk("late").expect("document must be valid");
    late.add_string("language", "rust")
        .expect("language must be valid");
    late.add_string("body", "workspace rust late lexical")
        .expect("body must be valid");
    collection.insert(&[&late]).expect("document must insert");
    let expected = lexical_ranking(&collection);
    collection.close().expect("collection must close");

    let cached = open_read_only(&path);
    assert!(cached.stats().expect("stats must succeed").index_cache_hit);
    assert_eq!(lexical_ranking(&cached), expected);
    cached.close().expect("collection must close");

    fs::remove_file(cache_path(&path)).expect("cache must be removable");
    let rebuilt = open_read_only(&path);
    assert!(!rebuilt.stats().expect("stats must succeed").index_cache_hit);
    assert_eq!(lexical_ranking(&rebuilt), expected);
}

#[test]
fn valid_cache_restores_ann_and_preserves_results_without_read_only_writes() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let expected = create_fixture(&path, 128);
    let before = fs::read(cache_path(&path)).expect("index cache must exist after close");
    assert!(
        !diskann_path(&path).exists(),
        "non-Vamana indexes must not require a DiskANN sidecar"
    );

    let reopened = open_read_only(&path);
    assert!(
        reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(
        ranking(&reopened, &exhaustive_query(127.0, 8, 128)),
        expected
    );
    reopened.close().expect("read-only close must succeed");

    assert_eq!(
        fs::read(cache_path(&path)).expect("index cache must remain readable"),
        before
    );
}

#[test]
fn corrupt_cache_is_ignored_and_read_only_open_does_not_rewrite_it() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    let expected = create_fixture(&path, 128);
    let cache = cache_path(&path);
    let mut corrupted = fs::read(&cache).expect("index cache must exist");
    let last = corrupted.last_mut().expect("index cache must not be empty");
    *last ^= 0x5a;
    fs::write(&cache, &corrupted).expect("corrupt cache fixture must write");

    let reopened = open_read_only(&path);
    assert!(
        !reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(
        ranking(&reopened, &exhaustive_query(127.0, 8, 128)),
        expected
    );
    reopened.close().expect("read-only close must succeed");
    assert_eq!(
        fs::read(&cache).expect("cache must remain readable"),
        corrupted
    );
}

#[test]
fn stale_cache_falls_back_then_writable_open_refreshes_it() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    create_fixture(&path, 128);
    let stale = fs::read(cache_path(&path)).expect("index cache must exist");

    let collection = Collection::open(path.to_str().expect("UTF-8 path"), None)
        .expect("collection must reopen writable");
    assert!(
        collection
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    let late = document("late", 10_000.0, "rust");
    collection
        .insert(&[&late])
        .expect("late document must insert");
    drop(collection);

    let read_only = open_read_only(&path);
    assert!(
        !read_only
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(read_only.count().expect("count must succeed"), 129);
    assert_eq!(
        read_only
            .fetch(&["late"])
            .expect("late document must be recoverable")
            .len(),
        1
    );
    read_only.close().expect("read-only close must succeed");
    assert_eq!(
        fs::read(cache_path(&path)).expect("stale cache must remain readable"),
        stale
    );

    let writable = Collection::open(path.to_str().expect("UTF-8 path"), None)
        .expect("collection must rebuild writable");
    assert!(
        !writable
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    let refreshed = fs::read(cache_path(&path)).expect("cache must be refreshed after fallback");
    assert_ne!(refreshed, stale);
    writable.close().expect("writable close must succeed");

    let cached = open_read_only(&path);
    assert!(cached.stats().expect("stats must succeed").index_cache_hit);
    assert_eq!(cached.count().expect("count must succeed"), 129);
}

#[test]
fn cached_delta_and_tombstones_preserve_incremental_ann_results() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("collection");
    create_fixture(&path, 128);

    let collection = Collection::open(path.to_str().expect("UTF-8 path"), None)
        .expect("collection must reopen writable");
    assert!(
        collection
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    let replacement = document("doc-000", 10_000.0, "rust");
    let late = document("late", 9_999.0, "rust");
    collection
        .upsert(&[&replacement, &late])
        .expect("delta vectors must publish");
    collection
        .delete(&["doc-001"])
        .expect("base vector must be tombstoned");
    collection.close().expect("collection must close");

    let reopened = open_read_only(&path);
    assert!(
        reopened
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    let ids: Vec<String> = reopened
        .query(&exhaustive_query(10_000.0, 2, 129))
        .expect("cached overlay query must succeed")
        .into_iter()
        .map(|doc| doc.get_pk().expect("result must have an id").to_string())
        .collect();
    assert_eq!(ids, vec!["doc-000", "late"]);
    assert!(reopened
        .fetch(&["doc-001"])
        .expect("fetch must succeed")
        .is_empty());
}

#[test]
fn vamana_generation_rebuilds_and_round_trips_with_incremental_overlays() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("vamana");
    let collection = Collection::create(
        path.to_str().expect("collection path must be UTF-8"),
        &vamana_schema(),
        None,
    )
    .expect("collection must be created");
    let docs = documents(128);
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    collection
        .rebuild_index("embedding")
        .expect("Vamana index must rebuild");

    let replacement = document("doc-000", 10_000.0, "rust");
    let late = document("late", 9_999.0, "rust");
    collection
        .upsert(&[&replacement, &late])
        .expect("delta vectors must publish");
    collection
        .delete(&["doc-001"])
        .expect("base vector must be tombstoned");
    let query = exhaustive_vamana_query(10_000.0, 8, 129);
    let expected = ranking(&collection, &query);
    collection.close().expect("collection must close");

    let diskann = fs::read(diskann_path(&path)).expect("DiskANN sidecar must exist after close");
    assert!(!diskann.is_empty());
    assert_eq!(diskann.len() % DISKANN_SECTOR_BYTES, 0);

    let reopened = open_read_only(&path);
    let stats = reopened.stats().expect("stats must succeed");
    assert!(stats.index_cache_hit);
    assert_eq!(stats.indexes.len(), 1);
    assert_eq!(stats.indexes[0].index_type, IndexType::Vamana);
    assert_eq!(ranking(&reopened, &query), expected);
    assert!(reopened
        .fetch(&["doc-001"])
        .expect("fetch must succeed")
        .is_empty());
}

#[test]
fn missing_or_corrupt_vamana_sidecar_falls_back_then_refreshes_writable() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("vamana-corruption");
    let collection = Collection::create(
        path.to_str().expect("collection path must be UTF-8"),
        &vamana_schema(),
        None,
    )
    .expect("collection must be created");
    let docs = documents(128);
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    let query = exhaustive_vamana_query(127.0, 8, 128);
    let expected = ranking(&collection, &query);
    collection.close().expect("collection must close");

    let path_to_sidecar = diskann_path(&path);
    let valid = fs::read(&path_to_sidecar).expect("DiskANN sidecar must exist");
    fs::remove_file(&path_to_sidecar).expect("sidecar must be removable");
    let missing = open_read_only(&path);
    assert!(!missing.stats().expect("stats must succeed").index_cache_hit);
    assert_eq!(ranking(&missing, &query), expected);
    missing.close().expect("collection must close");
    assert!(
        !path_to_sidecar.exists(),
        "read-only fallback must not recreate a missing sidecar"
    );

    let mut corrupted = valid;
    let last = corrupted.last_mut().expect("sidecar must not be empty");
    *last ^= 0x5a;
    fs::write(&path_to_sidecar, &corrupted).expect("corrupt sidecar fixture must write");

    let read_only = open_read_only(&path);
    assert!(
        !read_only
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(ranking(&read_only, &query), expected);
    read_only.close().expect("collection must close");
    assert_eq!(
        fs::read(&path_to_sidecar).expect("corrupt sidecar must remain readable"),
        corrupted
    );

    let writable = Collection::open(path.to_str().expect("UTF-8 path"), None)
        .expect("collection must rebuild writable");
    assert!(
        !writable
            .stats()
            .expect("stats must succeed")
            .index_cache_hit
    );
    assert_eq!(ranking(&writable, &query), expected);
    writable.close().expect("collection must close");

    let repaired = fs::read(&path_to_sidecar).expect("DiskANN sidecar must be refreshed");
    assert_ne!(repaired, corrupted);
    assert_eq!(repaired.len() % DISKANN_SECTOR_BYTES, 0);
    let cached = open_read_only(&path);
    assert!(cached.stats().expect("stats must succeed").index_cache_hit);
    assert_eq!(ranking(&cached, &query), expected);
}
