#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Filtered DiskANN over mmap sidecars must keep allow-list + exact re-rank (I5/I6).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc, Durability,
    FieldSchema, IndexParams, IoBackend, MetricType, SearchQuery,
};
use tempfile::tempdir;

const N: usize = 8_400;
const TOPK: usize = 8;

fn write_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

fn mmap_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_read_only(true).expect("ro");
    options.set_io_backend(IoBackend::Mmap).expect("mmap");
    options
}

fn schema() -> CollectionSchema {
    let mut scope = FieldSchema::new("scope", DataType::String, false, 0).expect("scope");
    scope
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"))
        .expect("attach");
    CollectionSchema::builder("filtered-mmap-diskann")
        .add_field(scope)
        .add_field(embedding)
        .build()
        .expect("schema")
}

#[test]
fn filtered_mmap_diskann_stays_inside_allow_list() {
    let temporary = tempdir().expect("temp");
    let path = temporary.path().join("fmd");
    let collection = Collection::create(
        path.to_str().expect("utf8"),
        &schema(),
        Some(&write_options()),
    )
    .expect("create");

    let docs: Vec<Doc> = (0..N)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("pk");
            let allowed = index >= N / 2;
            doc.add_string("scope", if allowed { "allowed" } else { "blocked" })
                .expect("scope");
            let local = (index % (N / 2)) as f32;
            doc.add_vector_f32(
                "embedding",
                &[if allowed { 1_000.0 + local } else { local }, local % 11.0],
            )
            .expect("vector");
            doc
        })
        .collect();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("insert");
    collection
        .rebuild_index("embedding")
        .expect("rebuild diskann");
    collection.close().expect("close");

    let mapped =
        Collection::open(path.to_str().expect("utf8"), Some(&mmap_options())).expect("mmap open");
    assert_eq!(mapped.stats().expect("stats").io_backend, IoBackend::Mmap);

    let mut query = SearchQuery::new("embedding", &[1_050.0, 1.0], TOPK as i32).expect("query");
    query.set_filter("scope = \"allowed\"").expect("filter");
    query
        .set_diskann_params(DiskannQueryParams::new(256))
        .expect("list_size");
    let before = mapped.stats_snapshot().expect("snap");
    let hits = mapped.query(&query).expect("filtered mmap diskann");
    let after = mapped.stats_snapshot().expect("snap");
    assert_eq!(hits.len(), TOPK);
    assert!(after.diskann_query_count > before.diskann_query_count);
    for hit in &hits {
        match hit.field("scope") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "allowed"),
            other => panic!("expected allowed, got {other:?}"),
        }
    }
}
