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
