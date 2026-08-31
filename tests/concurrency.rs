use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    HnswQueryParams, IndexParams, IndexType, MetricType, SearchQuery,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::thread;
use tempfile::tempdir;

fn concurrency_schema() -> CollectionSchema {
    CollectionSchema::builder("concurrency")
        .add_field(
            FieldSchema::new("epoch", DataType::Int32, false, 0)
                .expect("epoch schema must be valid"),
        )
        .add_field(
            FieldSchema::new("left", DataType::Int32, false, 0).expect("left schema must be valid"),
        )
        .add_field(
            FieldSchema::new("right", DataType::Int32, false, 0)
                .expect("right schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("collection options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn complete_doc(id: &str, epoch: i32, left: i32, right: i32) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_i32("epoch", epoch).expect("epoch must be valid");
    doc.add_i32("left", left).expect("left must be valid");
    doc.add_i32("right", right).expect("right must be valid");
    doc
}

fn patch_doc(id: &str, field: &str, value: i32) -> Doc {
    let mut doc = Doc::with_pk(id).expect("primary key must be valid");
    doc.add_i32(field, value).expect("patch must be valid");
    doc
}

fn epoch_pair(epoch: i32) -> (Doc, Doc) {
    (
        complete_doc("left-doc", epoch, epoch, epoch),
        complete_doc("right-doc", epoch, epoch, epoch),
    )
}

fn required_i32(doc: &Doc, field: &str) -> i32 {
    doc.get_i32(field)
        .expect("stored field must have its declared type")
        .expect("stored field must be present")
}

fn scalar_generation_schema() -> CollectionSchema {
    let mut bucket =
        FieldSchema::new("bucket", DataType::Int32, false, 0).expect("bucket field must be valid");
    bucket
        .set_index_params(
            &IndexParams::invert(false, false).expect("inverted descriptor must be valid"),
        )
        .expect("bucket must support an inverted index");
    CollectionSchema::builder("scalar-generation")
        .add_field(
            FieldSchema::new("epoch", DataType::Int32, false, 0)
                .expect("epoch field must be valid"),
        )
        .add_field(bucket)
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
                .expect("vector field must be valid"),
        )
        .build()
        .expect("scalar generation schema must be valid")
}

fn scalar_generation_doc(index: usize, epoch: i32) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:04}")).expect("document primary key must be valid");
    doc.add_i32("epoch", epoch).expect("epoch must be valid");
    let index_parity = i32::try_from(index % 2).expect("parity fits i32");
    doc.add_i32("bucket", (index_parity + epoch).rem_euclid(2))
        .expect("bucket must be valid");
    let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
    doc.add_vector_f32("embedding", &[coordinate, 1.0])
        .expect("embedding must be valid");
    doc
}

fn scalar_generation_query(document_count: usize) -> SearchQuery {
    let mut query = SearchQuery::new(
        "embedding",
        &[0.0, 1.0],
        i32::try_from(document_count).expect("document count fits i32"),
    )
    .expect("query must be valid");
    query
        .params
        .insert("metric".into(), serde_json::json!("l2"));
    query
        .set_filter("bucket == 1")
        .expect("filter must be valid");
    query
}

fn validate_scalar_generation(collection: &Collection, query: &SearchQuery, expected: usize) {
    let docs = collection
        .query(query)
        .expect("concurrent scalar query must succeed");
    assert_eq!(docs.len(), expected);
    let epoch = docs
        .first()
        .map(|doc| required_i32(doc, "epoch"))
        .expect("filtered generation must not be empty");
    assert!(docs
        .iter()
        .all(|doc| { required_i32(doc, "epoch") == epoch && required_i32(doc, "bucket") == 1 }));
}

#[test]
fn concurrent_disjoint_updates_are_serialized_without_lost_fields() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let collection = Collection::create(
        temporary
            .path()
            .join("serialized-writers")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &concurrency_schema(),
        Some(&options),
    )
    .expect("collection must be created");
    let initial = complete_doc("shared", 0, 0, 0);
    collection
        .insert(&[&initial])
        .expect("initial insert must succeed");

    let barrier = Arc::new(Barrier::new(3));
    let (left_result, right_result) = thread::scope(|scope| {
        let left_collection = collection.clone();
        let left_barrier = Arc::clone(&barrier);
        let left = scope.spawn(move || {
            let patch = patch_doc("shared", "left", 1);
            left_barrier.wait();
            left_collection.update(&[&patch])
        });
        let right_collection = collection.clone();
        let right_barrier = Arc::clone(&barrier);
        let right = scope.spawn(move || {
            let patch = patch_doc("shared", "right", 1);
            right_barrier.wait();
            right_collection.update(&[&patch])
        });
        barrier.wait();
        (
            left.join().expect("left writer must not panic"),
            right.join().expect("right writer must not panic"),
        )
    });

    assert_eq!(
        left_result.expect("left update must succeed").success_count,
        1
    );
    assert_eq!(
        right_result
            .expect("right update must succeed")
            .success_count,
        1
    );
    let stored = collection
        .fetch(&["shared"])
        .expect("updated document must be readable");
    assert_eq!(required_i32(&stored[0], "left"), 1);
    assert_eq!(required_i32(&stored[0], "right"), 1);
    let stats = collection.stats().expect("statistics must be readable");
    assert_eq!(stats.doc_count, 1);
    assert_eq!(stats.revision, 3);
}

#[test]
fn iterator_keeps_one_revision_while_a_writer_publishes_the_next() {
    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let collection = Collection::create(
        temporary
            .path()
            .join("iterator-snapshot")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &concurrency_schema(),
        Some(&options),
    )
    .expect("collection must be created");
    let (initial_left, initial_right) = epoch_pair(0);
    collection
        .insert(&[&initial_left, &initial_right])
        .expect("initial batch must succeed");

    let (ready_tx, ready_rx) = mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = mpsc::sync_channel(0);
    let (snapshot_revision, old_docs, write_result) = thread::scope(|scope| {
        let reader_collection = collection.clone();
        let reader = scope.spawn(move || {
            let iterator = reader_collection
                .iter()
                .expect("snapshot iterator must be created");
            let revision = iterator.revision();
            ready_tx
                .send(revision)
                .expect("snapshot revision must be delivered");
            resume_rx.recv().expect("reader must be resumed");
            iterator
                .map(|doc| doc.expect("snapshot document must be readable"))
                .collect::<Vec<_>>()
        });

        let revision = ready_rx.recv().expect("reader must publish its revision");
        let (next_left, next_right) = epoch_pair(1);
        let write_result = collection.upsert(&[&next_left, &next_right]);
        resume_tx.send(()).expect("reader must be resumed");
        let docs = reader.join().expect("reader must not panic");
        (revision, docs, write_result)
    });

    assert_eq!(snapshot_revision, 1);
    assert_eq!(
        write_result
            .expect("replacement batch must succeed")
            .success_count,
        2
    );
    assert_eq!(old_docs.len(), 2);
    assert!(old_docs.iter().all(|doc| required_i32(doc, "epoch") == 0));

    let current = collection.iter().expect("current iterator must be created");
    assert_eq!(current.revision(), 2);
    let current_docs: Vec<Doc> = current
        .map(|doc| doc.expect("current document must be readable"))
        .collect();
    assert!(current_docs
        .iter()
        .all(|doc| required_i32(doc, "epoch") == 1));
}

#[test]
fn readers_never_observe_a_partially_published_batch() {
    const ROUNDS: i32 = 64;

    let temporary = tempdir().expect("temporary directory must be available");
    let options = manual_options();
    let collection = Collection::create(
        temporary
            .path()
            .join("atomic-batches")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &concurrency_schema(),
        Some(&options),
    )
    .expect("collection must be created");
    let (initial_left, initial_right) = epoch_pair(0);
    collection
        .insert(&[&initial_left, &initial_right])
        .expect("initial batch must succeed");

    let round_barrier = Arc::new(Barrier::new(3));
    let (writer_error, reader_errors) = thread::scope(|scope| {
        let writer_collection = collection.clone();
        let writer_barrier = Arc::clone(&round_barrier);
        let writer = scope.spawn(move || {
            let mut first_error = None;
            for epoch in 1..=ROUNDS {
                writer_barrier.wait();
                if first_error.is_none() {
                    let (left, right) = epoch_pair(epoch);
                    match writer_collection.upsert(&[&left, &right]) {
                        Ok(result) if result.success_count == 2 => {}
                        Ok(result) => {
                            first_error = Some(format!(
                                "writer accepted {} of 2 documents at epoch {epoch}",
                                result.success_count
                            ));
                        }
                        Err(error) => first_error = Some(error.to_string()),
                    }
                }
                writer_barrier.wait();
            }
            first_error
        });

        let mut readers = Vec::new();
        for reader_id in 0..2 {
            let reader_collection = collection.clone();
            let reader_barrier = Arc::clone(&round_barrier);
            readers.push(scope.spawn(move || {
                let mut first_error = None;
                for round in 1..=ROUNDS {
                    reader_barrier.wait();
                    if first_error.is_none() {
                        match reader_collection.fetch(&["left-doc", "right-doc"]) {
                            Ok(docs) if docs.len() == 2 => {
                                let left_epoch = required_i32(&docs[0], "epoch");
                                let right_epoch = required_i32(&docs[1], "epoch");
                                if left_epoch != right_epoch {
                                    first_error = Some(format!(
                                        "reader {reader_id} saw epochs {left_epoch} and {right_epoch} in round {round}"
                                    ));
                                }
                            }
                            Ok(docs) => {
                                first_error = Some(format!(
                                    "reader {reader_id} saw {} documents in round {round}",
                                    docs.len()
                                ));
                            }
                            Err(error) => first_error = Some(error.to_string()),
                        }
                    }
                    reader_barrier.wait();
                }
                first_error
            }));
        }

        let writer_error = writer.join().expect("writer must not panic");
        let reader_errors = readers
            .into_iter()
            .map(|reader| reader.join().expect("reader must not panic"))
            .collect::<Vec<_>>();
        (writer_error, reader_errors)
    });

    assert_eq!(writer_error, None);
    assert!(reader_errors.iter().all(Option::is_none));
    let final_docs = collection
        .fetch(&["left-doc", "right-doc"])
        .expect("final batch must be readable");
    assert!(final_docs
        .iter()
        .all(|doc| required_i32(doc, "epoch") == ROUNDS));
    let stats = collection.stats().expect("statistics must be readable");
    assert_eq!(stats.doc_count, 2);
    assert_eq!(
        stats.revision,
        u64::try_from(ROUNDS + 1).expect("revision must fit u64")
    );
}

#[test]
fn ann_readers_observe_only_complete_index_generations() {
    const DOCUMENTS: usize = 256;

    let temporary = tempdir().expect("temporary directory must be available");
    let schema = CollectionSchema::builder("ann-generation")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
                .expect("vector field must be valid"),
        )
        .build()
        .expect("schema must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("ann-generation")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema,
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS)
        .map(|index| {
            let coordinate = f32::from(u16::try_from(index).expect("fixture index fits u16"));
            let mut doc = Doc::with_pk(format!("doc-{index:04}")).expect("document must be valid");
            doc.add_vector_f32("embedding", &[coordinate, 0.0])
                .expect("vector must be valid");
            doc
        })
        .collect();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::L2, 12, 64).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index must build");
    let mut query = SearchQuery::new("embedding", &[1_000.0, 0.0], 1).expect("query must be valid");
    query
        .set_hnsw_params(HnswQueryParams::new(
            i32::try_from(DOCUMENTS).expect("fixture size fits i32"),
            0.0,
            false,
            true,
        ))
        .expect("HNSW controls must be valid");
    assert_eq!(
        collection.query(&query).expect("old query must succeed")[0].get_pk(),
        Some("doc-0255")
    );

    let started = Arc::new(Barrier::new(2));
    let finished = Arc::new(AtomicBool::new(false));
    let (write_result, observations) = thread::scope(|scope| {
        let writer_collection = collection.clone();
        let writer_started = Arc::clone(&started);
        let writer_finished = Arc::clone(&finished);
        let writer = scope.spawn(move || {
            let mut replacement = Doc::with_pk("doc-0000").expect("document must be valid");
            replacement
                .add_vector_f32("embedding", &[1_000.0, 0.0])
                .expect("replacement vector must be valid");
            writer_started.wait();
            let result = writer_collection.upsert(&[&replacement]);
            writer_finished.store(true, Ordering::Release);
            result
        });

        started.wait();
        let mut observations = Vec::new();
        while !finished.load(Ordering::Acquire) {
            let result = collection
                .query(&query)
                .expect("concurrent ANN query must succeed");
            observations.push(
                result[0]
                    .get_pk()
                    .expect("result must have a primary key")
                    .to_string(),
            );
        }
        observations.push(
            collection.query(&query).expect("new query must succeed")[0]
                .get_pk()
                .expect("result must have a primary key")
                .to_string(),
        );
        (writer.join().expect("writer must not panic"), observations)
    });

    assert_eq!(
        write_result
            .expect("indexed upsert must succeed")
            .success_count,
        1
    );
    assert!(!observations.is_empty());
    assert!(observations
        .iter()
        .all(|id| matches!(id.as_str(), "doc-0255" | "doc-0000")));
    assert_eq!(observations.last().map(String::as_str), Some("doc-0000"));
    let stats = collection.stats().expect("stats must be available");
    assert_eq!(stats.indexes[0].source_revision, stats.revision);
}

#[test]
fn scalar_bitmap_readers_observe_only_complete_index_generations() {
    const DOCUMENTS: usize = 128;
    const ROUNDS: i32 = 32;
    const READERS: usize = 2;

    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("scalar-generation")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &scalar_generation_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let initial: Vec<Doc> = (0..DOCUMENTS)
        .map(|index| scalar_generation_doc(index, 0))
        .collect();
    collection
        .insert(&initial.iter().collect::<Vec<_>>())
        .expect("initial generation must be inserted");

    let started = Arc::new(Barrier::new(READERS + 1));
    let finished = Arc::new(AtomicBool::new(false));
    let query = scalar_generation_query(DOCUMENTS);
    thread::scope(|scope| {
        let writer_collection = collection.clone();
        let writer_started = Arc::clone(&started);
        let writer_finished = Arc::clone(&finished);
        let writer = scope.spawn(move || {
            writer_started.wait();
            for epoch in 1..=ROUNDS {
                let docs: Vec<Doc> = (0..DOCUMENTS)
                    .map(|index| scalar_generation_doc(index, epoch))
                    .collect();
                let result = writer_collection
                    .upsert(&docs.iter().collect::<Vec<_>>())
                    .expect("replacement generation must succeed");
                assert_eq!(
                    result.success_count,
                    u64::try_from(DOCUMENTS).expect("document count fits u64")
                );
            }
            writer_finished.store(true, Ordering::Release);
        });

        let readers = (0..READERS)
            .map(|_| {
                let reader_collection = collection.clone();
                let reader_started = Arc::clone(&started);
                let reader_finished = Arc::clone(&finished);
                let reader_query = query.clone();
                scope.spawn(move || {
                    reader_started.wait();
                    while !reader_finished.load(Ordering::Acquire) {
                        validate_scalar_generation(
                            &reader_collection,
                            &reader_query,
                            DOCUMENTS / 2,
                        );
                    }
                    validate_scalar_generation(&reader_collection, &reader_query, DOCUMENTS / 2);
                })
            })
            .collect::<Vec<_>>();

        writer.join().expect("writer must not panic");
        for reader in readers {
            reader.join().expect("reader must not panic");
        }
    });

    let stats = collection.stats().expect("stats must be available");
    let scalar = stats
        .indexes
        .iter()
        .find(|index| index.index_type == IndexType::Invert)
        .expect("scalar index stats must be available");
    assert_eq!(scalar.source_revision, stats.revision);
    assert_eq!(
        stats.revision,
        u64::try_from(ROUNDS + 1).expect("revision fits u64")
    );
}
