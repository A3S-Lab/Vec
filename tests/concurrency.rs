use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
};
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
