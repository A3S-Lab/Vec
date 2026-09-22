//! Checkpoint byte budget and published snapshot compatibility.

use a3s_vec::{Collection, CollectionSchema, DataType, Doc, FieldSchema, FieldValue};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const DOCUMENTS: usize = 2_000;
const DIMENSION: usize = 128;

fn collection_path(directory: &tempfile::TempDir) -> String {
    directory
        .path()
        .join("snapshots")
        .to_str()
        .expect("temporary path must be UTF-8")
        .to_string()
}

fn schema() -> CollectionSchema {
    CollectionSchema::builder("snapshot-budget")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 128).expect("vector field"),
        )
        .add_field(FieldSchema::new("label", DataType::String, false, 0).expect("label field"))
        .build()
        .expect("schema")
}

fn document(index: usize, label: &str) -> Doc {
    let id = format!("doc-{index:04}");
    let mut doc = Doc::with_pk(&id).expect("primary key");
    let vector: Vec<f32> = (0..DIMENSION)
        .map(|coordinate| {
            let coordinate = u16::try_from(coordinate).expect("dimension fits");
            f32::from(u16::try_from(index).expect("index fits") % 97)
                + f32::from(coordinate) * 0.015_625
        })
        .collect();
    doc.add_vector_f32("embedding", &vector).expect("vector");
    doc.add_string("label", label).expect("label");
    doc
}

fn snapshot_bins(root: &Path) -> Vec<(u64, u64, Vec<u8>)> {
    let mut files = Vec::new();
    for entry in fs::read_dir(root.join("segments")).expect("segments directory") {
        let entry = entry.expect("segment entry");
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(generation) = name
            .strip_prefix("snapshot-")
            .and_then(|rest| rest.strip_suffix(".bin"))
            .and_then(|rest| rest.parse::<u64>().ok())
        else {
            continue;
        };
        let bytes = fs::read(entry.path()).expect("snapshot bytes");
        let length = u64::try_from(bytes.len()).expect("snapshot length fits u64");
        files.push((generation, length, bytes));
    }
    files.sort_by_key(|(generation, _, _)| *generation);
    files
}

fn assert_documents(collection: &Collection, edited: bool) {
    assert_eq!(collection.count().expect("count"), DOCUMENTS);
    for index in 0..DOCUMENTS {
        let id = format!("doc-{index:04}");
        let docs = collection.fetch(&[&id]).expect("fetch");
        assert_eq!(docs.len(), 1);
        let expected = if edited && index == 0 { "edited" } else { "v1" };
        assert_eq!(
            docs[0].field("label"),
            Some(&FieldValue::String(expected.to_string()))
        );
        assert_eq!(
            docs[0].vector("embedding"),
            document(index, expected).vector("embedding")
        );
    }
}

#[test]
fn enterprise_ga_one_document_checkpoint_is_smaller_than_a_full_snapshot() {
    let temporary = tempdir().expect("temporary directory");
    let path = collection_path(&temporary);
    let collection = Collection::create(&path, &schema(), None).expect("create");
    let docs: Vec<Doc> = (0..DOCUMENTS).map(|index| document(index, "v1")).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    collection.insert(&refs).expect("insert");
    collection.flush().expect("full checkpoint");
    drop(collection);

    let full = snapshot_bins(Path::new(&path));
    assert_eq!(
        full.len(),
        1,
        "the first content checkpoint is one full file"
    );
    assert_eq!(
        full[0].2.first().copied(),
        Some(0x95),
        "published format-4 snapshots are a five-field MessagePack array"
    );
    assert_eq!(full[0].2.get(1).copied(), Some(0x04));
    let full_length = full[0].1;

    let opened = Collection::open(&path, None).expect("format 4 snapshot must open");
    assert_documents(&opened, false);
    let edited = document(0, "edited");
    opened.upsert(&[&edited]).expect("edit one document");
    opened.flush().expect("delta checkpoint");
    drop(opened);

    let files = snapshot_bins(Path::new(&path));
    assert!(
        files.len() >= 2,
        "the unchanged base snapshot must remain beside the delta"
    );
    let delta = files.last().expect("delta snapshot");
    assert_eq!(delta.2.first().copied(), Some(0x98));
    assert_eq!(delta.2.get(1).copied(), Some(0x05));
    assert!(delta.1 > 0);
    assert!(
        delta.1.saturating_mul(2) < full_length,
        "one-document checkpoint length {} is not under half of the full snapshot {full_length}",
        delta.1
    );

    let reopened = Collection::open(&path, None).expect("delta snapshot must open");
    assert_documents(&reopened, true);
}
