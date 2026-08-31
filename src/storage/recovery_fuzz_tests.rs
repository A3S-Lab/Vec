//! Fixed-seed mutation fuzzing for the persisted recovery surface.

use super::test_support::{doc, schema};
use super::{snapshot, wal, StorageHandle, WalOperation};
use crate::config::{ConfigBuilder, Durability};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const RANDOM_CASES_PER_FILE: usize = 256;

#[test]
fn persisted_recovery_mutation_corpus_never_returns_a_torn_state() {
    let temporary = tempdir().expect("temporary directory must be available");
    let root = temporary.path().join("collection");
    let expected_schema = schema();
    let expected_doc = doc("doc-1");
    let mut storage =
        StorageHandle::create(&root, &expected_schema, false).expect("storage must be created");
    storage
        .append(
            1,
            WalOperation::Insert {
                docs: vec![expected_doc.clone()],
            },
            &ConfigBuilder::default().durability(Durability::Always),
        )
        .expect("seed WAL record must commit");
    drop(storage);

    let persisted = [
        root.join("manifest.json"),
        root.join(snapshot::binary_relative_path(1)),
        wal::segment_path(&root, 1),
    ];
    for (file_index, path) in persisted.iter().enumerate() {
        let original = fs::read(path).expect("seed persistence file must be readable");
        exercise_structural_mutations(&root, path, &original, &expected_schema, &expected_doc);
        exercise_random_mutations(
            &root,
            path,
            &original,
            &expected_schema,
            &expected_doc,
            0xa35e_5eed_d1ff_3000_u64 ^ u64::try_from(file_index).unwrap_or(u64::MAX),
        );
        fs::write(path, original).expect("seed persistence file must be restored");
    }
}

fn exercise_structural_mutations(
    root: &Path,
    path: &Path,
    original: &[u8],
    expected_schema: &crate::CollectionSchema,
    expected_doc: &crate::Doc,
) {
    for offset in 0..original.len() {
        let mut mutated = original.to_vec();
        mutated[offset] ^= 1_u8 << (offset % 8);
        assert_recovery_is_atomic(root, path, &mutated, expected_schema, expected_doc);
    }
    for length in truncation_lengths(original.len()) {
        assert_recovery_is_atomic(
            root,
            path,
            &original[..length],
            expected_schema,
            expected_doc,
        );
    }
    let mut extended = original.to_vec();
    extended.extend_from_slice(b"uncommitted-tail");
    assert_recovery_is_atomic(root, path, &extended, expected_schema, expected_doc);
}

fn exercise_random_mutations(
    root: &Path,
    path: &Path,
    original: &[u8],
    expected_schema: &crate::CollectionSchema,
    expected_doc: &crate::Doc,
    mut state: u64,
) {
    for _ in 0..RANDOM_CASES_PER_FILE {
        let mut mutated = original.to_vec();
        let changes = usize::from(next_u8(&mut state) % 4) + 1;
        for _ in 0..changes {
            if mutated.is_empty() {
                mutated.push(next_u8(&mut state));
            } else {
                let index = next_index(&mut state, mutated.len());
                mutated[index] ^= next_u8(&mut state) | 1;
            }
        }
        match next_u8(&mut state) % 3 {
            0 if !mutated.is_empty() => {
                let length = next_index(&mut state, mutated.len());
                mutated.truncate(length);
            }
            1 => mutated.extend((0..8).map(|_| next_u8(&mut state))),
            _ => {}
        }
        assert_recovery_is_atomic(root, path, &mutated, expected_schema, expected_doc);
    }
}

fn assert_recovery_is_atomic(
    root: &Path,
    path: &Path,
    mutated: &[u8],
    expected_schema: &crate::CollectionSchema,
    expected_doc: &crate::Doc,
) {
    fs::write(path, mutated).expect("mutated persistence file must be writable");
    if let Ok((storage, recovered_schema, docs)) = StorageHandle::open(root, false) {
        assert_eq!(recovered_schema, *expected_schema);
        assert_eq!(storage.manifest.revision, 1);
        assert_eq!(docs.as_slice(), std::slice::from_ref(expected_doc));
        drop(storage);
    }
}

fn truncation_lengths(length: usize) -> Vec<usize> {
    let mut lengths = vec![0, 1, length / 4, length / 2, length.saturating_sub(1)];
    lengths.retain(|candidate| *candidate <= length);
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn next_u8(state: &mut u64) -> u8 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    state.to_le_bytes()[4]
}

fn next_index(state: &mut u64, upper: usize) -> usize {
    usize::from(next_u8(state)) % upper
}
