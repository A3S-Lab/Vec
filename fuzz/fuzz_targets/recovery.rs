#![no_main]

use a3s_vec::{Collection, CollectionSchema, DataType, Doc, FieldSchema};
use libfuzzer_sys::fuzz_target;
use std::fs;
use std::path::Path;

fuzz_target!(|input: &[u8]| {
    if input.len() < 2 {
        return;
    }
    let temporary = match tempfile::tempdir() {
        Ok(temporary) => temporary,
        Err(_) => return,
    };
    let root = temporary.path().join("collection");
    if seed_collection(&root).is_err() {
        return;
    }

    let targets = [
        root.join("manifest.json"),
        root.join("segments/snapshot-00000000000000000001.json"),
        root.join("wal/wal-00000000000000000001.bin"),
    ];
    let target = &targets[usize::from(input[0]) % targets.len()];
    let Ok(original) = fs::read(target) else {
        return;
    };
    let mutated = mutate(&original, input[1], &input[2..]);
    if fs::write(target, mutated).is_err() {
        return;
    }

    if let Ok(collection) = Collection::open(root.to_string_lossy().as_ref(), None) {
        assert_eq!(collection.count().expect("recovered count must load"), 1);
        assert_eq!(
            collection
                .stats()
                .expect("recovered statistics must load")
                .revision,
            1
        );
        let docs = collection
            .fetch(&["doc-1"])
            .expect("recovered document must load");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].get_pk(), Some("doc-1"));
    }
});

fn seed_collection(root: &Path) -> a3s_vec::Result<()> {
    let schema = CollectionSchema::builder("recovery-fuzz")
        .add_field(FieldSchema::new("title", DataType::String, false, 0)?)
        .build()?;
    let collection = Collection::create(root.to_string_lossy().as_ref(), &schema, None)?;
    let mut doc = Doc::with_pk("doc-1")?;
    doc.add_string("title", "seed")?;
    let result = collection.insert(&[&doc])?;
    if result.success_count != 1 {
        return Err(a3s_vec::Error::internal(
            "recovery fuzz seed document was rejected",
        ));
    }
    drop(collection);
    Ok(())
}

fn mutate(original: &[u8], mode: u8, payload: &[u8]) -> Vec<u8> {
    match mode % 4 {
        0 => payload.to_vec(),
        1 => {
            let mut mutated = original.to_vec();
            for (offset, byte) in payload.iter().copied().enumerate() {
                if mutated.is_empty() {
                    mutated.push(byte);
                } else {
                    let index = offset.wrapping_add(usize::from(byte)) % mutated.len();
                    mutated[index] ^= byte | 1;
                }
            }
            mutated
        }
        2 => {
            let keep = payload.first().map_or(0, |byte| {
                usize::from(*byte).saturating_mul(original.len()) / 255
            });
            original[..keep.min(original.len())].to_vec()
        }
        _ => {
            let mut mutated = original.to_vec();
            mutated.extend_from_slice(payload);
            mutated
        }
    }
}
