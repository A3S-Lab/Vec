use super::Collection;
use crate::{CollectionSchema, DataType, Doc, FieldSchema};
use std::sync::Arc;
use tempfile::tempdir;

fn vector_doc(id: &str, value: f32) -> Doc {
    let mut doc = Doc::with_pk(id).expect("document must be valid");
    doc.add_vector_f32("embedding", &[value, 0.0])
        .expect("vector must be valid");
    doc
}

#[test]
fn collection_generations_share_unchanged_documents() {
    let temporary = tempdir().expect("temporary directory must be available");
    let schema = CollectionSchema::builder("shared-documents")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
                .expect("vector field must be valid"),
        )
        .build()
        .expect("schema must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("shared-documents")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &schema,
        None,
    )
    .expect("collection must be created");
    let first = vector_doc("first", 1.0);
    let second = vector_doc("second", 2.0);
    collection
        .insert(&[&first, &second])
        .expect("documents must be inserted");
    let before = collection
        .inner
        .state
        .read()
        .expect("state lock must not be poisoned")
        .docs
        .clone();

    let replacement = vector_doc("first", 3.0);
    collection
        .upsert(&[&replacement])
        .expect("replacement must be published");
    let after = collection
        .inner
        .state
        .read()
        .expect("state lock must not be poisoned")
        .docs
        .clone();

    assert!(!Arc::ptr_eq(&before["first"], &after["first"]));
    assert!(Arc::ptr_eq(&before["second"], &after["second"]));
}

#[test]
fn document_generation_clone_shares_the_persistent_tree() {
    let mut original = crate::doc::DocumentMap::new();
    for index in 0..256 {
        let id = format!("doc-{index:04}");
        let value = f32::from(u16::try_from(index).expect("fixture index fits u16"));
        original.insert(id.clone(), Arc::new(vector_doc(&id, value)));
    }

    let mut next = original.clone();
    assert!(original.ptr_eq(&next));

    next.insert("new".into(), Arc::new(vector_doc("new", 999.0)));
    assert!(!original.ptr_eq(&next));
    assert_eq!(original.len(), 256);
    assert_eq!(next.len(), 257);
    assert!(Arc::ptr_eq(&original["doc-0128"], &next["doc-0128"]));
}
