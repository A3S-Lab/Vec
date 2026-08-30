use crate::doc::Doc;
use crate::schema::{CollectionSchema, FieldSchema};
use crate::types::DataType;

pub(super) fn schema() -> CollectionSchema {
    CollectionSchema::builder("storage-tests")
        .add_field(
            FieldSchema::new("title", DataType::String, false, 0)
                .expect("test field schema must be valid"),
        )
        .build()
        .expect("test collection schema must be valid")
}

pub(super) fn doc(id: &str) -> Doc {
    let mut doc = Doc::with_pk(id).expect("test primary key must be valid");
    doc.add_string("title", "stored title")
        .expect("test field value must be valid");
    doc
}
