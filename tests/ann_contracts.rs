use a3s_vec::{
    Collection, CollectionSchema, DataType, DiskannQueryParams, Doc, ErrorCode, FieldSchema,
    HnswQueryParams, IndexParams, IndexType, IvfQueryParams, IvfRabitqQueryParams, MetricType,
    QuantizeType, SearchQuery,
};
use tempfile::tempdir;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("ann-contracts")
        .add_field(
            FieldSchema::new("embedding", DataType::VectorFp32, false, 8)
                .expect("vector schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn vector_for(index: usize) -> Vec<f32> {
    (0..8)
        .map(|dimension| {
            let raw = (index * 31 + dimension * 17 + index * dimension * 3) % 101;
            (f32::from(u16::try_from(raw).expect("fixture value fits u16")) - 50.0) / 50.0
        })
        .collect()
}

fn doc(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:04}")).expect("test primary key must be valid");
    doc.add_vector_f32("embedding", &vector_for(index))
        .expect("test vector must be valid");
    doc
}

fn insert_docs(collection: &Collection, count: usize) {
    let docs: Vec<Doc> = (0..count).map(doc).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection.insert(&refs).expect("insert must succeed");
    assert_eq!(result.success_count, count as u64);
}

fn ids(docs: &[Doc]) -> Vec<String> {
    docs.iter()
        .map(|doc| {
            doc.get_pk()
                .expect("result must have a primary key")
                .to_string()
        })
        .collect()
}

fn assert_same_ranking(left: &[Doc], right: &[Doc]) {
    assert_eq!(ids(left), ids(right));
    assert_eq!(
        left.iter().map(Doc::get_score).collect::<Vec<_>>(),
        right.iter().map(Doc::get_score).collect::<Vec<_>>()
    );
}

include!("ann_contracts/graph.rs");
include!("ann_contracts/rabitq.rs");
include!("ann_contracts/lifecycle.rs");
