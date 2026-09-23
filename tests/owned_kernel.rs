//! Collection paths for the crate-owned filter, tokenizer, and quantizer.

use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams, MetricType,
    QuantizeType, SearchQuery,
};
use tempfile::tempdir;

fn collection(name: &str, schema: &CollectionSchema) -> (tempfile::TempDir, Collection) {
    let temporary = tempdir().expect("temporary directory");
    let path = temporary.path().join(name);
    let collection =
        Collection::create(path.to_str().expect("UTF-8 path"), schema, None).expect("collection");
    (temporary, collection)
}

fn ids(docs: &[Doc]) -> Vec<&str> {
    docs.iter().filter_map(Doc::get_pk).collect()
}

#[test]
fn collection_filter_keeps_the_accepted_document_selection() {
    let mut tag = FieldSchema::new("tag", DataType::String, false, 0).expect("tag");
    tag.set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("invert attaches");
    let embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("vector");
    let schema = CollectionSchema::builder("filter")
        .add_field(tag)
        .add_field(embedding)
        .build()
        .expect("schema");
    let (_temporary, collection) = collection("filter", &schema);
    for (id, tag) in [("doc-alpha", "alpha"), ("doc-beta", "beta")] {
        let mut doc = Doc::with_pk(id).expect("primary key");
        doc.add_string("tag", tag).expect("tag");
        doc.add_vector_f32("embedding", &[1.0, 0.0])
            .expect("vector");
        collection.insert(&[&doc]).expect("insert");
    }
    for expression in ["tag == \"alpha\"", "tag = 'alpha'", "not tag == 'beta'"] {
        let mut query = SearchQuery::new("embedding", &[1.0, 0.0], 4).expect("query");
        query.set_filter(expression).expect("filter");
        let hits = collection.query(&query).expect("filtered query");
        assert_eq!(ids(&hits), ["doc-alpha"], "{expression}");
    }
}

#[test]
fn whitespace_tokenizer_splits_terms_before_the_lowercase_filter() {
    let mut body = FieldSchema::new("body", DataType::String, false, 0).expect("body");
    body.set_index_params(&IndexParams::fts(Some("whitespace"), None, None).expect("fts"))
        .expect("fts attaches");
    let schema = CollectionSchema::builder("whitespace")
        .add_field(body)
        .build()
        .expect("schema");
    let (_temporary, collection) = collection("whitespace", &schema);
    for (id, text) in [("doc-red", "Red apple"), ("doc-blue", "blue berry")] {
        let mut doc = Doc::with_pk(id).expect("primary key");
        doc.add_string("body", text).expect("body");
        collection.insert(&[&doc]).expect("insert");
    }
    for (term, expected) in [
        ("apple", vec!["doc-red"]),
        ("berry", vec!["doc-blue"]),
        ("red", vec!["doc-red"]),
    ] {
        let mut fts = Fts::new().expect("fts");
        fts.set_match_string(term).expect("term");
        let hits = collection
            .query(&SearchQuery::fts("body", &fts, 4).expect("query"))
            .expect("query");
        assert_eq!(ids(&hits), expected, "{term}");
    }
}

#[test]
fn quantized_hnsw_search_keeps_exact_rerank_scores() {
    let docs = [
        ("doc-same", [1.0_f32, 0.0, 0.0, 0.0]),
        ("doc-mid", [1.0, 1.0, 0.0, 0.0]),
        ("doc-far", [0.0, 1.0, 0.0, 0.0]),
    ];
    let flat = ranked(&docs, QuantizeType::Undefined);
    for quantize in [QuantizeType::Fp16, QuantizeType::Int8, QuantizeType::Int4] {
        assert_eq!(ranked(&docs, quantize), flat, "{quantize:?}");
    }
}

fn ranked(docs: &[(&str, [f32; 4])], quantize: QuantizeType) -> Vec<(String, f32)> {
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("vector");
    let params = if quantize == QuantizeType::Undefined {
        IndexParams::flat(MetricType::Cosine).expect("flat")
    } else {
        IndexParams::hnsw_with_quantize(MetricType::Cosine, 8, 64, quantize).expect("hnsw")
    };
    embedding.set_index_params(&params).expect("index");
    let schema = CollectionSchema::builder("quantized")
        .add_field(embedding)
        .build()
        .expect("schema");
    let (_temporary, collection) = collection("quantized", &schema);
    let stored: Vec<Doc> = docs
        .iter()
        .map(|&(id, vector)| {
            let mut doc = Doc::with_pk(id).expect("primary key");
            doc.add_vector_f32("embedding", &vector).expect("vector");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = stored.iter().collect();
    collection.insert(&refs).expect("insert");
    let query = SearchQuery::new("embedding", &[1.0, 0.0, 0.0, 0.0], 3).expect("query");
    collection
        .query(&query)
        .expect("query")
        .iter()
        .map(|doc| (doc.get_pk().unwrap_or("").to_string(), doc.get_score()))
        .collect()
}
