#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Filtered HNSW/IVF RaBitQ must exact-rerank within the allow-list (I5).

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    HnswQueryParams, IndexParams, IndexType, IvfRabitqQueryParams, MetricType, SearchQuery,
};
use tempfile::tempdir;

const DOCUMENTS: usize = 8_400;
const ALLOWED_START: usize = DOCUMENTS / 2;
const TOPK: usize = 8;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

fn schema(name: &str, index: IndexType) -> CollectionSchema {
    let mut scope = FieldSchema::new("scope", DataType::String, false, 0).expect("scope");
    scope
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("embedding");
    let params = match index {
        IndexType::HnswRabitq => {
            IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw rabitq")
        }
        IndexType::IvfRabitq => {
            IndexParams::ivf_rabitq(MetricType::L2, 16, 8, 0).expect("ivf rabitq")
        }
        _ => panic!("rabitq only"),
    };
    embedding.set_index_params(&params).expect("attach ann");
    CollectionSchema::builder(name)
        .add_field(scope)
        .add_field(embedding)
        .build()
        .expect("schema")
}

fn document(index: usize) -> Doc {
    let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("pk");
    let allowed = index >= ALLOWED_START;
    doc.add_string("scope", if allowed { "allowed" } else { "blocked" })
        .expect("scope");
    let local = (index % 64) as f32;
    let offset = if allowed { 100.0 } else { 0.0 };
    doc.add_vector_f32("embedding", &[offset + local, local, 0.0, 1.0])
        .expect("vector");
    doc
}

fn insert_fixture(collection: &Collection) {
    let docs: Vec<Doc> = (0..DOCUMENTS).map(document).collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection.insert(&refs).expect("insert");
    assert_eq!(result.success_count, DOCUMENTS as u64);
    collection.optimize().expect("optimize");
}

fn filtered_query(index: IndexType) -> SearchQuery {
    let mut query =
        SearchQuery::new("embedding", &[100.0, 0.0, 0.0, 1.0], TOPK as i32).expect("query");
    query.set_filter("scope = \"allowed\"").expect("filter");
    let eligible = DOCUMENTS - ALLOWED_START;
    match index {
        IndexType::HnswRabitq => {
            query
                .set_hnsw_params(HnswQueryParams::new(
                    i32::try_from(eligible).expect("fits"),
                    0.0,
                    false,
                    true,
                ))
                .expect("hnsw");
        }
        IndexType::IvfRabitq => {
            let mut params = IvfRabitqQueryParams::new(
                i32::try_from(eligible.min(64)).expect("fits"),
                0.0,
                false,
                true,
            );
            params
                .set_scale_factor(f32::from(
                    u16::try_from(eligible.min(u16::MAX as usize)).expect("fits"),
                ))
                .expect("scale");
            query.set_ivf_rabitq_params(params).expect("ivf");
        }
        _ => {}
    }
    query
}

fn assert_filtered_oracle(ann: &[Doc], exact: &[Doc]) {
    assert_eq!(
        ids(ann),
        ids(exact),
        "filtered ANN ranking must match exact"
    );
    for (ann_doc, exact_doc) in ann.iter().zip(exact) {
        assert_eq!(ann_doc.get_pk(), exact_doc.get_pk());
        assert_eq!(
            ann_doc.get_score().to_bits(),
            exact_doc.get_score().to_bits(),
            "public scores must match DocumentMap re-rank"
        );
        match ann_doc.field("scope") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "allowed"),
            other => panic!("expected allowed scope, got {other:?}"),
        }
    }
}

fn ids(docs: &[Doc]) -> Vec<&str> {
    docs.iter().filter_map(Doc::get_pk).collect()
}

#[test]
fn filtered_hnsw_rabitq_stays_inside_allow_list_and_matches_exact() {
    let temporary = tempdir().expect("temp");
    let path = temporary.path().join("hnsw-rq");
    let collection = Collection::create(
        path.to_str().expect("utf8"),
        &schema("hnsw-rq", IndexType::HnswRabitq),
        Some(&options()),
    )
    .expect("create");
    insert_fixture(&collection);

    // Flat exact peer for the same filter.
    let mut flat_embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("embedding");
    flat_embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("flat");
    let mut scope = FieldSchema::new("scope", DataType::String, false, 0).expect("scope");
    scope
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach");
    let flat_schema = CollectionSchema::builder("flat-exact")
        .add_field(scope)
        .add_field(flat_embedding)
        .build()
        .expect("schema");
    let flat = Collection::create(
        temporary.path().join("flat").to_str().expect("utf8"),
        &flat_schema,
        Some(&options()),
    )
    .expect("flat");
    insert_fixture(&flat);

    let query = filtered_query(IndexType::HnswRabitq);
    let mut exact_query =
        SearchQuery::new("embedding", &[100.0, 0.0, 0.0, 1.0], TOPK as i32).expect("q");
    exact_query
        .set_filter("scope = \"allowed\"")
        .expect("filter");
    let ann = collection.query(&query).expect("ann");
    let exact = flat.query(&exact_query).expect("exact");
    assert_filtered_oracle(&ann, &exact);
}

#[test]
fn filtered_hnsw_rabitq_with_bounded_ef_stays_inside_allow_list() {
    // Exhaustive ef makes proportional_candidate_limit cover the allow-list and
    // skips the filtered ANN traversal. Bounded ef must still honor I5.
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary
            .path()
            .join("hnsw-rq-bounded")
            .to_str()
            .expect("utf8"),
        &schema("hnsw-rq-bounded", IndexType::HnswRabitq),
        Some(&options()),
    )
    .expect("create");
    insert_fixture(&collection);

    let mut query =
        SearchQuery::new("embedding", &[100.0, 0.0, 0.0, 1.0], TOPK as i32).expect("query");
    query.set_filter("scope = \"allowed\"").expect("filter");
    query
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
        .expect("bounded ef");
    let before = collection.stats_snapshot().expect("snap");
    let hits = collection.query(&query).expect("bounded filtered");
    let after = collection.stats_snapshot().expect("snap");
    assert_eq!(hits.len(), TOPK);
    assert!(
        after.ann_query_count > before.ann_query_count || after.query_count > before.query_count
    );
    for hit in &hits {
        match hit.field("scope") {
            Some(a3s_vec::FieldValue::String(value)) => assert_eq!(value, "allowed"),
            other => panic!("expected allowed, got {other:?}"),
        }
    }
}

#[test]
fn filtered_ivf_rabitq_stays_inside_allow_list_and_matches_exact() {
    let temporary = tempdir().expect("temp");
    let collection = Collection::create(
        temporary.path().join("ivf-rq").to_str().expect("utf8"),
        &schema("ivf-rq", IndexType::IvfRabitq),
        Some(&options()),
    )
    .expect("create");
    insert_fixture(&collection);

    let mut flat_embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 4).expect("embedding");
    flat_embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("flat");
    let mut scope = FieldSchema::new("scope", DataType::String, false, 0).expect("scope");
    scope
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach");
    let flat_schema = CollectionSchema::builder("flat-exact")
        .add_field(scope)
        .add_field(flat_embedding)
        .build()
        .expect("schema");
    let flat = Collection::create(
        temporary.path().join("flat").to_str().expect("utf8"),
        &flat_schema,
        Some(&options()),
    )
    .expect("flat");
    insert_fixture(&flat);

    let query = filtered_query(IndexType::IvfRabitq);
    let mut exact_query =
        SearchQuery::new("embedding", &[100.0, 0.0, 0.0, 1.0], TOPK as i32).expect("q");
    exact_query
        .set_filter("scope = \"allowed\"")
        .expect("filter");
    let ann = collection.query(&query).expect("ann");
    let exact = flat.query(&exact_query).expect("exact");
    assert_filtered_oracle(&ann, &exact);
}
