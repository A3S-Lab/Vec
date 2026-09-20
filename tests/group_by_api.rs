#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

//! Group-by API surface: route validation, null keys, and group-count trimming.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    GroupBySearchQuery, IndexParams, MetricType,
};
use tempfile::tempdir;

fn options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options");
    options.set_durability(Durability::Manual).expect("manual");
    options
}

#[test]
fn group_by_trims_to_group_count_and_handles_null_keys() {
    let temporary = tempdir().expect("temp");
    let mut category = FieldSchema::new("category", DataType::String, true, 0).expect("category");
    category
        .set_index_params(&IndexParams::invert(false, false).expect("invert"))
        .expect("attach");
    let mut embedding =
        FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("embedding");
    embedding
        .set_index_params(&IndexParams::flat(MetricType::L2).expect("flat"))
        .expect("attach");
    let schema = CollectionSchema::builder("group-by-api")
        .add_field(category)
        .add_field(embedding)
        .build()
        .expect("schema");
    let collection = Collection::create(
        temporary.path().join("gb").to_str().expect("utf8"),
        &schema,
        Some(&options()),
    )
    .expect("create");

    for (id, category, vector) in [
        ("a1", Some("alpha"), [1.0_f32, 0.0]),
        ("a2", Some("alpha"), [0.9, 0.1]),
        ("b1", Some("beta"), [0.0, 1.0]),
        ("b2", Some("beta"), [0.1, 0.9]),
        ("c1", Some("gamma"), [0.5, 0.5]),
        ("n1", None, [0.2, 0.8]),
    ] {
        let mut doc = Doc::with_pk(id).expect("pk");
        if let Some(category) = category {
            doc.add_string("category", category).expect("category");
        }
        doc.add_vector_f32("embedding", &vector).expect("vector");
        collection.insert(&[&doc]).expect("insert");
    }

    // Candidate window must be wide enough to materialize more groups than
    // group_count so the API trims after ranking (query_api retain path).
    let mut query =
        GroupBySearchQuery::new("embedding", "category", &[1.0, 0.0], 2, 3).expect("groupby");
    query.set_include_vector(true).expect("include");
    let groups = collection.group_by(&query).expect("group_by");
    assert!(
        groups.len() <= 2,
        "must trim to group_count, got {}",
        groups.len()
    );
    assert!(!groups.is_empty());
    let via_alias = collection.group_by_query(&query).expect("alias");
    assert_eq!(groups.len(), via_alias.len());

    // Null category values collapse into the sentinel key.
    let wide = GroupBySearchQuery::new("embedding", "category", &[0.2, 0.8], 6, 2).expect("wide");
    let wide_groups = collection.group_by(&wide).expect("wide");
    assert!(wide_groups.contains_key("__null__"));

    // Ambiguous dense+binary route fails closed.
    let mut ambiguous =
        GroupBySearchQuery::new("embedding", "category", &[1.0, 0.0], 1, 1).expect("amb");
    ambiguous.binary_vector = Some(vec![0; 4]);
    assert!(collection.group_by(&ambiguous).is_err());
}
