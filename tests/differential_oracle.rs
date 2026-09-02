use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, SearchQuery,
    SearchQueryBuilder,
};
use serde_json::json;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use tempfile::tempdir;

#[derive(Clone)]
struct DenseCase {
    id: String,
    vector: Vec<f32>,
    bucket: i32,
}

#[derive(Clone)]
struct SparseCase {
    id: String,
    vector: Vec<(u32, f32)>,
    bucket: i32,
}

fn vector_schema(data_type: DataType, dimension: u32) -> CollectionSchema {
    CollectionSchema::builder("differential-vector")
        .add_field(
            FieldSchema::new("bucket", DataType::Int32, false, 0)
                .expect("bucket schema must be valid"),
        )
        .add_field(
            FieldSchema::new("embedding", data_type, false, dimension)
                .expect("vector schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn dense_cases() -> Vec<DenseCase> {
    (0_i32..24)
        .map(|index| DenseCase {
            id: format!("doc-{index:02}"),
            vector: vec![
                small_i32_to_f32((index * 17 + 3) % 23 - 11) / 5.0,
                small_i32_to_f32((index * 11 + 7) % 19 - 9) / 4.0,
                small_i32_to_f32((index * 7 + 5) % 17 - 8) / 3.0,
                small_i32_to_f32((index * 5 + 2) % 13 - 6) / 2.0,
            ],
            bucket: index % 3,
        })
        .collect()
}

fn sparse_cases() -> Vec<SparseCase> {
    (0_u32..18)
        .map(|index| {
            let mut vector = Vec::new();
            for dimension in 0_u32..8 {
                if (index + dimension * 2) % 3 == 0 {
                    let raw = f32::from(
                        u16::try_from((index * 7 + dimension * 5) % 17)
                            .expect("generated value must fit u16"),
                    ) - 8.0;
                    vector.push((dimension, raw / 3.0));
                }
            }
            if vector.is_empty() {
                vector.push((index % 8, 1.0));
            }
            SparseCase {
                id: format!("sparse-{index:02}"),
                vector,
                bucket: i32::try_from(index % 3).expect("bucket must fit i32"),
            }
        })
        .collect()
}

fn small_i32_to_f32(value: i32) -> f32 {
    f32::from(i16::try_from(value).expect("generated value must fit i16"))
}

fn insert_dense(collection: &Collection, cases: &[DenseCase]) {
    let docs: Vec<Doc> = cases
        .iter()
        .map(|case| {
            let mut doc = Doc::with_pk(&case.id).expect("primary key must be valid");
            doc.add_i32("bucket", case.bucket)
                .expect("bucket must be valid");
            doc.add_vector_f32("embedding", &case.vector)
                .expect("vector must be valid");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection.insert(&refs).expect("insert must succeed");
    assert_eq!(
        result.success_count,
        u64::try_from(cases.len()).expect("case count must fit u64")
    );
}

fn insert_sparse(collection: &Collection, cases: &[SparseCase]) {
    let docs: Vec<Doc> = cases
        .iter()
        .map(|case| {
            let mut doc = Doc::with_pk(&case.id).expect("primary key must be valid");
            doc.add_i32("bucket", case.bucket)
                .expect("bucket must be valid");
            let (indices, values): (Vec<_>, Vec<_>) = case.vector.iter().copied().unzip();
            doc.add_sparse_vector_f32("embedding", &indices, &values)
                .expect("sparse vector must be valid");
            doc
        })
        .collect();
    let refs: Vec<&Doc> = docs.iter().collect();
    let result = collection.insert(&refs).expect("insert must succeed");
    assert_eq!(
        result.success_count,
        u64::try_from(cases.len()).expect("case count must fit u64")
    );
}

fn dense_score(query: &[f32], vector: &[f32], metric: &str) -> f64 {
    let query: Vec<f64> = query.iter().map(|value| f64::from(*value)).collect();
    let vector: Vec<f64> = vector.iter().map(|value| f64::from(*value)).collect();
    match metric {
        "l2" => -query
            .iter()
            .zip(&vector)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>(),
        "cosine" => {
            let dot = dot_product(&query, &vector);
            let query_norm = dot_product(&query, &query).sqrt();
            let vector_norm = dot_product(&vector, &vector).sqrt();
            if query_norm == 0.0 || vector_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * vector_norm)
            }
        }
        "ip" | "mips_l2" => dot_product(&query, &vector),
        _ => panic!("unknown test metric"),
    }
}

fn dot_product(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn sparse_score(query: &[(u32, f32)], vector: &[(u32, f32)], metric: &str) -> f64 {
    let query: BTreeMap<u32, f64> = query
        .iter()
        .map(|(index, value)| (*index, f64::from(*value)))
        .collect();
    let vector: BTreeMap<u32, f64> = vector
        .iter()
        .map(|(index, value)| (*index, f64::from(*value)))
        .collect();
    let dot = query
        .iter()
        .filter_map(|(index, value)| vector.get(index).map(|other| value * other))
        .sum::<f64>();
    match metric {
        "l2" => {
            let query_distance = query
                .iter()
                .map(|(index, value)| {
                    (value - vector.get(index).copied().unwrap_or_default()).powi(2)
                })
                .sum::<f64>();
            let vector_distance = vector
                .iter()
                .filter(|(index, _)| !query.contains_key(index))
                .map(|(_, value)| value.powi(2))
                .sum::<f64>();
            -(query_distance + vector_distance)
        }
        "cosine" => {
            let query_norm = query
                .values()
                .map(|value| value.powi(2))
                .sum::<f64>()
                .sqrt();
            let vector_norm = vector
                .values()
                .map(|value| value.powi(2))
                .sum::<f64>()
                .sqrt();
            if query_norm == 0.0 || vector_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * vector_norm)
            }
        }
        "ip" | "mips_l2" => dot,
        _ => panic!("unknown test metric"),
    }
}

fn sort_reference(values: &mut [(String, f64)]) {
    values.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
}

fn assert_ranked_result(actual: &[Doc], expected: &[(String, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, (expected_id, expected_score)) in actual.iter().zip(expected) {
        assert_eq!(actual.get_pk(), Some(expected_id.as_str()));
        let tolerance = (expected_score.abs() * 1.0e-6).max(1.0e-6);
        assert!(
            (f64::from(actual.get_score()) - expected_score).abs() <= tolerance,
            "id={expected_id}, expected={expected_score}, actual={}",
            actual.get_score()
        );
    }
}

#[test]
fn dense_exact_scan_matches_an_independent_reference() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("dense")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &vector_schema(DataType::VectorFp32, 4),
        None,
    )
    .expect("collection must be created");
    let cases = dense_cases();
    insert_dense(&collection, &cases);
    let query_vector = [0.75_f32, -1.25, 2.0, 0.5];

    for (metric, radius) in [
        ("l2", 3.0_f32),
        ("ip", 0.5),
        ("cosine", 0.1),
        ("mips_l2", 0.5),
    ] {
        let mut query =
            SearchQuery::new("embedding", &query_vector, 7).expect("query must be valid");
        query.params.insert("metric".into(), json!(metric));
        query.set_radius(radius).expect("radius must be valid");
        query
            .set_filter("bucket != 2")
            .expect("filter must be valid");
        let actual = collection.query(&query).expect("query must succeed");
        let mut expected: Vec<(String, f64)> = cases
            .iter()
            .filter(|case| case.bucket != 2)
            .map(|case| {
                (
                    case.id.clone(),
                    dense_score(&query_vector, &case.vector, metric),
                )
            })
            .filter(|(_, score)| {
                if metric == "l2" {
                    *score >= -f64::from(radius).powi(2)
                } else {
                    *score >= f64::from(radius)
                }
            })
            .collect();
        sort_reference(&mut expected);
        expected.truncate(7);
        assert_ranked_result(&actual, &expected);
    }
}

#[test]
fn include_doc_id_resolves_the_generation_ordinal_and_survives_reopen() {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary
        .path()
        .join("doc-id")
        .to_str()
        .expect("temporary path must be UTF-8")
        .to_string();
    let collection = Collection::create(&path, &vector_schema(DataType::VectorFp32, 4), None)
        .expect("collection must be created");
    let cases = dense_cases();
    insert_dense(&collection, &cases);

    let mut query =
        SearchQuery::new("embedding", &[0.75, -1.25, 2.0, 0.5], 7).expect("query must be valid");
    query
        .set_include_doc_id(true)
        .expect("include-doc-id control must be accepted");
    let first = collection.query(&query).expect("query must succeed");
    assert_eq!(first.len(), 7);
    let first_ids: Vec<_> = first
        .iter()
        .map(|doc| {
            (
                doc.get_pk().expect("query result must have a primary key"),
                doc.doc_id()
                    .expect("query result must include a document ID"),
            )
        })
        .collect();
    let mut distinct = first_ids.iter().map(|(_, id)| *id).collect::<Vec<_>>();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), first_ids.len());

    let hidden =
        SearchQuery::new("embedding", &[0.75, -1.25, 2.0, 0.5], 7).expect("query must be valid");
    let hidden_results = collection.query(&hidden).expect("query must succeed");
    assert!(hidden_results.iter().all(|doc| doc.doc_id().is_none()));

    collection.flush().expect("flush must succeed");
    collection.close().expect("collection must close");
    let reopened = Collection::open(&path, None).expect("collection must reopen");
    let reopened_results = reopened.query(&query).expect("reopened query must succeed");
    let reopened_ids: Vec<_> = reopened_results
        .iter()
        .map(|doc| {
            (
                doc.get_pk().expect("query result must have a primary key"),
                doc.doc_id()
                    .expect("query result must include a document ID"),
            )
        })
        .collect();
    assert_eq!(reopened_ids, first_ids);
}

#[test]
fn query_builder_dense_route_executes_controls_and_matches_direct_query() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("builder-dense")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &vector_schema(DataType::VectorFp32, 4),
        None,
    )
    .expect("collection must be created");
    let cases = dense_cases();
    insert_dense(&collection, &cases);
    let query_vector = [0.75_f32, -1.25, 2.0, 0.5];

    let mut direct = SearchQuery::new("embedding", &query_vector, 5).expect("query must be valid");
    direct
        .set_filter("bucket != 2")
        .expect("direct filter must be valid");
    direct
        .set_include_vector(true)
        .expect("direct vector projection must be valid");
    direct
        .set_include_doc_id(true)
        .expect("direct document IDs must be valid");
    direct
        .set_output_fields(&["bucket", "embedding"])
        .expect("direct projection must be valid");
    let direct = collection
        .query(&direct)
        .expect("direct dense query must succeed");
    let built = SearchQueryBuilder::new()
        .field_name("embedding")
        .vector(&query_vector)
        .topk(5)
        .filter("bucket != 2")
        .include_vector(true)
        .include_doc_id(true)
        .output_fields(&["bucket", "embedding"])
        .build()
        .expect("dense builder route must be valid");
    let filtered = collection
        .query(&built)
        .expect("dense builder route must execute");
    assert_eq!(filtered.len(), 5);
    assert!(filtered.iter().all(|doc| {
        doc.doc_id().is_some()
            && doc.vector("embedding").is_some()
            && doc.get_i32("bucket").expect("bucket getter").is_some()
    }));
    assert_eq!(
        filtered.iter().map(Doc::get_pk).collect::<Vec<_>>(),
        direct.iter().map(Doc::get_pk).collect::<Vec<_>>()
    );
    assert_eq!(
        filtered.iter().map(Doc::get_score).collect::<Vec<_>>(),
        direct.iter().map(Doc::get_score).collect::<Vec<_>>()
    );
    let mut expected = cases
        .iter()
        .filter(|case| case.bucket != 2)
        .map(|case| {
            (
                case.id.clone(),
                dense_score(&query_vector, &case.vector, "cosine"),
            )
        })
        .collect::<Vec<_>>();
    sort_reference(&mut expected);
    expected.truncate(5);
    assert_ranked_result(&filtered, &expected);
}

#[test]
fn sparse_exact_scan_matches_an_independent_reference() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("sparse")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &vector_schema(DataType::SparseVectorFp32, 8),
        None,
    )
    .expect("collection must be created");
    let cases = sparse_cases();
    insert_sparse(&collection, &cases);
    let query_vector = [(0_u32, 1.25_f32), (2, -0.75), (5, 2.0)];
    let (indices, values): (Vec<_>, Vec<_>) = query_vector.iter().copied().unzip();

    for (metric, radius) in [
        ("l2", 3.0_f32),
        ("ip", 0.0),
        ("cosine", 0.0),
        ("mips_l2", 0.0),
    ] {
        let mut query =
            SearchQuery::sparse("embedding", &indices, &values, 6).expect("query must be valid");
        query.params.insert("metric".into(), json!(metric));
        query.set_radius(radius).expect("radius must be valid");
        query
            .set_filter("bucket == 1")
            .expect("filter must be valid");
        let actual = collection.query(&query).expect("query must succeed");
        let mut expected: Vec<(String, f64)> = cases
            .iter()
            .filter(|case| case.bucket == 1)
            .map(|case| {
                (
                    case.id.clone(),
                    sparse_score(&query_vector, &case.vector, metric),
                )
            })
            .filter(|(_, score)| {
                if metric == "l2" {
                    *score >= -f64::from(radius).powi(2)
                } else {
                    *score >= f64::from(radius)
                }
            })
            .collect();
        sort_reference(&mut expected);
        expected.truncate(6);
        assert_ranked_result(&actual, &expected);
    }
}

#[test]
fn fp64_ranking_uses_the_exact_score_before_public_narrowing() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("fp64-rank")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &vector_schema(DataType::VectorFp64, 2),
        None,
    )
    .expect("collection must be created");
    let mut best = Doc::with_pk("z-best").expect("primary key must be valid");
    best.add_i32("bucket", 0).expect("bucket must be valid");
    best.add_vector_f64("embedding", &[1.000_000_002, 0.0])
        .expect("vector must be valid");
    let mut second = Doc::with_pk("a-second").expect("primary key must be valid");
    second.add_i32("bucket", 0).expect("bucket must be valid");
    second
        .add_vector_f64("embedding", &[1.000_000_001, 0.0])
        .expect("vector must be valid");
    let mut tied_best = Doc::with_pk("b-best").expect("primary key must be valid");
    tied_best
        .add_i32("bucket", 0)
        .expect("bucket must be valid");
    tied_best
        .add_vector_f64("embedding", &[1.000_000_002, 0.0])
        .expect("vector must be valid");
    collection
        .insert(&[&best, &second, &tied_best])
        .expect("insert must succeed");
    let mut query = SearchQuery::new("embedding", &[1.0, 0.0], 2).expect("query must be valid");
    query.params.insert("metric".into(), json!("ip"));
    let actual = collection.query(&query).expect("query must succeed");
    let actual_ids: Vec<_> = actual.iter().map(Doc::get_pk).collect();
    assert_eq!(actual_ids, [Some("b-best"), Some("z-best")]);
}

#[test]
fn topk_precedes_public_score_narrowing() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("fp64-topk")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &vector_schema(DataType::VectorFp64, 1),
        None,
    )
    .expect("collection must be created");
    let mut selected = Doc::with_pk("selected").expect("primary key must be valid");
    selected.add_i32("bucket", 0).expect("bucket must be valid");
    selected
        .add_vector_f64("embedding", &[1.0])
        .expect("vector must be valid");
    let mut excluded = Doc::with_pk("excluded").expect("primary key must be valid");
    excluded.add_i32("bucket", 0).expect("bucket must be valid");
    excluded
        .add_vector_f64("embedding", &[-f64::MAX])
        .expect("vector must be valid");
    collection
        .insert(&[&selected, &excluded])
        .expect("insert must succeed");
    let mut query = SearchQuery::new("embedding", &[1.0], 1).expect("query must be valid");
    query.params.insert("metric".into(), json!("ip"));
    let actual = collection
        .query(&query)
        .expect("an excluded score does not cross the public f32 boundary");
    assert_eq!(actual[0].get_pk(), Some("selected"));
    assert!((actual[0].get_score() - 1.0).abs() <= f32::EPSILON);
}

#[test]
fn negative_l2_radius_is_rejected() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection = Collection::create(
        temporary
            .path()
            .join("radius")
            .to_str()
            .expect("temporary path must be UTF-8"),
        &vector_schema(DataType::VectorFp32, 4),
        None,
    )
    .expect("collection must be created");
    let mut query = SearchQuery::new("embedding", &[0.0; 4], 10).expect("query must be valid");
    query.params.insert("metric".into(), json!("l2"));
    query
        .set_radius(-1.0)
        .expect("generic radius setter is valid");
    let error = collection
        .query(&query)
        .expect_err("negative Euclidean radius must fail");
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}
