use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams, SearchQuery,
};
use serde_json::json;
use std::cmp::Ordering;
use tempfile::tempdir;

const CORPUS_SIZE: usize = 256;
const VECTOR_DIMENSION: usize = 12;
const CORPUS_SEED: u64 = 0xa35e_5eed_d1ff_2026;
const VOCABULARY: [&str; 16] = [
    "agent",
    "async",
    "cache",
    "database",
    "durable",
    "embedding",
    "filter",
    "index",
    "memory",
    "query",
    "ranking",
    "rust",
    "search",
    "storage",
    "vector",
    "wal",
];

#[derive(Clone)]
struct GeneratedCase {
    id: String,
    vector: Vec<f32>,
    bucket: i32,
    active: bool,
    label: String,
    tags: Vec<String>,
    body: Option<String>,
}

#[derive(Clone, Copy)]
struct FilterCase {
    expression: Option<&'static str>,
    predicate: fn(&GeneratedCase) -> bool,
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.state >> 32).expect("upper half of u64 must fit u32")
    }

    fn bounded(&mut self, upper: u32) -> u32 {
        assert!(upper > 0, "random upper bound must be positive");
        self.next_u32() % upper
    }
}

fn generated_schema() -> CollectionSchema {
    let fts_params =
        IndexParams::fts(Some("whitespace"), None, None).expect("FTS parameters must be valid");
    let mut body =
        FieldSchema::new("body", DataType::String, true, 0).expect("body schema must be valid");
    body.set_index_params(&fts_params)
        .expect("FTS parameters must be attachable");

    CollectionSchema::builder("generated-differential")
        .add_field(
            FieldSchema::new("bucket", DataType::Int32, false, 0)
                .expect("bucket schema must be valid"),
        )
        .add_field(
            FieldSchema::new("active", DataType::Bool, false, 0)
                .expect("active schema must be valid"),
        )
        .add_field(
            FieldSchema::new("label", DataType::String, false, 0)
                .expect("label schema must be valid"),
        )
        .add_field(
            FieldSchema::new("tags", DataType::ArrayString, false, 0)
                .expect("tags schema must be valid"),
        )
        .add_field(body)
        .add_field(
            FieldSchema::new(
                "embedding",
                DataType::VectorFp32,
                false,
                u32::try_from(VECTOR_DIMENSION).expect("dimension must fit u32"),
            )
            .expect("vector schema must be valid"),
        )
        .build()
        .expect("collection schema must be valid")
}

fn generated_cases() -> Vec<GeneratedCase> {
    let mut rng = DeterministicRng::new(CORPUS_SEED);
    (0..CORPUS_SIZE)
        .map(|index| {
            let bucket = i32::try_from(rng.bounded(8)).expect("bucket must fit i32");
            let vector = (0..VECTOR_DIMENSION)
                .map(|_| {
                    let raw =
                        i32::try_from(rng.bounded(2_001)).expect("coordinate must fit i32") - 1_000;
                    f32::from(i16::try_from(raw).expect("coordinate must fit i16")) / 257.0
                })
                .collect();
            let active = rng.bounded(5) != 0;
            let parity = if index % 2 == 0 { "even" } else { "odd" };
            let temperature = if bucket < 3 { "hot" } else { "cold" };
            let tags = vec!["searchable".into(), parity.into(), temperature.into()];
            let label = format!("group-{bucket}-item-{index:03}");
            let body = if index % 13 == 0 {
                None
            } else {
                let token_count =
                    3 + usize::try_from(rng.bounded(18)).expect("token count must fit usize");
                let tokens: Vec<&str> = (0..token_count)
                    .map(|_| {
                        let vocabulary_index = usize::try_from(rng.bounded(
                            u32::try_from(VOCABULARY.len()).expect("vocabulary size must fit u32"),
                        ))
                        .expect("vocabulary index must fit usize");
                        VOCABULARY[vocabulary_index]
                    })
                    .collect();
                Some(tokens.join(" "))
            };
            GeneratedCase {
                id: format!("generated-{index:03}"),
                vector,
                bucket,
                active,
                label,
                tags,
                body,
            }
        })
        .collect()
}

fn insert_cases(collection: &Collection, cases: &[GeneratedCase]) {
    let docs: Vec<Doc> = cases
        .iter()
        .map(|case| {
            let mut doc = Doc::with_pk(&case.id).expect("primary key must be valid");
            doc.add_i32("bucket", case.bucket)
                .expect("bucket must be valid");
            doc.add_bool("active", case.active)
                .expect("active must be valid");
            doc.add_string("label", &case.label)
                .expect("label must be valid");
            let tags: Vec<&str> = case.tags.iter().map(String::as_str).collect();
            doc.add_array_string("tags", &tags)
                .expect("tags must be valid");
            if let Some(body) = &case.body {
                doc.add_string("body", body).expect("body must be valid");
            }
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

fn all_cases(_: &GeneratedCase) -> bool {
    true
}

fn active_selected_buckets(case: &GeneratedCase) -> bool {
    case.active && matches!(case.bucket, 1 | 3 | 5)
}

fn prefix_or_high_bucket(case: &GeneratedCase) -> bool {
    case.label.starts_with("group-2-") || case.bucket >= 6
}

fn searchable_even(case: &GeneratedCase) -> bool {
    case.tags.iter().any(|tag| tag == "searchable") && case.tags.iter().any(|tag| tag == "even")
}

fn missing_body(case: &GeneratedCase) -> bool {
    case.body.is_none()
}

fn filter_cases() -> [FilterCase; 5] {
    [
        FilterCase {
            expression: None,
            predicate: all_cases,
        },
        FilterCase {
            expression: Some("bucket in [1, 3, 5] and active == true"),
            predicate: active_selected_buckets,
        },
        FilterCase {
            expression: Some("label has_prefix 'group-2-' or bucket >= 6"),
            predicate: prefix_or_high_bucket,
        },
        FilterCase {
            expression: Some("tags contain_all ['searchable', 'even']"),
            predicate: searchable_even,
        },
        FilterCase {
            expression: Some("body is_null"),
            predicate: missing_body,
        },
    ]
}

fn dense_score(query: &[f32], vector: &[f32], metric: &str) -> f64 {
    let query: Vec<f64> = query.iter().map(|value| f64::from(*value)).collect();
    let vector: Vec<f64> = vector.iter().map(|value| f64::from(*value)).collect();
    let dot = query.iter().zip(&vector).map(|(a, b)| a * b).sum::<f64>();
    match metric {
        "l2" => -query
            .iter()
            .zip(&vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>(),
        "cosine" => {
            let query_norm = query.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
            let vector_norm = vector.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
            if query_norm == 0.0 || vector_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * vector_norm)
            }
        }
        "ip" | "mips_l2" => dot,
        _ => panic!("unknown test metric: {metric}"),
    }
}

fn count_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("generated corpus count must fit u32"))
}

fn reference_bm25(
    cases: &[GeneratedCase],
    terms: &[&str],
    predicate: fn(&GeneratedCase) -> bool,
) -> Vec<(String, f64)> {
    let corpus: Vec<Vec<&str>> = cases
        .iter()
        .filter_map(|case| {
            case.body
                .as_deref()
                .map(|body| body.split_whitespace().collect())
        })
        .collect();
    let document_count = count_to_f64(corpus.len());
    let average_length = count_to_f64(corpus.iter().map(Vec::len).sum()) / document_count;
    let mut scored = Vec::new();
    for case in cases.iter().filter(|case| predicate(case)) {
        let Some(body) = case.body.as_deref() else {
            continue;
        };
        let tokens: Vec<&str> = body.split_whitespace().collect();
        let mut score = 0.0;
        for term in terms {
            let frequency = count_to_f64(tokens.iter().filter(|token| **token == *term).count());
            if frequency == 0.0 {
                continue;
            }
            let document_frequency =
                count_to_f64(corpus.iter().filter(|tokens| tokens.contains(term)).count());
            let idf = ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                + 1.0)
                .ln();
            let normalized_length = count_to_f64(tokens.len()) / average_length;
            let denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * normalized_length);
            score += idf * (frequency * 2.2 / denominator);
        }
        if score > 0.0 {
            scored.push((case.id.clone(), score));
        }
    }
    sort_reference(&mut scored);
    scored
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

fn assert_ranked_result(actual: &[Doc], expected: &[(String, f64)], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (actual, (expected_id, expected_score)) in actual.iter().zip(expected) {
        assert_eq!(actual.get_pk(), Some(expected_id.as_str()), "{context}");
        let tolerance = (expected_score.abs() * 1.0e-6).max(1.0e-6);
        assert!(
            (f64::from(actual.get_score()) - expected_score).abs() <= tolerance,
            "{context}: id={expected_id}, expected={expected_score}, actual={}",
            actual.get_score()
        );
    }
}

fn create_populated_collection(cases: &[GeneratedCase]) -> (tempfile::TempDir, Collection) {
    let temporary = tempdir().expect("temporary directory must be available");
    let path = temporary.path().join("generated");
    let path = path.to_str().expect("temporary path must be UTF-8");
    let collection =
        Collection::create(path, &generated_schema(), None).expect("collection must be created");
    insert_cases(&collection, cases);
    collection.flush().expect("collection must be flushed");
    collection.close().expect("collection must be closed");
    let reopened = Collection::open(path, None).expect("collection must reopen");
    (temporary, reopened)
}

#[test]
fn generated_dense_queries_match_independent_metrics_and_filters() {
    let cases = generated_cases();
    let (_temporary, collection) = create_populated_collection(&cases);
    for (query_number, case_index) in [3_usize, 37, 89, 151, 233].into_iter().enumerate() {
        let query_vector = &cases[case_index].vector;
        let topk = 1 + i32::try_from((query_number * 11) % 47).expect("topk must fit i32");
        for metric in ["l2", "ip", "cosine", "mips_l2"] {
            for filter in filter_cases() {
                let mut query =
                    SearchQuery::new("embedding", query_vector, topk).expect("query must be valid");
                query.params.insert("metric".into(), json!(metric));
                if let Some(expression) = filter.expression {
                    query.set_filter(expression).expect("filter must be valid");
                }
                let actual = collection.query(&query).expect("query must succeed");
                let mut expected: Vec<(String, f64)> = cases
                    .iter()
                    .filter(|case| (filter.predicate)(case))
                    .map(|case| {
                        (
                            case.id.clone(),
                            dense_score(query_vector, &case.vector, metric),
                        )
                    })
                    .collect();
                sort_reference(&mut expected);
                expected.truncate(usize::try_from(topk).expect("topk must fit usize"));
                let context = format!(
                    "seed={CORPUS_SEED:#x}, query={case_index}, metric={metric}, filter={:?}",
                    filter.expression
                );
                assert_ranked_result(&actual, &expected, &context);
            }
        }
    }
}

#[test]
fn generated_fts_queries_match_independent_bm25_and_filters() {
    let cases = generated_cases();
    let (_temporary, collection) = create_populated_collection(&cases);
    let term_sets: [&[&str]; 6] = [
        &["rust"],
        &["vector", "search"],
        &["database", "storage", "wal"],
        &["agent", "async"],
        &["ranking", "query", "filter"],
        &["cache", "durable", "embedding", "index"],
    ];
    for terms in term_sets {
        for filter in filter_cases().into_iter().take(4) {
            let mut fts = Fts::new().expect("FTS payload must be valid");
            fts.set_match_string(&terms.join(" "))
                .expect("FTS expression must be valid");
            let mut query = SearchQuery::fts("body", &fts, 29).expect("FTS query must be valid");
            if let Some(expression) = filter.expression {
                query.set_filter(expression).expect("filter must be valid");
            }
            let actual = collection.query(&query).expect("FTS query must succeed");
            let mut expected = reference_bm25(&cases, terms, filter.predicate);
            expected.truncate(29);
            let context = format!(
                "seed={CORPUS_SEED:#x}, terms={terms:?}, filter={:?}",
                filter.expression
            );
            assert_ranked_result(&actual, &expected, &context);
        }
    }
}
