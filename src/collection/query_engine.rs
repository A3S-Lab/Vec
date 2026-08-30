//! Exact query execution used as the collection correctness oracle.

use super::query_contract::{query_metric, validate_query_contract, validate_tokenizer};
use crate::doc::{Doc, FieldValue, VectorValue};
use crate::error::{Error, Result};
use crate::query::SearchQuery;
use crate::schema::{CollectionSchema, IndexParams};
use crate::types::MetricType;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeMap;

struct ScoredDoc {
    exact_score: f64,
    doc: Doc,
}

impl ScoredDoc {
    fn new(exact_score: f64, doc: Doc) -> Result<Self> {
        if !exact_score.is_finite() {
            return Err(Error::resource_exhausted("query score is not finite"));
        }
        Ok(Self { exact_score, doc })
    }
}

/// Query scores are exposed as `f32`; reject non-finite and out-of-range
/// intermediates before performing the intentional narrowing cast.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn score_to_f32(value: f64) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(Error::resource_exhausted(
            "query score cannot be represented as f32",
        ));
    }
    Ok(value as f32)
}

/// Ranking statistics use floating-point arithmetic. Storage limits keep
/// these counts far below the exact-integer boundary of `f64`.
#[allow(clippy::cast_precision_loss)]
pub(super) fn count_to_f64(value: usize) -> f64 {
    value as f64
}

fn dense_score(query: &[f64], vector: &VectorValue, metric: MetricType) -> Option<f64> {
    let values = vector.to_dense_f64()?;
    if values.len() != query.len() {
        return None;
    }
    Some(match metric {
        MetricType::L2 => -query
            .iter()
            .zip(values.iter())
            .map(|(left, right)| {
                let difference = *left - *right;
                difference * difference
            })
            .sum::<f64>(),
        MetricType::Cosine => {
            let dot = query
                .iter()
                .zip(values.iter())
                .map(|(left, right)| *left * *right)
                .sum::<f64>();
            let query_norm = query.iter().map(|value| value * value).sum::<f64>().sqrt();
            let value_norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            if query_norm == 0.0 || value_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * value_norm)
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => query
            .iter()
            .zip(values.iter())
            .map(|(left, right)| *left * *right)
            .sum(),
    })
}

pub(super) fn sort_docs(docs: &mut [Doc]) {
    docs.sort_by(|left, right| {
        right
            .get_score()
            .partial_cmp(&left.get_score())
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.get_pk()
                    .unwrap_or_default()
                    .cmp(right.get_pk().unwrap_or_default())
            })
    });
}

fn sort_scored_docs(docs: &mut [ScoredDoc]) {
    docs.sort_by(|left, right| {
        right
            .exact_score
            .partial_cmp(&left.exact_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.doc
                    .get_pk()
                    .unwrap_or_default()
                    .cmp(right.doc.get_pk().unwrap_or_default())
            })
    });
}

pub(super) fn parse_filter_expression(expression: &str) -> Result<zvec_core::filter::FilterExpr> {
    if expression.trim().is_empty() {
        return Err(Error::invalid_argument(
            "filter expression must not be empty",
        ));
    }
    zvec_core::filter::parse_filter(expression)
        .map_err(|error| Error::invalid_argument(error.to_string()))
}

pub(super) fn matches_filter(doc: &Doc, expression: Option<&str>) -> bool {
    let Some(expression) = expression else {
        return true;
    };
    let Ok(parsed) = parse_filter_expression(expression) else {
        return false;
    };
    parsed.matches(&doc.to_core())
}

pub(super) fn execute_query(
    schema: &CollectionSchema,
    docs: &[Doc],
    query: &SearchQuery,
) -> Result<Vec<Doc>> {
    let field = validate_query_contract(schema, query)?;
    if let Some(filter) = query.filter.as_deref() {
        parse_filter_expression(filter)?;
    }
    let metric = query_metric(&field, query)?;
    let mut scored = if query.fts.is_some() {
        execute_fts(docs, query, field.index_params)?
    } else {
        execute_vector(docs, query, metric)?
    };
    sort_scored_docs(&mut scored);
    let topk = usize::try_from(query.topk)
        .map_err(|_| Error::invalid_argument("query topk must be positive"))?;
    scored.truncate(topk);
    let output_fields = query.output_fields.as_deref();
    scored
        .into_iter()
        .map(|mut scored| {
            scored.doc.set_score(score_to_f32(scored.exact_score)?)?;
            Ok(scored.doc.project(output_fields, query.include_vector))
        })
        .collect()
}

fn execute_vector(docs: &[Doc], query: &SearchQuery, metric: MetricType) -> Result<Vec<ScoredDoc>> {
    let dense_query = if let Some(vector) = &query.vector {
        Some(vector.iter().map(|value| f64::from(*value)).collect())
    } else if let Some(id) = &query.id {
        docs.iter()
            .find(|doc| doc.get_pk() == Some(id.as_str()))
            .and_then(|doc| doc.vector(&query.field_name))
            .and_then(VectorValue::to_dense_f64)
    } else {
        None
    };
    let sparse_query: Option<BTreeMap<u32, f64>> = query.sparse_vector.as_ref().map(|values| {
        values
            .iter()
            .map(|(index, value)| (*index, f64::from(*value)))
            .collect()
    });
    if dense_query.is_none() && sparse_query.is_none() {
        return Err(Error::invalid_argument(
            "query requires a dense vector, sparse vector, or source id",
        ));
    }
    let radius = query.params.get("radius").and_then(Value::as_f64);
    let mut result = Vec::new();
    for doc in docs {
        if !matches_filter(doc, query.filter.as_deref()) {
            continue;
        }
        let Some(vector) = doc.vector(&query.field_name) else {
            continue;
        };
        let score = if let Some(ref dense) = dense_query {
            let Some(score) = dense_score(dense, vector, metric) else {
                continue;
            };
            score
        } else {
            let Some(stored) = vector.to_sparse_f64() else {
                continue;
            };
            sparse_query
                .as_ref()
                .map_or(0.0, |candidate| sparse_score(candidate, &stored, metric))
        };
        if let Some(radius) = radius {
            let passes = if metric == MetricType::L2 {
                score >= -radius * radius
            } else {
                score >= radius
            };
            if !passes {
                continue;
            }
        }
        result.push(ScoredDoc::new(score, doc.clone())?);
    }
    Ok(result)
}

fn sparse_score(
    query: &BTreeMap<u32, f64>,
    stored: &BTreeMap<u32, f64>,
    metric: MetricType,
) -> f64 {
    let dot = query
        .iter()
        .filter_map(|(index, value)| stored.get(index).map(|other| value * other))
        .sum::<f64>();
    match metric {
        MetricType::L2 => {
            let query_distance = query
                .iter()
                .map(|(index, value)| {
                    let difference = value - stored.get(index).copied().unwrap_or_default();
                    difference * difference
                })
                .sum::<f64>();
            let stored_distance = stored
                .iter()
                .filter(|(index, _)| !query.contains_key(index))
                .map(|(_, value)| value * value)
                .sum::<f64>();
            -(query_distance + stored_distance)
        }
        MetricType::Cosine => {
            let query_norm = query
                .values()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let stored_norm = stored
                .values()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if query_norm == 0.0 || stored_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * stored_norm)
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => dot,
    }
}

fn execute_fts(
    docs: &[Doc],
    query: &SearchQuery,
    index_params: Option<&IndexParams>,
) -> Result<Vec<ScoredDoc>> {
    let fts = query
        .fts
        .as_ref()
        .ok_or_else(|| Error::invalid_argument("FTS payload is missing"))?;
    let (expression, advanced) = match (fts.match_string.as_deref(), fts.query_string.as_deref()) {
        (Some(expression), None) => (expression, false),
        (None, Some(expression)) => (expression, true),
        (Some(_), Some(_)) => {
            return Err(Error::invalid_argument(
                "FTS query must select exactly one expression form",
            ))
        }
        (None, None) => ("", false),
    };
    if expression.trim().is_empty() {
        return Err(Error::invalid_argument("FTS query is empty"));
    }
    if advanced {
        validate_simple_fts_syntax(expression)?;
    }
    let tokenizer = index_params
        .and_then(|params| params.params.get("tokenizer_name"))
        .and_then(Value::as_str)
        .unwrap_or("standard");
    validate_tokenizer(tokenizer)?;
    let terms = tokenize(expression, tokenizer);
    if terms.is_empty() {
        return Err(Error::invalid_argument("FTS query has no searchable terms"));
    }
    let corpus: Vec<(&Doc, Vec<String>)> = docs
        .iter()
        .filter_map(|doc| {
            text_value(doc, &query.field_name).map(|text| (doc, tokenize(text, tokenizer)))
        })
        .collect();
    if corpus.is_empty() {
        return Ok(Vec::new());
    }
    let document_count = count_to_f64(corpus.len());
    let average_length = corpus
        .iter()
        .map(|(_, tokens)| count_to_f64(tokens.len()))
        .sum::<f64>()
        / document_count;
    let document_frequency = document_frequencies(&corpus, &terms);
    let mut result = Vec::new();
    for (doc, tokens) in corpus {
        if !matches_filter(doc, query.filter.as_deref()) {
            continue;
        }
        let score = bm25(
            &tokens,
            &terms,
            &document_frequency,
            document_count,
            average_length,
        );
        if score <= 0.0 {
            continue;
        }
        result.push(ScoredDoc::new(score, doc.clone())?);
    }
    Ok(result)
}

fn text_value<'a>(doc: &'a Doc, field: &str) -> Option<&'a str> {
    match doc.field(field) {
        Some(FieldValue::String(value) | FieldValue::Json(Value::String(value))) => {
            Some(value.as_str())
        }
        _ => None,
    }
}

fn tokenize(text: &str, tokenizer: &str) -> Vec<String> {
    zvec_core::engine::fts::tokenize_with(text, tokenizer)
}

fn validate_simple_fts_syntax(expression: &str) -> Result<()> {
    let has_operator = expression
        .split_whitespace()
        .any(|term| matches!(term.to_ascii_uppercase().as_str(), "AND" | "OR" | "NOT"));
    let has_query_syntax = expression.chars().any(|character| {
        !(character.is_alphanumeric() || character.is_whitespace() || character == '_')
    });
    if has_operator || has_query_syntax {
        return Err(Error::not_supported(
            "FTS query_string supports only whitespace-separated terms",
        ));
    }
    Ok(())
}

fn document_frequencies(
    corpus: &[(&Doc, Vec<String>)],
    query_terms: &[String],
) -> BTreeMap<String, usize> {
    let mut frequencies = BTreeMap::new();
    for term in query_terms {
        frequencies.entry(term.clone()).or_insert_with(|| {
            corpus
                .iter()
                .filter(|(_, tokens)| tokens.contains(term))
                .count()
        });
    }
    frequencies
}

fn bm25(
    document_tokens: &[String],
    query_terms: &[String],
    document_frequencies: &BTreeMap<String, usize>,
    document_count: f64,
    average_length: f64,
) -> f64 {
    if document_tokens.is_empty() || query_terms.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    let document_length = count_to_f64(document_tokens.len());
    let normalized_length = if average_length == 0.0 {
        0.0
    } else {
        document_length / average_length
    };
    let k1 = 1.2;
    let b = 0.75;
    for term in query_terms {
        let frequency = count_to_f64(
            document_tokens
                .iter()
                .filter(|token| *token == term)
                .count(),
        );
        if frequency == 0.0 {
            continue;
        }
        let document_frequency =
            count_to_f64(document_frequencies.get(term).copied().unwrap_or_default());
        let idf =
            ((document_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).ln();
        let denominator = frequency + k1 * (1.0 - b + b * normalized_length);
        score += idf * (frequency * (k1 + 1.0) / denominator.max(1e-12));
    }
    score.max(0.0)
}

pub(super) fn normalize_scores(docs: &mut [Doc], method: &str) -> Result<()> {
    if docs.is_empty() || method == "none" {
        return Ok(());
    }
    let values: Vec<f64> = docs.iter().map(|doc| f64::from(doc.get_score())).collect();
    match method {
        "minmax" => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator = (max - min).max(1e-12);
            for doc in docs {
                let normalized = (f64::from(doc.get_score()) - min) / denominator;
                doc.set_score(score_to_f32(normalized)?)?;
            }
        }
        "zscore" => {
            let mean = values.iter().sum::<f64>() / count_to_f64(values.len());
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / count_to_f64(values.len());
            let denominator = variance.sqrt().max(1e-12);
            for doc in docs {
                let normalized = (f64::from(doc.get_score()) - mean) / denominator;
                doc.set_score(score_to_f32(normalized)?)?;
            }
        }
        _ => {
            return Err(Error::invalid_argument(
                "unknown score normalization method",
            ));
        }
    }
    Ok(())
}
