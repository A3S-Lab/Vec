//! Exact query execution used as the collection correctness oracle.

use super::query_contract::{query_metric, validate_query_contract};
use crate::doc::{Doc, DocumentMap, VectorValue};
use crate::error::{Error, Result};
use crate::index::{CandidateSelection, IndexRegistry, OrdinalScores};
use crate::query::{FtsDefaultOperator, SearchQuery};
use crate::schema::{CollectionSchema, IndexParams};
use crate::text::{
    bm25_term_score, contains_ordered_phrase, parse_fts_query, text_value, FtsEvalContext,
    Tokenizer,
};
use crate::types::MetricType;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use zvec_core::filter::FilterExpr;

struct ScoredDoc {
    exact_score: f64,
    doc: Doc,
}

struct ScoredCandidate<'a> {
    exact_score: f64,
    doc: &'a Doc,
}

enum ResolvedQueryVector {
    Dense { values: Vec<f64>, norm: f64 },
    Binary(Vec<u8>),
    Sparse(BTreeMap<u32, f64>),
}

struct TopKCollector<'a> {
    limit: usize,
    candidates: BinaryHeap<ScoredCandidate<'a>>,
}

impl ScoredDoc {
    fn new(exact_score: f64, doc: Doc) -> Result<Self> {
        if !exact_score.is_finite() {
            return Err(Error::resource_exhausted("query score is not finite"));
        }
        Ok(Self { exact_score, doc })
    }
}

impl ScoredCandidate<'_> {
    fn new(exact_score: f64, doc: &Doc) -> Result<ScoredCandidate<'_>> {
        if !exact_score.is_finite() {
            return Err(Error::resource_exhausted("query score is not finite"));
        }
        Ok(ScoredCandidate { exact_score, doc })
    }

    fn id(&self) -> &str {
        self.doc.get_pk().unwrap_or_default()
    }
}

impl PartialEq for ScoredCandidate<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoredCandidate<'_> {}

impl PartialOrd for ScoredCandidate<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredCandidate<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap keeps the greatest element at the root, so reverse the
        // score ordering and keep the lexicographically greatest tied ID as
        // the worst retained candidate.
        other
            .exact_score
            .total_cmp(&self.exact_score)
            .then_with(|| self.id().cmp(other.id()))
    }
}

impl<'a> TopKCollector<'a> {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            candidates: BinaryHeap::new(),
        }
    }

    fn push(&mut self, exact_score: f64, doc: &'a Doc) -> Result<()> {
        let candidate = ScoredCandidate::new(exact_score, doc)?;
        if self.limit == 0 {
            return Ok(());
        }
        if self.candidates.len() < self.limit {
            self.candidates.push(candidate);
            return Ok(());
        }
        if self
            .candidates
            .peek()
            .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
        {
            self.candidates.pop();
            self.candidates.push(candidate);
        }
        Ok(())
    }

    fn into_scored_docs(self) -> Result<Vec<ScoredDoc>> {
        self.candidates
            .into_iter()
            .map(|candidate| ScoredDoc::new(candidate.exact_score, candidate.doc.clone()))
            .collect()
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

fn binary_score(query: &[u8], vector: &VectorValue) -> Option<f64> {
    let (VectorValue::Binary32(stored) | VectorValue::Binary64(stored)) = vector else {
        return None;
    };
    if stored.len() != query.len() {
        return None;
    }
    Some(
        -query
            .iter()
            .zip(stored)
            .map(|(left, right)| f64::from((left ^ right).count_ones()))
            .sum::<f64>(),
    )
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

pub(super) fn parse_optional_filter(expression: Option<&str>) -> Result<Option<FilterExpr>> {
    expression.map(parse_filter_expression).transpose()
}

pub(super) fn matches_filter(doc: &Doc, filter: Option<&FilterExpr>) -> bool {
    filter.map_or(true, |filter| filter.matches(&doc.to_core()))
}

pub(super) fn execute_query_with_candidates(
    schema: &CollectionSchema,
    docs: &DocumentMap,
    indexes: &IndexRegistry,
    query: &SearchQuery,
    candidate_ids: Option<&CandidateSelection>,
    fts_scores: Option<&OrdinalScores>,
    filter: Option<&FilterExpr>,
) -> Result<Vec<Doc>> {
    let field = validate_query_contract(schema, query)?;
    let metric = query_metric(&field, query)?;
    let topk = usize::try_from(query.topk)
        .map_err(|_| Error::invalid_argument("query topk must be positive"))?;
    let mut scored = if query.fts.is_some() {
        execute_fts(
            docs,
            query,
            field.index_params,
            candidate_ids,
            fts_scores,
            filter,
            topk,
        )?
    } else {
        execute_vector(docs, query, metric, candidate_ids, filter, topk)?
    };
    sort_scored_docs(&mut scored);
    scored.truncate(topk);
    let output_fields = query.output_fields.as_deref();
    scored
        .into_iter()
        .map(|mut scored| {
            scored.doc.set_score(score_to_f32(scored.exact_score)?)?;
            if query.include_doc_id {
                let pk = scored
                    .doc
                    .get_pk()
                    .ok_or_else(|| Error::internal("query result has no primary key"))?;
                let doc_id = indexes.document_ordinal(pk).ok_or_else(|| {
                    Error::internal(format!(
                        "query result primary key '{pk}' has no document ID"
                    ))
                })?;
                scored.doc.set_internal_id(Some(doc_id));
            } else {
                scored.doc.set_internal_id(None);
            }
            Ok(scored.doc.project(output_fields, query.include_vector))
        })
        .collect()
}

fn execute_vector(
    docs: &DocumentMap,
    query: &SearchQuery,
    metric: MetricType,
    candidate_ids: Option<&CandidateSelection>,
    filter: Option<&FilterExpr>,
    topk: usize,
) -> Result<Vec<ScoredDoc>> {
    let query_vector = resolve_query_vector(docs, query)?;
    let radius = query.params.get("radius").and_then(Value::as_f64);
    let mut result = TopKCollector::new(topk);
    if let Some(candidate_ids) = candidate_ids {
        for id in candidate_ids.ids() {
            if let Some(doc) = docs.get(id) {
                score_vector_document(
                    &mut result,
                    doc,
                    query,
                    metric,
                    &query_vector,
                    radius,
                    filter,
                )?;
            }
        }
    } else {
        for doc in docs.values() {
            score_vector_document(
                &mut result,
                doc,
                query,
                metric,
                &query_vector,
                radius,
                filter,
            )?;
        }
    }
    result.into_scored_docs()
}

fn resolve_query_vector(docs: &DocumentMap, query: &SearchQuery) -> Result<ResolvedQueryVector> {
    if let Some(vector) = &query.vector {
        let values: Vec<f64> = vector.iter().map(|value| f64::from(*value)).collect();
        return Ok(ResolvedQueryVector::Dense {
            norm: dense_query_norm(&values),
            values,
        });
    }
    if let Some(vector) = &query.binary_vector {
        return Ok(ResolvedQueryVector::Binary(vector.clone()));
    }
    if let Some(values) = &query.sparse_vector {
        return Ok(ResolvedQueryVector::Sparse(
            values
                .iter()
                .map(|(index, value)| (*index, f64::from(*value)))
                .collect(),
        ));
    }
    let id = query.id.as_deref().ok_or_else(|| {
        Error::invalid_argument(
            "query requires a dense vector, binary vector, sparse vector, or source id",
        )
    })?;
    let source = docs
        .get(id)
        .ok_or_else(|| Error::not_found(format!("source document '{id}' not found")))?;
    let vector = source.vector(&query.field_name).ok_or_else(|| {
        Error::failed_precondition(format!(
            "source document '{id}' has no vector in field '{}'",
            query.field_name
        ))
    })?;
    if let Some(vector) = vector.to_dense_f64() {
        return Ok(ResolvedQueryVector::Dense {
            norm: dense_query_norm(&vector),
            values: vector,
        });
    }
    if let Some(vector) = vector.to_sparse_f64() {
        return Ok(ResolvedQueryVector::Sparse(vector));
    }
    if let VectorValue::Binary32(values) | VectorValue::Binary64(values) = vector {
        return Ok(ResolvedQueryVector::Binary(values.clone()));
    }
    Err(Error::failed_precondition(format!(
        "source document '{id}' has no searchable vector in field '{}'",
        query.field_name
    )))
}

#[allow(clippy::too_many_arguments)]
fn score_vector_document<'a>(
    result: &mut TopKCollector<'a>,
    doc: &'a Doc,
    query: &SearchQuery,
    metric: MetricType,
    query_vector: &ResolvedQueryVector,
    radius: Option<f64>,
    filter: Option<&FilterExpr>,
) -> Result<()> {
    if !matches_filter(doc, filter) {
        return Ok(());
    }
    let Some(vector) = doc.vector(&query.field_name) else {
        return Ok(());
    };
    let score = match query_vector {
        ResolvedQueryVector::Dense { values, norm } => {
            let Some(score) = vector.dense_score(values, *norm, metric) else {
                return Ok(());
            };
            score
        }
        ResolvedQueryVector::Binary(query) => {
            let Some(score) = binary_score(query, vector) else {
                return Ok(());
            };
            score
        }
        ResolvedQueryVector::Sparse(query) => {
            let Some(stored) = vector.to_sparse_f64() else {
                return Ok(());
            };
            sparse_score(query, &stored, metric)
        }
    };
    if radius.is_some_and(|radius| {
        if metric == MetricType::L2 {
            score < -radius * radius
        } else {
            score < radius
        }
    }) {
        return Ok(());
    }
    result.push(score, doc)
}

fn dense_query_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
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
    docs: &DocumentMap,
    query: &SearchQuery,
    index_params: Option<&IndexParams>,
    candidate_ids: Option<&CandidateSelection>,
    indexed_scores: Option<&OrdinalScores>,
    filter: Option<&FilterExpr>,
    topk: usize,
) -> Result<Vec<ScoredDoc>> {
    if let Some(indexed_scores) = indexed_scores {
        let mut result = TopKCollector::new(topk);
        for (id, score) in indexed_scores.entries() {
            let Some(doc) = docs.get(id) else {
                continue;
            };
            if matches_filter(doc, filter) {
                result.push(score, doc)?;
            }
        }
        return result.into_scored_docs();
    }
    let tokenizer = Tokenizer::from_index_params(index_params)?;
    let mut parsed = parse_fts_query(query, &tokenizer)?;
    let corpus: Vec<(&Doc, Vec<String>)> = docs
        .values()
        .filter_map(|doc| {
            text_value(doc, &query.field_name).map(|text| (doc.as_ref(), tokenizer.tokenize(text)))
        })
        .collect();
    if corpus.is_empty() {
        return Ok(Vec::new());
    }
    parsed.expand_terms(
        corpus
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().map(String::as_str)),
    );
    let document_count = count_to_f64(corpus.len());
    let average_length = corpus
        .iter()
        .map(|(_, tokens)| count_to_f64(tokens.len()))
        .sum::<f64>()
        / document_count;
    let document_frequency = document_frequencies(&corpus, parsed.all_terms());
    let mut result = TopKCollector::new(topk);
    for (doc, tokens) in corpus {
        if candidate_ids.is_some_and(|ids| doc.get_pk().map_or(true, |id| !ids.contains(id))) {
            continue;
        }
        if !matches_filter(doc, filter) {
            continue;
        }
        let score = if let Some((terms, operator)) = parsed.simple() {
            if operator == FtsDefaultOperator::And
                && terms.iter().any(|term| !tokens.contains(term))
            {
                continue;
            }
            bm25(
                &tokens,
                terms,
                &document_frequency,
                document_count,
                average_length,
            )
        } else {
            let mut context = ScanFtsEvalContext {
                tokens: &tokens,
                document_frequency: &document_frequency,
                document_count,
                average_length,
            };
            let Some(score) = parsed.score(&mut context) else {
                continue;
            };
            score
        };
        if score <= 0.0 {
            continue;
        }
        result.push(score, doc)?;
    }
    result.into_scored_docs()
}

struct ScanFtsEvalContext<'a> {
    tokens: &'a [String],
    document_frequency: &'a BTreeMap<String, usize>,
    document_count: f64,
    average_length: f64,
}

impl FtsEvalContext for ScanFtsEvalContext<'_> {
    fn contains_term(&mut self, term: &str) -> bool {
        self.tokens.iter().any(|token| token == term)
    }

    fn contains_phrase(&mut self, terms: &[String], slop: u32) -> bool {
        contains_ordered_phrase(self.tokens, terms, slop)
    }

    fn term_score(&mut self, term: &str) -> f64 {
        let frequency = count_to_f64(self.tokens.iter().filter(|token| *token == term).count());
        if frequency == 0.0 {
            return 0.0;
        }
        let document_frequency = self
            .document_frequency
            .get(term)
            .copied()
            .map_or(0.0, count_to_f64);
        bm25_term_score(
            frequency,
            document_frequency,
            self.document_count,
            count_to_f64(self.tokens.len()),
            self.average_length,
        )
    }
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
        score += bm25_term_score(
            frequency,
            document_frequency,
            document_count,
            document_length,
            average_length,
        );
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

#[cfg(test)]
mod tests {
    use super::{sort_scored_docs, TopKCollector};
    use crate::doc::Doc;

    #[test]
    fn bounded_topk_keeps_exact_scores_and_primary_key_ties() {
        let docs: Vec<Doc> = ["z-low", "b-tie", "a-tie", "best", "c-tie"]
            .into_iter()
            .map(|id| Doc::with_pk(id).expect("document must be valid"))
            .collect();
        let mut collector = TopKCollector::new(3);
        for (doc, score) in docs.iter().zip([1.0, 2.0, 2.0, 3.0, 2.0]) {
            collector
                .push(score, doc)
                .expect("finite score must be accepted");
        }
        assert!(collector.push(f64::NAN, &docs[0]).is_err());

        let mut ranked = collector
            .into_scored_docs()
            .expect("retained scores must be valid");
        sort_scored_docs(&mut ranked);
        assert_eq!(
            ranked
                .iter()
                .map(|scored| scored.doc.get_pk().unwrap_or_default())
                .collect::<Vec<_>>(),
            vec!["best", "a-tie", "b-tie"]
        );
    }
}
