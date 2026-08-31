//! Immutable, revisioned full-text postings with exact BM25 statistics.

mod document_lengths;
mod posting_list;
mod query_context;
mod term_dictionary;

#[cfg(test)]
mod tests;

use super::ordinals::{OrdinalSet, OrdinalTable};
use crate::doc::DocumentMap;
use crate::error::{Error, Result};
use crate::query::{FtsDefaultOperator, SearchQuery};
use crate::schema::{CollectionSchema, FieldSchema, IndexParams};
use crate::stats::IndexStat;
use crate::text::{
    bm25_term_score, parse_fts_query, text_value, FtsExpr, FtsExprKind, FtsModifier,
    ParsedFtsQuery, Tokenizer,
};
use crate::types::IndexType;
use document_lengths::DocumentLengths;
use posting_list::PostingList;
use query_context::IndexedEvalContext;
use roaring::RoaringTreemap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use term_dictionary::TermDictionary;

const DENSE_SCORE_MIN_VISITS: usize = 4_096;
const DENSE_SCORE_MAX_SPAN_FACTOR: usize = 8;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct FtsIndexRegistry {
    source_revision: u64,
    indexes: BTreeMap<String, FtsIndex>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FtsIndex {
    #[serde(with = "super::cache::index_params_serde")]
    params: IndexParams,
    tokenizer: Tokenizer,
    postings: TermDictionary,
    document_lengths: DocumentLengths,
    total_tokens: u64,
}

/// Co-locates both BM25 inputs so a posting hit does not need a second tree
/// lookup. A layout test guards the unchanged aligned B-tree entry footprint.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct PostingEntry {
    frequency: u32,
    document_length: u32,
}

impl FtsIndexRegistry {
    pub(super) fn build(
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        let configured: Vec<_> = schema
            .fields
            .iter()
            .filter_map(|field| {
                field
                    .index_params
                    .as_ref()
                    .filter(|params| params.index_type == IndexType::Fts)
                    .map(|params| (field, params))
            })
            .collect();
        if configured.is_empty() {
            return Ok(Self {
                source_revision,
                ..Self::default()
            });
        }

        let mut indexes = BTreeMap::new();
        for (field, params) in configured {
            indexes.insert(
                field.name.clone(),
                FtsIndex::build(&field.name, params, docs, ordinals)?,
            );
        }
        Ok(Self {
            source_revision,
            indexes,
        })
    }

    pub(super) fn rebuild_field(
        &self,
        field: &FieldSchema,
        docs: &DocumentMap,
        source_revision: u64,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        let params = field
            .index_params
            .as_ref()
            .filter(|params| params.index_type == IndexType::Fts)
            .ok_or_else(|| Error::internal("FTS rebuild requires an FTS index field"))?;
        let mut next = self.clone();
        next.source_revision = source_revision;
        next.indexes.insert(
            field.name.clone(),
            FtsIndex::build(&field.name, params, docs, ordinals)?,
        );
        Ok(next)
    }

    pub(super) fn apply_document_changes(
        &self,
        schema: &CollectionSchema,
        previous_docs: &DocumentMap,
        docs: &DocumentMap,
        source_revision: u64,
        changed_ids: &BTreeSet<String>,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        if !self.matches_schema(schema) {
            return Self::build(schema, docs, source_revision, ordinals);
        }
        if previous_docs.is_empty() && !docs.is_empty() && changed_ids.len() == docs.len() {
            return Self::build(schema, docs, source_revision, ordinals);
        }
        if self.indexes.is_empty() {
            return Ok(Self {
                source_revision,
                ..Self::default()
            });
        }

        let mut indexes = BTreeMap::new();
        for field in &schema.fields {
            let Some(params) = field
                .index_params
                .as_ref()
                .filter(|params| params.index_type == IndexType::Fts)
            else {
                continue;
            };
            let Some(current) = self
                .indexes
                .get(&field.name)
                .filter(|index| index.params == *params)
            else {
                return Self::build(schema, docs, source_revision, ordinals);
            };
            let mut next = current.clone();
            for id in changed_ids {
                let previous = previous_docs
                    .get(id)
                    .and_then(|doc| text_value(doc, &field.name));
                let current = docs.get(id).and_then(|doc| text_value(doc, &field.name));
                if previous == current {
                    continue;
                }
                let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                    Error::internal(format!("FTS ordinal is missing for document '{id}'"))
                })?;
                if let Some(text) = previous {
                    next.remove_text(text, ordinal)?;
                }
                if let Some(text) = current {
                    next.insert_text(text, ordinal)?;
                }
            }
            next.document_lengths.finish_changes()?;
            next.postings.finish_changes();
            indexes.insert(field.name.clone(), next);
        }
        Ok(Self {
            source_revision,
            indexes,
        })
    }

    pub(super) fn search(
        &self,
        source_revision: u64,
        query: &SearchQuery,
        candidates: Option<&OrdinalSet>,
        docs: &DocumentMap,
        ordinals: &OrdinalTable,
    ) -> Result<Option<Vec<(u64, f64)>>> {
        if self.source_revision != source_revision {
            return Ok(None);
        }
        let Some(index) = self.indexes.get(&query.field_name) else {
            return Ok(None);
        };
        let parsed = parse_fts_query(query, &index.tokenizer)?;
        let allowed = candidates.map(OrdinalSet::bitmap);
        let scores = if let Some((terms, operator)) = parsed.simple() {
            index.search(terms, allowed, operator)?
        } else {
            if allowed.is_none() && index.prefers_scan_for_expression(&parsed) {
                return Ok(None);
            }
            index.search_expression(&parsed, allowed, docs, ordinals, &query.field_name)?
        };
        Ok(Some(scores))
    }

    pub(super) fn stats(&self) -> Vec<IndexStat> {
        self.indexes
            .iter()
            .map(|(name, index)| IndexStat {
                name: name.clone(),
                index_type: IndexType::Fts,
                completeness: 1.0,
                source_revision: self.source_revision,
                document_count: u64::try_from(index.document_lengths.len()).unwrap_or(u64::MAX),
                estimated_payload_bytes: None,
                state: "ready".into(),
            })
            .collect()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    pub(super) fn validates(
        &self,
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
        ordinals: &OrdinalTable,
    ) -> bool {
        self.source_revision == source_revision
            && self.matches_schema(schema)
            && self
                .indexes
                .iter()
                .all(|(field_name, index)| index.validates(field_name, docs, ordinals))
    }

    fn matches_schema(&self, schema: &CollectionSchema) -> bool {
        let configured: Vec<_> = schema
            .fields
            .iter()
            .filter_map(|field| {
                field
                    .index_params
                    .as_ref()
                    .filter(|params| params.index_type == IndexType::Fts)
                    .map(|params| (&field.name, params))
            })
            .collect();
        configured.len() == self.indexes.len()
            && configured.into_iter().all(|(name, params)| {
                self.indexes
                    .get(name)
                    .is_some_and(|index| index.params == *params)
            })
    }
}

impl FtsIndex {
    fn build(
        field_name: &str,
        params: &IndexParams,
        docs: &DocumentMap,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_index_params(Some(params))?;
        let mut postings = BTreeMap::<String, BTreeMap<u64, PostingEntry>>::new();
        let mut document_lengths = BTreeMap::<u64, u32>::new();
        let mut total_tokens = 0_u64;
        for (id, doc) in docs {
            let Some(text) = text_value(doc, field_name) else {
                continue;
            };
            let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                Error::internal(format!("FTS ordinal is missing for document '{id}'"))
            })?;
            let tokens = tokenizer.tokenize(text);
            let length = u32::try_from(tokens.len())
                .map_err(|_| Error::resource_exhausted("FTS document has too many tokens"))?;
            if document_lengths.insert(ordinal, length).is_some() {
                return Err(Error::internal(format!(
                    "FTS document ordinal {ordinal} is already indexed"
                )));
            }
            total_tokens = total_tokens
                .checked_add(u64::from(length))
                .ok_or_else(|| Error::resource_exhausted("FTS token count overflow"))?;
            let mut frequencies = BTreeMap::<String, u32>::new();
            for token in tokens {
                let frequency = frequencies.entry(token).or_default();
                *frequency = frequency
                    .checked_add(1)
                    .ok_or_else(|| Error::resource_exhausted("FTS term frequency overflow"))?;
            }
            for (term, frequency) in frequencies {
                postings.entry(term).or_default().insert(
                    ordinal,
                    PostingEntry {
                        frequency,
                        document_length: length,
                    },
                );
            }
        }
        Ok(Self {
            params: params.clone(),
            tokenizer,
            postings: TermDictionary::from_sorted_entries(postings.into_iter().map(
                |(term, posting)| (term, Arc::new(PostingList::from_sorted_entries(posting))),
            )),
            document_lengths: DocumentLengths::from_sorted_entries(document_lengths)?,
            total_tokens,
        })
    }

    fn insert_text(&mut self, text: &str, ordinal: u64) -> Result<()> {
        let tokens = self.tokenizer.tokenize(text);
        let length = u32::try_from(tokens.len())
            .map_err(|_| Error::resource_exhausted("FTS document has too many tokens"))?;
        if self.document_lengths.contains_key(ordinal) {
            return Err(Error::internal(format!(
                "FTS document ordinal {ordinal} is already indexed"
            )));
        }
        self.total_tokens = self
            .total_tokens
            .checked_add(u64::from(length))
            .ok_or_else(|| Error::resource_exhausted("FTS token count overflow"))?;
        self.document_lengths.insert(ordinal, length)?;

        let mut frequencies = BTreeMap::<String, u32>::new();
        for token in tokens {
            let frequency = frequencies.entry(token).or_default();
            *frequency = frequency
                .checked_add(1)
                .ok_or_else(|| Error::resource_exhausted("FTS term frequency overflow"))?;
        }
        for (term, frequency) in frequencies {
            let entry = PostingEntry {
                frequency,
                document_length: length,
            };
            if let Some(posting) = self.postings.get(&term).cloned() {
                let mut posting = posting;
                Arc::make_mut(&mut posting).insert(ordinal, entry)?;
                self.postings.insert(term, posting)?;
            } else {
                self.postings
                    .insert(term, Arc::new(PostingList::single(ordinal, entry)))?;
            }
        }
        Ok(())
    }

    fn remove_text(&mut self, text: &str, ordinal: u64) -> Result<()> {
        let length = self.document_lengths.remove(ordinal).ok_or_else(|| {
            Error::internal(format!("FTS document ordinal {ordinal} is not indexed"))
        })?;
        self.total_tokens = self
            .total_tokens
            .checked_sub(u64::from(length))
            .ok_or_else(|| Error::internal("FTS token count underflow"))?;

        let terms: BTreeSet<String> = self.tokenizer.tokenize(text).into_iter().collect();
        for term in terms {
            if let Some(posting) = self.postings.get(&term).cloned() {
                let mut posting = posting;
                Arc::make_mut(&mut posting).remove(ordinal);
                if posting.is_empty() {
                    self.postings.remove(&term);
                } else {
                    self.postings.insert(term, posting)?;
                }
            }
        }
        Ok(())
    }

    fn validates(&self, field_name: &str, docs: &DocumentMap, ordinals: &OrdinalTable) -> bool {
        if Tokenizer::from_index_params(Some(&self.params)).as_ref() != Ok(&self.tokenizer)
            || self
                .document_lengths
                .keys()
                .any(|ordinal| !ordinals.live().contains(ordinal))
        {
            return false;
        }
        let expected_documents = docs
            .iter()
            .filter(|(_, doc)| text_value(doc, field_name).is_some())
            .count();
        if self.document_lengths.len() != expected_documents
            || docs.iter().any(|(id, doc)| {
                let Some(ordinal) = ordinals.ordinal(id) else {
                    return true;
                };
                self.document_lengths.contains_key(ordinal) != text_value(doc, field_name).is_some()
            })
        {
            return false;
        }
        let total_tokens = self
            .document_lengths
            .values()
            .try_fold(0_u64, |total, length| total.checked_add(u64::from(length)));
        total_tokens == Some(self.total_tokens)
            && self
                .document_lengths
                .validates(ordinals.allocated_len(), ordinals.live())
            && self.postings.validates()
            && self.postings.iter().all(|(_, posting)| {
                !posting.is_empty() && posting.validates(&self.document_lengths)
            })
    }

    fn search(
        &self,
        terms: &[String],
        allowed: Option<&RoaringTreemap>,
        operator: FtsDefaultOperator,
    ) -> Result<Vec<(u64, f64)>> {
        if self.document_lengths.is_empty() || allowed.is_some_and(RoaringTreemap::is_empty) {
            return Ok(Vec::new());
        }
        let document_count = count_to_f64(
            u64::try_from(self.document_lengths.len())
                .map_err(|_| Error::resource_exhausted("FTS document count exceeds u64"))?,
        );
        let average_length = count_to_f64(self.total_tokens) / document_count;
        if let [term] = terms {
            return self.search_single(term, allowed, document_count, average_length);
        }
        if operator == FtsDefaultOperator::And {
            return self.search_conjunctive(terms, allowed, document_count, average_length);
        }
        let posting_visits = terms.iter().fold(0_usize, |visits, term| {
            visits.saturating_add(self.postings.get(term).map_or(0, |posting| posting.len()))
        });
        let allowed_visits = allowed.map_or(posting_visits, |allowed| {
            usize::try_from(allowed.len())
                .unwrap_or(usize::MAX)
                .saturating_mul(terms.len())
                .min(posting_visits)
        });
        if allowed_visits >= DENSE_SCORE_MIN_VISITS {
            let ordinal_span = self
                .document_lengths
                .keys()
                .next_back()
                .and_then(|ordinal| usize::try_from(ordinal).ok())
                .and_then(|ordinal| ordinal.checked_add(1));
            if let Some(ordinal_span) = ordinal_span
                .filter(|ordinal_span| use_dense_score_scratch(allowed_visits, *ordinal_span))
            {
                return self.search_dense(
                    terms,
                    allowed,
                    document_count,
                    average_length,
                    allowed_visits,
                    ordinal_span,
                );
            }
        }
        self.search_sparse(terms, allowed, document_count, average_length)
    }

    fn search_expression(
        &self,
        query: &ParsedFtsQuery,
        allowed: Option<&RoaringTreemap>,
        docs: &DocumentMap,
        ordinals: &OrdinalTable,
        field_name: &str,
    ) -> Result<Vec<(u64, f64)>> {
        if self.document_lengths.is_empty() || allowed.is_some_and(RoaringTreemap::is_empty) {
            return Ok(Vec::new());
        }
        let mut candidates = self.expression_candidates(&query.root)?;
        if let Some(allowed) = allowed {
            candidates &= allowed;
        }
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let document_count = count_to_f64(
            u64::try_from(self.document_lengths.len())
                .map_err(|_| Error::resource_exhausted("FTS document count exceeds u64"))?,
        );
        let average_length = count_to_f64(self.total_tokens) / document_count;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(usize::try_from(candidates.len()).unwrap_or(usize::MAX))
            .map_err(|_| {
                Error::resource_exhausted("FTS score result exceeds addressable memory")
            })?;
        for ordinal in candidates {
            let id = ordinals.id(ordinal).ok_or_else(|| {
                Error::internal(format!(
                    "FTS candidate ordinal {ordinal} has no primary key"
                ))
            })?;
            let text = docs
                .get(id)
                .and_then(|doc| text_value(doc, field_name))
                .ok_or_else(|| {
                    Error::internal(format!("FTS candidate document '{id}' has no indexed text"))
                })?;
            let mut context =
                IndexedEvalContext::new(self, ordinal, text, document_count, average_length);
            if let Some(score) = query.score(&mut context).filter(|score| *score > 0.0) {
                scores.push((ordinal, score));
            }
        }
        Ok(scores)
    }

    fn expression_candidates(&self, expression: &FtsExpr) -> Result<RoaringTreemap> {
        match &expression.kind {
            FtsExprKind::Empty => Ok(RoaringTreemap::new()),
            FtsExprKind::Term(term) => Ok(self.term_candidates(term)),
            FtsExprKind::Phrase(terms) => Ok(self.phrase_candidates(terms)),
            FtsExprKind::And(children) => self.and_candidates(children),
            FtsExprKind::Or(children) => self.or_candidates(children),
        }
    }

    fn prefers_scan_for_expression(&self, query: &ParsedFtsQuery) -> bool {
        let documents = self.document_lengths.len();
        let threshold = if query.has_phrase() {
            documents.saturating_add(1) / 2
        } else {
            documents.saturating_sub(documents / 4)
        };
        self.estimated_candidates(&query.root) >= threshold
    }

    fn term_candidates(&self, term: &str) -> RoaringTreemap {
        self.postings
            .get(term)
            .map(|posting| posting.iter().map(|(ordinal, _)| ordinal).collect())
            .unwrap_or_default()
    }

    fn phrase_candidates(&self, terms: &[String]) -> RoaringTreemap {
        let mut postings = Vec::with_capacity(terms.len());
        for term in terms.iter().collect::<BTreeSet<_>>() {
            let Some(posting) = self.postings.get(term) else {
                return RoaringTreemap::new();
            };
            postings.push(posting);
        }
        postings.sort_unstable_by_key(|posting| posting.len());
        let Some(first) = postings.first() else {
            return RoaringTreemap::new();
        };
        first
            .iter()
            .map(|(ordinal, _)| ordinal)
            .filter(|ordinal| {
                postings
                    .iter()
                    .skip(1)
                    .all(|posting| posting.get(*ordinal).is_some())
            })
            .collect()
    }

    fn and_candidates(&self, children: &[FtsExpr]) -> Result<RoaringTreemap> {
        let mut positive: Vec<_> = children
            .iter()
            .filter(|child| child.modifier != FtsModifier::MustNot)
            .collect();
        let negative: Vec<_> = children
            .iter()
            .filter(|child| child.modifier == FtsModifier::MustNot)
            .collect();
        if positive.is_empty() {
            return Ok(RoaringTreemap::new());
        }
        positive.sort_unstable_by_key(|child| self.estimated_candidates(child));
        let mut candidates = self.expression_candidates(positive.remove(0))?;
        for required in positive {
            candidates = candidates
                .into_iter()
                .filter(|ordinal| self.candidate_matches(required, *ordinal))
                .collect();
            if candidates.is_empty() {
                return Ok(candidates);
            }
        }
        for prohibited in negative {
            candidates = candidates
                .into_iter()
                .filter(|ordinal| !self.candidate_matches(prohibited, *ordinal))
                .collect();
        }
        Ok(candidates)
    }

    fn or_candidates(&self, children: &[FtsExpr]) -> Result<RoaringTreemap> {
        let mut required: Vec<_> = children
            .iter()
            .filter(|child| child.modifier == FtsModifier::Must)
            .collect();
        let optional: Vec<_> = children
            .iter()
            .filter(|child| child.modifier == FtsModifier::None)
            .collect();
        let negative: Vec<_> = children
            .iter()
            .filter(|child| child.modifier == FtsModifier::MustNot)
            .collect();
        let mut candidates = if required.is_empty() {
            let mut union = RoaringTreemap::new();
            for candidate in optional {
                union |= self.expression_candidates(candidate)?;
            }
            union
        } else {
            required.sort_unstable_by_key(|child| self.estimated_candidates(child));
            let mut intersection = self.expression_candidates(required.remove(0))?;
            for candidate in required {
                intersection = intersection
                    .into_iter()
                    .filter(|ordinal| self.candidate_matches(candidate, *ordinal))
                    .collect();
                if intersection.is_empty() {
                    break;
                }
            }
            intersection
        };
        for prohibited in negative {
            candidates = candidates
                .into_iter()
                .filter(|ordinal| !self.candidate_matches(prohibited, *ordinal))
                .collect();
        }
        Ok(candidates)
    }

    fn estimated_candidates(&self, expression: &FtsExpr) -> usize {
        match &expression.kind {
            FtsExprKind::Empty => 0,
            FtsExprKind::Term(term) => self.postings.get(term).map_or(0, |posting| posting.len()),
            FtsExprKind::Phrase(terms) => terms
                .iter()
                .map(|term| self.postings.get(term).map_or(0, |posting| posting.len()))
                .min()
                .unwrap_or(0),
            FtsExprKind::And(children) => children
                .iter()
                .filter(|child| child.modifier != FtsModifier::MustNot)
                .map(|child| self.estimated_candidates(child))
                .min()
                .unwrap_or(0),
            FtsExprKind::Or(children) => {
                let required: Vec<_> = children
                    .iter()
                    .filter(|child| child.modifier == FtsModifier::Must)
                    .collect();
                if required.is_empty() {
                    children
                        .iter()
                        .filter(|child| child.modifier == FtsModifier::None)
                        .fold(0_usize, |total, child| {
                            total.saturating_add(self.estimated_candidates(child))
                        })
                } else {
                    required
                        .into_iter()
                        .map(|child| self.estimated_candidates(child))
                        .min()
                        .unwrap_or(0)
                }
            }
        }
    }

    fn candidate_matches(&self, expression: &FtsExpr, ordinal: u64) -> bool {
        match &expression.kind {
            FtsExprKind::Empty => false,
            FtsExprKind::Term(term) => self
                .postings
                .get(term)
                .and_then(|posting| posting.get(ordinal))
                .is_some(),
            FtsExprKind::Phrase(terms) => {
                !terms.is_empty()
                    && terms.iter().all(|term| {
                        self.postings
                            .get(term)
                            .and_then(|posting| posting.get(ordinal))
                            .is_some()
                    })
            }
            FtsExprKind::And(children) => {
                let mut has_positive = false;
                for child in children {
                    let matched = self.candidate_matches(child, ordinal);
                    if child.modifier == FtsModifier::MustNot {
                        if matched {
                            return false;
                        }
                    } else {
                        has_positive = true;
                        if !matched {
                            return false;
                        }
                    }
                }
                has_positive
            }
            FtsExprKind::Or(children) => {
                if children.iter().any(|child| {
                    child.modifier == FtsModifier::MustNot && self.candidate_matches(child, ordinal)
                }) {
                    return false;
                }
                let required: Vec<_> = children
                    .iter()
                    .filter(|child| child.modifier == FtsModifier::Must)
                    .collect();
                if required.is_empty() {
                    children.iter().any(|child| {
                        child.modifier == FtsModifier::None
                            && self.candidate_matches(child, ordinal)
                    })
                } else {
                    required
                        .into_iter()
                        .all(|child| self.candidate_matches(child, ordinal))
                }
            }
        }
    }

    fn search_conjunctive(
        &self,
        terms: &[String],
        allowed: Option<&RoaringTreemap>,
        document_count: f64,
        average_length: f64,
    ) -> Result<Vec<(u64, f64)>> {
        let unique_terms: BTreeSet<&str> = terms.iter().map(String::as_str).collect();
        let mut required = Vec::with_capacity(unique_terms.len());
        for term in unique_terms {
            let Some(posting) = self.postings.get(term) else {
                return Ok(Vec::new());
            };
            required.push((term, posting));
        }
        required.sort_unstable_by(|left, right| {
            left.1
                .len()
                .cmp(&right.1.len())
                .then_with(|| left.0.cmp(right.0))
        });
        let Some((_, driver)) = required.first() else {
            return Ok(Vec::new());
        };
        let scoring: Vec<_> = terms
            .iter()
            .map(|term| {
                let posting = self.postings.get(term).ok_or_else(|| {
                    Error::internal(format!("FTS posting is missing for required term '{term}'"))
                })?;
                let document_frequency = count_to_f64(
                    u64::try_from(posting.len())
                        .map_err(|_| Error::resource_exhausted("FTS posting count exceeds u64"))?,
                );
                Ok((posting, document_frequency))
            })
            .collect::<Result<_>>()?;
        let capacity = allowed.map_or(driver.len(), |allowed| {
            driver
                .len()
                .min(usize::try_from(allowed.len()).unwrap_or(usize::MAX))
        });
        let mut scores = Vec::new();
        scores.try_reserve_exact(capacity).map_err(|_| {
            Error::resource_exhausted("FTS score result exceeds addressable memory")
        })?;
        for (ordinal, _) in driver.iter() {
            if allowed.is_some_and(|allowed| !allowed.contains(ordinal))
                || required
                    .iter()
                    .skip(1)
                    .any(|(_, posting)| posting.get(ordinal).is_none())
            {
                continue;
            }
            let mut score = 0.0_f64;
            for (posting, document_frequency) in &scoring {
                let entry = posting.get(ordinal).ok_or_else(|| {
                    Error::internal(format!(
                        "FTS posting is missing required document ordinal {ordinal}"
                    ))
                })?;
                score += bm25_term_score(
                    f64::from(entry.frequency),
                    *document_frequency,
                    document_count,
                    f64::from(entry.document_length),
                    average_length,
                );
            }
            if score > 0.0 {
                scores.push((ordinal, score));
            }
        }
        Ok(scores)
    }

    fn search_single(
        &self,
        term: &str,
        allowed: Option<&RoaringTreemap>,
        document_count: f64,
        average_length: f64,
    ) -> Result<Vec<(u64, f64)>> {
        let Some(posting) = self.postings.get(term) else {
            return Ok(Vec::new());
        };
        let document_frequency = count_to_f64(
            u64::try_from(posting.len())
                .map_err(|_| Error::resource_exhausted("FTS posting count exceeds u64"))?,
        );
        let capacity = allowed.map_or(posting.len(), |allowed| {
            posting
                .len()
                .min(usize::try_from(allowed.len()).unwrap_or(usize::MAX))
        });
        let mut scores = Vec::new();
        scores.try_reserve_exact(capacity).map_err(|_| {
            Error::resource_exhausted("FTS score result exceeds addressable memory")
        })?;
        for (ordinal, entry) in posting.iter() {
            if allowed.is_some_and(|allowed| !allowed.contains(ordinal)) {
                continue;
            }
            let score = bm25_term_score(
                f64::from(entry.frequency),
                document_frequency,
                document_count,
                f64::from(entry.document_length),
                average_length,
            );
            if score > 0.0 {
                scores.push((ordinal, score));
            }
        }
        Ok(scores)
    }

    fn search_sparse(
        &self,
        terms: &[String],
        allowed: Option<&RoaringTreemap>,
        document_count: f64,
        average_length: f64,
    ) -> Result<Vec<(u64, f64)>> {
        let mut scores = BTreeMap::<u64, f64>::new();
        for term in terms {
            let Some(posting) = self.postings.get(term) else {
                continue;
            };
            let document_frequency = count_to_f64(
                u64::try_from(posting.len())
                    .map_err(|_| Error::resource_exhausted("FTS posting count exceeds u64"))?,
            );
            for (ordinal, entry) in posting.iter() {
                if allowed.is_some_and(|allowed| !allowed.contains(ordinal)) {
                    continue;
                }
                let contribution = bm25_term_score(
                    f64::from(entry.frequency),
                    document_frequency,
                    document_count,
                    f64::from(entry.document_length),
                    average_length,
                );
                *scores.entry(ordinal).or_default() += contribution;
            }
        }
        Ok(scores
            .into_iter()
            .filter(|(_, score)| *score > 0.0)
            .collect())
    }

    fn search_dense(
        &self,
        terms: &[String],
        allowed: Option<&RoaringTreemap>,
        document_count: f64,
        average_length: f64,
        estimated_visits: usize,
        ordinal_span: usize,
    ) -> Result<Vec<(u64, f64)>> {
        let mut scores = Vec::new();
        scores.try_reserve_exact(ordinal_span).map_err(|_| {
            Error::resource_exhausted("FTS direct score scratch exceeds addressable memory")
        })?;
        scores.resize(ordinal_span, 0.0_f64);
        let mut touched = Vec::new();
        touched
            .try_reserve(estimated_visits.min(self.document_lengths.len()))
            .map_err(|_| {
                Error::resource_exhausted("FTS touched-ordinal scratch exceeds addressable memory")
            })?;
        for term in terms {
            let Some(posting) = self.postings.get(term) else {
                continue;
            };
            let document_frequency = count_to_f64(
                u64::try_from(posting.len())
                    .map_err(|_| Error::resource_exhausted("FTS posting count exceeds u64"))?,
            );
            for (ordinal, entry) in posting.iter() {
                if allowed.is_some_and(|allowed| !allowed.contains(ordinal)) {
                    continue;
                }
                let slot = usize::try_from(ordinal)
                    .ok()
                    .and_then(|ordinal| scores.get_mut(ordinal))
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "FTS score slot is missing for document ordinal {ordinal}"
                        ))
                    })?;
                let contribution = bm25_term_score(
                    f64::from(entry.frequency),
                    document_frequency,
                    document_count,
                    f64::from(entry.document_length),
                    average_length,
                );
                if *slot == 0.0 {
                    touched.push(ordinal);
                }
                *slot += contribution;
            }
        }
        touched.sort_unstable();
        let mut output = Vec::new();
        output.try_reserve_exact(touched.len()).map_err(|_| {
            Error::resource_exhausted("FTS score result exceeds addressable memory")
        })?;
        for ordinal in touched {
            let score = usize::try_from(ordinal)
                .ok()
                .and_then(|ordinal| scores.get(ordinal).copied())
                .ok_or_else(|| {
                    Error::internal(format!(
                        "FTS score result is missing for document ordinal {ordinal}"
                    ))
                })?;
            if score > 0.0 {
                output.push((ordinal, score));
            }
        }
        Ok(output)
    }
}

fn use_dense_score_scratch(estimated_visits: usize, ordinal_span: usize) -> bool {
    estimated_visits >= DENSE_SCORE_MIN_VISITS
        && ordinal_span <= estimated_visits.saturating_mul(DENSE_SCORE_MAX_SPAN_FACTOR)
}

/// Storage limits keep document/token counts below f64's exact integer range.
#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: u64) -> f64 {
    value as f64
}
