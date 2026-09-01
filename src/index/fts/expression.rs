//! Candidate planning for structured full-text expressions.

use super::FtsIndex;
use crate::error::{Error, Result};
use crate::text::{FtsExpr, FtsExprKind, FtsModifier, ParsedFtsQuery};
use roaring::RoaringTreemap;
use std::collections::BTreeSet;

impl FtsIndex {
    pub(super) fn expression_candidates(&self, expression: &FtsExpr) -> Result<RoaringTreemap> {
        match &expression.kind {
            FtsExprKind::Empty => Ok(RoaringTreemap::new()),
            FtsExprKind::MatchAll => Ok(self.document_lengths.keys().collect()),
            FtsExprKind::Term(term) => Ok(self.term_candidates(term)),
            FtsExprKind::ExpandedTerms(terms) => {
                let mut candidates = RoaringTreemap::new();
                for term in terms {
                    candidates |= self.term_candidates(term);
                }
                Ok(candidates)
            }
            FtsExprKind::TermMatcher(_) => Err(Error::internal(
                "FTS term matcher reached the index before vocabulary expansion",
            )),
            FtsExprKind::Phrase { terms, .. } => Ok(self.phrase_candidates(terms)),
            FtsExprKind::And(children) => self.and_candidates(children),
            FtsExprKind::Or(children) => self.or_candidates(children),
        }
    }

    pub(super) fn prefers_scan_for_expression(&self, query: &ParsedFtsQuery) -> bool {
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
            FtsExprKind::MatchAll | FtsExprKind::TermMatcher(_) => self.document_lengths.len(),
            FtsExprKind::Term(term) => self.postings.get(term).map_or(0, |posting| posting.len()),
            FtsExprKind::ExpandedTerms(terms) => terms
                .iter()
                .fold(0_usize, |total, term| {
                    total.saturating_add(self.postings.get(term).map_or(0, |posting| posting.len()))
                })
                .min(self.document_lengths.len()),
            FtsExprKind::Phrase { terms, .. } => terms
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
            FtsExprKind::Empty | FtsExprKind::TermMatcher(_) => false,
            FtsExprKind::MatchAll => true,
            FtsExprKind::Term(term) => self
                .postings
                .get(term)
                .and_then(|posting| posting.get(ordinal))
                .is_some(),
            FtsExprKind::ExpandedTerms(terms) => terms.iter().any(|term| {
                self.postings
                    .get(term)
                    .and_then(|posting| posting.get(ordinal))
                    .is_some()
            }),
            FtsExprKind::Phrase { terms, .. } => {
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
}
