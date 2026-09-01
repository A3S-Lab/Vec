//! Structured full-text query parsing, expansion, and boolean evaluation.

mod lexer;
mod parser;
mod pattern;

#[cfg(test)]
mod tests;

use self::parser::Parser;
use self::pattern::FtsTermMatcher;
use super::Tokenizer;
use crate::error::{Error, Result};
use crate::query::{fts_default_operator, FtsDefaultOperator, SearchQuery};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FtsModifier {
    None,
    Must,
    MustNot,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FtsExpr {
    pub(crate) kind: FtsExprKind,
    pub(crate) modifier: FtsModifier,
    pub(crate) boost: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FtsExprKind {
    Empty,
    MatchAll,
    Term(String),
    ExpandedTerms(Vec<String>),
    TermMatcher(FtsTermMatcher),
    Phrase { terms: Vec<String>, slop: u32 },
    And(Vec<FtsExpr>),
    Or(Vec<FtsExpr>),
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedFtsQuery {
    pub(crate) root: FtsExpr,
    all_terms: Vec<String>,
    simple: Option<(Vec<String>, FtsDefaultOperator)>,
    has_phrase: bool,
    default_operator: FtsDefaultOperator,
}

pub(crate) trait FtsEvalContext {
    fn contains_term(&mut self, term: &str) -> bool;
    fn contains_phrase(&mut self, terms: &[String], slop: u32) -> bool;
    fn term_score(&mut self, term: &str) -> f64;
}

pub(crate) fn contains_ordered_phrase(tokens: &[String], phrase: &[String], slop: u32) -> bool {
    let Some(first) = phrase.first() else {
        return false;
    };
    let extra = usize::try_from(slop).unwrap_or(usize::MAX);
    tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| *token == first)
        .any(|(start, _)| {
            let maximum_end = start
                .saturating_add(phrase.len().saturating_sub(1))
                .saturating_add(extra)
                .min(tokens.len().saturating_sub(1));
            let mut next = start.saturating_add(1);
            for expected in phrase.iter().skip(1) {
                let Some(offset) = tokens
                    .get(next..=maximum_end)
                    .and_then(|window| window.iter().position(|token| token == expected))
                else {
                    return false;
                };
                next = next.saturating_add(offset).saturating_add(1);
            }
            true
        })
}

#[derive(Debug, Clone, Copy)]
struct Evaluation {
    matched: bool,
    score: f64,
}

pub(crate) fn parse_fts_query(
    query: &SearchQuery,
    tokenizer: &Tokenizer,
) -> Result<ParsedFtsQuery> {
    let fts = query
        .fts
        .as_ref()
        .ok_or_else(|| Error::invalid_argument("FTS payload is missing"))?;
    let default_operator = fts_default_operator(query)?;
    let root = match (fts.match_string.as_deref(), fts.query_string.as_deref()) {
        (Some(expression), None) => {
            if expression.trim().is_empty() {
                return Err(Error::invalid_argument("FTS query is empty"));
            }
            expression_from_terms(tokenizer.tokenize(expression), default_operator)
        }
        (None, Some(expression)) => {
            if expression.trim().is_empty() {
                return Err(Error::invalid_argument("FTS query is empty"));
            }
            Parser::new(expression, tokenizer, &query.field_name, default_operator)?.parse()?
        }
        (Some(_), Some(_)) => {
            return Err(Error::invalid_argument(
                "FTS query must select exactly one expression form",
            ));
        }
        (None, None) => return Err(Error::invalid_argument("FTS query is empty")),
    };
    if root.modifier == FtsModifier::MustNot {
        return Err(Error::invalid_argument(
            "FTS query cannot contain only a top-level negative clause",
        ));
    }
    validate_positive_clause(&root)?;
    let mut parsed = ParsedFtsQuery {
        root,
        all_terms: Vec::new(),
        simple: None,
        has_phrase: false,
        default_operator,
    };
    parsed.refresh_metadata();
    Ok(parsed)
}

impl FtsExpr {
    pub(super) fn new(kind: FtsExprKind) -> Self {
        Self {
            kind,
            modifier: FtsModifier::None,
            boost: 1.0,
        }
    }
}

impl ParsedFtsQuery {
    pub(crate) fn expand_terms<'a>(&mut self, vocabulary: impl IntoIterator<Item = &'a str>) {
        if !contains_term_matcher(&self.root) {
            return;
        }
        let mut vocabulary: Vec<&str> = vocabulary.into_iter().collect();
        if !vocabulary.windows(2).all(|pair| pair[0] < pair[1]) {
            vocabulary.sort_unstable();
            vocabulary.dedup();
        }
        expand_expression(&mut self.root, &vocabulary);
        self.refresh_metadata();
    }

    pub(crate) fn all_terms(&self) -> &[String] {
        &self.all_terms
    }

    pub(crate) fn simple(&self) -> Option<(&[String], FtsDefaultOperator)> {
        self.simple
            .as_ref()
            .map(|(terms, operator)| (terms.as_slice(), *operator))
    }

    pub(crate) fn has_phrase(&self) -> bool {
        self.has_phrase
    }

    pub(crate) fn score<C: FtsEvalContext>(&self, context: &mut C) -> Option<f64> {
        let evaluation = evaluate(&self.root, context);
        evaluation.matched.then_some(evaluation.score)
    }

    fn refresh_metadata(&mut self) {
        let mut all_terms = BTreeSet::new();
        collect_terms(&self.root, &mut all_terms);
        self.all_terms = all_terms.into_iter().collect();
        self.has_phrase = contains_multi_term_phrase(&self.root);
        let simple_operator = match &self.root.kind {
            FtsExprKind::And(_) => FtsDefaultOperator::And,
            FtsExprKind::Or(_) => FtsDefaultOperator::Or,
            _ => self.default_operator,
        };
        let mut simple_terms = Vec::new();
        self.simple = collect_simple_terms(&self.root, simple_operator, &mut simple_terms)
            .then_some((simple_terms, simple_operator));
    }
}

fn contains_term_matcher(expression: &FtsExpr) -> bool {
    match &expression.kind {
        FtsExprKind::TermMatcher(_) => true,
        FtsExprKind::And(children) | FtsExprKind::Or(children) => {
            children.iter().any(contains_term_matcher)
        }
        _ => false,
    }
}

pub(super) fn expression_from_terms(terms: Vec<String>, operator: FtsDefaultOperator) -> FtsExpr {
    let children: Vec<_> = terms
        .into_iter()
        .map(|term| FtsExpr::new(FtsExprKind::Term(term)))
        .collect();
    if children.is_empty() {
        FtsExpr::new(FtsExprKind::Empty)
    } else {
        combine(operator, children)
    }
}

pub(super) fn combine(operator: FtsDefaultOperator, mut children: Vec<FtsExpr>) -> FtsExpr {
    if children.len() == 1 {
        if let Some(child) = children.pop() {
            return child;
        }
    }
    FtsExpr::new(match operator {
        FtsDefaultOperator::And => FtsExprKind::And(children),
        FtsDefaultOperator::Or => FtsExprKind::Or(children),
    })
}

fn expand_expression(expression: &mut FtsExpr, vocabulary: &[&str]) {
    match &mut expression.kind {
        FtsExprKind::TermMatcher(matcher) => {
            let terms = vocabulary
                .iter()
                .filter(|term| matcher.matches(term))
                .map(|term| (*term).to_string())
                .collect();
            expression.kind = FtsExprKind::ExpandedTerms(terms);
        }
        FtsExprKind::And(children) | FtsExprKind::Or(children) => {
            for child in children {
                expand_expression(child, vocabulary);
            }
        }
        FtsExprKind::Empty
        | FtsExprKind::MatchAll
        | FtsExprKind::Term(_)
        | FtsExprKind::ExpandedTerms(_)
        | FtsExprKind::Phrase { .. } => {}
    }
}

fn validate_positive_clause(expression: &FtsExpr) -> Result<()> {
    let (FtsExprKind::And(children) | FtsExprKind::Or(children)) = &expression.kind else {
        return Ok(());
    };
    if children
        .iter()
        .all(|child| child.modifier == FtsModifier::MustNot)
    {
        return Err(Error::invalid_argument(
            "FTS boolean group must contain a positive clause",
        ));
    }
    for child in children {
        validate_positive_clause(child)?;
    }
    Ok(())
}

fn collect_terms(expression: &FtsExpr, terms: &mut BTreeSet<String>) {
    match &expression.kind {
        FtsExprKind::Term(term) => {
            terms.insert(term.clone());
        }
        FtsExprKind::ExpandedTerms(expanded) => terms.extend(expanded.iter().cloned()),
        FtsExprKind::Phrase { terms: phrase, .. } => terms.extend(phrase.iter().cloned()),
        FtsExprKind::And(children) | FtsExprKind::Or(children) => {
            for child in children {
                collect_terms(child, terms);
            }
        }
        FtsExprKind::Empty | FtsExprKind::MatchAll | FtsExprKind::TermMatcher(_) => {}
    }
}

fn contains_multi_term_phrase(expression: &FtsExpr) -> bool {
    match &expression.kind {
        FtsExprKind::Phrase { terms, .. } => terms.len() > 1,
        FtsExprKind::And(children) | FtsExprKind::Or(children) => {
            children.iter().any(contains_multi_term_phrase)
        }
        _ => false,
    }
}

fn collect_simple_terms(
    expression: &FtsExpr,
    operator: FtsDefaultOperator,
    terms: &mut Vec<String>,
) -> bool {
    if expression.modifier != FtsModifier::None || expression.boost.to_bits() != 1.0_f64.to_bits() {
        return false;
    }
    match &expression.kind {
        FtsExprKind::Term(term) => {
            terms.push(term.clone());
            true
        }
        FtsExprKind::And(children) if operator == FtsDefaultOperator::And => children
            .iter()
            .all(|child| collect_simple_terms(child, operator, terms)),
        FtsExprKind::Or(children) if operator == FtsDefaultOperator::Or => children
            .iter()
            .all(|child| collect_simple_terms(child, operator, terms)),
        FtsExprKind::Empty => true,
        _ => false,
    }
}

fn evaluate<C: FtsEvalContext>(expression: &FtsExpr, context: &mut C) -> Evaluation {
    let mut evaluation = match &expression.kind {
        FtsExprKind::Empty | FtsExprKind::TermMatcher(_) => Evaluation {
            matched: false,
            score: 0.0,
        },
        FtsExprKind::MatchAll => Evaluation {
            matched: true,
            score: 1.0,
        },
        FtsExprKind::Term(term) => evaluate_terms(std::slice::from_ref(term), context),
        FtsExprKind::ExpandedTerms(terms) => evaluate_terms(terms, context),
        FtsExprKind::Phrase { terms, slop } => {
            let matched = !terms.is_empty() && context.contains_phrase(terms, *slop);
            Evaluation {
                matched,
                score: if matched {
                    terms.iter().fold(0.0, |score, term| {
                        finite_add(score, context.term_score(term))
                    })
                } else {
                    0.0
                },
            }
        }
        FtsExprKind::And(children) => evaluate_and(children, context),
        FtsExprKind::Or(children) => evaluate_or(children, context),
    };
    if evaluation.matched {
        evaluation.score = finite_multiply(evaluation.score, expression.boost);
    }
    evaluation
}

fn evaluate_terms<C: FtsEvalContext>(terms: &[String], context: &mut C) -> Evaluation {
    let mut matched = false;
    let mut score = 0.0;
    for term in terms {
        if context.contains_term(term) {
            matched = true;
            score = finite_add(score, context.term_score(term));
        }
    }
    Evaluation { matched, score }
}

fn evaluate_and<C: FtsEvalContext>(children: &[FtsExpr], context: &mut C) -> Evaluation {
    let mut score = 0.0;
    let mut has_positive = false;
    for child in children {
        let evaluation = evaluate(child, context);
        if child.modifier == FtsModifier::MustNot {
            if evaluation.matched {
                return Evaluation {
                    matched: false,
                    score: 0.0,
                };
            }
        } else {
            has_positive = true;
            if !evaluation.matched {
                return Evaluation {
                    matched: false,
                    score: 0.0,
                };
            }
            score = finite_add(score, evaluation.score);
        }
    }
    Evaluation {
        matched: has_positive,
        score,
    }
}

fn evaluate_or<C: FtsEvalContext>(children: &[FtsExpr], context: &mut C) -> Evaluation {
    let mut score = 0.0;
    let mut required = 0_usize;
    let mut required_matches = 0_usize;
    let mut optional_matches = 0_usize;
    for child in children {
        let evaluation = evaluate(child, context);
        match child.modifier {
            FtsModifier::MustNot => {
                if evaluation.matched {
                    return Evaluation {
                        matched: false,
                        score: 0.0,
                    };
                }
            }
            FtsModifier::Must => {
                required += 1;
                if evaluation.matched {
                    required_matches += 1;
                    score = finite_add(score, evaluation.score);
                }
            }
            FtsModifier::None => {
                if evaluation.matched {
                    optional_matches += 1;
                    score = finite_add(score, evaluation.score);
                }
            }
        }
    }
    let matched = if required == 0 {
        optional_matches > 0
    } else {
        required_matches == required
    };
    Evaluation {
        matched,
        score: if matched { score } else { 0.0 },
    }
}

fn finite_add(left: f64, right: f64) -> f64 {
    let result = left + right;
    if result.is_finite() {
        result
    } else {
        f64::MAX
    }
}

fn finite_multiply(score: f64, boost: f64) -> f64 {
    let result = score * boost;
    if result.is_finite() {
        result
    } else {
        f64::MAX
    }
}
