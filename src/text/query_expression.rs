//! Structured full-text query parser and document-level boolean evaluator.

use super::Tokenizer;
use crate::error::{Error, Result};
use crate::query::{fts_default_operator, FtsDefaultOperator, SearchQuery};
use std::collections::BTreeSet;
use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FtsModifier {
    None,
    Must,
    MustNot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FtsExpr {
    pub(crate) kind: FtsExprKind,
    pub(crate) modifier: FtsModifier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FtsExprKind {
    Empty,
    Term(String),
    Phrase(Vec<String>),
    And(Vec<FtsExpr>),
    Or(Vec<FtsExpr>),
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedFtsQuery {
    pub(crate) root: FtsExpr,
    all_terms: Vec<String>,
    simple: Option<(Vec<String>, FtsDefaultOperator)>,
    has_phrase: bool,
}

pub(crate) trait FtsEvalContext {
    fn contains_term(&mut self, term: &str) -> bool;
    fn contains_phrase(&mut self, terms: &[String]) -> bool;
    fn term_score(&mut self, term: &str) -> f64;
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    And,
    Or,
    Not,
    Plus,
    Minus,
    LeftParen,
    RightParen,
    Word(String),
    Phrase(String),
}

struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    tokenizer: &'a Tokenizer,
    default_operator: FtsDefaultOperator,
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
            Parser::new(expression, tokenizer, default_operator)?.parse()?
        }
        (Some(_), Some(_)) => {
            return Err(Error::invalid_argument(
                "FTS query must select exactly one expression form",
            ))
        }
        (None, None) => return Err(Error::invalid_argument("FTS query is empty")),
    };
    if root.modifier == FtsModifier::MustNot {
        return Err(Error::invalid_argument(
            "FTS query cannot contain only a top-level negative clause",
        ));
    }
    validate_positive_clause(&root)?;
    let mut all_terms = BTreeSet::new();
    collect_terms(&root, &mut all_terms);
    let has_phrase = contains_multi_term_phrase(&root);
    let mut simple_terms = Vec::new();
    let simple_operator = match &root.kind {
        FtsExprKind::And(_) => FtsDefaultOperator::And,
        FtsExprKind::Or(_) => FtsDefaultOperator::Or,
        FtsExprKind::Empty | FtsExprKind::Term(_) | FtsExprKind::Phrase(_) => default_operator,
    };
    let simple = collect_simple_terms(&root, simple_operator, &mut simple_terms)
        .then_some((simple_terms, simple_operator));
    Ok(ParsedFtsQuery {
        root,
        all_terms: all_terms.into_iter().collect(),
        simple,
        has_phrase,
    })
}

impl ParsedFtsQuery {
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
}

impl<'a> Parser<'a> {
    fn new(
        expression: &str,
        tokenizer: &'a Tokenizer,
        default_operator: FtsDefaultOperator,
    ) -> Result<Self> {
        Ok(Self {
            tokens: lex(expression)?,
            position: 0,
            tokenizer,
            default_operator,
        })
    }

    fn parse(mut self) -> Result<FtsExpr> {
        if self.tokens.is_empty() {
            return Err(Error::invalid_argument("FTS query is empty"));
        }
        let expression = self.parse_or()?;
        if self.position != self.tokens.len() {
            return Err(Error::invalid_argument(format!(
                "unexpected token in FTS query at position {}",
                self.position
            )));
        }
        Ok(expression)
    }

    fn parse_or(&mut self) -> Result<FtsExpr> {
        let mut children = vec![self.parse_and()?];
        while self.consume(&Token::Or) {
            children.push(self.parse_and()?);
        }
        Ok(combine(FtsDefaultOperator::Or, children))
    }

    fn parse_and(&mut self) -> Result<FtsExpr> {
        let mut children = vec![self.parse_sequence()?];
        loop {
            let must_not = if self.consume(&Token::And) {
                self.consume(&Token::Not)
            } else if self.consume(&Token::Not) {
                true
            } else {
                break;
            };
            let mut child = self.parse_sequence()?;
            if must_not {
                if child.modifier == FtsModifier::Must {
                    return Err(Error::invalid_argument(
                        "FTS clause cannot be both required and prohibited",
                    ));
                }
                child.modifier = FtsModifier::MustNot;
            }
            children.push(child);
        }
        Ok(combine(FtsDefaultOperator::And, children))
    }

    fn parse_sequence(&mut self) -> Result<FtsExpr> {
        let mut children = Vec::new();
        while self.next_starts_unary() {
            children.push(self.parse_unary()?);
        }
        if children.is_empty() {
            return Err(Error::invalid_argument(format!(
                "expected an FTS term at position {}",
                self.position
            )));
        }
        Ok(combine(self.default_operator, children))
    }

    fn parse_unary(&mut self) -> Result<FtsExpr> {
        let modifier = if self.consume(&Token::Plus) {
            FtsModifier::Must
        } else if self.consume(&Token::Minus) {
            FtsModifier::MustNot
        } else {
            FtsModifier::None
        };
        let mut expression = self.parse_atom()?;
        if modifier != FtsModifier::None && expression.modifier != FtsModifier::None {
            return Err(Error::invalid_argument(
                "FTS clause has conflicting unary modifiers",
            ));
        }
        expression.modifier = modifier;
        Ok(expression)
    }

    fn parse_atom(&mut self) -> Result<FtsExpr> {
        let token = self
            .tokens
            .get(self.position)
            .cloned()
            .ok_or_else(|| Error::invalid_argument("FTS query ended while parsing a term"))?;
        self.position += 1;
        match token {
            Token::Word(word) => Ok(expression_from_terms(
                self.tokenizer.tokenize(&word),
                self.default_operator,
            )),
            Token::Phrase(phrase) => {
                let terms = self.tokenizer.tokenize(&phrase);
                Ok(if let [term] = terms.as_slice() {
                    FtsExpr {
                        kind: FtsExprKind::Term(term.clone()),
                        modifier: FtsModifier::None,
                    }
                } else {
                    FtsExpr {
                        kind: FtsExprKind::Phrase(terms),
                        modifier: FtsModifier::None,
                    }
                })
            }
            Token::LeftParen => {
                let expression = self.parse_or()?;
                if !self.consume(&Token::RightParen) {
                    return Err(Error::invalid_argument("unclosed parenthesis in FTS query"));
                }
                Ok(expression)
            }
            _ => Err(Error::invalid_argument(format!(
                "expected an FTS term at position {}",
                self.position - 1
            ))),
        }
    }

    fn next_starts_unary(&self) -> bool {
        matches!(
            self.tokens.get(self.position),
            Some(Token::Plus | Token::Minus | Token::LeftParen | Token::Word(_) | Token::Phrase(_))
        )
    }

    fn consume(&mut self, expected: &Token) -> bool {
        if self.tokens.get(self.position) == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }
}

fn expression_from_terms(terms: Vec<String>, operator: FtsDefaultOperator) -> FtsExpr {
    let children: Vec<_> = terms
        .into_iter()
        .map(|term| FtsExpr {
            kind: FtsExprKind::Term(term),
            modifier: FtsModifier::None,
        })
        .collect();
    if children.is_empty() {
        FtsExpr {
            kind: FtsExprKind::Empty,
            modifier: FtsModifier::None,
        }
    } else {
        combine(operator, children)
    }
}

fn combine(operator: FtsDefaultOperator, mut children: Vec<FtsExpr>) -> FtsExpr {
    if children.len() == 1 {
        return children.pop().expect("one FTS child must exist");
    }
    FtsExpr {
        kind: match operator {
            FtsDefaultOperator::And => FtsExprKind::And(children),
            FtsDefaultOperator::Or => FtsExprKind::Or(children),
        },
        modifier: FtsModifier::None,
    }
}

fn validate_positive_clause(expression: &FtsExpr) -> Result<()> {
    let children = match &expression.kind {
        FtsExprKind::And(children) | FtsExprKind::Or(children) => children,
        FtsExprKind::Empty | FtsExprKind::Term(_) | FtsExprKind::Phrase(_) => return Ok(()),
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
        FtsExprKind::Phrase(phrase) => terms.extend(phrase.iter().cloned()),
        FtsExprKind::And(children) | FtsExprKind::Or(children) => {
            for child in children {
                collect_terms(child, terms);
            }
        }
        FtsExprKind::Empty => {}
    }
}

fn contains_multi_term_phrase(expression: &FtsExpr) -> bool {
    match &expression.kind {
        FtsExprKind::Phrase(terms) => terms.len() > 1,
        FtsExprKind::And(children) | FtsExprKind::Or(children) => {
            children.iter().any(contains_multi_term_phrase)
        }
        FtsExprKind::Empty | FtsExprKind::Term(_) => false,
    }
}

fn collect_simple_terms(
    expression: &FtsExpr,
    operator: FtsDefaultOperator,
    terms: &mut Vec<String>,
) -> bool {
    if expression.modifier != FtsModifier::None {
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
        FtsExprKind::Phrase(_) | FtsExprKind::And(_) | FtsExprKind::Or(_) => false,
    }
}

fn evaluate<C: FtsEvalContext>(expression: &FtsExpr, context: &mut C) -> Evaluation {
    match &expression.kind {
        FtsExprKind::Empty => Evaluation {
            matched: false,
            score: 0.0,
        },
        FtsExprKind::Term(term) => {
            let matched = context.contains_term(term);
            Evaluation {
                matched,
                score: if matched {
                    context.term_score(term)
                } else {
                    0.0
                },
            }
        }
        FtsExprKind::Phrase(terms) => {
            let matched = !terms.is_empty() && context.contains_phrase(terms);
            Evaluation {
                matched,
                score: if matched {
                    terms.iter().map(|term| context.term_score(term)).sum()
                } else {
                    0.0
                },
            }
        }
        FtsExprKind::And(children) => evaluate_and(children, context),
        FtsExprKind::Or(children) => evaluate_or(children, context),
    }
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
            score += evaluation.score;
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
                    score += evaluation.score;
                }
            }
            FtsModifier::None => {
                if evaluation.matched {
                    optional_matches += 1;
                    score += evaluation.score;
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

fn lex(expression: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut characters = expression.chars().peekable();
    while let Some(character) = characters.peek().copied() {
        if character.is_whitespace() {
            characters.next();
            continue;
        }
        match character {
            '+' => {
                characters.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                characters.next();
                tokens.push(Token::Minus);
            }
            '(' => {
                characters.next();
                tokens.push(Token::LeftParen);
            }
            ')' => {
                characters.next();
                tokens.push(Token::RightParen);
            }
            '"' => tokens.push(Token::Phrase(read_phrase(&mut characters)?)),
            _ => tokens.push(read_word(&mut characters)?),
        }
    }
    Ok(tokens)
}

fn read_phrase(characters: &mut Peekable<Chars<'_>>) -> Result<String> {
    let Some('"') = characters.next() else {
        return Err(Error::internal("FTS phrase lexer lost its opening quote"));
    };
    let mut phrase = String::new();
    while let Some(character) = characters.next() {
        match character {
            '"' => return Ok(phrase),
            '\\' => {
                let escaped = characters
                    .next()
                    .ok_or_else(|| Error::invalid_argument("dangling escape in FTS phrase"))?;
                phrase.push(escaped);
            }
            '\n' | '\r' => {
                return Err(Error::invalid_argument(
                    "FTS phrase cannot contain a line break",
                ))
            }
            _ => phrase.push(character),
        }
    }
    Err(Error::invalid_argument("unclosed phrase in FTS query"))
}

fn read_word(characters: &mut Peekable<Chars<'_>>) -> Result<Token> {
    let mut word = String::new();
    let mut escaped = false;
    while let Some(character) = characters.peek().copied() {
        if character.is_whitespace() || matches!(character, '(' | ')' | '"') {
            break;
        }
        characters.next();
        if character == '\\' {
            let literal = characters
                .next()
                .ok_or_else(|| Error::invalid_argument("dangling escape in FTS term"))?;
            word.push(literal);
            escaped = true;
            continue;
        }
        if matches!(
            character,
            '*' | '?' | ':' | '^' | '[' | ']' | '{' | '}' | '~' | '&' | '|'
        ) {
            return Err(Error::not_supported(format!(
                "unsupported FTS query syntax '{character}'"
            )));
        }
        word.push(character);
    }
    if word.is_empty() {
        return Err(Error::invalid_argument("empty term in FTS query"));
    }
    if !escaped {
        match word.to_ascii_uppercase().as_str() {
            "AND" => return Ok(Token::And),
            "OR" => return Ok(Token::Or),
            "NOT" => return Ok(Token::Not),
            _ => {}
        }
    }
    Ok(Token::Word(word))
}

#[cfg(test)]
mod tests {
    use super::{parse_fts_query, FtsExprKind, FtsModifier};
    use crate::query::{Fts, FtsQueryParams, SearchQuery};
    use crate::schema::IndexParams;
    use crate::text::Tokenizer;

    fn parse(expression: &str, operator: Option<&str>) -> super::ParsedFtsQuery {
        let params =
            IndexParams::fts(Some("whitespace"), None, None).expect("FTS params must be valid");
        let tokenizer =
            Tokenizer::from_index_params(Some(&params)).expect("tokenizer must be valid");
        let mut fts = Fts::new().expect("FTS payload must be valid");
        fts.set_query_string(expression)
            .expect("query string must be valid");
        let mut query = SearchQuery::fts("body", &fts, 10).expect("query must be valid");
        if let Some(operator) = operator {
            query
                .set_fts_params(
                    FtsQueryParams::new(Some(operator)).expect("operator must be valid"),
                )
                .expect("operator must be accepted");
        }
        parse_fts_query(&query, &tokenizer).expect("expression must parse")
    }

    #[test]
    fn boolean_precedence_and_modifiers_are_structural() {
        let parsed = parse("+Rust database OR python AND NOT legacy", None);
        let FtsExprKind::Or(children) = &parsed.root.kind else {
            panic!("root must be OR");
        };
        assert_eq!(children.len(), 2);
        let FtsExprKind::Or(left) = &children[0].kind else {
            panic!("left branch must use the implicit OR shape");
        };
        assert_eq!(left[0].modifier, FtsModifier::Must);
        let FtsExprKind::And(right) = &children[1].kind else {
            panic!("right branch must be AND");
        };
        assert_eq!(right[1].modifier, FtsModifier::MustNot);
    }

    #[test]
    fn explicit_operator_overrides_the_default_operator() {
        let parsed = parse("rust OR python database", Some("AND"));
        assert!(matches!(parsed.root.kind, FtsExprKind::Or(_)));
        assert!(parsed.simple().is_none());
    }
}
