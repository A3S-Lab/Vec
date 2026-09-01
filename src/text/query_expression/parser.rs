//! Lexer and recursive-descent parser for structured FTS expressions.

use super::lexer::{lex, RawWord, RawWordPart, Token};
use super::pattern::{FtsTermMatcher, WildcardAtom, WildcardPattern};
use super::{expression_from_terms, FtsExpr, FtsExprKind, FtsModifier};
use crate::error::{Error, Result};
use crate::query::FtsDefaultOperator;
use crate::text::Tokenizer;

const MAX_BOOST: f64 = 1_000_000.0;
const MAX_FUZZY_DISTANCE: u8 = 2;
const MAX_FUZZY_TERM_CHARS: usize = 256;
const MAX_PHRASE_SLOP: u32 = 1_024;

pub(super) struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    tokenizer: &'a Tokenizer,
    field_name: &'a str,
    default_operator: FtsDefaultOperator,
}

impl<'a> Parser<'a> {
    pub(super) fn new(
        expression: &str,
        tokenizer: &'a Tokenizer,
        field_name: &'a str,
        default_operator: FtsDefaultOperator,
    ) -> Result<Self> {
        Ok(Self {
            tokens: lex(expression)?,
            position: 0,
            tokenizer,
            field_name,
            default_operator,
        })
    }

    pub(super) fn parse(mut self) -> Result<FtsExpr> {
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
        Ok(super::combine(FtsDefaultOperator::Or, children))
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
        Ok(super::combine(FtsDefaultOperator::And, children))
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
        Ok(super::combine(self.default_operator, children))
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
        let field = self.field_qualifier()?;
        if let Some(field) = field {
            if field != self.field_name {
                return Err(Error::invalid_argument(format!(
                    "FTS field qualifier '{field}' does not match query field '{}'",
                    self.field_name
                )));
            }
        }
        let mut expression = self.parse_primary()?;
        if self.consume(&Token::Tilde) {
            let value = self.u32_suffix("FTS fuzzy/proximity suffix")?;
            expression.kind = match expression.kind {
                FtsExprKind::Term(term) => {
                    let distance = u8::try_from(value)
                        .ok()
                        .filter(|value| *value > 0 && *value <= MAX_FUZZY_DISTANCE);
                    let Some(distance) = distance else {
                        return Err(Error::invalid_argument(format!(
                            "FTS fuzzy distance must be in 1..={MAX_FUZZY_DISTANCE}"
                        )));
                    };
                    if term.chars().count() > MAX_FUZZY_TERM_CHARS {
                        return Err(Error::invalid_argument(format!(
                            "FTS fuzzy term must contain at most {MAX_FUZZY_TERM_CHARS} characters"
                        )));
                    }
                    FtsExprKind::TermMatcher(FtsTermMatcher::Fuzzy { term, distance })
                }
                FtsExprKind::Phrase { terms, slop: 0 } => {
                    if value > MAX_PHRASE_SLOP {
                        return Err(Error::invalid_argument(format!(
                            "FTS phrase slop must be at most {MAX_PHRASE_SLOP}"
                        )));
                    }
                    FtsExprKind::Phrase { terms, slop: value }
                }
                FtsExprKind::TermMatcher(FtsTermMatcher::Wildcard(_)) => {
                    return Err(Error::invalid_argument(
                        "FTS wildcard and fuzzy suffix cannot be combined",
                    ));
                }
                _ => {
                    return Err(Error::invalid_argument(
                        "FTS fuzzy/proximity suffix requires one term or phrase",
                    ));
                }
            };
        }
        if self.consume(&Token::Caret) {
            expression.boost = self.positive_f64_suffix("FTS boost")?;
        }
        Ok(expression)
    }

    fn field_qualifier(&mut self) -> Result<Option<String>> {
        let Some(Token::Word(word)) = self.tokens.get(self.position) else {
            return Ok(None);
        };
        if self.tokens.get(self.position + 1) != Some(&Token::Colon) {
            return Ok(None);
        }
        let Some(field) = word.plain().filter(|field| !field.is_empty()) else {
            return Err(Error::invalid_argument(
                "FTS field qualifier must be a plain field name",
            ));
        };
        let field = field.to_string();
        self.position += 2;
        if !self.next_starts_primary() {
            return Err(Error::invalid_argument(
                "FTS field qualifier must be followed by a term, phrase, range, or group",
            ));
        }
        Ok(Some(field))
    }

    fn parse_primary(&mut self) -> Result<FtsExpr> {
        let token = self
            .tokens
            .get(self.position)
            .cloned()
            .ok_or_else(|| Error::invalid_argument("FTS query ended while parsing a term"))?;
        self.position += 1;
        match token {
            Token::Word(word) => self.word_expression(word),
            Token::Phrase(phrase) => {
                let terms = self.tokenizer.tokenize(&phrase);
                Ok(match terms.as_slice() {
                    [] => FtsExpr::new(FtsExprKind::Empty),
                    [term] => FtsExpr::new(FtsExprKind::Term(term.clone())),
                    _ => FtsExpr::new(FtsExprKind::Phrase { terms, slop: 0 }),
                })
            }
            Token::LeftParen => {
                let expression = self.parse_or()?;
                if !self.consume(&Token::RightParen) {
                    return Err(Error::invalid_argument("unclosed parenthesis in FTS query"));
                }
                Ok(expression)
            }
            Token::LeftBracket => self.range_expression(true),
            Token::LeftBrace => self.range_expression(false),
            _ => Err(Error::invalid_argument(format!(
                "expected an FTS term at position {}",
                self.position - 1
            ))),
        }
    }

    fn word_expression(&self, word: RawWord) -> Result<FtsExpr> {
        if !word.has_wildcard() {
            let value = word
                .plain()
                .ok_or_else(|| Error::invalid_argument("invalid FTS term"))?;
            return Ok(expression_from_terms(
                self.tokenizer.tokenize(value),
                self.default_operator,
            ));
        }
        if word.is_unbounded() {
            return Ok(FtsExpr::new(FtsExprKind::MatchAll));
        }
        let mut atoms = Vec::new();
        for part in word.parts {
            match part {
                RawWordPart::Literal(literal) => {
                    let normalized = self.normalize_single_term(&literal, "wildcard fragment")?;
                    atoms.extend(normalized.chars().map(WildcardAtom::Literal));
                }
                RawWordPart::AnyOne => atoms.push(WildcardAtom::AnyOne),
                RawWordPart::AnyMany => atoms.push(WildcardAtom::AnyMany),
            }
        }
        Ok(FtsExpr::new(FtsExprKind::TermMatcher(
            FtsTermMatcher::Wildcard(WildcardPattern::new(atoms)),
        )))
    }

    fn range_expression(&mut self, lower_inclusive: bool) -> Result<FtsExpr> {
        let lower = self.range_bound("lower")?;
        let Some(Token::Word(to)) = self.tokens.get(self.position) else {
            return Err(Error::invalid_argument(
                "FTS range requires the TO separator",
            ));
        };
        if to.plain().map(str::to_ascii_uppercase).as_deref() != Some("TO") || to.had_escape {
            return Err(Error::invalid_argument(
                "FTS range requires the TO separator",
            ));
        }
        self.position += 1;
        let upper = self.range_bound("upper")?;
        let upper_inclusive = match self.tokens.get(self.position) {
            Some(Token::RightBracket) => true,
            Some(Token::RightBrace) => false,
            _ => {
                return Err(Error::invalid_argument(
                    "FTS range is missing its closing bound",
                ))
            }
        };
        self.position += 1;
        if lower.is_none() && upper.is_none() {
            return Ok(FtsExpr::new(FtsExprKind::MatchAll));
        }
        Ok(FtsExpr::new(FtsExprKind::TermMatcher(
            FtsTermMatcher::Range {
                lower: lower.map(|value| (value, lower_inclusive)),
                upper: upper.map(|value| (value, upper_inclusive)),
            },
        )))
    }

    fn range_bound(&mut self, label: &str) -> Result<Option<String>> {
        let token = self.tokens.get(self.position).cloned().ok_or_else(|| {
            Error::invalid_argument(format!("FTS range is missing its {label} bound"))
        })?;
        self.position += 1;
        match token {
            Token::Word(word) if word.is_unbounded() => Ok(None),
            Token::Word(word) if !word.has_wildcard() => self
                .normalize_single_term(word.plain().unwrap_or_default(), "range bound")
                .map(Some),
            Token::Phrase(phrase) => self.normalize_single_term(&phrase, "range bound").map(Some),
            _ => Err(Error::invalid_argument(format!(
                "FTS range {label} bound must be one term or '*'"
            ))),
        }
    }

    fn normalize_single_term(&self, value: &str, role: &str) -> Result<String> {
        let terms = self.tokenizer.tokenize(value);
        match terms.as_slice() {
            [term] => Ok(term.clone()),
            _ => Err(Error::invalid_argument(format!(
                "FTS {role} must analyze to exactly one term"
            ))),
        }
    }

    fn u32_suffix(&mut self, role: &str) -> Result<u32> {
        let value = self.plain_suffix(role)?;
        value
            .parse::<u32>()
            .map_err(|_| Error::invalid_argument(format!("{role} must be a non-negative integer")))
    }

    fn positive_f64_suffix(&mut self, role: &str) -> Result<f64> {
        let value = self.plain_suffix(role)?;
        value
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= MAX_BOOST)
            .ok_or_else(|| {
                Error::invalid_argument(format!("{role} must be finite and in (0, {MAX_BOOST}]"))
            })
    }

    fn plain_suffix(&mut self, role: &str) -> Result<String> {
        let Some(Token::Word(word)) = self.tokens.get(self.position) else {
            return Err(Error::invalid_argument(format!("{role} requires a value")));
        };
        let Some(value) = word.plain().filter(|value| !value.is_empty()) else {
            return Err(Error::invalid_argument(format!(
                "{role} requires a plain value"
            )));
        };
        let value = value.to_string();
        self.position += 1;
        Ok(value)
    }

    fn next_starts_unary(&self) -> bool {
        matches!(
            self.tokens.get(self.position),
            Some(Token::Plus | Token::Minus)
        ) || self.next_starts_primary()
    }

    fn next_starts_primary(&self) -> bool {
        matches!(
            self.tokens.get(self.position),
            Some(
                Token::LeftParen
                    | Token::LeftBracket
                    | Token::LeftBrace
                    | Token::Word(_)
                    | Token::Phrase(_)
            )
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
