//! Scalar filter language evaluated against collection documents.
//!
//! Precedence, low to high: `or`, `and`, `not`, then a parenthesized expression
//! or a field predicate. A missing field makes a comparison false, so `not`
//! around that comparison is true. `id` reads the document primary key.

use crate::doc::{Doc, FieldValue};
use crate::error::{Error, Result};
use serde_json::Value;
use std::cmp::Ordering;

/// A parsed filter expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FilterExpr {
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
    Not(Box<FilterExpr>),
    Compare {
        field: String,
        op: CmpOp,
        value: Literal,
    },
    In {
        field: String,
        values: Vec<Literal>,
        negated: bool,
    },
    Like {
        field: String,
        pattern: String,
        negated: bool,
    },
    IsNull {
        field: String,
        negated: bool,
    },
    ContainAll {
        field: String,
        values: Vec<Literal>,
        negated: bool,
    },
    HasPrefix {
        field: String,
        prefix: String,
        negated: bool,
    },
    HasSuffix {
        field: String,
        suffix: String,
        negated: bool,
    },
}

/// Comparison operators in a filter predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

/// A literal on the right-hand side of a predicate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Literal {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

/// Parses one filter expression. An empty string is rejected by the caller.
pub(crate) fn parse_filter(input: &str) -> Result<FilterExpr> {
    let tokens = tokenize(input)?;
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.parse_or()?;
    if parser.pos != parser.tokens.len() {
        return Err(invalid(format!(
            "unexpected trailing tokens in filter near position {}",
            parser.pos
        )));
    }
    Ok(expr)
}

impl FilterExpr {
    pub(crate) fn matches(&self, doc: &Doc) -> bool {
        match self {
            Self::And(left, right) => left.matches(doc) && right.matches(doc),
            Self::Or(left, right) => left.matches(doc) || right.matches(doc),
            Self::Not(inner) => !inner.matches(doc),
            Self::Compare { field, op, value } => {
                field_json(doc, field).is_some_and(|stored| compare(&stored, *op, value))
            }
            Self::In {
                field,
                values,
                negated,
            } => {
                let hit = field_json(doc, field).is_some_and(|stored| {
                    values.iter().any(|literal| values_equal(&stored, literal))
                });
                hit ^ negated
            }
            Self::Like {
                field,
                pattern,
                negated,
            } => {
                let hit = field_json(doc, field)
                    .and_then(|stored| stored.as_str().map(|text| like_match(text, pattern)))
                    .unwrap_or(false);
                hit ^ negated
            }
            Self::IsNull { field, negated } => {
                let is_null = match field_json(doc, field) {
                    None => true,
                    Some(stored) => stored.is_null(),
                };
                is_null ^ negated
            }
            Self::ContainAll {
                field,
                values,
                negated,
            } => {
                let hit = field_json(doc, field)
                    .and_then(|stored| stored.as_array().cloned())
                    .is_some_and(|items| {
                        values
                            .iter()
                            .all(|literal| items.iter().any(|item| values_equal(item, literal)))
                    });
                hit ^ negated
            }
            Self::HasPrefix {
                field,
                prefix,
                negated,
            } => {
                let hit = field_json(doc, field)
                    .and_then(|stored| stored.as_str().map(|text| text.starts_with(prefix)))
                    .unwrap_or(false);
                hit ^ negated
            }
            Self::HasSuffix {
                field,
                suffix,
                negated,
            } => {
                let hit = field_json(doc, field)
                    .and_then(|stored| stored.as_str().map(|text| text.ends_with(suffix)))
                    .unwrap_or(false);
                hit ^ negated
            }
        }
    }
}

fn field_json(doc: &Doc, field: &str) -> Option<Value> {
    if field == "id" {
        return Some(Value::String(doc.get_pk().unwrap_or("").to_string()));
    }
    doc.field(field).map(FieldValue::to_json)
}

fn compare(stored: &Value, op: CmpOp, literal: &Literal) -> bool {
    match op {
        CmpOp::Eq => values_equal(stored, literal),
        CmpOp::Ne => !values_equal(stored, literal),
        CmpOp::Gt | CmpOp::Ge | CmpOp::Lt | CmpOp::Le => match ordering(stored, literal) {
            Some(order) => match op {
                CmpOp::Gt => order.is_gt(),
                CmpOp::Ge => order.is_ge(),
                CmpOp::Lt => order.is_lt(),
                CmpOp::Le => order.is_le(),
                CmpOp::Eq | CmpOp::Ne => false,
            },
            None => false,
        },
    }
}

fn values_equal(stored: &Value, literal: &Literal) -> bool {
    match literal {
        Literal::Str(text) => stored.as_str() == Some(text.as_str()),
        Literal::Num(number) => stored.as_f64() == Some(*number),
        Literal::Bool(flag) => stored.as_bool() == Some(*flag),
        Literal::Null => stored.is_null(),
    }
}

fn ordering(stored: &Value, literal: &Literal) -> Option<Ordering> {
    match literal {
        Literal::Num(number) => stored.as_f64().and_then(|value| value.partial_cmp(number)),
        Literal::Str(text) => stored.as_str().map(|value| value.cmp(text.as_str())),
        Literal::Bool(_) | Literal::Null => None,
    }
}

/// SQL-like match. `%` and `*` match any run; `_` matches one character.
pub(crate) fn like_match(text: &str, pattern: &str) -> bool {
    let text: Vec<char> = text.chars().collect();
    let pattern: Vec<char> = pattern.chars().collect();
    let (mut text_index, mut pattern_index) = (0, 0);
    let mut wildcard_pattern = None;
    let mut wildcard_text = 0;
    while text_index < text.len() {
        if pattern_index < pattern.len() && matches!(pattern[pattern_index], '%' | '*') {
            while pattern_index < pattern.len() && matches!(pattern[pattern_index], '%' | '*') {
                pattern_index += 1;
            }
            if pattern_index == pattern.len() {
                return true;
            }
            wildcard_pattern = Some(pattern_index);
            wildcard_text = text_index;
        } else if pattern_index < pattern.len()
            && (pattern[pattern_index] == '_' || pattern[pattern_index] == text[text_index])
        {
            text_index += 1;
            pattern_index += 1;
        } else if let Some(saved_pattern) = wildcard_pattern {
            wildcard_text += 1;
            text_index = wildcard_text;
            pattern_index = saved_pattern;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && matches!(pattern[pattern_index], '%' | '*') {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid_argument(format!("invalid filter: {}", message.into()))
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Str(String),
    Num(f64),
    Op(CmpOp),
    And,
    Or,
    Not,
    In,
    Like,
    IsNull,
    ContainAll,
    HasPrefix,
    HasSuffix,
    True,
    False,
    Null,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = input.chars().collect();
    let mut index = 0;
    let mut tokens = Vec::new();
    while index < chars.len() {
        let character = chars[index];
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        match character {
            '(' => push_single(&mut tokens, Token::LParen, &mut index),
            ')' => push_single(&mut tokens, Token::RParen, &mut index),
            '[' => push_single(&mut tokens, Token::LBracket, &mut index),
            ']' => push_single(&mut tokens, Token::RBracket, &mut index),
            ',' => push_single(&mut tokens, Token::Comma, &mut index),
            '\'' | '"' => tokens.push(read_string(&chars, &mut index, character)?),
            '=' | '!' | '<' | '>' => tokens.push(read_operator(&chars, &mut index)?),
            '&' | '|' => tokens.push(read_logic(&chars, &mut index)?),
            _ if is_number_start(&chars, index) => tokens.push(read_number(&chars, &mut index)?),
            _ if character.is_alphabetic() || character == '_' => {
                tokens.push(read_word(&chars, &mut index));
            }
            _ => return Err(invalid(format!("unexpected character: {character}"))),
        }
    }
    Ok(tokens)
}

fn push_single(tokens: &mut Vec<Token>, token: Token, index: &mut usize) {
    tokens.push(token);
    *index += 1;
}

fn is_number_start(chars: &[char], index: usize) -> bool {
    chars[index].is_ascii_digit()
        || (chars[index] == '-' && chars.get(index + 1).is_some_and(char::is_ascii_digit))
}

fn read_string(chars: &[char], index: &mut usize, quote: char) -> Result<Token> {
    *index += 1;
    let mut text = String::new();
    while *index < chars.len() {
        let character = chars[*index];
        if character == '\\' && *index + 1 < chars.len() {
            text.push(chars[*index + 1]);
            *index += 2;
            continue;
        }
        if character == quote {
            *index += 1;
            return Ok(Token::Str(text));
        }
        text.push(character);
        *index += 1;
    }
    Err(invalid("unterminated string literal"))
}

fn read_operator(chars: &[char], index: &mut usize) -> Result<Token> {
    let next = chars.get(*index + 1).copied();
    let (token, length) = match (chars[*index], next) {
        ('=', Some('=')) => (Token::Op(CmpOp::Eq), 2),
        ('!', Some('=')) => (Token::Op(CmpOp::Ne), 2),
        ('>', Some('=')) => (Token::Op(CmpOp::Ge), 2),
        ('<', Some('=')) => (Token::Op(CmpOp::Le), 2),
        ('&', Some('&')) => (Token::And, 2),
        ('|', Some('|')) => (Token::Or, 2),
        ('>', _) => (Token::Op(CmpOp::Gt), 1),
        ('<', _) => (Token::Op(CmpOp::Lt), 1),
        ('!', _) => (Token::Not, 1),
        ('=', _) => (Token::Op(CmpOp::Eq), 1),
        (character, _) => {
            return Err(invalid(format!("unexpected operator: {character}")));
        }
    };
    *index += length;
    Ok(token)
}

fn read_logic(chars: &[char], index: &mut usize) -> Result<Token> {
    let pair: String = chars.iter().skip(*index).take(2).collect();
    let token = match pair.as_str() {
        "&&" => Token::And,
        "||" => Token::Or,
        _ => return Err(invalid(format!("unexpected operator: {}", chars[*index]))),
    };
    *index += 2;
    Ok(token)
}

fn read_number(chars: &[char], index: &mut usize) -> Result<Token> {
    let start = *index;
    *index += 1;
    while chars
        .get(*index)
        .is_some_and(|character| character.is_ascii_digit() || *character == '.')
    {
        *index += 1;
    }
    let text: String = chars[start..*index].iter().collect();
    let value = text
        .parse::<f64>()
        .map_err(|_| invalid(format!("invalid number: {text}")))?;
    Ok(Token::Num(value))
}

fn read_word(chars: &[char], index: &mut usize) -> Token {
    let start = *index;
    *index += 1;
    while chars.get(*index).is_some_and(|character| {
        character.is_alphanumeric() || *character == '_' || *character == '.'
    }) {
        *index += 1;
    }
    let word: String = chars[start..*index].iter().collect();
    match word.to_lowercase().as_str() {
        "and" => Token::And,
        "or" => Token::Or,
        "not" => Token::Not,
        "in" => Token::In,
        "like" => Token::Like,
        "is_null" => Token::IsNull,
        "contain_all" => Token::ContainAll,
        "has_prefix" => Token::HasPrefix,
        "has_suffix" => Token::HasSuffix,
        "true" => Token::True,
        "false" => Token::False,
        "null" => Token::Null,
        _ => Token::Ident(word),
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn parse_or(&mut self) -> Result<FilterExpr> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Or)) {
            self.next();
            let right = self.parse_and()?;
            left = FilterExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<FilterExpr> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.next();
            let right = self.parse_not()?;
            left = FilterExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<FilterExpr> {
        if matches!(self.peek(), Some(Token::Not)) {
            self.next();
            return Ok(FilterExpr::Not(Box::new(self.parse_not()?)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<FilterExpr> {
        if matches!(self.peek(), Some(Token::LParen)) {
            self.next();
            let expr = self.parse_or()?;
            return match self.next() {
                Some(Token::RParen) => Ok(expr),
                _ => Err(invalid("expected ')'")),
            };
        }
        self.parse_predicate()
    }

    fn parse_predicate(&mut self) -> Result<FilterExpr> {
        let field = match self.next() {
            Some(Token::Ident(name)) => name,
            other => return Err(invalid(format!("expected field name, got {other:?}"))),
        };
        let negated = if matches!(self.peek(), Some(Token::Not)) {
            self.next();
            true
        } else {
            false
        };
        match self.next() {
            Some(Token::Op(op)) if !negated => Ok(FilterExpr::Compare {
                field,
                op,
                value: self.parse_literal()?,
            }),
            Some(Token::In) => Ok(FilterExpr::In {
                field,
                values: self.parse_list()?,
                negated,
            }),
            Some(Token::Like) => Ok(FilterExpr::Like {
                field,
                pattern: self.parse_string_literal("like pattern must be a string")?,
                negated,
            }),
            Some(Token::IsNull) => Ok(FilterExpr::IsNull { field, negated }),
            Some(Token::ContainAll) => Ok(FilterExpr::ContainAll {
                field,
                values: self.parse_list()?,
                negated,
            }),
            Some(Token::HasPrefix) => Ok(FilterExpr::HasPrefix {
                field,
                prefix: self.parse_string_literal("has_prefix argument must be a string")?,
                negated,
            }),
            Some(Token::HasSuffix) => Ok(FilterExpr::HasSuffix {
                field,
                suffix: self.parse_string_literal("has_suffix argument must be a string")?,
                negated,
            }),
            other => Err(invalid(format!(
                "expected comparison/in/like/is_null/contain_all/has_prefix/has_suffix operator, got {other:?}"
            ))),
        }
    }

    fn parse_list(&mut self) -> Result<Vec<Literal>> {
        self.expect(&Token::LBracket)?;
        let mut values = Vec::new();
        if !matches!(self.peek(), Some(Token::RBracket)) {
            loop {
                values.push(self.parse_literal()?);
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.next();
                } else {
                    break;
                }
            }
        }
        self.expect(&Token::RBracket)?;
        Ok(values)
    }

    fn parse_string_literal(&mut self, message: &str) -> Result<String> {
        match self.parse_literal()? {
            Literal::Str(text) => Ok(text),
            _ => Err(invalid(message)),
        }
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        match self.next() {
            Some(Token::Str(text)) => Ok(Literal::Str(text)),
            Some(Token::Num(number)) => Ok(Literal::Num(number)),
            Some(Token::True) => Ok(Literal::Bool(true)),
            Some(Token::False) => Ok(Literal::Bool(false)),
            Some(Token::Null) => Ok(Literal::Null),
            other => Err(invalid(format!("expected literal, got {other:?}"))),
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        match self.next() {
            Some(token) if &token == expected => Ok(()),
            other => Err(invalid(format!("expected {expected:?}, got {other:?}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_filter, FilterExpr};
    use crate::doc::Doc;

    fn document(id: &str, tag: &str, rank: i32) -> Doc {
        let mut doc = Doc::with_pk(id).expect("primary key");
        doc.add_string("tag", tag).expect("tag");
        doc.add_i32("rank", rank).expect("rank");
        doc.add_array_string("labels", &["red", "blue"])
            .expect("labels");
        doc
    }

    #[test]
    fn accepted_filters_select_the_same_documents() {
        let alpha = document("doc-alpha", "alpha", 2);
        let beta = document("doc-beta", "beta", 9);
        let cases = [
            ("tag == \"alpha\"", true, false),
            ("tag = 'alpha'", true, false),
            ("tag != 'alpha' and rank > 1", false, true),
            ("not tag == 'beta'", true, false),
            ("rank >= 2 and rank <= 2", true, false),
            ("tag in ['alpha', 'missing']", true, false),
            ("tag not in ['beta']", true, false),
            ("tag like 'a%'", true, false),
            ("id == 'doc-beta'", false, true),
            ("missing is_null", true, true),
            ("tag is_null", false, false),
            ("labels contain_all ['red']", true, true),
            ("tag has_prefix 'al'", true, false),
            ("tag has_suffix 'eta'", false, true),
        ];
        for (expression, alpha_hit, beta_hit) in cases {
            let filter = parse_filter(expression).expect(expression);
            assert_eq!(filter.matches(&alpha), alpha_hit, "{expression}");
            assert_eq!(filter.matches(&beta), beta_hit, "{expression}");
        }
        assert!(parse_filter("tag ==").is_err());
        let _typed: FilterExpr = parse_filter("rank < 3").expect("range");
    }
}
