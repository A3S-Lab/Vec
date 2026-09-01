//! Tokenization for structured full-text query strings.

use crate::error::{Error, Result};
use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RawWordPart {
    Literal(String),
    AnyOne,
    AnyMany,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RawWord {
    pub(super) parts: Vec<RawWordPart>,
    pub(super) had_escape: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Token {
    And,
    Or,
    Not,
    Plus,
    Minus,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Caret,
    Tilde,
    Word(RawWord),
    Phrase(String),
}

impl RawWord {
    pub(super) fn plain(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [RawWordPart::Literal(value)] => Some(value),
            _ => None,
        }
    }

    pub(super) fn is_unbounded(&self) -> bool {
        self.parts == [RawWordPart::AnyMany]
    }

    pub(super) fn has_wildcard(&self) -> bool {
        self.parts
            .iter()
            .any(|part| !matches!(part, RawWordPart::Literal(_)))
    }
}

pub(super) fn lex(expression: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut characters = expression.chars().peekable();
    while let Some(character) = characters.peek().copied() {
        if character.is_whitespace() {
            characters.next();
            continue;
        }
        let token = match character {
            '+' => single(&mut characters, Token::Plus),
            '-' => single(&mut characters, Token::Minus),
            '(' => single(&mut characters, Token::LeftParen),
            ')' => single(&mut characters, Token::RightParen),
            '[' => single(&mut characters, Token::LeftBracket),
            ']' => single(&mut characters, Token::RightBracket),
            '{' => single(&mut characters, Token::LeftBrace),
            '}' => single(&mut characters, Token::RightBrace),
            ':' => single(&mut characters, Token::Colon),
            '^' => single(&mut characters, Token::Caret),
            '~' => single(&mut characters, Token::Tilde),
            '"' | '\'' => Token::Phrase(read_phrase(&mut characters, character)?),
            _ => read_word(&mut characters)?,
        };
        tokens.push(token);
    }
    Ok(tokens)
}

fn single(characters: &mut Peekable<Chars<'_>>, token: Token) -> Token {
    characters.next();
    token
}

fn read_phrase(characters: &mut Peekable<Chars<'_>>, quote: char) -> Result<String> {
    if characters.next() != Some(quote) {
        return Err(Error::internal("FTS phrase lexer lost its opening quote"));
    }
    let mut phrase = String::new();
    while let Some(character) = characters.next() {
        match character {
            character if character == quote => return Ok(phrase),
            '\\' => {
                let escaped = characters
                    .next()
                    .ok_or_else(|| Error::invalid_argument("dangling escape in FTS phrase"))?;
                phrase.push(escaped);
            }
            '\n' | '\r' => {
                return Err(Error::invalid_argument(
                    "FTS phrase cannot contain a line break",
                ));
            }
            _ => phrase.push(character),
        }
    }
    Err(Error::invalid_argument("unclosed phrase in FTS query"))
}

fn read_word(characters: &mut Peekable<Chars<'_>>) -> Result<Token> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut had_escape = false;
    while let Some(character) = characters.peek().copied() {
        if character.is_whitespace()
            || matches!(
                character,
                '(' | ')' | '[' | ']' | '{' | '}' | ':' | '^' | '~' | '"' | '\'' | '+' | '-'
            )
        {
            break;
        }
        characters.next();
        match character {
            '\\' => {
                let escaped = characters
                    .next()
                    .ok_or_else(|| Error::invalid_argument("dangling escape in FTS term"))?;
                literal.push(escaped);
                had_escape = true;
            }
            '*' | '?' => {
                if !literal.is_empty() {
                    parts.push(RawWordPart::Literal(std::mem::take(&mut literal)));
                }
                parts.push(if character == '*' {
                    RawWordPart::AnyMany
                } else {
                    RawWordPart::AnyOne
                });
            }
            '&' | '|' => {
                return Err(Error::not_supported(format!(
                    "unsupported FTS query syntax '{character}'"
                )));
            }
            _ => literal.push(character),
        }
    }
    if !literal.is_empty() {
        parts.push(RawWordPart::Literal(literal));
    }
    if parts.is_empty() {
        return Err(Error::invalid_argument("empty term in FTS query"));
    }
    let word = RawWord { parts, had_escape };
    if !word.had_escape && !word.has_wildcard() {
        match word
            .plain()
            .unwrap_or_default()
            .to_ascii_uppercase()
            .as_str()
        {
            "AND" => return Ok(Token::And),
            "OR" => return Ok(Token::Or),
            "NOT" => return Ok(Token::Not),
            _ => {}
        }
    }
    Ok(Token::Word(word))
}
