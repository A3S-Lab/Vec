//! Shared full-text tokenization, query parsing, and BM25 primitives.

mod filters;
mod query_expression;

pub(crate) use query_expression::{
    parse_fts_query, FtsEvalContext, FtsExpr, FtsExprKind, FtsModifier, ParsedFtsQuery,
};

use crate::doc::{Doc, FieldValue};
use crate::error::{Error, Result};
use crate::schema::IndexParams;
use filters::{StemmerAlgorithm, TokenFilter};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use unicode_general_category::{get_general_category, GeneralCategory};

const DEFAULT_NGRAM_LENGTH: u32 = 2;
const DEFAULT_STANDARD_MAX_TOKEN_LENGTH: u32 = 255;
const MAX_STANDARD_TOKEN_LENGTH: u32 = 1_048_576;
const MAX_NGRAM_RANGE: u32 = 1;
const MAX_INITIAL_TOKEN_CAPACITY: usize = 4_096;
const TOKEN_CHAR_LETTER: u8 = 1 << 0;
const TOKEN_CHAR_DIGIT: u8 = 1 << 1;
const TOKEN_CHAR_WHITESPACE: u8 = 1 << 2;
const TOKEN_CHAR_PUNCTUATION: u8 = 1 << 3;
const TOKEN_CHAR_SYMBOL: u8 = 1 << 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Tokenizer {
    name: String,
    ngram: Option<NgramConfig>,
    standard_max_token_length: Option<u32>,
    filters: Vec<TokenFilter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct NgramConfig {
    min: u32,
    max: u32,
    token_char_mask: u8,
}

impl Tokenizer {
    pub(crate) fn from_index_params(index_params: Option<&IndexParams>) -> Result<Self> {
        Self::from_index_params_with_availability(index_params, true)
    }

    fn from_index_params_with_availability(
        index_params: Option<&IndexParams>,
        require_available: bool,
    ) -> Result<Self> {
        let name = tokenizer_name(index_params)?;
        let extra_params = parse_extra_params(index_params)?;
        let filters = parse_filters(index_params, &extra_params)?;
        let has_stemmer = filters
            .iter()
            .any(|filter| matches!(filter, TokenFilter::Stemmer(_)));
        match name {
            "standard" => {
                validate_extra_parameter_names(
                    &extra_params,
                    if has_stemmer {
                        &["max_token_length", "stemmer_lang"]
                    } else {
                        &["max_token_length"]
                    },
                )?;
                Ok(Self {
                    name: name.to_string(),
                    ngram: None,
                    standard_max_token_length: Some(standard_max_token_length(&extra_params)?),
                    filters,
                })
            }
            "whitespace" => {
                validate_extra_parameter_names(
                    &extra_params,
                    if has_stemmer { &["stemmer_lang"] } else { &[] },
                )?;
                Ok(Self {
                    name: name.to_string(),
                    ngram: None,
                    standard_max_token_length: None,
                    filters,
                })
            }
            "ngram" => {
                validate_extra_parameter_names(
                    &extra_params,
                    if has_stemmer {
                        &["ngram_min", "ngram_max", "token_chars", "stemmer_lang"]
                    } else {
                        &["ngram_min", "ngram_max", "token_chars"]
                    },
                )?;
                Ok(Self {
                    name: name.to_string(),
                    ngram: Some(parse_ngram_config(&extra_params)?),
                    standard_max_token_length: None,
                    filters,
                })
            }
            "jieba" | "jieba_accurate" => {
                validate_extra_parameter_names(
                    &extra_params,
                    if has_stemmer { &["stemmer_lang"] } else { &[] },
                )?;
                if require_available && !cfg!(feature = "jieba") {
                    return Err(Error::not_supported(
                        "jieba tokenizer requires the optional 'jieba' feature",
                    ));
                }
                Ok(Self {
                    name: name.to_string(),
                    ngram: None,
                    standard_max_token_length: None,
                    filters,
                })
            }
            _ => Err(Error::invalid_argument(format!(
                "unknown FTS tokenizer '{name}'"
            ))),
        }
    }

    pub(crate) fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = if let Some(config) = self.ngram {
            tokenize_ngram(text, config)
        } else if let Some(max_token_length) = self.standard_max_token_length {
            tokenize_standard(text, max_token_length)
        } else {
            zvec_core::engine::fts::tokenize_with(text, &self.name)
        };
        for filter in &self.filters {
            tokens = filter.apply(tokens);
        }
        tokens
    }
}

fn tokenizer_name(index_params: Option<&IndexParams>) -> Result<&str> {
    index_params
        .and_then(|params| params.params.get("tokenizer_name"))
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| Error::invalid_argument("FTS tokenizer_name must be a string"))
        })
        .transpose()
        .map(|name| name.unwrap_or("standard"))
}

pub(crate) fn validate_tokenizer_params(index_params: Option<&IndexParams>) -> Result<()> {
    Tokenizer::from_index_params_with_availability(index_params, false).map(|_| ())
}

fn parse_extra_params(index_params: Option<&IndexParams>) -> Result<Map<String, Value>> {
    let Some(value) = index_params.and_then(|params| params.params.get("extra_params")) else {
        return Ok(Map::new());
    };
    let extra_params = value
        .as_str()
        .ok_or_else(|| Error::invalid_argument("FTS extra_params parameter must be a string"))?;
    if extra_params.trim().is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_str::<Value>(extra_params)
        .map_err(|error| {
            Error::invalid_argument(format!("FTS extra_params must be valid JSON: {error}"))
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| Error::invalid_argument("FTS extra_params must be a JSON object"))
}

fn parse_filters(
    index_params: Option<&IndexParams>,
    extra_params: &Map<String, Value>,
) -> Result<Vec<TokenFilter>> {
    let values = match index_params.and_then(|params| params.params.get("filters")) {
        Some(value) => value
            .as_array()
            .ok_or_else(|| Error::invalid_argument("FTS filters parameter must be an array"))?,
        None => return Ok(vec![TokenFilter::Lowercase]),
    };
    values
        .iter()
        .map(|value| {
            let name = value
                .as_str()
                .ok_or_else(|| Error::invalid_argument("FTS filters entries must be strings"))?;
            match name {
                "lowercase" => Ok(TokenFilter::Lowercase),
                "ascii_folding" => Ok(TokenFilter::AsciiFolding),
                "stemmer" => {
                    let language = extra_params
                        .get("stemmer_lang")
                        .map(|value| {
                            value
                                .as_str()
                                .filter(|value| !value.is_empty())
                                .ok_or_else(|| {
                                    Error::invalid_argument(
                                        "FTS stemmer_lang must be a non-empty string",
                                    )
                                })
                        })
                        .transpose()?
                        .unwrap_or("english");
                    Ok(TokenFilter::Stemmer(StemmerAlgorithm::parse(language)?))
                }
                _ => Err(Error::invalid_argument(format!(
                    "unknown FTS token filter '{name}'"
                ))),
            }
        })
        .collect()
}

fn validate_extra_parameter_names(object: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(Error::invalid_argument(format!(
            "unsupported FTS extra_params key '{key}'"
        )));
    }
    Ok(())
}

fn standard_max_token_length(object: &Map<String, Value>) -> Result<u32> {
    let length = positive_u32_parameter(
        object,
        "max_token_length",
        DEFAULT_STANDARD_MAX_TOKEN_LENGTH,
    )?;
    if length > MAX_STANDARD_TOKEN_LENGTH {
        return Err(Error::invalid_argument(format!(
            "FTS max_token_length must be at most {MAX_STANDARD_TOKEN_LENGTH}"
        )));
    }
    Ok(length)
}

fn parse_ngram_config(object: &Map<String, Value>) -> Result<NgramConfig> {
    let min = positive_u32_parameter(object, "ngram_min", DEFAULT_NGRAM_LENGTH)?;
    let max = positive_u32_parameter(object, "ngram_max", DEFAULT_NGRAM_LENGTH)?;
    if min > max {
        return Err(Error::invalid_argument(
            "FTS ngram_min must be less than or equal to ngram_max",
        ));
    }
    if max - min > MAX_NGRAM_RANGE {
        return Err(Error::invalid_argument(
            "FTS ngram_max minus ngram_min must be at most 1",
        ));
    }
    Ok(NgramConfig {
        min,
        max,
        token_char_mask: token_char_mask(object)?,
    })
}

fn tokenize_standard(text: &str, max_token_length: u32) -> Vec<String> {
    let max_token_length = usize::try_from(max_token_length).unwrap_or(usize::MAX);
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .flat_map(|token| split_standard_token(token, max_token_length))
        .collect()
}

fn split_standard_token(token: &str, max_token_length: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut characters = 0;
    for (offset, _) in token.char_indices() {
        if characters == max_token_length {
            chunks.push(token[start..offset].to_string());
            start = offset;
            characters = 0;
        }
        characters += 1;
    }
    if start < token.len() {
        chunks.push(token[start..].to_string());
    }
    chunks
}

fn positive_u32_parameter(object: &Map<String, Value>, key: &str, default: u32) -> Result<u32> {
    let Some(value) = object.get(key) else {
        return Ok(default);
    };
    let value = value
        .as_u64()
        .ok_or_else(|| Error::invalid_argument(format!("FTS {key} must be a positive integer")))?;
    u32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::invalid_argument(format!("FTS {key} must be a positive u32")))
}

fn token_char_mask(object: &Map<String, Value>) -> Result<u8> {
    let Some(value) = object.get("token_chars") else {
        return Ok(0);
    };
    let values = value
        .as_array()
        .ok_or_else(|| Error::invalid_argument("FTS ngram token_chars must be an array"))?;
    values.iter().try_fold(0_u8, |mask, value| {
        let name = value.as_str().ok_or_else(|| {
            Error::invalid_argument("FTS ngram token_chars entries must be strings")
        })?;
        let bit = match name {
            "letter" => TOKEN_CHAR_LETTER,
            "digit" => TOKEN_CHAR_DIGIT,
            "whitespace" => TOKEN_CHAR_WHITESPACE,
            "punctuation" => TOKEN_CHAR_PUNCTUATION,
            "symbol" => TOKEN_CHAR_SYMBOL,
            _ => {
                return Err(Error::invalid_argument(format!(
                    "unsupported FTS ngram token_chars value '{name}'"
                )))
            }
        };
        Ok(mask | bit)
    })
}

fn tokenize_ngram(text: &str, config: NgramConfig) -> Vec<String> {
    let estimated = (text.len() / 4 + 1).min(MAX_INITIAL_TOKEN_CAPACITY);
    let mut tokens = Vec::with_capacity(estimated);
    let mut segment = Vec::with_capacity(estimated);
    for (start, character) in text.char_indices() {
        let end = start + character.len_utf8();
        if is_ngram_token_char(character, config.token_char_mask) {
            segment.push((start, end));
        } else {
            emit_ngram_segment(text, &segment, config, &mut tokens);
            segment.clear();
        }
    }
    emit_ngram_segment(text, &segment, config, &mut tokens);
    tokens
}

fn emit_ngram_segment(
    text: &str,
    segment: &[(usize, usize)],
    config: NgramConfig,
    tokens: &mut Vec<String>,
) {
    let Ok(min) = usize::try_from(config.min) else {
        return;
    };
    let Ok(max) = usize::try_from(config.max) else {
        return;
    };
    if segment.len() < min {
        return;
    }
    for start in 0..segment.len() {
        let upper = max.min(segment.len() - start);
        for length in min..=upper {
            let first = segment[start].0;
            let last = segment[start + length - 1].1;
            tokens.push(text[first..last].to_string());
        }
    }
}

fn is_ngram_token_char(character: char, mask: u8) -> bool {
    mask == 0 || ngram_token_char_bit(character) & mask != 0
}

fn ngram_token_char_bit(character: char) -> u8 {
    let category = get_general_category(character);
    if is_ngram_whitespace(character, category) {
        return TOKEN_CHAR_WHITESPACE;
    }
    match category {
        GeneralCategory::DecimalNumber => TOKEN_CHAR_DIGIT,
        GeneralCategory::UppercaseLetter
        | GeneralCategory::LowercaseLetter
        | GeneralCategory::TitlecaseLetter
        | GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter => TOKEN_CHAR_LETTER,
        GeneralCategory::ConnectorPunctuation
        | GeneralCategory::DashPunctuation
        | GeneralCategory::OpenPunctuation
        | GeneralCategory::ClosePunctuation
        | GeneralCategory::InitialPunctuation
        | GeneralCategory::FinalPunctuation
        | GeneralCategory::OtherPunctuation => TOKEN_CHAR_PUNCTUATION,
        GeneralCategory::MathSymbol
        | GeneralCategory::CurrencySymbol
        | GeneralCategory::ModifierSymbol
        | GeneralCategory::OtherSymbol => TOKEN_CHAR_SYMBOL,
        _ => 0,
    }
}

fn is_ngram_whitespace(character: char, category: GeneralCategory) -> bool {
    let codepoint = u32::from(character);
    if (0x0009..=0x000d).contains(&codepoint) || (0x001c..=0x001f).contains(&codepoint) {
        return true;
    }
    match category {
        GeneralCategory::LineSeparator | GeneralCategory::ParagraphSeparator => true,
        GeneralCategory::SpaceSeparator => !matches!(codepoint, 0x00a0 | 0x2007 | 0x202f),
        _ => false,
    }
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            min: DEFAULT_NGRAM_LENGTH,
            max: DEFAULT_NGRAM_LENGTH,
            token_char_mask: 0,
        }
    }
}

pub(crate) fn text_value<'a>(doc: &'a Doc, field: &str) -> Option<&'a str> {
    match doc.field(field) {
        Some(FieldValue::String(value) | FieldValue::Json(Value::String(value))) => {
            Some(value.as_str())
        }
        _ => None,
    }
}

pub(crate) fn bm25_term_score(
    frequency: f64,
    document_frequency: f64,
    document_count: f64,
    document_length: f64,
    average_length: f64,
) -> f64 {
    if frequency <= 0.0 || document_count <= 0.0 {
        return 0.0;
    }
    let normalized_length = if average_length == 0.0 {
        0.0
    } else {
        document_length / average_length
    };
    let idf = ((document_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).ln();
    let denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * normalized_length);
    idf * (frequency * 2.2 / denominator.max(1e-12))
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;
    use crate::error::ErrorCode;
    use crate::schema::IndexParams;

    fn tokenizer(extra_params: Option<&str>) -> Tokenizer {
        let params =
            IndexParams::fts(Some("ngram"), None, extra_params).expect("FTS params must be valid");
        Tokenizer::from_index_params(Some(&params)).expect("tokenizer must be valid")
    }

    #[test]
    fn ngram_defaults_to_unicode_bigrams() {
        let tokenizer = tokenizer(None);
        assert_eq!(tokenizer.tokenize("hello"), ["he", "el", "ll", "lo"]);
        assert_eq!(tokenizer.tokenize("中文分词"), ["中文", "文分", "分词"]);
        assert_eq!(
            tokenizer.tokenize("foobar未跟踪文件"),
            ["fo", "oo", "ob", "ba", "ar", "r未", "未跟", "跟踪", "踪文", "文件"]
        );
    }

    #[test]
    fn ngram_range_and_character_classes_match_the_public_contract() {
        let range = tokenizer(Some(r#"{"ngram_min":2,"ngram_max":3}"#));
        assert_eq!(
            range.tokenize("hello"),
            ["he", "hel", "el", "ell", "ll", "llo", "lo"]
        );

        let letters = tokenizer(Some(
            r#"{"ngram_min":2,"ngram_max":2,"token_chars":["letter","digit"]}"#,
        ));
        assert_eq!(
            letters.tokenize("ab cd,中文!de"),
            ["ab", "cd", "中文", "de"]
        );

        let separators = tokenizer(Some(
            r#"{"ngram_min":1,"ngram_max":1,"token_chars":["whitespace"]}"#,
        ));
        assert_eq!(
            separators.tokenize("\t \n\u{2028}\u{00a0}"),
            ["\t", " ", "\n", "\u{2028}"]
        );
    }

    #[test]
    fn standard_tokenizer_honors_unicode_character_limits_before_filters() {
        let params = IndexParams::fts(
            Some("standard"),
            Some(&[]),
            Some(r#"{"max_token_length":3}"#),
        )
        .expect("FTS params must be valid");
        let tokenizer =
            Tokenizer::from_index_params(Some(&params)).expect("tokenizer must be valid");

        assert_eq!(
            tokenizer.tokenize("abcdef 中文测试"),
            ["abc", "def", "中文测", "试"]
        );
    }

    #[test]
    fn standard_tokenizer_rejects_out_of_range_character_limits() {
        for extra_params in [
            r#"{"max_token_length":0}"#,
            r#"{"max_token_length":1048577}"#,
        ] {
            let params = IndexParams::fts(Some("standard"), None, Some(extra_params))
                .expect("FTS params must be syntactically valid");
            let error = Tokenizer::from_index_params(Some(&params))
                .expect_err("invalid token length must fail");
            assert_eq!(error.code, ErrorCode::InvalidArgument);
        }
    }
}
