//! Ordered, serializable full-text token filters.

use crate::error::{Error, Result};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum TokenFilter {
    Lowercase,
    AsciiFolding,
    Stemmer(StemmerAlgorithm),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum StemmerAlgorithm {
    Arabic,
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

impl TokenFilter {
    pub(super) fn apply(self, tokens: Vec<String>) -> Vec<String> {
        match self {
            Self::Lowercase => tokens
                .into_iter()
                .map(|token| token.to_lowercase())
                .collect(),
            Self::AsciiFolding => tokens
                .into_iter()
                .map(|token| fold_ascii(&token))
                .filter(|token| !token.is_empty())
                .collect(),
            Self::Stemmer(algorithm) => {
                let stemmer = Stemmer::create(algorithm.into());
                tokens
                    .into_iter()
                    .map(|token| stemmer.stem(&token).into_owned())
                    .collect()
            }
        }
    }
}

impl StemmerAlgorithm {
    pub(super) fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "arabic" => Ok(Self::Arabic),
            "danish" => Ok(Self::Danish),
            "dutch" => Ok(Self::Dutch),
            "english" => Ok(Self::English),
            "finnish" => Ok(Self::Finnish),
            "french" => Ok(Self::French),
            "german" => Ok(Self::German),
            "greek" => Ok(Self::Greek),
            "hungarian" => Ok(Self::Hungarian),
            "italian" => Ok(Self::Italian),
            "norwegian" => Ok(Self::Norwegian),
            "portuguese" => Ok(Self::Portuguese),
            "romanian" => Ok(Self::Romanian),
            "russian" => Ok(Self::Russian),
            "spanish" => Ok(Self::Spanish),
            "swedish" => Ok(Self::Swedish),
            "tamil" => Ok(Self::Tamil),
            "turkish" => Ok(Self::Turkish),
            _ => Err(Error::invalid_argument(format!(
                "unsupported FTS stemmer language '{value}'"
            ))),
        }
    }
}

impl From<StemmerAlgorithm> for Algorithm {
    fn from(value: StemmerAlgorithm) -> Self {
        match value {
            StemmerAlgorithm::Arabic => Self::Arabic,
            StemmerAlgorithm::Danish => Self::Danish,
            StemmerAlgorithm::Dutch => Self::Dutch,
            StemmerAlgorithm::English => Self::English,
            StemmerAlgorithm::Finnish => Self::Finnish,
            StemmerAlgorithm::French => Self::French,
            StemmerAlgorithm::German => Self::German,
            StemmerAlgorithm::Greek => Self::Greek,
            StemmerAlgorithm::Hungarian => Self::Hungarian,
            StemmerAlgorithm::Italian => Self::Italian,
            StemmerAlgorithm::Norwegian => Self::Norwegian,
            StemmerAlgorithm::Portuguese => Self::Portuguese,
            StemmerAlgorithm::Romanian => Self::Romanian,
            StemmerAlgorithm::Russian => Self::Russian,
            StemmerAlgorithm::Spanish => Self::Spanish,
            StemmerAlgorithm::Swedish => Self::Swedish,
            StemmerAlgorithm::Tamil => Self::Tamil,
            StemmerAlgorithm::Turkish => Self::Turkish,
        }
    }
}

fn fold_ascii(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for character in input.nfkd() {
        if is_combining_mark(character) {
            continue;
        }
        if let Some(replacement) = special_fold(character) {
            output.push_str(replacement);
        } else {
            output.push(character);
        }
    }
    output
}

fn special_fold(character: char) -> Option<&'static str> {
    match character {
        'Æ' => Some("AE"),
        'æ' => Some("ae"),
        'Ð' | 'Đ' => Some("D"),
        'ð' | 'đ' => Some("d"),
        'Ø' => Some("O"),
        'ø' => Some("o"),
        'Þ' => Some("TH"),
        'þ' => Some("th"),
        'ß' => Some("ss"),
        'ẞ' => Some("SS"),
        'Ħ' => Some("H"),
        'ħ' => Some("h"),
        'ı' => Some("i"),
        'Ĳ' => Some("IJ"),
        'ĳ' => Some("ij"),
        'Ł' => Some("L"),
        'ł' => Some("l"),
        'Ŋ' => Some("N"),
        'ŋ' => Some("n"),
        'Œ' => Some("OE"),
        'œ' => Some("oe"),
        'Ŧ' => Some("T"),
        'ŧ' => Some("t"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{StemmerAlgorithm, TokenFilter};

    #[test]
    fn filters_apply_in_place_without_changing_token_cardinality() {
        assert_eq!(
            TokenFilter::Lowercase.apply(vec!["RÉSUMÉ".into()]),
            ["résumé"]
        );
        assert_eq!(
            TokenFilter::AsciiFolding.apply(vec!["café".into(), "Straße".into()]),
            ["cafe", "Strasse"]
        );
        assert_eq!(
            TokenFilter::Stemmer(StemmerAlgorithm::English)
                .apply(vec!["running".into(), "repositories".into()]),
            ["run", "repositori"]
        );
    }
}
