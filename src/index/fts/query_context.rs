//! Indexed document evaluation for structured full-text expressions.

use super::{count_to_f64, FtsIndex};
use crate::text::{bm25_term_score, contains_ordered_phrase, FtsEvalContext};

pub(super) struct IndexedEvalContext<'a> {
    index: &'a FtsIndex,
    ordinal: u64,
    text: &'a str,
    tokens: Option<Vec<String>>,
    document_count: f64,
    average_length: f64,
}

impl<'a> IndexedEvalContext<'a> {
    pub(super) fn new(
        index: &'a FtsIndex,
        ordinal: u64,
        text: &'a str,
        document_count: f64,
        average_length: f64,
    ) -> Self {
        Self {
            index,
            ordinal,
            text,
            tokens: None,
            document_count,
            average_length,
        }
    }
}

impl FtsEvalContext for IndexedEvalContext<'_> {
    fn contains_term(&mut self, term: &str) -> bool {
        self.index
            .postings
            .get(term)
            .and_then(|posting| posting.get(self.ordinal))
            .is_some()
    }

    fn contains_phrase(&mut self, terms: &[String], slop: u32) -> bool {
        if self.tokens.is_none() {
            self.tokens = Some(self.index.tokenizer.tokenize(self.text));
        }
        self.tokens
            .as_ref()
            .is_some_and(|tokens| contains_ordered_phrase(tokens, terms, slop))
    }

    fn term_score(&mut self, term: &str) -> f64 {
        let Some(posting) = self.index.postings.get(term) else {
            return 0.0;
        };
        let Some(entry) = posting.get(self.ordinal) else {
            return 0.0;
        };
        bm25_term_score(
            f64::from(entry.frequency),
            count_to_f64(u64::try_from(posting.len()).unwrap_or(u64::MAX)),
            self.document_count,
            f64::from(entry.document_length),
            self.average_length,
        )
    }
}
