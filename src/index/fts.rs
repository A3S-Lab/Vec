//! Full-text token/posting helpers.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default)]
pub struct FtsIndex {
    pub field_name: String,
    pub tokenizer: String,
    pub postings: BTreeMap<String, BTreeSet<String>>,
    pub document_lengths: BTreeMap<String, usize>,
}
impl FtsIndex {
    pub fn new(field_name: impl Into<String>, tokenizer: impl Into<String>) -> Self { Self { field_name: field_name.into(), tokenizer: tokenizer.into(), ..Self::default() } }
    pub fn add(&mut self, id: impl Into<String>, text: &str) { let id = id.into(); let tokens = tokenize(text, &self.tokenizer); self.document_lengths.insert(id.clone(), tokens.len()); for token in tokens { self.postings.entry(token).or_default().insert(id.clone()); } }
    pub fn search(&self, terms: &[String]) -> BTreeSet<String> { terms.iter().flat_map(|term| self.postings.get(term).into_iter().flatten().cloned()).collect() }
}
pub fn tokenize(text: &str, tokenizer: &str) -> Vec<String> { zvec_core::engine::fts::tokenize_with(text, tokenizer) }

