//! Character-trigram → term posting for FTS matcher expansion pruning.
//!
//! Wildcard leaves with a literal run of length ≥ 3 must contain those
//! overlapping trigrams. Intersecting the inverted lists yields a recall-safe
//! candidate set before the full pattern matcher runs. Patterns without a long
//! enough literal run fall back to a full vocabulary scan.

use std::collections::{BTreeMap, BTreeSet};

/// Maps each character trigram to the terms that contain it.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct TrigramTermIndex {
    postings: BTreeMap<String, BTreeSet<String>>,
}

impl TrigramTermIndex {
    pub(super) fn from_terms<'a>(terms: impl IntoIterator<Item = &'a str>) -> Self {
        let mut index = Self::default();
        for term in terms {
            index.insert_term(term);
        }
        index
    }

    pub(super) fn insert_term(&mut self, term: &str) {
        for gram in char_trigrams(term) {
            self.postings
                .entry(gram)
                .or_default()
                .insert(term.to_string());
        }
    }

    pub(super) fn remove_term(&mut self, term: &str) {
        for gram in char_trigrams(term) {
            let Some(terms) = self.postings.get_mut(&gram) else {
                continue;
            };
            terms.remove(term);
            if terms.is_empty() {
                self.postings.remove(&gram);
            }
        }
    }

    /// Terms that contain every required trigram (AND). Empty input yields empty.
    pub(super) fn terms_containing_all(&self, required: &[String]) -> BTreeSet<String> {
        let mut required = required.iter();
        let Some(first) = required.next() else {
            return BTreeSet::new();
        };
        let mut candidates = self.postings.get(first).cloned().unwrap_or_default();
        for gram in required {
            let Some(terms) = self.postings.get(gram) else {
                return BTreeSet::new();
            };
            candidates.retain(|term| terms.contains(term));
            if candidates.is_empty() {
                break;
            }
        }
        candidates
    }

    /// Terms that share at least `minimum` trigrams with `query_grams` (fuzzy prefilter).
    pub(super) fn terms_sharing_at_least(
        &self,
        query_grams: &[String],
        minimum: usize,
    ) -> BTreeSet<String> {
        if minimum == 0 || query_grams.is_empty() {
            return BTreeSet::new();
        }
        let mut counts = BTreeMap::<String, usize>::new();
        for gram in query_grams {
            let Some(terms) = self.postings.get(gram) else {
                continue;
            };
            for term in terms {
                *counts.entry(term.clone()).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter(|(_, count)| *count >= minimum)
            .map(|(term, _)| term)
            .collect()
    }
}

pub(super) fn char_trigrams(term: &str) -> Vec<String> {
    let chars: Vec<char> = term.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    chars
        .windows(3)
        .map(|window| window.iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{char_trigrams, TrigramTermIndex};

    #[test]
    fn char_trigrams_are_overlapping_and_unicode_aware() {
        assert_eq!(char_trigrams("ab"), Vec::<String>::new());
        assert_eq!(
            char_trigrams("rust"),
            vec!["rus".to_string(), "ust".to_string()]
        );
        assert_eq!(
            char_trigrams("\u{4e2d}\u{56fd}\u{6587}"),
            vec!["\u{4e2d}\u{56fd}\u{6587}".to_string()]
        );
    }

    #[test]
    fn terms_containing_all_intersects_postings() {
        let index = TrigramTermIndex::from_terms(["rust", "rusty", "crust", "python"]);
        let required = vec!["rus".into(), "ust".into()];
        let hits = index.terms_containing_all(&required);
        assert!(hits.contains("rust"));
        assert!(hits.contains("rusty"));
        assert!(hits.contains("crust"));
        assert!(!hits.contains("python"));
    }

    #[test]
    fn remove_term_drops_empty_trigram_postings() {
        let mut index = TrigramTermIndex::from_terms(["rust", "dust"]);
        index.remove_term("rust");
        assert!(index.terms_containing_all(&["rus".into()]).is_empty());
        assert!(index.terms_containing_all(&["ust".into()]).contains("dust"));
        index.remove_term("dust");
        assert!(index.postings.is_empty());
    }
}
