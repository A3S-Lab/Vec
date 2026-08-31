//! Persistent FTS term dictionary with a contiguous immutable base.

use super::posting_list::PostingList;
use crate::error::{Error, Result};
use im::OrdMap;
use std::cmp::Ordering;
use std::iter::Peekable;
use std::slice;
use std::sync::Arc;

const MIN_DELTA_COMPACTION: usize = 64;
const MAX_DELTA_COMPACTION: usize = 2_048;
const DELTA_COMPACTION_DIVISOR: usize = 8;

/// Term lookups binary-search the sorted contiguous base and consult only the
/// small persistent change map. Index generations therefore share the large
/// dictionary while document mutations copy only changed term paths.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct TermDictionary {
    base: Arc<Vec<(String, Arc<PostingList>)>>,
    changes: OrdMap<String, Option<Arc<PostingList>>>,
    live_len: usize,
}

pub(super) enum TermIter<'a> {
    Base(slice::Iter<'a, (String, Arc<PostingList>)>),
    Merged(MergedTermIter<'a>),
}

pub(super) struct MergedTermIter<'a> {
    base: Peekable<slice::Iter<'a, (String, Arc<PostingList>)>>,
    changes: Peekable<im::ordmap::Iter<'a, String, Option<Arc<PostingList>>>>,
}

impl TermDictionary {
    pub(super) fn from_sorted_entries(
        entries: impl IntoIterator<Item = (String, Arc<PostingList>)>,
    ) -> Self {
        let base: Vec<_> = entries.into_iter().collect();
        debug_assert!(base.windows(2).all(|pair| pair[0].0 < pair[1].0));
        Self {
            live_len: base.len(),
            base: Arc::new(base),
            changes: OrdMap::new(),
        }
    }

    pub(super) fn get(&self, term: &str) -> Option<&Arc<PostingList>> {
        if let Some(posting) = self.changes.get(term) {
            return posting.as_ref();
        }
        self.base_entry(term)
    }

    pub(super) fn insert(&mut self, term: String, posting: Arc<PostingList>) -> Result<()> {
        let existed = self.get(&term).is_some();
        let live_len = if existed {
            self.live_len
        } else {
            self.live_len
                .checked_add(1)
                .ok_or_else(|| Error::resource_exhausted("FTS term count overflow"))?
        };
        self.changes.insert(term, Some(posting));
        self.live_len = live_len;
        Ok(())
    }

    pub(super) fn remove(&mut self, term: &str) -> bool {
        if self.get(term).is_none() {
            return false;
        }
        if self.base_entry(term).is_some() {
            self.changes.insert(term.to_string(), None);
        } else {
            self.changes.remove(term);
        }
        self.live_len -= 1;
        true
    }

    /// Defers compaction until the full document batch has been applied, so a
    /// large mutation folds the dictionary at most once.
    pub(super) fn finish_changes(&mut self) {
        if self.changes.len() < compaction_limit(self.base.len()) {
            return;
        }
        if self.base.is_empty() {
            let changes = std::mem::take(&mut self.changes);
            self.base = Arc::new(
                changes
                    .into_iter()
                    .filter_map(|(term, posting)| posting.map(|posting| (term, posting)))
                    .collect(),
            );
        } else {
            self.base = Arc::new(
                self.iter()
                    .map(|(term, posting)| (term.to_string(), Arc::clone(posting)))
                    .collect(),
            );
            self.changes = OrdMap::new();
        }
        debug_assert_eq!(self.live_len, self.base.len());
    }

    pub(super) fn iter(&self) -> TermIter<'_> {
        if self.changes.is_empty() {
            TermIter::Base(self.base.iter())
        } else {
            TermIter::Merged(MergedTermIter {
                base: self.base.iter().peekable(),
                changes: self.changes.iter().peekable(),
            })
        }
    }

    pub(super) fn validates(&self) -> bool {
        if self.changes.len() >= compaction_limit(self.base.len())
            || !self.base.windows(2).all(|pair| pair[0].0 < pair[1].0)
            || self.changes.iter().any(|(term, posting)| {
                term.is_empty() || (posting.is_none() && self.base_entry(term).is_none())
            })
        {
            return false;
        }
        let mut count = 0_usize;
        let mut previous: Option<&str> = None;
        for (term, _) in self.iter() {
            if term.is_empty() || previous.is_some_and(|previous| previous >= term) {
                return false;
            }
            previous = Some(term);
            count = match count.checked_add(1) {
                Some(count) => count,
                None => return false,
            };
        }
        count == self.live_len
    }

    fn base_entry(&self, term: &str) -> Option<&Arc<PostingList>> {
        self.base
            .binary_search_by(|(candidate, _)| candidate.as_str().cmp(term))
            .ok()
            .map(|position| &self.base[position].1)
    }

    #[cfg(test)]
    fn base_ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.base, &other.base)
    }

    #[cfg(test)]
    fn change_len(&self) -> usize {
        self.changes.len()
    }
}

impl<'a> Iterator for TermIter<'a> {
    type Item = (&'a str, &'a Arc<PostingList>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Base(base) => base.next().map(|(term, posting)| (term.as_str(), posting)),
            Self::Merged(merged) => merged.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Base(base) => base.size_hint(),
            Self::Merged(merged) => merged.size_hint(),
        }
    }
}

impl<'a> Iterator for MergedTermIter<'a> {
    type Item = (&'a str, &'a Arc<PostingList>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let base_term = self.base.peek().map(|(term, _)| term.as_str());
            let change_term = self.changes.peek().map(|(term, _)| term.as_str());
            match (base_term, change_term) {
                (None, None) => return None,
                (Some(_), None) => {
                    return self
                        .base
                        .next()
                        .map(|(term, posting)| (term.as_str(), posting));
                }
                (None, Some(_)) => {
                    let (term, posting) = self.changes.next()?;
                    if let Some(posting) = posting {
                        return Some((term.as_str(), posting));
                    }
                }
                (Some(base_term), Some(change_term)) => match base_term.cmp(change_term) {
                    Ordering::Less => {
                        return self
                            .base
                            .next()
                            .map(|(term, posting)| (term.as_str(), posting));
                    }
                    Ordering::Greater => {
                        let (term, posting) = self.changes.next()?;
                        if let Some(posting) = posting {
                            return Some((term.as_str(), posting));
                        }
                    }
                    Ordering::Equal => {
                        self.base.next();
                        let (term, posting) = self.changes.next()?;
                        if let Some(posting) = posting {
                            return Some((term.as_str(), posting));
                        }
                    }
                },
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.base.len().saturating_add(self.changes.len())))
    }
}

fn compaction_limit(base_len: usize) -> usize {
    let fractional =
        base_len.saturating_add(DELTA_COMPACTION_DIVISOR - 1) / DELTA_COMPACTION_DIVISOR;
    fractional.clamp(MIN_DELTA_COMPACTION, MAX_DELTA_COMPACTION)
}

#[cfg(test)]
mod tests {
    use super::{TermDictionary, MIN_DELTA_COMPACTION};
    use crate::index::fts::posting_list::PostingList;
    use crate::index::fts::PostingEntry;
    use std::sync::Arc;

    fn posting(ordinal: u64) -> Arc<PostingList> {
        Arc::new(PostingList::single(
            ordinal,
            PostingEntry {
                frequency: 1,
                document_length: 1,
            },
        ))
    }

    #[test]
    fn cloned_dictionaries_share_the_base_and_merge_changes_in_term_order() {
        let original = TermDictionary::from_sorted_entries([
            ("alpha".to_string(), posting(0)),
            ("gamma".to_string(), posting(1)),
        ]);
        let mut changed = original.clone();
        assert!(original.base_ptr_eq(&changed));

        changed
            .insert("beta".to_string(), posting(2))
            .expect("insert must succeed");
        changed
            .insert("gamma".to_string(), posting(3))
            .expect("replace must succeed");
        assert!(changed.remove("alpha"));
        assert!(!changed.remove("missing"));

        assert!(original.get("alpha").is_some());
        assert!(changed.get("alpha").is_none());
        assert!(changed.get("beta").is_some());
        let terms: Vec<_> = changed.iter().map(|(term, _)| term).collect();
        assert_eq!(terms, vec!["beta", "gamma"]);
    }

    #[test]
    fn a_large_batch_compacts_once_after_changes_finish() {
        let mut dictionary = TermDictionary::from_sorted_entries([]);
        for ordinal in 0..u64::try_from(MIN_DELTA_COMPACTION).expect("limit fits u64") {
            dictionary
                .insert(format!("term-{ordinal:03}"), posting(ordinal))
                .expect("insert must succeed");
        }
        assert_eq!(dictionary.change_len(), MIN_DELTA_COMPACTION);
        dictionary.finish_changes();

        assert_eq!(dictionary.change_len(), 0);
        assert_eq!(dictionary.base.len(), MIN_DELTA_COMPACTION);
        assert!(dictionary.validates());
    }
}
