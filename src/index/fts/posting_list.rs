//! Persistent full-text posting lists with a contiguous immutable base.

use super::document_lengths::DocumentLengths;
use super::PostingEntry;
use crate::error::{Error, Result};
use im::OrdMap;
use std::cmp::Ordering;
use std::iter::Peekable;
use std::slice;
use std::sync::Arc;

const MIN_DELTA_COMPACTION: usize = 64;
const MAX_DELTA_COMPACTION: usize = 2_048;
const DELTA_COMPACTION_DIVISOR: usize = 8;

/// Query traversal reads the sorted contiguous base and merges only the small
/// persistent change map. Cloned index generations share the base, while
/// insertions, replacements, and removals copy only changed delta paths.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct PostingList {
    base: Arc<Vec<(u64, PostingEntry)>>,
    changes: OrdMap<u64, Option<PostingEntry>>,
    live_len: usize,
}

pub(super) enum PostingIter<'a> {
    Base(slice::Iter<'a, (u64, PostingEntry)>),
    Merged(MergedPostingIter<'a>),
}

pub(super) struct MergedPostingIter<'a> {
    base: Peekable<slice::Iter<'a, (u64, PostingEntry)>>,
    changes: Peekable<im::ordmap::Iter<'a, u64, Option<PostingEntry>>>,
}

impl PostingList {
    pub(super) fn from_sorted_entries(
        entries: impl IntoIterator<Item = (u64, PostingEntry)>,
    ) -> Self {
        let base: Vec<_> = entries.into_iter().collect();
        debug_assert!(base.windows(2).all(|pair| pair[0].0 < pair[1].0));
        Self {
            live_len: base.len(),
            base: Arc::new(base),
            changes: OrdMap::new(),
        }
    }

    pub(super) fn single(ordinal: u64, entry: PostingEntry) -> Self {
        Self::from_sorted_entries([(ordinal, entry)])
    }

    pub(super) fn len(&self) -> usize {
        self.live_len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.live_len == 0
    }

    pub(super) fn insert(&mut self, ordinal: u64, entry: PostingEntry) -> Result<()> {
        let existed = self.get(ordinal).is_some();
        let live_len = if existed {
            self.live_len
        } else {
            self.live_len
                .checked_add(1)
                .ok_or_else(|| Error::resource_exhausted("FTS posting length overflow"))?
        };
        if self.base_entry(ordinal).copied() == Some(entry) {
            self.changes.remove(&ordinal);
        } else {
            self.changes.insert(ordinal, Some(entry));
        }
        self.live_len = live_len;
        self.compact_if_needed();
        Ok(())
    }

    pub(super) fn remove(&mut self, ordinal: u64) -> bool {
        if self.get(ordinal).is_none() {
            return false;
        }
        if self.base_entry(ordinal).is_some() {
            self.changes.insert(ordinal, None);
        } else {
            self.changes.remove(&ordinal);
        }
        self.live_len -= 1;
        self.compact_if_needed();
        true
    }

    pub(super) fn get(&self, ordinal: u64) -> Option<&PostingEntry> {
        if let Some(entry) = self.changes.get(&ordinal) {
            return entry.as_ref();
        }
        self.base_entry(ordinal)
    }

    pub(super) fn iter(&self) -> PostingIter<'_> {
        if self.changes.is_empty() {
            PostingIter::Base(self.base.iter())
        } else {
            PostingIter::Merged(MergedPostingIter {
                base: self.base.iter().peekable(),
                changes: self.changes.iter().peekable(),
            })
        }
    }

    pub(super) fn validates(&self, document_lengths: &DocumentLengths) -> bool {
        if self.changes.len() >= compaction_limit(self.base.len())
            || !self.base.windows(2).all(|pair| pair[0].0 < pair[1].0)
            || self.changes.iter().any(|(ordinal, changed)| {
                match (self.base_entry(*ordinal), changed.as_ref()) {
                    (None, None) => true,
                    (Some(base), Some(changed)) => base == changed,
                    (None, Some(_)) | (Some(_), None) => false,
                }
            })
        {
            return false;
        }
        let mut count = 0_usize;
        let mut previous = None;
        for (ordinal, entry) in self.iter() {
            if previous.is_some_and(|previous| previous >= ordinal)
                || entry.frequency == 0
                || document_lengths.get(ordinal) != Some(&entry.document_length)
                || entry.frequency > entry.document_length
            {
                return false;
            }
            previous = Some(ordinal);
            count = match count.checked_add(1) {
                Some(count) => count,
                None => return false,
            };
        }
        count == self.live_len
    }

    fn base_entry(&self, ordinal: u64) -> Option<&PostingEntry> {
        self.base
            .binary_search_by_key(&ordinal, |(ordinal, _)| *ordinal)
            .ok()
            .map(|position| &self.base[position].1)
    }

    fn compact_if_needed(&mut self) {
        if self.changes.len() < compaction_limit(self.base.len()) {
            return;
        }
        let base = self.iter().collect();
        self.base = Arc::new(base);
        self.changes = OrdMap::new();
        debug_assert_eq!(self.live_len, self.base.len());
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

impl Iterator for PostingIter<'_> {
    type Item = (u64, PostingEntry);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Base(base) => base.next().copied(),
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

impl Iterator for MergedPostingIter<'_> {
    type Item = (u64, PostingEntry);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let base_ordinal = self.base.peek().map(|(ordinal, _)| *ordinal);
            let change_ordinal = self.changes.peek().map(|(ordinal, _)| **ordinal);
            match (base_ordinal, change_ordinal) {
                (None, None) => return None,
                (Some(_), None) => return self.base.next().copied(),
                (None, Some(_)) => {
                    let (ordinal, entry) = self.changes.next()?;
                    if let Some(entry) = entry {
                        return Some((*ordinal, *entry));
                    }
                }
                (Some(base_ordinal), Some(change_ordinal)) => {
                    match base_ordinal.cmp(&change_ordinal) {
                        Ordering::Less => return self.base.next().copied(),
                        Ordering::Greater => {
                            let (ordinal, entry) = self.changes.next()?;
                            if let Some(entry) = entry {
                                return Some((*ordinal, *entry));
                            }
                        }
                        Ordering::Equal => {
                            self.base.next();
                            let (ordinal, entry) = self.changes.next()?;
                            if let Some(entry) = entry {
                                return Some((*ordinal, *entry));
                            }
                        }
                    }
                }
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
    use super::{compaction_limit, PostingList, MIN_DELTA_COMPACTION};
    use crate::index::fts::PostingEntry;

    fn entry(frequency: u32) -> PostingEntry {
        PostingEntry {
            frequency,
            document_length: 8,
        }
    }

    #[test]
    fn cloned_postings_share_the_base_and_merge_changes_in_ordinal_order() {
        let original = PostingList::from_sorted_entries(
            (0_u64..128)
                .filter(|ordinal| ordinal % 2 == 0)
                .map(|ordinal| (ordinal, entry(1))),
        );
        let mut changed = original.clone();
        assert!(original.base_ptr_eq(&changed));

        changed.insert(3, entry(2)).expect("insert must succeed");
        changed.insert(4, entry(3)).expect("replace must succeed");
        assert!(changed.remove(6));
        assert!(!changed.remove(7));

        assert_eq!(original.get(3).copied(), None);
        assert_eq!(original.get(4).copied(), Some(entry(1)));
        assert_eq!(original.get(6).copied(), Some(entry(1)));
        assert_eq!(changed.get(3).copied(), Some(entry(2)));
        assert_eq!(changed.get(4).copied(), Some(entry(3)));
        assert_eq!(changed.get(6).copied(), None);
        assert_eq!(changed.len(), original.len());
        let ordinals: Vec<_> = changed.iter().map(|(ordinal, _)| ordinal).collect();
        assert!(ordinals.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn bounded_changes_compact_into_a_new_shared_base() {
        let original =
            PostingList::from_sorted_entries((0_u64..512).map(|ordinal| (ordinal, entry(1))));
        assert_eq!(compaction_limit(original.len()), MIN_DELTA_COMPACTION);
        let mut changed = original.clone();
        for ordinal in 0..u64::try_from(MIN_DELTA_COMPACTION).expect("limit fits u64") {
            changed
                .insert(ordinal, entry(2))
                .expect("replace must succeed");
        }

        assert_eq!(changed.change_len(), 0);
        assert!(!original.base_ptr_eq(&changed));
        assert_eq!(changed.len(), original.len());
        assert!(changed
            .iter()
            .take(MIN_DELTA_COMPACTION)
            .all(|(_, value)| value == entry(2)));
    }
}
