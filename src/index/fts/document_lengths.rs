//! Persistent direct-address storage for FTS document lengths.

use crate::error::{Error, Result};
use im::OrdMap;
use roaring::RoaringTreemap;
use std::sync::Arc;

const MIN_DELTA_COMPACTION: usize = 64;
const MAX_DELTA_COMPACTION: usize = 2_048;
const DELTA_COMPACTION_DIVISOR: usize = 8;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct DocumentLengths {
    base: Arc<Vec<Option<u32>>>,
    changes: OrdMap<u64, Option<u32>>,
    live_len: usize,
}

impl DocumentLengths {
    pub(super) fn from_sorted_entries(
        entries: impl IntoIterator<Item = (u64, u32)>,
    ) -> Result<Self> {
        let mut base = Vec::new();
        let mut live_len = 0_usize;
        for (ordinal, length) in entries {
            let index = usize::try_from(ordinal).map_err(|_| {
                Error::resource_exhausted("FTS document ordinal exceeds addressable memory")
            })?;
            let required = index.checked_add(1).ok_or_else(|| {
                Error::resource_exhausted("FTS document ordinal exceeds addressable memory")
            })?;
            if base.len() < required {
                base.try_reserve(required - base.len()).map_err(|_| {
                    Error::resource_exhausted("FTS document lengths exceed addressable memory")
                })?;
                base.resize(required, None);
            }
            if base[index].replace(length).is_some() {
                return Err(Error::internal(format!(
                    "FTS document ordinal {ordinal} is already indexed"
                )));
            }
            live_len = live_len
                .checked_add(1)
                .ok_or_else(|| Error::resource_exhausted("FTS document count overflow"))?;
        }
        Ok(Self {
            base: Arc::new(base),
            changes: OrdMap::new(),
            live_len,
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.live_len == 0
    }

    pub(super) fn len(&self) -> usize {
        self.live_len
    }

    pub(super) fn get(&self, ordinal: u64) -> Option<&u32> {
        if let Some(length) = self.changes.get(&ordinal) {
            return length.as_ref();
        }
        self.base_entry(ordinal)
    }

    pub(super) fn contains_key(&self, ordinal: u64) -> bool {
        self.get(ordinal).is_some()
    }

    pub(super) fn insert(&mut self, ordinal: u64, length: u32) -> Result<()> {
        let index = usize::try_from(ordinal).map_err(|_| {
            Error::resource_exhausted("FTS document ordinal exceeds addressable memory")
        })?;
        index.checked_add(1).ok_or_else(|| {
            Error::resource_exhausted("FTS document ordinal exceeds addressable memory")
        })?;
        if self.get(ordinal).is_some() {
            return Err(Error::internal(format!(
                "FTS document ordinal {ordinal} is already indexed"
            )));
        }
        let live_len = self
            .live_len
            .checked_add(1)
            .ok_or_else(|| Error::resource_exhausted("FTS document count overflow"))?;
        if self.base_entry(ordinal) == Some(&length) {
            self.changes.remove(&ordinal);
        } else {
            self.changes.insert(ordinal, Some(length));
        }
        self.live_len = live_len;
        Ok(())
    }

    pub(super) fn remove(&mut self, ordinal: u64) -> Option<u32> {
        let length = *self.get(ordinal)?;
        if self.base_entry(ordinal).is_some() {
            self.changes.insert(ordinal, None);
        } else {
            self.changes.remove(&ordinal);
        }
        self.live_len -= 1;
        Some(length)
    }

    pub(super) fn finish_changes(&mut self) -> Result<()> {
        if self.changes.len() < compaction_limit(self.base.len()) {
            return Ok(());
        }
        let mut base = (*self.base).clone();
        let required = self.span();
        if base.len() < required {
            base.try_reserve(required - base.len()).map_err(|_| {
                Error::resource_exhausted("FTS document lengths exceed addressable memory")
            })?;
            base.resize(required, None);
        }
        for (ordinal, length) in &self.changes {
            let index = usize::try_from(*ordinal).map_err(|_| {
                Error::resource_exhausted("FTS document ordinal exceeds addressable memory")
            })?;
            base[index] = *length;
        }
        while base.last() == Some(&None) {
            base.pop();
        }
        self.base = Arc::new(base);
        self.changes = OrdMap::new();
        debug_assert_eq!(self.live_len, self.iter().count());
        Ok(())
    }

    pub(super) fn keys(&self) -> impl DoubleEndedIterator<Item = u64> + '_ {
        self.iter().map(|(ordinal, _)| ordinal)
    }

    pub(super) fn values(&self) -> impl DoubleEndedIterator<Item = u32> + '_ {
        self.iter().map(|(_, length)| length)
    }

    pub(super) fn validates(&self, maximum_slots: usize, live: &RoaringTreemap) -> bool {
        if self.base.len() > maximum_slots
            || self.changes.len() >= compaction_limit(self.base.len())
            || self.changes.iter().any(|(ordinal, length)| {
                let Some(index) = usize::try_from(*ordinal).ok() else {
                    return true;
                };
                index >= maximum_slots
                    || match (self.base.get(index).and_then(Option::as_ref), length) {
                        (None, None) => true,
                        (Some(base), Some(changed)) => base == changed,
                        (None, Some(_)) | (Some(_), None) => false,
                    }
            })
        {
            return false;
        }
        let mut count = 0_usize;
        for (ordinal, _) in self.iter() {
            if !live.contains(ordinal) {
                return false;
            }
            count = match count.checked_add(1) {
                Some(count) => count,
                None => return false,
            };
        }
        count == self.live_len
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = (u64, u32)> + '_ {
        (0..self.span()).filter_map(|index| {
            let ordinal = u64::try_from(index).ok()?;
            self.get(ordinal).copied().map(|length| (ordinal, length))
        })
    }

    fn span(&self) -> usize {
        self.changes
            .keys()
            .next_back()
            .and_then(|ordinal| usize::try_from(*ordinal).ok())
            .and_then(|ordinal| ordinal.checked_add(1))
            .map_or(self.base.len(), |span| span.max(self.base.len()))
    }

    fn base_entry(&self, ordinal: u64) -> Option<&u32> {
        usize::try_from(ordinal)
            .ok()
            .and_then(|ordinal| self.base.get(ordinal))
            .and_then(Option::as_ref)
    }
}

fn compaction_limit(base_len: usize) -> usize {
    let fractional =
        base_len.saturating_add(DELTA_COMPACTION_DIVISOR - 1) / DELTA_COMPACTION_DIVISOR;
    fractional.clamp(MIN_DELTA_COMPACTION, MAX_DELTA_COMPACTION)
}

#[cfg(test)]
mod tests {
    use super::{DocumentLengths, MIN_DELTA_COMPACTION};
    use roaring::RoaringTreemap;

    #[test]
    fn cloned_lengths_share_the_base_and_shadow_small_changes() {
        let original =
            DocumentLengths::from_sorted_entries([(1, 3), (4, 7)]).expect("lengths must build");
        let mut changed = original.clone();
        assert!(std::sync::Arc::ptr_eq(&original.base, &changed.base));

        assert_eq!(changed.remove(1), Some(3));
        changed.insert(2, 5).expect("insert must succeed");
        changed.insert(1, 9).expect("reinsert must succeed");

        assert_eq!(original.get(1), Some(&3));
        assert_eq!(changed.get(1), Some(&9));
        assert_eq!(changed.get(2), Some(&5));
        assert_eq!(changed.keys().collect::<Vec<_>>(), vec![1, 2, 4]);
    }

    #[test]
    fn bounded_changes_compact_into_direct_address_slots() {
        let mut lengths = DocumentLengths::from_sorted_entries([]).expect("lengths must build");
        for ordinal in 0..u64::try_from(MIN_DELTA_COMPACTION).expect("limit fits u64") {
            lengths.insert(ordinal, 4).expect("insert must succeed");
        }
        lengths.finish_changes().expect("compaction must succeed");

        assert!(lengths.changes.is_empty());
        assert_eq!(lengths.base.len(), MIN_DELTA_COMPACTION);
        let live: RoaringTreemap =
            (0..u64::try_from(MIN_DELTA_COMPACTION).expect("limit fits u64")).collect();
        assert!(lengths.validates(MIN_DELTA_COMPACTION, &live));
    }
}
