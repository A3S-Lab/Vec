//! Direct-address storage for values keyed by stable document ordinals.

/// Sparse values stored at their ordinal offset. Ordinals stay compact because
/// the registry rebuilds all derived indexes when retired IDs cross the shared
/// compaction threshold. This removes tree-node allocations and logarithmic
/// lookup from ANN base vectors and graph layers without changing public IDs.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct OrdinalMap<T> {
    slots: Vec<Option<T>>,
    len: usize,
}

impl<T> Default for OrdinalMap<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
        }
    }
}

impl<T> OrdinalMap<T> {
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(super) fn get(&self, ordinal: u64) -> Option<&T> {
        let index = usize::try_from(ordinal).ok()?;
        self.slots.get(index)?.as_ref()
    }

    #[cfg(test)]
    pub(super) fn get_mut(&mut self, ordinal: u64) -> Option<&mut T> {
        let index = usize::try_from(ordinal).ok()?;
        self.slots.get_mut(index)?.as_mut()
    }

    pub(super) fn contains_key(&self, ordinal: u64) -> bool {
        self.get(ordinal).is_some()
    }

    pub(super) fn insert(&mut self, ordinal: u64, value: T) -> Option<T> {
        let index = usize::try_from(ordinal).ok()?;
        if self.slots.len() <= index {
            self.slots.resize_with(index.saturating_add(1), || None);
        }
        let previous = self.slots[index].replace(value);
        if previous.is_none() {
            self.len = self.len.saturating_add(1);
        }
        previous
    }

    #[cfg(test)]
    pub(super) fn remove(&mut self, ordinal: u64) -> Option<T> {
        let index = usize::try_from(ordinal).ok()?;
        let removed = self.slots.get_mut(index)?.take();
        if removed.is_some() {
            self.len = self.len.saturating_sub(1);
        }
        removed
    }

    pub(super) fn get_or_insert_default(&mut self, ordinal: u64) -> Option<&mut T>
    where
        T: Default,
    {
        let index = usize::try_from(ordinal).ok()?;
        if self.slots.len() <= index {
            self.slots.resize_with(index.saturating_add(1), || None);
        }
        if self.slots[index].is_none() {
            self.slots[index] = Some(T::default());
            self.len = self.len.saturating_add(1);
        }
        self.slots[index].as_mut()
    }

    pub(super) fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.as_ref().and_then(|_| u64::try_from(index).ok()))
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(Option::as_ref)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (u64, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, value)| Some((u64::try_from(index).ok()?, value.as_ref()?)))
    }

    pub(super) fn validates(&self, maximum_slots: usize) -> bool {
        self.slots.len() <= maximum_slots
            && self.len == self.slots.iter().filter(|value| value.is_some()).count()
    }
}

impl<T> FromIterator<(u64, T)> for OrdinalMap<T> {
    fn from_iter<I: IntoIterator<Item = (u64, T)>>(iter: I) -> Self {
        let mut values = Self::default();
        for (ordinal, value) in iter {
            values.insert(ordinal, value);
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::OrdinalMap;

    #[test]
    fn sparse_slots_iterate_in_ordinal_order_and_track_membership() {
        let mut values = OrdinalMap::default();
        values.insert(4, "four");
        values.insert(1, "one");
        assert_eq!(values.len(), 2);
        assert_eq!(values.slot_count(), 5);
        assert_eq!(values.keys().collect::<Vec<_>>(), vec![1, 4]);
        assert_eq!(values.get(4), Some(&"four"));
        assert_eq!(values.remove(1), Some("one"));
        assert!(!values.contains_key(1));
        assert!(values.validates(5));
        values.slots.push(None);
        assert!(!values.validates(5));
        values.slots.pop();
        values.len = 2;
        assert!(!values.validates(5));
    }
}
