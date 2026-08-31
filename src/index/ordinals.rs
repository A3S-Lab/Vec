//! Stable document ordinals shared by bitmap and posting-list indexes.

use crate::doc::DocumentMap;
use crate::error::{Error, Result};
use im::{OrdMap, Vector};
use roaring::RoaringTreemap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

const MIN_COMPACTION: usize = 64;
const MAX_COMPACTION: usize = 2_048;
const COMPACTION_DIVISOR: usize = 8;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct OrdinalTable {
    by_id: OrdMap<String, u64>,
    by_ordinal: Vector<String>,
    live: Arc<RoaringTreemap>,
}

/// An immutable candidate bitmap paired with the persistent ordinal generation
/// that can resolve it. Cloning the table shares its persistent lookup
/// structures and live bitmap, so query planning does not materialize or
/// duplicate primary keys.
#[derive(Debug, Clone)]
pub(super) struct OrdinalSet {
    table: OrdinalTable,
    bitmap: RoaringTreemap,
}

/// Immutable index scores paired with the ordinal generation that produced
/// them. Query execution resolves borrowed primary keys on demand instead of
/// materializing an owned primary-key map for every search.
#[derive(Debug, Clone)]
pub(crate) struct OrdinalScores {
    table: OrdinalTable,
    scores: Vec<(u64, f64)>,
    candidate_count: usize,
}

#[derive(Debug)]
struct RankedOrdinal<'a> {
    ordinal: u64,
    score: f64,
    id: &'a str,
}

impl OrdinalTable {
    pub(super) fn build(docs: &DocumentMap) -> Result<Self> {
        let mut table = Self::default();
        for id in docs.keys() {
            table.ensure_live(id)?;
        }
        Ok(table)
    }

    pub(super) fn ensure_live(&mut self, id: &str) -> Result<u64> {
        if let Some(ordinal) = self.by_id.get(id).copied() {
            if !self.live.contains(ordinal) {
                Arc::make_mut(&mut self.live).insert(ordinal);
            }
            return Ok(ordinal);
        }
        let ordinal = u64::try_from(self.by_ordinal.len())
            .map_err(|_| Error::resource_exhausted("document ordinal space exhausted"))?;
        self.by_id.insert(id.to_string(), ordinal);
        self.by_ordinal.push_back(id.to_string());
        Arc::make_mut(&mut self.live).insert(ordinal);
        Ok(ordinal)
    }

    pub(super) fn remove_live(&mut self, id: &str) {
        if let Some(ordinal) = self.by_id.get(id).copied() {
            Arc::make_mut(&mut self.live).remove(ordinal);
        }
    }

    pub(super) fn ordinal(&self, id: &str) -> Option<u64> {
        self.by_id.get(id).copied()
    }

    pub(super) fn id(&self, ordinal: u64) -> Option<&str> {
        usize::try_from(ordinal)
            .ok()
            .and_then(|ordinal| self.by_ordinal.get(ordinal))
            .map(String::as_str)
    }

    pub(super) fn live(&self) -> &RoaringTreemap {
        &self.live
    }

    pub(super) fn allocated_len(&self) -> usize {
        self.by_ordinal.len()
    }

    pub(super) fn validates(&self, docs: &DocumentMap) -> bool {
        if self.by_id.len() != self.by_ordinal.len()
            || self
                .by_id
                .iter()
                .any(|(id, ordinal)| self.id(*ordinal) != Some(id.as_str()))
            || self.by_ordinal.iter().enumerate().any(|(ordinal, id)| {
                let Ok(ordinal) = u64::try_from(ordinal) else {
                    return true;
                };
                self.by_id.get(id) != Some(&ordinal)
            })
        {
            return false;
        }
        let expected_live: Option<RoaringTreemap> =
            docs.keys().map(|id| self.ordinal(id)).collect();
        expected_live.is_some_and(|expected| expected == *self.live)
    }

    pub(super) fn should_compact(&self) -> bool {
        let live = usize::try_from(self.live.len()).unwrap_or(usize::MAX);
        let retired = self.by_id.len().saturating_sub(live);
        let fractional = live.saturating_add(COMPACTION_DIVISOR - 1) / COMPACTION_DIVISOR;
        retired >= fractional.clamp(MIN_COMPACTION, MAX_COMPACTION)
    }
}

impl OrdinalSet {
    pub(super) fn new(table: &OrdinalTable, bitmap: RoaringTreemap) -> Self {
        Self {
            table: table.clone(),
            bitmap,
        }
    }

    pub(super) fn len(&self) -> usize {
        usize::try_from(self.bitmap.len()).unwrap_or(usize::MAX)
    }

    pub(super) fn contains(&self, id: &str) -> bool {
        self.table
            .ordinal(id)
            .is_some_and(|ordinal| self.bitmap.contains(ordinal))
    }

    pub(super) fn bitmap(&self) -> &RoaringTreemap {
        &self.bitmap
    }

    pub(super) fn ids(&self) -> impl Iterator<Item = &str> {
        self.bitmap
            .iter()
            .filter_map(|ordinal| self.table.id(ordinal))
    }

    pub(super) fn retain_ids(&mut self, mut keep: impl FnMut(&str) -> bool) {
        self.bitmap = self
            .bitmap
            .iter()
            .filter(|ordinal| self.table.id(*ordinal).is_some_and(&mut keep))
            .collect();
    }
}

impl OrdinalScores {
    pub(super) fn new(table: &OrdinalTable, scores: Vec<(u64, f64)>) -> Result<Self> {
        Self::build(table, scores, None)
    }

    pub(super) fn new_topk(
        table: &OrdinalTable,
        scores: Vec<(u64, f64)>,
        limit: usize,
    ) -> Result<Self> {
        Self::build(table, scores, Some(limit))
    }

    fn build(table: &OrdinalTable, scores: Vec<(u64, f64)>, limit: Option<usize>) -> Result<Self> {
        let candidate_count = scores.len();
        let mut previous = None;
        let mut retained = limit
            .filter(|limit| *limit < candidate_count)
            .map(|_| BinaryHeap::<RankedOrdinal<'_>>::new());
        for (ordinal, score) in &scores {
            let id = validate_score(table, previous, *ordinal, *score)?;
            previous = Some(*ordinal);
            if let (Some(limit), Some(retained)) = (limit, retained.as_mut()) {
                let candidate = RankedOrdinal {
                    ordinal: *ordinal,
                    score: *score,
                    id,
                };
                if retained.len() < limit {
                    retained.push(candidate);
                } else if retained
                    .peek()
                    .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
                {
                    retained.pop();
                    retained.push(candidate);
                }
            }
        }
        let scores = retained.map_or(scores, |retained| {
            let mut scores: Vec<_> = retained
                .into_iter()
                .map(|candidate| (candidate.ordinal, candidate.score))
                .collect();
            scores.sort_unstable_by_key(|(ordinal, _)| *ordinal);
            scores
        });
        Ok(Self {
            table: table.clone(),
            scores,
            candidate_count,
        })
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.scores.len()
    }

    pub(crate) fn candidate_count(&self) -> u64 {
        u64::try_from(self.candidate_count).unwrap_or(u64::MAX)
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&str, f64)> {
        self.scores
            .iter()
            .filter_map(|(ordinal, score)| self.table.id(*ordinal).map(|id| (id, *score)))
    }
}

impl PartialEq for RankedOrdinal<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedOrdinal<'_> {}

impl PartialOrd for RankedOrdinal<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedOrdinal<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(other.id))
    }
}

fn validate_score(
    table: &OrdinalTable,
    previous: Option<u64>,
    ordinal: u64,
    score: f64,
) -> Result<&str> {
    if previous.is_some_and(|previous| ordinal <= previous) {
        return Err(Error::internal(
            "scored document ordinals must be unique and ordered",
        ));
    }
    let id = table.id(ordinal).ok_or_else(|| {
        Error::internal(format!(
            "document mapping is missing for scored ordinal {ordinal}"
        ))
    })?;
    if !table.live().contains(ordinal) {
        return Err(Error::internal(format!(
            "scored document ordinal {ordinal} is not live"
        )));
    }
    if !score.is_finite() || score <= 0.0 {
        return Err(Error::internal(format!(
            "document ordinal {ordinal} has an invalid index score"
        )));
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::{OrdinalScores, OrdinalSet, OrdinalTable};
    use crate::doc::{Doc, DocumentMap};
    use roaring::RoaringTreemap;
    use std::sync::Arc;

    fn docs() -> DocumentMap {
        ["doc-a", "doc-b", "doc-c"]
            .into_iter()
            .map(|id| {
                let doc = Doc::with_pk(id).expect("document id must be valid");
                (id.to_string(), Arc::new(doc))
            })
            .collect()
    }

    #[test]
    fn ordinal_set_resolves_members_without_materializing_primary_keys() {
        let table = OrdinalTable::build(&docs()).expect("ordinal table must build");
        let bitmap: RoaringTreemap = [
            table.ordinal("doc-a").expect("doc-a ordinal"),
            table.ordinal("doc-c").expect("doc-c ordinal"),
        ]
        .into_iter()
        .collect();
        let mut set = OrdinalSet::new(&table, bitmap);

        assert!(set.contains("doc-a"));
        assert!(!set.contains("doc-b"));
        assert_eq!(set.ids().collect::<Vec<_>>(), vec!["doc-a", "doc-c"]);

        set.retain_ids(|id| id.ends_with('c'));
        assert_eq!(set.ids().collect::<Vec<_>>(), vec!["doc-c"]);
    }

    #[test]
    fn ordinal_reverse_lookup_is_persistent_and_append_only() {
        let documents: DocumentMap = (0..128)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let doc = Doc::with_pk(&id).expect("document id must be valid");
                (id, Arc::new(doc))
            })
            .collect();
        let table = OrdinalTable::build(&documents).expect("ordinal table must build");
        let mut next = table.clone();
        assert!(table.by_ordinal.ptr_eq(&next.by_ordinal));

        next.remove_live("doc-000");
        next.ensure_live("doc-000")
            .expect("existing ordinal must revive");
        assert!(table.by_ordinal.ptr_eq(&next.by_ordinal));

        let appended = next
            .ensure_live("doc-new")
            .expect("new ordinal must append");
        assert_eq!(appended, 128);
        assert_eq!(next.id(appended), Some("doc-new"));
        assert_eq!(table.id(appended), None);
        assert_eq!(table.allocated_len(), 128);
        assert_eq!(next.allocated_len(), 129);
        assert!(table.validates(&documents));
    }

    #[test]
    fn ordinal_scores_resolve_borrowed_primary_keys() {
        let table = OrdinalTable::build(&docs()).expect("ordinal table must build");
        let scores = [
            (table.ordinal("doc-a").expect("doc-a ordinal"), 2.0),
            (table.ordinal("doc-c").expect("doc-c ordinal"), 1.0),
        ]
        .into_iter()
        .collect();
        let scores = OrdinalScores::new(&table, scores).expect("score mapping must be valid");

        assert_eq!(scores.len(), 2);
        assert_eq!(
            scores.entries().collect::<Vec<_>>(),
            vec![("doc-a", 2.0), ("doc-c", 1.0)]
        );
    }

    #[test]
    fn ordinal_scores_reject_invalid_generations() {
        let table = OrdinalTable::build(&docs()).expect("ordinal table must build");
        let first = table.ordinal("doc-a").expect("doc-a ordinal");
        let second = table.ordinal("doc-b").expect("doc-b ordinal");

        assert!(OrdinalScores::new(&table, vec![(second, 1.0), (first, 2.0)]).is_err());
        assert!(OrdinalScores::new(&table, vec![(first, f64::NAN)]).is_err());
        assert!(OrdinalScores::new(&table, vec![(99, 1.0)]).is_err());

        let mut retired = table;
        retired.remove_live("doc-a");
        assert!(OrdinalScores::new(&retired, vec![(first, 1.0)]).is_err());
    }

    #[test]
    fn ordinal_scores_topk_uses_primary_key_ties_and_keeps_candidate_count() {
        let initial: DocumentMap = [(
            "z-base".to_string(),
            Arc::new(Doc::with_pk("z-base").expect("document must be valid")),
        )]
        .into_iter()
        .collect();
        let mut table = OrdinalTable::build(&initial).expect("ordinal table must build");
        let base = table.ordinal("z-base").expect("base ordinal must exist");
        let later = table
            .ensure_live("a-later")
            .expect("later ordinal must be allocated");
        let scores = OrdinalScores::new_topk(&table, vec![(base, 1.0), (later, 1.0)], 1)
            .expect("top-k scores must build");

        assert_eq!(scores.len(), 1);
        assert_eq!(scores.candidate_count(), 2);
        assert_eq!(scores.entries().collect::<Vec<_>>(), vec![("a-later", 1.0)]);
    }
}
