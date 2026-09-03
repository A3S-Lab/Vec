//! Allocation-conscious deterministic HNSW traversal.

use crate::index::ordinal_map::OrdinalMap;
use crate::index::ordinals::OrdinalTable;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

const DENSE_VISITED_MAX_SLOTS: usize = 1 << 24;

/// Tracks graph membership with a compact ordinal bitset when the ordinal
/// space is reasonably dense. Very sparse/retired ordinal spaces retain the
/// hash-set fallback so a query cannot allocate an attacker-sized bitmap.
enum VisitedSet {
    Dense(Vec<u64>),
    Sparse(HashSet<u64>),
}

impl VisitedSet {
    fn new(slot_count: usize, capacity: usize) -> Self {
        if slot_count > 0 && slot_count <= DENSE_VISITED_MAX_SLOTS {
            Self::Dense(vec![0; slot_count.saturating_add(63) / 64])
        } else {
            Self::Sparse(HashSet::with_capacity(capacity))
        }
    }

    fn insert(&mut self, ordinal: u64) -> bool {
        match self {
            Self::Dense(bits) => {
                let Ok(index) = usize::try_from(ordinal) else {
                    return false;
                };
                let word = index / 64;
                let mask = 1_u64 << (index % 64);
                let Some(value) = bits.get_mut(word) else {
                    return false;
                };
                let was_new = *value & mask == 0;
                *value |= mask;
                was_new
            }
            Self::Sparse(values) => values.insert(ordinal),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ScoredNode<'a> {
    ordinal: u64,
    score: f64,
    ordinals: &'a OrdinalTable,
}

impl PartialEq for ScoredNode<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoredNode<'_> {}

impl PartialOrd for ScoredNode<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            // Resolve primary keys only for exact score ties. The old eager
            // lookup paid for a persistent reverse-map access on every heap
            // comparison, even when scores were distinct.
            .then_with(|| {
                other
                    .ordinals
                    .id(other.ordinal)
                    .unwrap_or_default()
                    .cmp(self.ordinals.id(self.ordinal).unwrap_or_default())
            })
            .then_with(|| other.ordinal.cmp(&self.ordinal))
    }
}

fn scored_node(ordinal: u64, score: f64, ordinals: &OrdinalTable) -> ScoredNode<'_> {
    ScoredNode {
        ordinal,
        score,
        ordinals,
    }
}

pub(super) fn greedy_search_by(
    layer: &OrdinalMap<Vec<u64>>,
    ordinals: &OrdinalTable,
    entry: u64,
    score_for: &impl Fn(u64) -> Option<f64>,
) -> u64 {
    let mut current = entry;
    let mut current_score = score_for(entry).unwrap_or(f64::NEG_INFINITY);
    loop {
        let mut best = scored_node(current, current_score, ordinals);
        for neighbor in layer.get(current).into_iter().flatten().copied() {
            let Some(score) = score_for(neighbor) else {
                continue;
            };
            let candidate = scored_node(neighbor, score, ordinals);
            if candidate > best {
                best = candidate;
            }
        }
        if best.ordinal == current {
            return current;
        }
        current = best.ordinal;
        current_score = best.score;
    }
}

pub(super) fn search_layer_by(
    layer: &OrdinalMap<Vec<u64>>,
    entries: &[u64],
    ef: usize,
    ordinals: &OrdinalTable,
    score_for: &impl Fn(u64) -> Option<f64>,
) -> Vec<u64> {
    bounded_graph_search(layer, entries, ef, ordinals, score_for)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_layer_filtered_by(
    layer: &OrdinalMap<Vec<u64>>,
    entries: &[u64],
    result_limit: usize,
    traversal_limit: usize,
    ordinals: &OrdinalTable,
    score_for: &impl Fn(u64) -> Option<f64>,
    is_allowed: &impl Fn(u64) -> bool,
) -> Vec<u64> {
    bounded_filtered_graph_search(
        layer,
        entries,
        result_limit,
        traversal_limit,
        ordinals,
        score_for,
        is_allowed,
    )
}

fn bounded_graph_search(
    layer: &OrdinalMap<Vec<u64>>,
    entries: &[u64],
    ef: usize,
    ordinals: &OrdinalTable,
    score_for: impl Fn(u64) -> Option<f64>,
) -> Vec<u64> {
    let ef = ef.max(1);
    let expansion_limit = ef.saturating_mul(8).max(entries.len());
    let mut visited = VisitedSet::new(layer.slot_count(), expansion_limit.min(layer.len()));
    let mut frontier = BinaryHeap::with_capacity(ef.saturating_mul(2).min(layer.len()));
    let mut best = BinaryHeap::with_capacity(ef.saturating_add(1));
    for ordinal in entries.iter().copied() {
        if !visited.insert(ordinal) {
            continue;
        }
        let Some(score) = score_for(ordinal) else {
            continue;
        };
        let candidate = scored_node(ordinal, score, ordinals);
        frontier.push(candidate);
        retain_best(&mut best, candidate, ef);
    }
    let mut expanded = 0;

    while expanded < expansion_limit {
        let Some(current) = frontier.pop() else {
            break;
        };
        if best.len() >= ef && best.peek().is_some_and(|Reverse(worst)| current < *worst) {
            break;
        }
        expanded += 1;
        for neighbor in layer.get(current.ordinal).into_iter().flatten().copied() {
            if !visited.insert(neighbor) {
                continue;
            }
            let Some(candidate_score) = score_for(neighbor) else {
                continue;
            };
            let candidate = scored_node(neighbor, candidate_score, ordinals);
            frontier.push(candidate);
            retain_best(&mut best, candidate, ef);
        }
    }
    ordered_ordinals(best)
}

fn bounded_filtered_graph_search(
    layer: &OrdinalMap<Vec<u64>>,
    entries: &[u64],
    result_limit: usize,
    traversal_limit: usize,
    ordinals: &OrdinalTable,
    score_for: impl Fn(u64) -> Option<f64>,
    is_allowed: impl Fn(u64) -> bool,
) -> Vec<u64> {
    let result_limit = result_limit.max(1);
    let traversal_limit = traversal_limit.max(result_limit);
    let expansion_limit = traversal_limit.saturating_mul(8).max(entries.len());
    let mut visited = VisitedSet::new(layer.slot_count(), expansion_limit.min(layer.len()));
    let mut frontier =
        BinaryHeap::with_capacity(traversal_limit.saturating_mul(2).min(layer.len()));
    let mut best = BinaryHeap::with_capacity(result_limit.saturating_add(1));
    for ordinal in entries.iter().copied() {
        if !visited.insert(ordinal) {
            continue;
        }
        let Some(score) = score_for(ordinal) else {
            continue;
        };
        let candidate = scored_node(ordinal, score, ordinals);
        frontier.push(candidate);
        if is_allowed(ordinal) {
            retain_best(&mut best, candidate, result_limit);
        }
    }
    let mut expanded = 0;

    while expanded < expansion_limit {
        let Some(current) = frontier.pop() else {
            break;
        };
        if best.len() >= result_limit && best.peek().is_some_and(|Reverse(worst)| current < *worst)
        {
            break;
        }
        expanded += 1;
        for neighbor in layer.get(current.ordinal).into_iter().flatten().copied() {
            if !visited.insert(neighbor) {
                continue;
            }
            let Some(candidate_score) = score_for(neighbor) else {
                continue;
            };
            let candidate = scored_node(neighbor, candidate_score, ordinals);
            frontier.push(candidate);
            if is_allowed(neighbor) {
                retain_best(&mut best, candidate, result_limit);
            }
        }
    }
    ordered_ordinals(best)
}

fn retain_best<'a>(
    best: &mut BinaryHeap<Reverse<ScoredNode<'a>>>,
    candidate: ScoredNode<'a>,
    limit: usize,
) {
    best.push(Reverse(candidate));
    if best.len() > limit {
        best.pop();
    }
}

fn ordered_ordinals(best: BinaryHeap<Reverse<ScoredNode<'_>>>) -> Vec<u64> {
    let mut nodes: Vec<ScoredNode<'_>> = best.into_iter().map(|Reverse(node)| node).collect();
    nodes.sort_unstable_by(|left, right| right.cmp(left));
    nodes.into_iter().map(|node| node.ordinal).collect()
}

#[cfg(test)]
mod tests {
    use super::{ordered_ordinals, retain_best, ScoredNode, VisitedSet};
    use crate::doc::{Doc, DocumentMap};
    use crate::index::ordinals::OrdinalTable;
    use std::collections::BinaryHeap;
    use std::sync::Arc;

    #[test]
    fn heaps_order_scores_then_primary_keys_deterministically() {
        let docs: DocumentMap = ["doc-a", "doc-b", "doc-high", "doc-low"]
            .into_iter()
            .map(|id| {
                let doc = Doc::with_pk(id).expect("document ID must be valid");
                (id.to_string(), Arc::new(doc))
            })
            .collect();
        let table = OrdinalTable::build(&docs).expect("ordinal table must build");
        let mut best = BinaryHeap::new();
        for candidate in [
            ScoredNode {
                ordinal: table.ordinal("doc-b").expect("doc-b ordinal"),
                score: 1.0,
                ordinals: &table,
            },
            ScoredNode {
                ordinal: table.ordinal("doc-low").expect("doc-low ordinal"),
                score: 0.0,
                ordinals: &table,
            },
            ScoredNode {
                ordinal: table.ordinal("doc-a").expect("doc-a ordinal"),
                score: 1.0,
                ordinals: &table,
            },
            ScoredNode {
                ordinal: table.ordinal("doc-high").expect("doc-high ordinal"),
                score: 2.0,
                ordinals: &table,
            },
        ] {
            retain_best(&mut best, candidate, 3);
        }
        assert_eq!(
            ordered_ordinals(best),
            vec![
                table.ordinal("doc-high").expect("doc-high ordinal"),
                table.ordinal("doc-a").expect("doc-a ordinal"),
                table.ordinal("doc-b").expect("doc-b ordinal"),
            ]
        );
    }

    #[test]
    fn visited_set_deduplicates_dense_ordinals() {
        let mut visited = VisitedSet::new(130, 8);
        assert!(visited.insert(0));
        assert!(visited.insert(64));
        assert!(visited.insert(129));
        assert!(!visited.insert(64));
        assert!(!visited.insert(0));
    }

    #[test]
    fn visited_set_uses_hash_fallback_for_empty_or_huge_spaces() {
        let mut empty = VisitedSet::new(0, 8);
        assert!(empty.insert(u64::MAX));
        assert!(!empty.insert(u64::MAX));

        let mut huge = VisitedSet::new(super::DENSE_VISITED_MAX_SLOTS + 1, 8);
        assert!(huge.insert(17));
        assert!(!huge.insert(17));
    }
}
