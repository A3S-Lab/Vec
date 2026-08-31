//! Allocation-conscious deterministic HNSW traversal.

use super::HnswFilter;
use crate::index::ordinal_map::OrdinalMap;
use crate::index::ordinals::OrdinalTable;
use crate::index::quantization::{score, score_dense, QuantizedVector};
use crate::types::MetricType;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug)]
struct ScoredNode<'a> {
    ordinal: u64,
    primary_key: &'a str,
    score: f64,
}

impl PartialEq for ScoredNode<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && self.score.total_cmp(&other.score) == Ordering::Equal
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
            .then_with(|| other.primary_key.cmp(self.primary_key))
            .then_with(|| other.ordinal.cmp(&self.ordinal))
    }
}

fn scored_node(ordinal: u64, score: f64, ordinals: &OrdinalTable) -> ScoredNode<'_> {
    ScoredNode {
        ordinal,
        primary_key: ordinals.id(ordinal).unwrap_or_default(),
        score,
    }
}

pub(super) fn greedy_search(
    layer: &OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<Vec<f32>>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entry: u64,
    metric: MetricType,
) -> u64 {
    let mut current = entry;
    let mut current_score = vectors.get(entry).map_or(f64::NEG_INFINITY, |vector| {
        score_dense(query, vector, metric)
    });
    loop {
        let mut best = scored_node(current, current_score, ordinals);
        for neighbor in layer.get(current).into_iter().flatten().copied() {
            let Some(vector) = vectors.get(neighbor) else {
                continue;
            };
            let candidate = scored_node(neighbor, score_dense(query, vector, metric), ordinals);
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

pub(super) fn greedy_search_quantized(
    layer: &OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<QuantizedVector>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entry: u64,
    metric: MetricType,
) -> u64 {
    let mut current = entry;
    let mut current_score = vectors
        .get(entry)
        .map_or(f64::NEG_INFINITY, |vector| score(query, vector, metric));
    loop {
        let mut best = scored_node(current, current_score, ordinals);
        for neighbor in layer.get(current).into_iter().flatten().copied() {
            let Some(vector) = vectors.get(neighbor) else {
                continue;
            };
            let candidate = scored_node(neighbor, score(query, vector, metric), ordinals);
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

pub(super) fn search_layer(
    layer: &OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<Vec<f32>>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entries: &[u64],
    ef: usize,
    metric: MetricType,
) -> Vec<u64> {
    bounded_graph_search(layer, entries, ef, ordinals, |ordinal| {
        vectors
            .get(ordinal)
            .map(|vector| score_dense(query, vector, metric))
    })
}

pub(super) fn search_layer_quantized(
    layer: &OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<QuantizedVector>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entries: &[u64],
    ef: usize,
    metric: MetricType,
) -> Vec<u64> {
    bounded_graph_search(layer, entries, ef, ordinals, |ordinal| {
        vectors
            .get(ordinal)
            .map(|vector| score(query, vector, metric))
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_layer_quantized_filtered(
    layer: &OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<QuantizedVector>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entries: &[u64],
    result_limit: usize,
    traversal_limit: usize,
    metric: MetricType,
    filter: HnswFilter<'_>,
) -> Vec<u64> {
    bounded_filtered_graph_search(
        layer,
        entries,
        result_limit,
        traversal_limit,
        ordinals,
        |ordinal| {
            vectors
                .get(ordinal)
                .map(|vector| score(query, vector, metric))
        },
        |ordinal| filter.allowed.contains(ordinal) && !filter.excluded.contains(ordinal),
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
    let mut visited = HashSet::with_capacity(expansion_limit.min(layer.len()));
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
    let mut visited = HashSet::with_capacity(expansion_limit.min(layer.len()));
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
    use super::{ordered_ordinals, retain_best, ScoredNode};
    use std::collections::BinaryHeap;

    #[test]
    fn heaps_order_scores_then_primary_keys_deterministically() {
        let mut best = BinaryHeap::new();
        for candidate in [
            ScoredNode {
                ordinal: 2,
                primary_key: "doc-b",
                score: 1.0,
            },
            ScoredNode {
                ordinal: 3,
                primary_key: "doc-low",
                score: 0.0,
            },
            ScoredNode {
                ordinal: 1,
                primary_key: "doc-a",
                score: 1.0,
            },
            ScoredNode {
                ordinal: 4,
                primary_key: "doc-high",
                score: 2.0,
            },
        ] {
            retain_best(&mut best, candidate, 3);
        }
        assert_eq!(ordered_ordinals(best), vec![4, 1, 2]);
    }
}
