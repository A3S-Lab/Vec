//! Deterministic in-memory Vamana graph construction and search.
//!
//! The build follows the two-pass GreedySearch/RobustPrune sequence from the
//! `DiskANN` paper while retaining the crate's immutable ordinal generations.

use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::quantization::{score, score_dense, QuantizedVector};
use crate::types::MetricType;
use roaring::RoaringTreemap;
use std::collections::{BTreeSet, HashSet};

const INITIAL_GRAPH_SEED: u64 = 0x6a09_e667_f3bc_c909;
const BUILD_ORDER_SEED: u64 = 0xbb67_ae85_84ca_a73b;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct VamanaIndex {
    graph: OrdinalMap<Vec<u64>>,
    entry_ordinal: Option<u64>,
    default_list_size: usize,
    max_degree: usize,
    alpha: f64,
}

#[derive(Clone, Copy, Debug)]
struct ScoredOrdinal {
    ordinal: u64,
    score: f64,
}

#[derive(Debug)]
struct GraphSearch {
    candidates: Vec<u64>,
    visited: Vec<u64>,
}

impl VamanaIndex {
    pub(super) fn nodes(&self) -> impl Iterator<Item = (u64, &[u64])> {
        self.graph
            .iter()
            .map(|(ordinal, neighbors)| (ordinal, neighbors.as_slice()))
    }

    pub(super) fn neighbors(&self, ordinal: u64) -> Option<&[u64]> {
        self.graph.get(ordinal).map(Vec::as_slice)
    }

    pub(super) fn entry_ordinal(&self) -> Option<u64> {
        self.entry_ordinal
    }

    pub(super) fn default_list_size(&self) -> usize {
        self.default_list_size
    }

    pub(super) fn max_degree(&self) -> usize {
        self.max_degree
    }

    pub(super) fn alpha(&self) -> f64 {
        self.alpha
    }

    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        max_degree: usize,
        list_size: usize,
        alpha: f64,
        metric: MetricType,
    ) -> Self {
        let decoded: OrdinalMap<Vec<f32>> = vectors
            .iter()
            .map(|(ordinal, vector)| (ordinal, vector.decode()))
            .collect();
        if decoded.is_empty() {
            return Self {
                graph: OrdinalMap::default(),
                entry_ordinal: None,
                default_list_size: list_size,
                max_degree,
                alpha,
            };
        }

        // MIPS uses the standard norm-augmentation reduction to turn inner
        // product ordering into a non-negative L2 distance for graph
        // construction.  The bound is derived once from the immutable base
        // and is shared by every RobustPrune invocation.
        let mips_norm_bound = decoded
            .values()
            .map(|vector| vector_norm(vector))
            .fold(0.0_f64, f64::max);
        let entry_ordinal = medoid(&decoded, ordinals);
        let mut graph = initial_graph(&decoded, max_degree);
        let build_order = shuffled_ordinals(decoded.keys().collect(), BUILD_ORDER_SEED);
        for pass_alpha in [1.0, alpha] {
            for ordinal in build_order.iter().copied() {
                let Some(query) = decoded.get(ordinal) else {
                    continue;
                };
                let search = greedy_search(
                    &graph,
                    &decoded,
                    ordinals,
                    query,
                    entry_ordinal,
                    list_size,
                    metric,
                );
                robust_prune(
                    &mut graph,
                    &decoded,
                    ordinals,
                    ordinal,
                    &search.visited,
                    pass_alpha,
                    max_degree,
                    metric,
                    mips_norm_bound,
                );

                let neighbors = graph.get(ordinal).cloned().unwrap_or_default();
                for neighbor in neighbors {
                    let mut candidates = graph.get(neighbor).cloned().unwrap_or_default();
                    if !candidates.contains(&ordinal) {
                        candidates.push(ordinal);
                    }
                    if candidates.len() > max_degree {
                        robust_prune(
                            &mut graph,
                            &decoded,
                            ordinals,
                            neighbor,
                            &candidates,
                            pass_alpha,
                            max_degree,
                            metric,
                            mips_norm_bound,
                        );
                    } else {
                        graph.insert(neighbor, candidates);
                    }
                }
            }
        }

        Self {
            graph,
            entry_ordinal: Some(entry_ordinal),
            default_list_size: list_size,
            max_degree,
            alpha,
        }
    }

    pub(super) fn candidates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        query: &[f32],
        requested_list_size: Option<usize>,
        topk: usize,
        metric: MetricType,
    ) -> RoaringTreemap {
        self.scored_candidates(
            vectors.len(),
            ordinals,
            requested_list_size,
            topk,
            &|ordinal| {
                vectors
                    .get(ordinal)
                    .map(|vector| score(query, vector, metric))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn filtered_candidates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        query: &[f32],
        result_limit: usize,
        traversal_limit: usize,
        metric: MetricType,
        allowed: &RoaringTreemap,
        excluded: &RoaringTreemap,
    ) -> RoaringTreemap {
        self.scored_filtered_candidates(
            vectors.len(),
            ordinals,
            result_limit,
            traversal_limit,
            allowed,
            excluded,
            &|ordinal| {
                vectors
                    .get(ordinal)
                    .map(|vector| score(query, vector, metric))
            },
        )
    }

    pub(super) fn scored_candidates(
        &self,
        vector_count: usize,
        ordinals: &OrdinalTable,
        requested_list_size: Option<usize>,
        topk: usize,
        score_for: &impl Fn(u64) -> Option<f64>,
    ) -> RoaringTreemap {
        let Some(entry) = self.entry_ordinal else {
            return RoaringTreemap::new();
        };
        let limit = self.candidate_limit(requested_list_size, topk, vector_count);
        if limit >= vector_count {
            return self.graph.keys().collect();
        }
        bounded_search(&self.graph, ordinals, entry, limit, score_for)
            .candidates
            .into_iter()
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scored_filtered_candidates(
        &self,
        vector_count: usize,
        ordinals: &OrdinalTable,
        result_limit: usize,
        traversal_limit: usize,
        allowed: &RoaringTreemap,
        excluded: &RoaringTreemap,
        score_for: &impl Fn(u64) -> Option<f64>,
    ) -> RoaringTreemap {
        let Some(entry) = self.entry_ordinal else {
            return RoaringTreemap::new();
        };
        let search = bounded_search(
            &self.graph,
            ordinals,
            entry,
            traversal_limit.min(vector_count).max(1),
            score_for,
        );
        let mut scored: Vec<ScoredOrdinal> = search
            .visited
            .into_iter()
            .chain(search.candidates)
            .filter(|ordinal| allowed.contains(*ordinal) && !excluded.contains(*ordinal))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|ordinal| score_for(ordinal).map(|score| ScoredOrdinal { ordinal, score }))
            .collect();
        sort_scored(&mut scored, ordinals);
        scored
            .into_iter()
            .take(result_limit)
            .map(|candidate| candidate.ordinal)
            .collect()
    }

    pub(super) fn candidate_limit(
        &self,
        requested_list_size: Option<usize>,
        topk: usize,
        vector_count: usize,
    ) -> usize {
        requested_list_size
            .unwrap_or(self.default_list_size)
            .max(topk)
            .min(vector_count)
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        let ordinal_bytes = std::mem::size_of::<u64>();
        self.graph.iter().fold(
            self.graph
                .slot_count()
                .saturating_add(self.entry_ordinal.map_or(0, |_| ordinal_bytes)),
            |total, (_, neighbors)| {
                total.saturating_add(neighbors.len().saturating_mul(ordinal_bytes))
            },
        )
    }

    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        max_degree: usize,
        list_size: usize,
        alpha: f64,
    ) -> bool {
        if !vectors.validates(vectors.slot_count())
            || !self.graph.validates(vectors.slot_count())
            || self.max_degree != max_degree
            || self.default_list_size != list_size
            || self.alpha.to_bits() != alpha.to_bits()
            || max_degree == 0
            || list_size == 0
            || !alpha.is_finite()
            || alpha < 1.0
        {
            return false;
        }
        if vectors.is_empty() {
            return self.entry_ordinal.is_none() && self.graph.is_empty();
        }
        let Some(entry) = self.entry_ordinal else {
            return false;
        };
        if !vectors.contains_key(entry) || self.graph.keys().ne(vectors.keys()) {
            return false;
        }
        self.graph.iter().all(|(ordinal, neighbors)| {
            neighbors.len() <= max_degree
                && neighbors.iter().all(|neighbor| {
                    *neighbor != ordinal
                        && vectors.contains_key(*neighbor)
                        && self.graph.contains_key(*neighbor)
                })
                && neighbors.iter().copied().collect::<BTreeSet<_>>().len() == neighbors.len()
        })
    }
}

fn initial_graph(vectors: &OrdinalMap<Vec<f32>>, max_degree: usize) -> OrdinalMap<Vec<u64>> {
    let order = shuffled_ordinals(vectors.keys().collect(), INITIAL_GRAPH_SEED);
    let degree = max_degree.min(order.len().saturating_sub(1));
    order
        .iter()
        .enumerate()
        .map(|(index, ordinal)| {
            let neighbors = (1..=degree)
                .map(|offset| order[(index + offset) % order.len()])
                .collect();
            (*ordinal, neighbors)
        })
        .collect()
}

fn medoid(vectors: &OrdinalMap<Vec<f32>>, ordinals: &OrdinalTable) -> u64 {
    let dimension = vectors.values().next().map_or(0, Vec::len);
    let mut centroid = vec![0.0_f64; dimension];
    for vector in vectors.values() {
        for (total, value) in centroid.iter_mut().zip(vector) {
            *total += f64::from(*value);
        }
    }
    let count = usize_to_f64(vectors.len()).max(1.0);
    for value in &mut centroid {
        *value /= count;
    }

    vectors
        .iter()
        .map(|(ordinal, vector)| {
            let distance = centroid
                .iter()
                .zip(vector)
                .map(|(left, right)| {
                    let delta = *left - f64::from(*right);
                    delta * delta
                })
                .sum::<f64>();
            (ordinal, distance)
        })
        .min_by(|left, right| {
            left.1.total_cmp(&right.1).then_with(|| {
                ordinals
                    .id(left.0)
                    .unwrap_or_default()
                    .cmp(ordinals.id(right.0).unwrap_or_default())
                    .then_with(|| left.0.cmp(&right.0))
            })
        })
        .map_or(0, |(ordinal, _)| ordinal)
}

#[allow(clippy::too_many_arguments)]
fn robust_prune(
    graph: &mut OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<Vec<f32>>,
    ordinals: &OrdinalTable,
    source: u64,
    candidates: &[u64],
    alpha: f64,
    max_degree: usize,
    metric: MetricType,
    mips_norm_bound: f64,
) {
    let Some(source_vector) = vectors.get(source) else {
        return;
    };
    let mut pool: Vec<u64> = graph
        .get(source)
        .into_iter()
        .flatten()
        .copied()
        .chain(candidates.iter().copied())
        .filter(|candidate| *candidate != source && vectors.contains_key(*candidate))
        .collect();
    pool.sort_unstable();
    pool.dedup();
    let mut selected = Vec::with_capacity(max_degree.min(pool.len()));
    let alpha_squared = alpha * alpha;

    while !pool.is_empty() && selected.len() < max_degree {
        let closest_index = pool
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                let left_distance = vectors.get(**left).map_or(f64::INFINITY, |vector| {
                    metric_distance(source_vector, vector, metric, mips_norm_bound)
                });
                let right_distance = vectors.get(**right).map_or(f64::INFINITY, |vector| {
                    metric_distance(source_vector, vector, metric, mips_norm_bound)
                });
                left_distance.total_cmp(&right_distance).then_with(|| {
                    ordinals
                        .id(**left)
                        .unwrap_or_default()
                        .cmp(ordinals.id(**right).unwrap_or_default())
                        .then_with(|| left.cmp(right))
                })
            })
            .map_or(0, |(index, _)| index);
        let closest = pool[closest_index];
        selected.push(closest);
        let Some(closest_vector) = vectors.get(closest) else {
            break;
        };
        pool.retain(|candidate| {
            // Remove the selected node by identity.  Metric distances can be
            // equal up to a tiny floating-point residual (for example, two
            // collinear cosine vectors), and the occlusion inequality alone
            // could otherwise retain the same ordinal repeatedly.
            if *candidate == closest {
                return false;
            }
            let Some(candidate_vector) = vectors.get(*candidate) else {
                return false;
            };
            alpha_squared
                * metric_distance(closest_vector, candidate_vector, metric, mips_norm_bound)
                > metric_distance(source_vector, candidate_vector, metric, mips_norm_bound)
        });
    }
    graph.insert(source, selected);
}

fn greedy_search(
    graph: &OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<Vec<f32>>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entry: u64,
    list_size: usize,
    metric: MetricType,
) -> GraphSearch {
    bounded_search(graph, ordinals, entry, list_size, &|ordinal| {
        vectors
            .get(ordinal)
            .map(|vector| score_dense(query, vector, metric))
    })
}

fn bounded_search(
    graph: &OrdinalMap<Vec<u64>>,
    ordinals: &OrdinalTable,
    entry: u64,
    list_size: usize,
    score_for: &impl Fn(u64) -> Option<f64>,
) -> GraphSearch {
    let limit = list_size.max(1);
    let mut pool = score_for(entry).map_or_else(Vec::new, |score| {
        vec![ScoredOrdinal {
            ordinal: entry,
            score,
        }]
    });
    let mut visited = HashSet::with_capacity(limit.saturating_mul(2).min(graph.len()));
    let mut visited_order = Vec::with_capacity(limit.saturating_mul(2).min(graph.len()));

    loop {
        sort_scored(&mut pool, ordinals);
        let Some(current) = pool
            .iter()
            .find(|candidate| !visited.contains(&candidate.ordinal))
            .copied()
        else {
            break;
        };
        visited.insert(current.ordinal);
        visited_order.push(current.ordinal);
        for neighbor in graph.get(current.ordinal).into_iter().flatten().copied() {
            if visited.contains(&neighbor)
                || pool.iter().any(|candidate| candidate.ordinal == neighbor)
            {
                continue;
            }
            if let Some(score) = score_for(neighbor) {
                pool.push(ScoredOrdinal {
                    ordinal: neighbor,
                    score,
                });
            }
        }
        sort_scored(&mut pool, ordinals);
        pool.truncate(limit);
    }
    sort_scored(&mut pool, ordinals);
    GraphSearch {
        candidates: pool
            .into_iter()
            .map(|candidate| candidate.ordinal)
            .collect(),
        visited: visited_order,
    }
}

fn sort_scored(values: &mut [ScoredOrdinal], ordinals: &OrdinalTable) {
    values.sort_unstable_by(|left, right| {
        right.score.total_cmp(&left.score).then_with(|| {
            ordinals
                .id(left.ordinal)
                .unwrap_or_default()
                .cmp(ordinals.id(right.ordinal).unwrap_or_default())
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        })
    });
}

fn squared_l2(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = f64::from(*left) - f64::from(*right);
            delta * delta
        })
        .sum()
}

/// Returns a non-negative distance suitable for Vamana `RobustPrune`.
///
/// The public query contract ranks by similarity for IP/MIPS/Cosine.  Graph
/// pruning, however, needs a proper distance and cannot safely compare raw
/// (possibly negative) similarities.  Cosine uses angular distance directly.
/// Inner product uses the standard norm-augmentation reduction: with a bound
/// `R >= ||x||` for every base vector, augmenting `x` by
/// `sqrt(R² - ||x||²)` makes squared L2 ordering equivalent to maximizing
/// `q·x` for a query whose extra coordinate is zero.
fn metric_distance(left: &[f32], right: &[f32], metric: MetricType, mips_norm_bound: f64) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    match metric {
        MetricType::L2 => squared_l2(left, right),
        MetricType::Cosine => {
            let left_norm = vector_norm(left);
            let right_norm = vector_norm(right);
            if left_norm == 0.0 || right_norm == 0.0 {
                1.0
            } else {
                let dot = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>();
                (1.0 - dot / (left_norm * right_norm)).max(0.0)
            }
        }
        MetricType::Ip | MetricType::MipsL2 | MetricType::Undefined => {
            let bound = mips_norm_bound
                .max(vector_norm(left))
                .max(vector_norm(right));
            let left_extra = (bound * bound - vector_norm_squared(left)).max(0.0).sqrt();
            let right_extra = (bound * bound - vector_norm_squared(right)).max(0.0).sqrt();
            squared_l2(left, right) + (left_extra - right_extra).powi(2)
        }
    }
}

fn vector_norm_squared(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum()
}

fn vector_norm(vector: &[f32]) -> f64 {
    vector_norm_squared(vector).sqrt()
}

fn shuffled_ordinals(mut values: Vec<u64>, seed: u64) -> Vec<u64> {
    let mut state = seed;
    for index in (1..values.len()).rev() {
        state = splitmix64(state);
        let bound = u64::try_from(index.saturating_add(1)).unwrap_or(u64::MAX);
        let swap = usize::try_from(state % bound).unwrap_or(0);
        values.swap(index, swap);
    }
    values
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^ (state >> 31)
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{metric_distance, VamanaIndex};
    use crate::doc::{Doc, DocumentMap};
    use crate::index::ordinal_map::OrdinalMap;
    use crate::index::ordinals::OrdinalTable;
    use crate::index::quantization::QuantizedVector;
    use crate::types::{MetricType, QuantizeType};
    use roaring::RoaringTreemap;
    use std::sync::Arc;

    fn fixture() -> (OrdinalTable, OrdinalMap<QuantizedVector>) {
        let docs: DocumentMap = (0_u16..96)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let doc = Doc::with_pk(&id).expect("document id must be valid");
                (id, Arc::new(doc))
            })
            .collect();
        let ordinals = OrdinalTable::build(&docs).expect("ordinals must build");
        let vectors = (0_u16..96)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let values = vec![
                    f32::from(index),
                    f32::from(index % 11),
                    f32::from((index * 7) % 13),
                ];
                (
                    ordinals.ordinal(&id).expect("document ordinal must exist"),
                    QuantizedVector::encode(values, QuantizeType::Undefined)
                        .expect("encoding must succeed"),
                )
            })
            .collect();
        (ordinals, vectors)
    }

    #[test]
    fn two_pass_graph_is_deterministic_and_degree_bounded() {
        let (ordinals, vectors) = fixture();
        let first = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, MetricType::L2);
        let second = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, MetricType::L2);
        assert_eq!(first.graph, second.graph);
        assert_eq!(first.entry_ordinal, second.entry_ordinal);
        assert!(first.graph.values().all(|neighbors| neighbors.len() <= 12));
        assert!(first.validates(&vectors, 12, 48, 1.2));
    }

    #[test]
    fn metric_aware_graphs_are_deterministic_and_validated() {
        let (ordinals, vectors) = fixture();
        for metric in [
            MetricType::L2,
            MetricType::Ip,
            MetricType::Cosine,
            MetricType::MipsL2,
        ] {
            let first = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, metric);
            let second = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, metric);
            assert_eq!(first.graph, second.graph, "metric={metric:?}");
            assert!(first.validates(&vectors, 12, 48, 1.2), "metric={metric:?}");
            assert!(first.graph.values().all(|neighbors| {
                neighbors
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == neighbors.len()
            }));
        }
    }

    #[test]
    fn exhaustive_list_returns_every_ordinal() {
        let (ordinals, vectors) = fixture();
        let index = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, MetricType::L2);
        let candidates = index.candidates(
            &vectors,
            &ordinals,
            &[41.0, 8.0, 1.0],
            Some(vectors.len()),
            10,
            MetricType::L2,
        );
        assert_eq!(
            candidates.len(),
            u64::try_from(vectors.len()).unwrap_or(u64::MAX)
        );
    }

    #[test]
    fn cosine_pruning_keeps_duplicate_vectors_canonical() {
        let docs: DocumentMap = (0_u8..6)
            .map(|index| {
                let id = format!("duplicate-{index}");
                let doc = Doc::with_pk(&id).expect("document id must be valid");
                (id, Arc::new(doc))
            })
            .collect();
        let ordinals = OrdinalTable::build(&docs).expect("ordinals must build");
        let values = [
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 2.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let vectors = values
            .into_iter()
            .enumerate()
            .map(|(index, values)| {
                let id = format!("duplicate-{index}");
                (
                    ordinals.ordinal(&id).expect("ordinal must exist"),
                    QuantizedVector::encode(values, QuantizeType::Undefined)
                        .expect("vector must encode"),
                )
            })
            .collect();
        let index = VamanaIndex::build(&vectors, &ordinals, 4, 6, 1.2, MetricType::Cosine);
        assert!(index.validates(&vectors, 4, 6, 1.2));
        assert!(index.graph.values().all(|neighbors| neighbors
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == neighbors.len()));
    }

    #[test]
    fn metric_distance_is_finite_non_negative_and_orders_similarity_neighbors() {
        let query = [1.0_f32, 0.0];
        let aligned = [2.0_f32, 0.0];
        let orthogonal = [0.0_f32, 2.0];
        for metric in [MetricType::Cosine, MetricType::Ip, MetricType::MipsL2] {
            let aligned_distance = metric_distance(&query, &aligned, metric, 2.0);
            let orthogonal_distance = metric_distance(&query, &orthogonal, metric, 2.0);
            assert!(aligned_distance.is_finite() && aligned_distance >= 0.0);
            assert!(orthogonal_distance.is_finite() && orthogonal_distance >= 0.0);
            assert!(aligned_distance < orthogonal_distance, "metric={metric:?}");
        }
    }

    #[test]
    fn filtered_similarity_search_matches_the_exact_eligible_oracle() {
        let (ordinals, vectors) = fixture();
        let allowed: RoaringTreemap = vectors.keys().filter(|ordinal| ordinal % 2 == 0).collect();
        let excluded: RoaringTreemap = [40_u64].into_iter().collect();
        let query = [41.0, 8.0, 1.0];
        for metric in [MetricType::Ip, MetricType::Cosine, MetricType::MipsL2] {
            let index = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, metric);
            let actual = index.filtered_candidates(
                &vectors,
                &ordinals,
                &query,
                10,
                vectors.len(),
                metric,
                &allowed,
                &excluded,
            );
            let mut expected: Vec<_> = vectors
                .iter()
                .filter(|(ordinal, _)| allowed.contains(*ordinal) && !excluded.contains(*ordinal))
                .map(|(ordinal, vector)| {
                    (
                        ordinal,
                        super::super::quantization::score(&query, vector, metric),
                    )
                })
                .collect();
            expected.sort_unstable_by(|left, right| {
                right.1.total_cmp(&left.1).then_with(|| {
                    ordinals
                        .id(left.0)
                        .unwrap_or_default()
                        .cmp(ordinals.id(right.0).unwrap_or_default())
                })
            });
            let expected: RoaringTreemap = expected
                .into_iter()
                .take(10)
                .map(|(ordinal, _)| ordinal)
                .collect();
            assert_eq!(actual, expected, "metric={metric:?}");
        }
    }

    #[test]
    fn filtered_search_keeps_navigation_bridges_and_exact_eligible_candidates() {
        let (ordinals, vectors) = fixture();
        let index = VamanaIndex::build(&vectors, &ordinals, 12, 48, 1.2, MetricType::L2);
        let allowed: RoaringTreemap = vectors.keys().filter(|ordinal| ordinal % 2 == 0).collect();
        let excluded: RoaringTreemap = [40_u64].into_iter().collect();
        let query = [41.0, 8.0, 1.0];
        let actual = index.filtered_candidates(
            &vectors,
            &ordinals,
            &query,
            10,
            vectors.len(),
            MetricType::L2,
            &allowed,
            &excluded,
        );
        let mut scored: Vec<_> = vectors
            .iter()
            .filter(|(ordinal, _)| allowed.contains(*ordinal) && !excluded.contains(*ordinal))
            .map(|(ordinal, vector)| {
                (
                    ordinal,
                    crate::index::quantization::score(&query, vector, MetricType::L2),
                )
            })
            .collect();
        scored.sort_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| {
                ordinals
                    .id(left.0)
                    .unwrap_or_default()
                    .cmp(ordinals.id(right.0).unwrap_or_default())
            })
        });
        let expected: RoaringTreemap = scored
            .into_iter()
            .take(10)
            .map(|(ordinal, _)| ordinal)
            .collect();
        assert_eq!(actual, expected);
    }
}
