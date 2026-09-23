//! Deterministic in-memory HNSW graph construction and traversal.

mod search;

use self::search::{
    greedy_search_by, prefetch_f32_at, search_layer_by, search_layer_filtered_by,
    serial_neighbor_batch,
};
use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::quantization::{
    dense_candidate_norm, dense_query_norm, dense_query_norm_fast, score_ann, score_dense_cosine,
    score_dense_fast, score_dense_with_query_norm, QuantizedVector,
};
use crate::types::MetricType;
use roaring::RoaringTreemap;
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct HnswIndex {
    entry_ordinal: Option<u64>,
    /// Layer zero is the complete base graph; later entries are progressively
    /// sparser upper layers.
    layers: Vec<OrdinalMap<Vec<u64>>>,
    default_ef: usize,
}

#[derive(Clone, Copy)]
pub(super) struct HnswFilter<'a> {
    pub allowed: &'a RoaringTreemap,
    pub excluded: &'a RoaringTreemap,
    pub eligible_count: usize,
}

fn prefetch_navigation(packed: Option<(usize, &[f32])>, ordinal: u64) {
    let Some((dimension, values)) = packed else {
        return;
    };
    let Some(index) = usize::try_from(ordinal).ok() else {
        return;
    };
    let Some(start) = index.checked_mul(dimension) else {
        return;
    };
    if start >= values.len() || dimension == 0 {
        return;
    }
    let end = start.saturating_add(dimension).min(values.len());
    let mut offset = start;
    while offset < end {
        prefetch_f32_at(values, offset);
        offset = offset.saturating_add(16);
    }
}

fn stored_cosine_norm(norms: Option<&[f32]>, ordinal: u64) -> Option<f32> {
    let norms = norms?;
    let index = usize::try_from(ordinal).ok()?;
    let value = *norms.get(index)?;
    value.is_finite().then_some(value)
}

#[allow(clippy::too_many_arguments)]
fn navigation_score(
    query: &[f32],
    vectors: &OrdinalMap<QuantizedVector>,
    ordinal: u64,
    metric: MetricType,
    query_norm_f32: f32,
    query_norm_f64: f64,
    candidate_norms: Option<&[f32]>,
    packed: Option<(usize, &[f32])>,
) -> Option<f64> {
    let candidate_norm = stored_cosine_norm(candidate_norms, ordinal);
    if let Some((dimension, values)) = packed {
        if let Some(score) = finite_packed_score(
            query,
            values,
            dimension,
            ordinal,
            metric,
            query_norm_f32,
            candidate_norm,
        ) {
            return Some(score);
        }
    }
    vectors.get(ordinal).map(|vector| {
        score_ann(
            query,
            vector,
            metric,
            query_norm_f32,
            query_norm_f64,
            candidate_norm,
        )
    })
}

fn finite_packed_score(
    query: &[f32],
    values: &[f32],
    dimension: usize,
    ordinal: u64,
    metric: MetricType,
    query_norm: f32,
    candidate_norm: Option<f32>,
) -> Option<f64> {
    let index = usize::try_from(ordinal).ok()?;
    let start = index.checked_mul(dimension)?;
    let end = start.checked_add(dimension)?;
    let candidate = values.get(start..end)?;
    let fast = score_dense_fast(query, candidate, metric, query_norm, candidate_norm);
    fast.is_finite().then_some(f64::from(fast))
}

impl HnswIndex {
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        m: usize,
        ef_construction: usize,
        metric: MetricType,
    ) -> Self {
        Self::build_with(vectors, ordinals, m, ef_construction, metric, true, true)
    }

    /// `cache_scores` reuses the f64 link score already computed for an edge.
    /// The false path is the original rescore, kept so tests can prove the
    /// published neighbor lists do not change.
    ///
    /// `parallel_scores` scores one neighbor list concurrently. Candidates are
    /// still admitted in neighbor-list order with the same `f64` scores, so
    /// the published graph matches the serial insertion.
    fn build_with(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        m: usize,
        ef_construction: usize,
        metric: MetricType,
        cache_scores: bool,
        parallel_scores: bool,
    ) -> Self {
        Self::build_timed(
            vectors,
            ordinals,
            m,
            ef_construction,
            metric,
            cache_scores,
            parallel_scores,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn build_timed(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        m: usize,
        ef_construction: usize,
        metric: MetricType,
        cache_scores: bool,
        parallel_scores: bool,
    ) -> Self {
        if vectors.is_empty() {
            return Self {
                entry_ordinal: None,
                layers: vec![OrdinalMap::default()],
                default_ef: ef_construction,
            };
        }
        let decoded: OrdinalMap<Vec<f32>> = vectors
            .iter()
            .map(|(ordinal, vector)| (ordinal, vector.decode()))
            .collect();
        let norms = if metric == MetricType::Cosine {
            cosine_norms(&decoded)
        } else {
            Vec::new()
        };
        let slab = pack_decoded(&decoded);
        let mut graph = Self {
            entry_ordinal: None,
            layers: vec![OrdinalMap::default()],
            default_ef: ef_construction,
        };
        let mut edge_scores = cache_scores.then(|| vec![OrdinalMap::<Vec<f64>>::default()]);
        let mut maximum_level = 0;

        for (position, (ordinal, query)) in decoded.iter().enumerate() {
            let level = deterministic_level(position, m);
            while graph.layers.len() <= level {
                graph.layers.push(OrdinalMap::default());
                if let Some(scores) = edge_scores.as_mut() {
                    scores.push(OrdinalMap::default());
                }
            }
            for layer in 0..=level {
                graph.layers[layer].get_or_insert_default(ordinal);
                if let Some(scores) = edge_scores.as_mut() {
                    scores[layer].get_or_insert_default(ordinal);
                }
            }

            let Some(mut entry) = graph.entry_ordinal else {
                graph.entry_ordinal = Some(ordinal);
                maximum_level = level;
                continue;
            };

            let query_norm_f64 = if metric == MetricType::Cosine {
                dense_query_norm(query)
            } else {
                0.0
            };
            let score_for = |candidate: u64| {
                let vector = slab
                    .as_ref()
                    .and_then(|packed| slab_slice(packed, candidate))
                    .or_else(|| decoded.get(candidate).map(Vec::as_slice))?;
                Some(if metric == MetricType::Cosine {
                    let candidate_norm = cached_cosine_norm(&norms, candidate)
                        .unwrap_or_else(|| dense_candidate_norm(vector));
                    score_dense_cosine(query, vector, query_norm_f64, candidate_norm)
                } else {
                    score_dense_with_query_norm(query, vector, metric, query_norm_f64)
                })
            };
            let prefetch_for = |candidate: u64| {
                if let Some(packed) = slab.as_ref() {
                    prefetch_navigation(
                        Some((packed.dimension, packed.values.as_slice())),
                        candidate,
                    );
                }
            };
            let relative_error =
                crate::score_f64::f32_cosine_score_error(query.len(), query_norm_f64);
            let absolute_error = crate::score_f64::f32_dot_absolute_error(query.len());
            let batch_score = |neighbors: &[u64], out: &mut [Option<f64>], floor: f64| -> bool {
                if metric != MetricType::Cosine || out.len() != neighbors.len() {
                    return false;
                }
                let Some(packed) = slab.as_ref() else {
                    return false;
                };
                let key_floor = floor * query_norm_f64;
                let can_reject = query_norm_f64.is_finite()
                    && query_norm_f64 > 0.0
                    && floor.is_finite()
                    && key_floor.is_finite();
                let mut pending = [0_u64; 8];
                let mut pending_at = [0_usize; 8];
                let mut pending_len = 0_usize;
                for (index, candidate) in neighbors.iter().copied().enumerate() {
                    if let Some(ahead) = neighbors.get(index + 8).copied() {
                        prefetch_for(ahead);
                    }
                    if can_reject {
                        let dominated = slab_slice(packed, candidate).is_some_and(|slice| {
                            cached_cosine_norm(&norms, candidate).is_some_and(|norm| {
                                cosine_cannot_beat(
                                    query,
                                    slice,
                                    norm,
                                    relative_error,
                                    absolute_error,
                                    key_floor,
                                )
                            })
                        });
                        if dominated {
                            out[index] = None;
                            continue;
                        }
                    }
                    pending[pending_len] = candidate;
                    pending_at[pending_len] = index;
                    pending_len += 1;
                    if pending_len == 8 {
                        write_exact_group(
                            query,
                            packed,
                            &norms,
                            query_norm_f64,
                            &pending,
                            &pending_at,
                            out,
                            &score_for,
                        );
                        pending_len = 0;
                    }
                }
                if pending_len > 0 {
                    write_exact_group(
                        query,
                        packed,
                        &norms,
                        query_norm_f64,
                        &pending[..pending_len],
                        &pending_at[..pending_len],
                        out,
                        &score_for,
                    );
                }
                true
            };
            if maximum_level > level {
                for layer in ((level + 1)..=maximum_level).rev() {
                    entry = greedy_search_by(
                        &graph.layers[layer],
                        ordinals,
                        entry,
                        &score_for,
                        &prefetch_for,
                        parallel_scores,
                        &batch_score,
                    );
                }
            }

            let connection_top = level.min(maximum_level);
            for layer in (0..=connection_top).rev() {
                let found = search_layer_by(
                    &graph.layers[layer],
                    &[entry],
                    ef_construction,
                    ordinals,
                    &score_for,
                    &prefetch_for,
                    parallel_scores,
                    &batch_score,
                );
                let degree = if layer == 0 { m.saturating_mul(2) } else { m }.max(1);
                let mut neighbor_ids = Vec::with_capacity(degree);
                let mut neighbor_scores = Vec::with_capacity(degree);
                for (candidate, score) in &found {
                    if *candidate == ordinal {
                        continue;
                    }
                    neighbor_ids.push(*candidate);
                    neighbor_scores.push(*score);
                    if neighbor_ids.len() == degree {
                        break;
                    }
                }
                graph.layers[layer].insert(ordinal, neighbor_ids.clone());
                if let Some(scores) = edge_scores.as_mut() {
                    scores[layer].insert(ordinal, neighbor_scores.clone());
                }
                for (neighbor, score) in neighbor_ids.into_iter().zip(neighbor_scores) {
                    let added =
                        if let Some(edges) = graph.layers[layer].get_or_insert_default(neighbor) {
                            if edges.contains(&ordinal) {
                                false
                            } else {
                                edges.push(ordinal);
                                true
                            }
                        } else {
                            false
                        };
                    if added {
                        let edge_len = graph.layers[layer].get(neighbor).map_or(0, Vec::len);
                        if let Some(scores) = edge_scores.as_mut() {
                            remember_scored_link(&mut scores[layer], neighbor, score, edge_len);
                        }
                    }
                    prune_edges(
                        &mut graph.layers[layer],
                        edge_scores.as_mut().map(|scores| &mut scores[layer]),
                        &decoded,
                        &norms,
                        ordinals,
                        neighbor,
                        degree,
                        metric,
                    );
                }
                if let Some((best, _)) = found.first() {
                    entry = *best;
                }
            }

            if level > maximum_level {
                graph.entry_ordinal = Some(ordinal);
                maximum_level = level;
            }
        }
        graph
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn candidates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        query: &[f32],
        requested_ef: Option<usize>,
        topk: usize,
        metric: MetricType,
        candidate_norms: Option<&[f32]>,
        packed: Option<(usize, &[f32])>,
    ) -> RoaringTreemap {
        if vectors.is_empty() {
            return RoaringTreemap::new();
        }
        let query_norm_f32 = if metric == MetricType::Cosine {
            dense_query_norm_fast(query)
        } else {
            0.0
        };
        let query_norm_f64 = if metric == MetricType::Cosine {
            dense_query_norm(query)
        } else {
            0.0
        };
        self.candidates_by(
            ordinals,
            requested_ef,
            topk,
            &|ordinal| {
                navigation_score(
                    query,
                    vectors,
                    ordinal,
                    metric,
                    query_norm_f32,
                    query_norm_f64,
                    candidate_norms,
                    packed,
                )
            },
            &|ordinal| prefetch_navigation(packed, ordinal),
            true,
        )
    }

    pub(super) fn candidates_by(
        &self,
        ordinals: &OrdinalTable,
        requested_ef: Option<usize>,
        topk: usize,
        score_for: &(impl Fn(u64) -> Option<f64> + Sync),
        prefetch_for: &impl Fn(u64),
        parallel: bool,
    ) -> RoaringTreemap {
        let vector_count = self.layers.first().map_or(0, OrdinalMap::len);
        if vector_count == 0 {
            return RoaringTreemap::new();
        }
        let ef = self.candidate_limit(requested_ef, topk, vector_count);
        if ef >= vector_count {
            return self.layers[0].keys().collect();
        }
        let Some(mut entry) = self.entry_ordinal else {
            return RoaringTreemap::new();
        };
        for layer in (1..self.layers.len()).rev() {
            entry = greedy_search_by(
                &self.layers[layer],
                ordinals,
                entry,
                score_for,
                prefetch_for,
                parallel,
                &serial_neighbor_batch,
            );
        }
        search_layer_by(
            &self.layers[0],
            &[entry],
            ef,
            ordinals,
            score_for,
            prefetch_for,
            parallel,
            &serial_neighbor_batch,
        )
        .into_iter()
        .map(|(ordinal, _)| ordinal)
        .take(ef)
        .collect()
    }

    /// Searches for eligible results while retaining every graph node as a
    /// possible navigation bridge. Filtering only the result heap prevents a
    /// namespace or language predicate from disconnecting the graph.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn filtered_candidates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        query: &[f32],
        result_limit: usize,
        traversal_limit: usize,
        metric: MetricType,
        filter: HnswFilter<'_>,
        candidate_norms: Option<&[f32]>,
        packed: Option<(usize, &[f32])>,
    ) -> RoaringTreemap {
        if vectors.is_empty() || filter.eligible_count == 0 || result_limit == 0 {
            return RoaringTreemap::new();
        }
        let query_norm_f32 = if metric == MetricType::Cosine {
            dense_query_norm_fast(query)
        } else {
            0.0
        };
        let query_norm_f64 = if metric == MetricType::Cosine {
            dense_query_norm(query)
        } else {
            0.0
        };
        self.filtered_candidates_by(
            ordinals,
            result_limit,
            traversal_limit,
            filter,
            &|ordinal| {
                navigation_score(
                    query,
                    vectors,
                    ordinal,
                    metric,
                    query_norm_f32,
                    query_norm_f64,
                    candidate_norms,
                    packed,
                )
            },
            &|ordinal| prefetch_navigation(packed, ordinal),
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn filtered_candidates_by(
        &self,
        ordinals: &OrdinalTable,
        result_limit: usize,
        traversal_limit: usize,
        filter: HnswFilter<'_>,
        score_for: &(impl Fn(u64) -> Option<f64> + Sync),
        prefetch_for: &impl Fn(u64),
        parallel: bool,
    ) -> RoaringTreemap {
        if self.layers.first().map_or(true, OrdinalMap::is_empty)
            || filter.eligible_count == 0
            || result_limit == 0
        {
            return RoaringTreemap::new();
        }
        if result_limit >= filter.eligible_count {
            return self.layers[0]
                .keys()
                .filter(|ordinal| {
                    filter.allowed.contains(*ordinal) && !filter.excluded.contains(*ordinal)
                })
                .collect();
        }
        let Some(mut entry) = self.entry_ordinal else {
            return RoaringTreemap::new();
        };
        for layer in (1..self.layers.len()).rev() {
            entry = greedy_search_by(
                &self.layers[layer],
                ordinals,
                entry,
                score_for,
                prefetch_for,
                parallel,
                &serial_neighbor_batch,
            );
        }
        search_layer_filtered_by(
            &self.layers[0],
            &[entry],
            result_limit,
            traversal_limit,
            ordinals,
            score_for,
            prefetch_for,
            &|ordinal| filter.allowed.contains(ordinal) && !filter.excluded.contains(ordinal),
            parallel,
            &serial_neighbor_batch,
        )
        .into_iter()
        .collect()
    }

    pub(super) fn candidate_limit(
        &self,
        requested_ef: Option<usize>,
        topk: usize,
        vector_count: usize,
    ) -> usize {
        requested_ef
            .unwrap_or(self.default_ef)
            .max(topk)
            .min(vector_count)
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        let ordinal_bytes = std::mem::size_of::<u64>();
        self.layers.iter().fold(
            self.entry_ordinal.map_or(0, |_| ordinal_bytes),
            |total, layer| {
                layer.iter().fold(
                    total.saturating_add(layer.slot_count()),
                    |total, (_, neighbors)| {
                        total.saturating_add(neighbors.len().saturating_mul(ordinal_bytes))
                    },
                )
            },
        )
    }

    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        m: usize,
        ef_construction: usize,
    ) -> bool {
        if !vectors.validates(vectors.slot_count())
            || self.layers.is_empty()
            || self.layers.len() > 33
            || self.default_ef != ef_construction
            || m == 0
            || ef_construction == 0
        {
            return false;
        }
        if vectors.is_empty() {
            return self.entry_ordinal.is_none()
                && self.layers.len() == 1
                && self.layers[0].is_empty();
        }
        let Some(entry) = self.entry_ordinal else {
            return false;
        };
        if self.layers[0].keys().ne(vectors.keys())
            || !self
                .layers
                .last()
                .is_some_and(|layer| layer.contains_key(entry))
        {
            return false;
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if !layer.validates(vectors.slot_count())
                || layer_index > 0
                    && layer
                        .keys()
                        .any(|ordinal| !self.layers[layer_index - 1].contains_key(ordinal))
            {
                return false;
            }
            let degree = if layer_index == 0 {
                m.saturating_mul(2)
            } else {
                m
            }
            .max(1);
            for (ordinal, neighbors) in layer.iter() {
                if neighbors.len() > degree
                    || neighbors
                        .iter()
                        .any(|neighbor| *neighbor == ordinal || !layer.contains_key(*neighbor))
                    || neighbors.iter().copied().collect::<BTreeSet<_>>().len() != neighbors.len()
                {
                    return false;
                }
            }
        }
        true
    }
}

fn score_link(
    vectors: &OrdinalMap<Vec<f32>>,
    norms: &[f64],
    node: u64,
    candidate: u64,
    metric: MetricType,
) -> Option<f64> {
    let query = vectors.get(node)?;
    let vector = vectors.get(candidate)?;
    let query_norm = if metric == MetricType::Cosine {
        cached_cosine_norm(norms, node).unwrap_or_else(|| dense_candidate_norm(query))
    } else {
        0.0
    };
    let score = if metric == MetricType::Cosine {
        let candidate_norm =
            cached_cosine_norm(norms, candidate).unwrap_or_else(|| dense_candidate_norm(vector));
        score_dense_cosine(query, vector, query_norm, candidate_norm)
    } else {
        score_dense_with_query_norm(query, vector, metric, query_norm)
    };
    Some(score)
}

fn remember_scored_link(scores: &mut OrdinalMap<Vec<f64>>, node: u64, score: f64, edge_len: usize) {
    let Some(slot) = scores.get_or_insert_default(node) else {
        return;
    };
    if slot.len() + 1 != edge_len {
        slot.clear();
        return;
    }
    slot.push(score);
}

fn prune_with_cache(
    layer: &mut OrdinalMap<Vec<u64>>,
    scores: &mut OrdinalMap<Vec<f64>>,
    ordinals: &OrdinalTable,
    node: u64,
    degree: usize,
) -> bool {
    let Some(len) = layer.get(node).map(Vec::len) else {
        return false;
    };
    let Some(score_len) = scores.get(node).map(Vec::len) else {
        return false;
    };
    if len != score_len {
        return false;
    }
    if len == 0 {
        return true;
    }
    if len <= 64 {
        let Some(neighbors) = layer.get_mut(node) else {
            return false;
        };
        let Some(cached) = scores.get_mut(node) else {
            return false;
        };
        if neighbors.len() != len || cached.len() != len || neighbors.is_empty() {
            return false;
        }
        // The previous prune left `..len-1` sorted. One appended edge only
        // has to move into that order, which is the same sequence as a full sort.
        let mut index = neighbors.len() - 1;
        while index > 0
            && link_cmp(
                neighbors[index],
                cached[index],
                neighbors[index - 1],
                cached[index - 1],
                ordinals,
            ) == Ordering::Less
        {
            neighbors.swap(index, index - 1);
            cached.swap(index, index - 1);
            index -= 1;
        }
        if neighbors.len() > degree {
            neighbors.truncate(degree);
            cached.truncate(degree);
        }
        return true;
    }
    let Some(mut paired) = paired_cache(layer, scores, node) else {
        return false;
    };
    sort_scored(&mut paired, ordinals);
    paired.truncate(degree);
    let (ids, values): (Vec<u64>, Vec<f64>) = paired.into_iter().unzip();
    layer.insert(node, ids);
    scores.insert(node, values);
    true
}

fn paired_cache(
    layer: &OrdinalMap<Vec<u64>>,
    scores: &OrdinalMap<Vec<f64>>,
    node: u64,
) -> Option<Vec<(u64, f64)>> {
    let neighbors = layer.get(node)?;
    let cached = scores.get(node)?;
    if cached.len() != neighbors.len() {
        return None;
    }
    Some(
        neighbors
            .iter()
            .copied()
            .zip(cached.iter().copied())
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn prune_edges(
    layer: &mut OrdinalMap<Vec<u64>>,
    mut scores: Option<&mut OrdinalMap<Vec<f64>>>,
    vectors: &OrdinalMap<Vec<f32>>,
    norms: &[f64],
    ordinals: &OrdinalTable,
    node: u64,
    degree: usize,
    metric: MetricType,
) {
    if vectors.get(node).is_none() {
        return;
    }
    if let Some(score_map) = scores.as_deref_mut() {
        if prune_with_cache(layer, score_map, ordinals, node, degree) {
            return;
        }
    }
    let mut ranked: Vec<(u64, f64)> = layer
        .get(node)
        .into_iter()
        .flatten()
        .filter_map(|candidate| {
            score_link(vectors, norms, node, *candidate, metric).map(|score| (*candidate, score))
        })
        .collect();
    sort_scored(&mut ranked, ordinals);
    ranked.truncate(degree);
    if let Some(score_map) = scores.as_mut() {
        score_map.insert(node, ranked.iter().map(|(_, score)| *score).collect());
    }
    layer.insert(
        node,
        ranked.into_iter().map(|(ordinal, _)| ordinal).collect(),
    );
}

struct DecodedSlab {
    dimension: usize,
    values: Vec<f32>,
}

fn pack_decoded(decoded: &OrdinalMap<Vec<f32>>) -> Option<DecodedSlab> {
    let mut dimension = None;
    for vector in decoded.values() {
        if vector.is_empty() {
            return None;
        }
        match dimension {
            None => dimension = Some(vector.len()),
            Some(expected) if expected != vector.len() => return None,
            Some(_) => {}
        }
    }
    let dimension = dimension?;
    let len = decoded.slot_count().checked_mul(dimension)?;
    let mut values = vec![0.0_f32; len];
    for (ordinal, vector) in decoded.iter() {
        let index = usize::try_from(ordinal).ok()?;
        let start = index.checked_mul(dimension)?;
        let destination = values.get_mut(start..start.saturating_add(dimension))?;
        if destination.len() != vector.len() {
            return None;
        }
        destination.copy_from_slice(vector);
    }
    Some(DecodedSlab { dimension, values })
}

/// `true` when the exact cosine is strictly below `key_floor` on the
/// `dot / ||candidate||` scale. Ties stay on the exact path.
fn cosine_cannot_beat(
    query: &[f32],
    candidate: &[f32],
    candidate_norm: f64,
    relative_error: f64,
    absolute_error: f64,
    key_floor: f64,
) -> bool {
    if candidate.len() != query.len() || !candidate_norm.is_finite() || candidate_norm <= 0.0 {
        return false;
    }
    let approximate =
        f64::from(crate::score_f64::dot_f32_approx(query, candidate)) / candidate_norm;
    if !approximate.is_finite() {
        return false;
    }
    let allowance = relative_error + absolute_error / candidate_norm;
    approximate + allowance < key_floor
}

#[allow(clippy::too_many_arguments)]
fn write_exact_group(
    query: &[f32],
    slab: &DecodedSlab,
    norms: &[f64],
    query_norm: f64,
    ordinals: &[u64],
    slots: &[usize],
    out: &mut [Option<f64>],
    score_for: &impl Fn(u64) -> Option<f64>,
) {
    let mut offset = 0;
    if ordinals.len() == 8 && slots.len() == 8 {
        if let Some(chunk) = cosine_chunk8(query, slab, norms, query_norm, ordinals) {
            for (slot, score) in slots.iter().copied().zip(chunk) {
                if let Some(destination) = out.get_mut(slot) {
                    *destination = score;
                }
            }
            return;
        }
    }
    while offset + 4 <= ordinals.len() && offset + 4 <= slots.len() {
        if let Some(chunk) = cosine_chunk4(
            query,
            slab,
            norms,
            query_norm,
            &ordinals[offset..offset + 4],
        ) {
            for (slot, score) in slots[offset..offset + 4].iter().copied().zip(chunk) {
                if let Some(destination) = out.get_mut(slot) {
                    *destination = score;
                }
            }
            offset += 4;
        } else {
            break;
        }
    }
    for (slot, candidate) in slots[offset..]
        .iter()
        .copied()
        .zip(ordinals[offset..].iter().copied())
    {
        if let Some(destination) = out.get_mut(slot) {
            *destination = score_for(candidate);
        }
    }
}

#[inline]
fn cosine_chunk8(
    query: &[f32],
    slab: &DecodedSlab,
    norms: &[f64],
    query_norm: f64,
    ordinals: &[u64],
) -> Option<[Option<f64>; 8]> {
    let mut coordinates: [&[f32]; 8] = [&[], &[], &[], &[], &[], &[], &[], &[]];
    let mut candidate_norms = [0.0_f64; 8];
    for (index, ordinal) in ordinals.iter().copied().enumerate() {
        let slice = slab_slice(slab, ordinal)?;
        if slice.len() != query.len() {
            return None;
        }
        let norm = cached_cosine_norm(norms, ordinal)?;
        if !norm.is_finite() || norm <= 0.0 {
            return None;
        }
        coordinates[index] = slice;
        candidate_norms[index] = norm;
    }
    let dots = crate::score_f64::dot_f32_x8(query, coordinates);
    let mut scores = [None; 8];
    for index in 0..8 {
        let score = if query_norm == 0.0 {
            0.0
        } else {
            dots[index] / (query_norm * candidate_norms[index])
        };
        scores[index] = Some(score);
    }
    Some(scores)
}

#[inline]
fn cosine_chunk4(
    query: &[f32],
    slab: &DecodedSlab,
    norms: &[f64],
    query_norm: f64,
    ordinals: &[u64],
) -> Option<[Option<f64>; 4]> {
    let mut coordinates: [&[f32]; 4] = [&[], &[], &[], &[]];
    let mut candidate_norms = [0.0_f64; 4];
    for (index, ordinal) in ordinals.iter().copied().enumerate() {
        let slice = slab_slice(slab, ordinal)?;
        if slice.len() != query.len() {
            return None;
        }
        let norm = cached_cosine_norm(norms, ordinal)?;
        if !norm.is_finite() || norm <= 0.0 {
            return None;
        }
        coordinates[index] = slice;
        candidate_norms[index] = norm;
    }
    let dots = crate::score_f64::dot_f32_x4(query, coordinates);
    let mut scores = [None; 4];
    for index in 0..4 {
        let score = if query_norm == 0.0 {
            0.0
        } else {
            dots[index] / (query_norm * candidate_norms[index])
        };
        scores[index] = Some(score);
    }
    Some(scores)
}

fn slab_slice(slab: &DecodedSlab, ordinal: u64) -> Option<&[f32]> {
    let index = usize::try_from(ordinal).ok()?;
    let start = index.checked_mul(slab.dimension)?;
    let end = start.checked_add(slab.dimension)?;
    slab.values.get(start..end)
}

fn cosine_norms(vectors: &OrdinalMap<Vec<f32>>) -> Vec<f64> {
    let mut norms = vec![f64::NAN; vectors.slot_count()];
    for (ordinal, vector) in vectors.iter() {
        let Ok(index) = usize::try_from(ordinal) else {
            continue;
        };
        if let Some(slot) = norms.get_mut(index) {
            *slot = dense_candidate_norm(vector);
        }
    }
    norms
}

fn cached_cosine_norm(norms: &[f64], ordinal: u64) -> Option<f64> {
    let index = usize::try_from(ordinal).ok()?;
    let value = *norms.get(index)?;
    value.is_finite().then_some(value)
}

fn link_cmp(
    left_id: u64,
    left_score: f64,
    right_id: u64,
    right_score: f64,
    ordinals: &OrdinalTable,
) -> Ordering {
    right_score.total_cmp(&left_score).then_with(|| {
        ordinals
            .id(left_id)
            .unwrap_or_default()
            .cmp(ordinals.id(right_id).unwrap_or_default())
            .then_with(|| left_id.cmp(&right_id))
    })
}

fn sort_scored(values: &mut [(u64, f64)], ordinals: &OrdinalTable) {
    values.sort_by(|left, right| link_cmp(left.0, left.1, right.0, right.1, ordinals));
}

/// Stable, fixed-seed level assignment. Capping protects against an
/// adversarially long tower without affecting realistic collections.
#[allow(clippy::cast_precision_loss)]
fn deterministic_level(ordinal: usize, m: usize) -> usize {
    let ordinal = u64::try_from(ordinal).unwrap_or(u64::MAX);
    let mut seed = ordinal.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    seed ^= seed >> 30;
    seed = seed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    seed ^= seed >> 27;
    seed = seed.wrapping_mul(0x94d0_49bb_1331_11eb);
    seed ^= seed >> 31;
    let unit = (((seed >> 11) as f64) + 1.0) / (((1_u64 << 53) as f64) + 1.0);
    let mut level = 0;
    let mut threshold = 1.0 / (m.max(2) as f64);
    while level < 32 && unit < threshold {
        level += 1;
        threshold /= m.max(2) as f64;
    }
    level
}

#[cfg(test)]
mod tests {
    use super::{HnswFilter, HnswIndex};
    use crate::doc::{Doc, DocumentMap};
    use crate::index::ordinal_map::OrdinalMap;
    use crate::index::ordinals::OrdinalTable;
    use crate::index::quantization::QuantizedVector;
    use crate::types::{MetricType, QuantizeType};
    use roaring::RoaringTreemap;
    use std::sync::Arc;

    fn fixture() -> (OrdinalTable, OrdinalMap<QuantizedVector>) {
        let docs: DocumentMap = (0_u16..64)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let doc = Doc::with_pk(&id).expect("document id must be valid");
                (id, Arc::new(doc))
            })
            .collect();
        let ordinals = OrdinalTable::build(&docs).expect("ordinals must build");
        let vectors = (0_u16..64)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let values = vec![f32::from(index), f32::from(index % 7)];
                (
                    ordinals.ordinal(&id).expect("document ordinal must exist"),
                    QuantizedVector::encode(values, QuantizeType::Undefined)
                        .expect("encoding must succeed"),
                )
            })
            .collect();
        (ordinals, vectors)
    }

    fn varied_fixture(count: u16, dimension: usize) -> (OrdinalTable, OrdinalMap<QuantizedVector>) {
        let docs: DocumentMap = (0..count)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let doc = Doc::with_pk(&id).expect("document id must be valid");
                (id, Arc::new(doc))
            })
            .collect();
        let ordinals = OrdinalTable::build(&docs).expect("ordinals must build");
        let vectors = (0..count)
            .map(|index| {
                let id = format!("doc-{index:03}");
                let values = (0..dimension)
                    .map(|axis| {
                        let axis = u16::try_from(axis).expect("fixture axis fits");
                        f32::from(index.wrapping_mul(3).wrapping_add(axis)) * 0.01
                    })
                    .collect();
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
    fn cached_edge_scores_match_the_rescored_graph() {
        let (ordinals, vectors) = varied_fixture(180, 8);
        for metric in [MetricType::Cosine, MetricType::L2] {
            let cached = HnswIndex::build_with(&vectors, &ordinals, 8, 24, metric, true, false);
            let fresh = HnswIndex::build_with(&vectors, &ordinals, 8, 24, metric, false, false);
            assert_eq!(cached.entry_ordinal, fresh.entry_ordinal);
            assert_eq!(cached.layers, fresh.layers);
        }
    }

    #[test]
    fn serial_f64_neighbor_fingerprint_stays_put() {
        let (ordinals, vectors) = varied_fixture(48, 8);
        let index = HnswIndex::build(&vectors, &ordinals, 8, 24, MetricType::Cosine);
        let mut fingerprint = 0u64;
        for layer in &index.layers {
            for (ordinal, neighbors) in layer.iter() {
                fingerprint = fingerprint
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(ordinal);
                for neighbor in neighbors {
                    fingerprint = fingerprint
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(*neighbor);
                }
            }
        }
        assert_eq!(fingerprint, 15_032_561_765_147_509_337);
    }

    #[test]
    fn enterprise_ga_parallel_neighbor_sets_match_serial_f64() {
        let (ordinals, vectors) = varied_fixture(180, 8);
        for metric in [MetricType::Cosine, MetricType::L2] {
            let serial = HnswIndex::build_with(&vectors, &ordinals, 8, 24, metric, true, false);
            let parallel = HnswIndex::build_with(&vectors, &ordinals, 8, 24, metric, true, true);
            assert_eq!(serial.entry_ordinal, parallel.entry_ordinal);
            assert_eq!(serial.layers, parallel.layers);
        }
        let (ordinals, vectors) = varied_fixture(96, 4);
        let index = HnswIndex::build(&vectors, &ordinals, 8, 64, MetricType::Cosine);
        assert_eq!(index.default_ef, 64);
    }

    #[test]
    fn graph_build_and_search_are_deterministic_and_bounded() {
        let (ordinals, vectors) = fixture();
        let first = HnswIndex::build(&vectors, &ordinals, 8, 32, MetricType::L2);
        let second = HnswIndex::build(&vectors, &ordinals, 8, 32, MetricType::L2);
        assert_eq!(first.layers, second.layers);
        assert_eq!(first.entry_ordinal, second.entry_ordinal);
        let candidates = first.candidates(
            &vectors,
            &ordinals,
            &[31.0, 3.0],
            Some(12),
            5,
            MetricType::L2,
            None,
            None,
        );
        assert!(candidates.len() <= 12);
        assert!(candidates.contains(
            ordinals
                .ordinal("doc-031")
                .expect("document ordinal must exist")
        ));
    }

    #[test]
    fn cosine_graph_build_is_deterministic_and_valid() {
        let (ordinals, vectors) = fixture();
        let first = HnswIndex::build(&vectors, &ordinals, 8, 32, MetricType::Cosine);
        let second = HnswIndex::build(&vectors, &ordinals, 8, 32, MetricType::Cosine);
        assert_eq!(first.layers, second.layers);
        assert!(first.validates(&vectors, 8, 32));
        assert!(second.entry_ordinal.is_some());
    }

    #[test]
    fn packed_f32_navigation_matches_the_quantized_lookup() {
        let (ordinals, vectors) = fixture();
        let index = HnswIndex::build(&vectors, &ordinals, 8, 32, MetricType::Cosine);
        let query = [31.0_f32, 3.0];
        let unpacked = index.candidates(
            &vectors,
            &ordinals,
            &query,
            Some(12),
            5,
            MetricType::Cosine,
            None,
            None,
        );
        let dimension = 2;
        let mut packed = vec![0.0_f32; vectors.slot_count() * dimension];
        for (ordinal, vector) in vectors.iter() {
            let QuantizedVector::F32(values) = vector else {
                continue;
            };
            let start = usize::try_from(ordinal).expect("ordinal fits") * dimension;
            packed[start..start + dimension].copy_from_slice(values);
        }
        let packed_hits = index.candidates(
            &vectors,
            &ordinals,
            &query,
            Some(12),
            5,
            MetricType::Cosine,
            None,
            Some((dimension, packed.as_slice())),
        );
        assert_eq!(unpacked, packed_hits);
    }

    #[test]
    fn filtered_search_uses_excluded_nodes_as_navigation_bridges() {
        let (ordinals, vectors) = fixture();
        let index = HnswIndex::build(&vectors, &ordinals, 8, 32, MetricType::L2);
        let allowed: RoaringTreemap = (40_u16..64)
            .filter_map(|value| ordinals.ordinal(&format!("doc-{value:03}")))
            .collect();
        let excluded = RoaringTreemap::new();
        let candidates = index.filtered_candidates(
            &vectors,
            &ordinals,
            &[0.0, 0.0],
            5,
            32,
            MetricType::L2,
            HnswFilter {
                allowed: &allowed,
                excluded: &excluded,
                eligible_count: usize::try_from(allowed.len()).expect("fixture size fits usize"),
            },
            None,
            None,
        );
        assert_eq!(candidates.len(), 5);
        assert!(candidates.iter().all(|ordinal| allowed.contains(ordinal)));
    }
}
