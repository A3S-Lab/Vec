//! Deterministic in-memory HNSW graph construction and traversal.

mod search;

use self::search::{greedy_search_by, prefetch_f32_at, search_layer_by, search_layer_filtered_by};
use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::quantization::{
    dense_candidate_norm, dense_query_norm, dense_query_norm_fast, score_ann, score_dense_cosine,
    score_dense_fast, score_dense_with_query_norm, QuantizedVector,
};
use crate::types::MetricType;
use roaring::RoaringTreemap;
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
        Self::build_with(vectors, ordinals, m, ef_construction, metric, true)
    }

    /// `cache_scores` reuses the f64 link score already computed for an edge.
    /// The false path is the original rescore, kept so tests can prove the
    /// published neighbor lists do not change.
    #[allow(clippy::too_many_lines)]
    fn build_with(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        m: usize,
        ef_construction: usize,
        metric: MetricType,
        cache_scores: bool,
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
            if maximum_level > level {
                for layer in ((level + 1)..=maximum_level).rev() {
                    entry = greedy_search_by(
                        &graph.layers[layer],
                        ordinals,
                        entry,
                        &score_for,
                        &prefetch_for,
                    );
                }
            }

            let connection_top = level.min(maximum_level);
            for layer in (0..=connection_top).rev() {
                let candidates = search_layer_by(
                    &graph.layers[layer],
                    &[entry],
                    ef_construction,
                    ordinals,
                    &score_for,
                    &prefetch_for,
                );
                let degree = if layer == 0 { m.saturating_mul(2) } else { m }.max(1);
                let neighbors: Vec<u64> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| *candidate != ordinal)
                    .take(degree)
                    .collect();
                graph.layers[layer].insert(ordinal, neighbors.clone());
                if let Some(scores) = edge_scores.as_mut() {
                    scores[layer].insert(
                        ordinal,
                        cached_link_scores(&decoded, &norms, ordinal, &neighbors, metric),
                    );
                }
                for neighbor in neighbors {
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
                            remember_link(
                                &mut scores[layer],
                                &decoded,
                                &norms,
                                neighbor,
                                ordinal,
                                metric,
                                edge_len,
                            );
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
                if let Some(best) = candidates.first() {
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
        )
    }

    pub(super) fn candidates_by(
        &self,
        ordinals: &OrdinalTable,
        requested_ef: Option<usize>,
        topk: usize,
        score_for: &impl Fn(u64) -> Option<f64>,
        prefetch_for: &impl Fn(u64),
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
            );
        }
        search_layer_by(
            &self.layers[0],
            &[entry],
            ef,
            ordinals,
            score_for,
            prefetch_for,
        )
        .into_iter()
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
        )
    }

    pub(super) fn filtered_candidates_by(
        &self,
        ordinals: &OrdinalTable,
        result_limit: usize,
        traversal_limit: usize,
        filter: HnswFilter<'_>,
        score_for: &impl Fn(u64) -> Option<f64>,
        prefetch_for: &impl Fn(u64),
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

fn cached_link_scores(
    vectors: &OrdinalMap<Vec<f32>>,
    norms: &[f64],
    node: u64,
    neighbors: &[u64],
    metric: MetricType,
) -> Vec<f64> {
    let mut values = Vec::with_capacity(neighbors.len());
    for candidate in neighbors {
        let Some(score) = score_link(vectors, norms, node, *candidate, metric) else {
            return Vec::new();
        };
        values.push(score);
    }
    values
}

fn remember_link(
    scores: &mut OrdinalMap<Vec<f64>>,
    vectors: &OrdinalMap<Vec<f32>>,
    norms: &[f64],
    node: u64,
    neighbor: u64,
    metric: MetricType,
    edge_len: usize,
) {
    let Some(slot) = scores.get_or_insert_default(node) else {
        return;
    };
    if slot.len() + 1 != edge_len {
        slot.clear();
        return;
    }
    let Some(score) = score_link(vectors, norms, node, neighbor, metric) else {
        slot.clear();
        return;
    };
    slot.push(score);
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
    if let Some(score_map) = scores.as_mut() {
        if let Some(mut paired) = paired_cache(layer, score_map, node) {
            sort_scored(&mut paired, ordinals);
            paired.truncate(degree);
            let (ids, values): (Vec<u64>, Vec<f64>) = paired.into_iter().unzip();
            layer.insert(node, ids);
            score_map.insert(node, values);
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

fn sort_scored(values: &mut [(u64, f64)], ordinals: &OrdinalTable) {
    values.sort_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| {
            ordinals
                .id(left.0)
                .unwrap_or_default()
                .cmp(ordinals.id(right.0).unwrap_or_default())
                .then_with(|| left.0.cmp(&right.0))
        })
    });
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
            let cached = HnswIndex::build_with(&vectors, &ordinals, 8, 24, metric, true);
            let fresh = HnswIndex::build_with(&vectors, &ordinals, 8, 24, metric, false);
            assert_eq!(cached.entry_ordinal, fresh.entry_ordinal);
            assert_eq!(cached.layers, fresh.layers);
        }
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
