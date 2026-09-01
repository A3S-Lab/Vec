//! Deterministic in-memory HNSW graph construction and traversal.

mod search;

use self::search::{
    greedy_search, greedy_search_by, search_layer, search_layer_by, search_layer_filtered_by,
};
use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::quantization::{score_dense, QuantizedVector};
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

impl HnswIndex {
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        m: usize,
        ef_construction: usize,
        metric: MetricType,
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
        let mut graph = Self {
            entry_ordinal: None,
            layers: vec![OrdinalMap::default()],
            default_ef: ef_construction,
        };
        let mut maximum_level = 0;

        for (position, (ordinal, query)) in decoded.iter().enumerate() {
            let level = deterministic_level(position, m);
            while graph.layers.len() <= level {
                graph.layers.push(OrdinalMap::default());
            }
            for layer in 0..=level {
                graph.layers[layer].get_or_insert_default(ordinal);
            }

            let Some(mut entry) = graph.entry_ordinal else {
                graph.entry_ordinal = Some(ordinal);
                maximum_level = level;
                continue;
            };

            if maximum_level > level {
                for layer in ((level + 1)..=maximum_level).rev() {
                    entry = greedy_search(
                        &graph.layers[layer],
                        &decoded,
                        ordinals,
                        query,
                        entry,
                        metric,
                    );
                }
            }

            let connection_top = level.min(maximum_level);
            for layer in (0..=connection_top).rev() {
                let candidates = search_layer(
                    &graph.layers[layer],
                    &decoded,
                    ordinals,
                    query,
                    &[entry],
                    ef_construction,
                    metric,
                );
                let degree = if layer == 0 { m.saturating_mul(2) } else { m }.max(1);
                let neighbors: Vec<u64> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| *candidate != ordinal)
                    .take(degree)
                    .collect();
                graph.layers[layer].insert(ordinal, neighbors.clone());
                for neighbor in neighbors {
                    if let Some(edges) = graph.layers[layer].get_or_insert_default(neighbor) {
                        if !edges.contains(&ordinal) {
                            edges.push(ordinal);
                        }
                    }
                    prune_edges(
                        &mut graph.layers[layer],
                        &decoded,
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

    pub(super) fn candidates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        query: &[f32],
        requested_ef: Option<usize>,
        topk: usize,
        metric: MetricType,
    ) -> RoaringTreemap {
        if vectors.is_empty() {
            return RoaringTreemap::new();
        }
        self.candidates_by(ordinals, requested_ef, topk, &|ordinal| {
            vectors
                .get(ordinal)
                .map(|vector| super::quantization::score(query, vector, metric))
        })
    }

    pub(super) fn candidates_by(
        &self,
        ordinals: &OrdinalTable,
        requested_ef: Option<usize>,
        topk: usize,
        score_for: &impl Fn(u64) -> Option<f64>,
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
            entry = greedy_search_by(&self.layers[layer], ordinals, entry, score_for);
        }
        search_layer_by(&self.layers[0], &[entry], ef, ordinals, score_for)
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
    ) -> RoaringTreemap {
        if vectors.is_empty() || filter.eligible_count == 0 || result_limit == 0 {
            return RoaringTreemap::new();
        }
        self.filtered_candidates_by(
            ordinals,
            result_limit,
            traversal_limit,
            filter,
            &|ordinal| {
                vectors
                    .get(ordinal)
                    .map(|vector| super::quantization::score(query, vector, metric))
            },
        )
    }

    pub(super) fn filtered_candidates_by(
        &self,
        ordinals: &OrdinalTable,
        result_limit: usize,
        traversal_limit: usize,
        filter: HnswFilter<'_>,
        score_for: &impl Fn(u64) -> Option<f64>,
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
            entry = greedy_search_by(&self.layers[layer], ordinals, entry, score_for);
        }
        search_layer_filtered_by(
            &self.layers[0],
            &[entry],
            result_limit,
            traversal_limit,
            ordinals,
            score_for,
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

fn prune_edges(
    layer: &mut OrdinalMap<Vec<u64>>,
    vectors: &OrdinalMap<Vec<f32>>,
    ordinals: &OrdinalTable,
    node: u64,
    degree: usize,
    metric: MetricType,
) {
    let Some(query) = vectors.get(node) else {
        return;
    };
    let mut scored: Vec<(u64, f64)> = layer
        .get(node)
        .into_iter()
        .flatten()
        .filter_map(|candidate| {
            Some((
                *candidate,
                score_dense(query, vectors.get(*candidate)?, metric),
            ))
        })
        .collect();
    sort_scored(&mut scored, ordinals);
    scored.truncate(degree);
    layer.insert(
        node,
        scored.into_iter().map(|(ordinal, _)| ordinal).collect(),
    );
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
        );
        assert!(candidates.len() <= 12);
        assert!(candidates.contains(
            ordinals
                .ordinal("doc-031")
                .expect("document ordinal must exist")
        ));
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
        );
        assert_eq!(candidates.len(), 5);
        assert!(candidates.iter().all(|ordinal| allowed.contains(ordinal)));
    }
}
