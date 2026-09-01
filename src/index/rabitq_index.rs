//! HNSW and IVF execution adapters backed by shared `RaBitQ` codes.

use super::hnsw::{HnswFilter, HnswIndex};
use super::ivf::IvfIndex;
use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::quantization::QuantizedVector;
use super::rabitq::RabitqQuantizer;
use crate::error::Result;
use crate::types::MetricType;
use roaring::RoaringTreemap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct HnswRabitqIndex {
    graph: HnswIndex,
    quantizer: RabitqQuantizer,
}

impl HnswRabitqIndex {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        dimension: usize,
        m: usize,
        ef_construction: usize,
        total_bits: usize,
        num_clusters: usize,
        sample_count: usize,
        metric: MetricType,
    ) -> Result<Self> {
        Ok(Self {
            graph: HnswIndex::build(vectors, ordinals, m, ef_construction, metric),
            quantizer: RabitqQuantizer::build(
                vectors,
                dimension,
                total_bits,
                num_clusters,
                sample_count,
                metric,
            )?,
        })
    }

    pub(super) fn candidates(
        &self,
        ordinals: &OrdinalTable,
        query: &[f32],
        requested_ef: Option<usize>,
        topk: usize,
    ) -> RoaringTreemap {
        let Some(prepared) = self.quantizer.prepare_query(query) else {
            return RoaringTreemap::new();
        };
        self.graph
            .candidates_by(ordinals, requested_ef, topk, &|ordinal| {
                self.quantizer.score(&prepared, ordinal)
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn filtered_candidates(
        &self,
        ordinals: &OrdinalTable,
        query: &[f32],
        result_limit: usize,
        traversal_limit: usize,
        filter: HnswFilter<'_>,
    ) -> RoaringTreemap {
        let Some(prepared) = self.quantizer.prepare_query(query) else {
            return RoaringTreemap::new();
        };
        self.graph.filtered_candidates_by(
            ordinals,
            result_limit,
            traversal_limit,
            filter,
            &|ordinal| self.quantizer.score(&prepared, ordinal),
        )
    }

    pub(super) fn candidate_limit(
        &self,
        requested_ef: Option<usize>,
        topk: usize,
        vector_count: usize,
    ) -> usize {
        self.graph.candidate_limit(requested_ef, topk, vector_count)
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        self.graph
            .estimated_payload_bytes()
            .saturating_add(self.quantizer.estimated_payload_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        m: usize,
        ef_construction: usize,
        total_bits: usize,
        num_clusters: usize,
        sample_count: usize,
        metric: MetricType,
    ) -> bool {
        self.graph.validates(vectors, m, ef_construction)
            && self.quantizer.validates(
                vectors,
                dimension,
                total_bits,
                num_clusters,
                sample_count,
                metric,
            )
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct IvfRabitqIndex {
    ivf: IvfIndex,
    quantizer: RabitqQuantizer,
}

impl IvfRabitqIndex {
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        n_list: usize,
        total_bits: usize,
        sample_count: usize,
        metric: MetricType,
    ) -> Result<Self> {
        Ok(Self {
            ivf: IvfIndex::build(vectors, n_list, 5),
            quantizer: RabitqQuantizer::build(
                vectors,
                dimension,
                total_bits,
                n_list,
                sample_count,
                metric,
            )?,
        })
    }

    pub(super) fn candidates(
        &self,
        query: &[f32],
        requested_nprobe: Option<usize>,
        limit: usize,
        ordinals: &OrdinalTable,
    ) -> RoaringTreemap {
        self.rank(
            self.ivf.candidates(query, requested_nprobe),
            query,
            limit,
            ordinals,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn filtered_candidates(
        &self,
        query: &[f32],
        requested_nprobe: Option<usize>,
        minimum_candidates: usize,
        limit: usize,
        allowed: &RoaringTreemap,
        excluded: &RoaringTreemap,
        ordinals: &OrdinalTable,
    ) -> RoaringTreemap {
        self.rank(
            self.ivf.filtered_candidates(
                query,
                requested_nprobe,
                minimum_candidates,
                allowed,
                excluded,
            ),
            query,
            limit,
            ordinals,
        )
    }

    fn rank(
        &self,
        candidates: RoaringTreemap,
        query: &[f32],
        limit: usize,
        ordinals: &OrdinalTable,
    ) -> RoaringTreemap {
        if candidates.len() <= u64::try_from(limit).unwrap_or(u64::MAX) {
            return candidates;
        }
        let Some(prepared) = self.quantizer.prepare_query(query) else {
            return RoaringTreemap::new();
        };
        let mut scored: Vec<(u64, f64)> = candidates
            .iter()
            .filter_map(|ordinal| Some((ordinal, self.quantizer.score(&prepared, ordinal)?)))
            .collect();
        scored.sort_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| {
                ordinals
                    .id(left.0)
                    .unwrap_or_default()
                    .cmp(ordinals.id(right.0).unwrap_or_default())
                    .then_with(|| left.0.cmp(&right.0))
            })
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(ordinal, _)| ordinal)
            .collect()
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        self.ivf
            .estimated_payload_bytes()
            .saturating_add(self.quantizer.estimated_payload_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        n_list: usize,
        total_bits: usize,
        sample_count: usize,
        metric: MetricType,
    ) -> bool {
        self.ivf.validates(vectors, dimension, n_list)
            && self.quantizer.validates(
                vectors,
                dimension,
                total_bits,
                n_list,
                sample_count,
                metric,
            )
    }
}
