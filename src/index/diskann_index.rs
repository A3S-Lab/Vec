//! `DiskANN` generation: a Vamana graph with optional product-quantized search.

use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::product_quantization::ProductQuantizer;
use super::quantization::QuantizedVector;
use super::vamana::VamanaIndex;
use crate::error::Result;
use crate::types::MetricType;
use roaring::RoaringTreemap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct DiskannIndex {
    graph: VamanaIndex,
    pq: Option<ProductQuantizer>,
}

impl DiskannIndex {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        dimension: usize,
        max_degree: usize,
        list_size: usize,
        pq_chunk_num: usize,
        alpha: f64,
        metric: MetricType,
    ) -> Result<Self> {
        let graph = VamanaIndex::build(vectors, ordinals, max_degree, list_size, alpha, metric);
        let pq = (pq_chunk_num > 0)
            .then(|| ProductQuantizer::build(vectors, dimension, pq_chunk_num))
            .transpose()?;
        Ok(Self { graph, pq })
    }

    pub(super) fn graph(&self) -> &VamanaIndex {
        &self.graph
    }

    pub(super) fn quantizer(&self) -> Option<&ProductQuantizer> {
        self.pq.as_ref()
    }

    pub(super) fn candidates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        ordinals: &OrdinalTable,
        query: &[f32],
        requested_list_size: Option<usize>,
        topk: usize,
        metric: MetricType,
    ) -> Result<RoaringTreemap> {
        let Some(pq) = &self.pq else {
            return Ok(self.graph.candidates(
                vectors,
                ordinals,
                query,
                requested_list_size,
                topk,
                metric,
            ));
        };
        let table = pq.table(query, metric)?;
        Ok(self.graph.scored_candidates(
            vectors.len(),
            ordinals,
            requested_list_size,
            topk,
            &|ordinal| pq.score(&table, ordinal),
        ))
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
    ) -> Result<RoaringTreemap> {
        let Some(pq) = &self.pq else {
            return Ok(self.graph.filtered_candidates(
                vectors,
                ordinals,
                query,
                result_limit,
                traversal_limit,
                metric,
                allowed,
                excluded,
            ));
        };
        let table = pq.table(query, metric)?;
        Ok(self.graph.scored_filtered_candidates(
            vectors.len(),
            ordinals,
            result_limit,
            traversal_limit,
            allowed,
            excluded,
            &|ordinal| pq.score(&table, ordinal),
        ))
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        self.graph.estimated_payload_bytes().saturating_add(
            self.pq
                .as_ref()
                .map_or(0, ProductQuantizer::estimated_payload_bytes),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        max_degree: usize,
        list_size: usize,
        pq_chunk_num: usize,
        alpha: f64,
    ) -> bool {
        self.graph.validates(vectors, max_degree, list_size, alpha)
            && match (&self.pq, pq_chunk_num) {
                (None, 0) => true,
                (Some(pq), chunks) if chunks > 0 => pq.validates(vectors, dimension, chunks),
                _ => false,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::DiskannIndex;
    use crate::doc::{Doc, DocumentMap};
    use crate::index::ordinal_map::OrdinalMap;
    use crate::index::ordinals::OrdinalTable;
    use crate::index::quantization::QuantizedVector;
    use crate::types::{MetricType, QuantizeType};
    use std::sync::Arc;

    fn fixture() -> (OrdinalTable, OrdinalMap<QuantizedVector>) {
        let docs: DocumentMap = (0_u16..96)
            .map(|value| {
                let id = format!("doc-{value:03}");
                (id.clone(), Arc::new(Doc::with_pk(&id).expect("valid id")))
            })
            .collect();
        let ordinals = OrdinalTable::build(&docs).expect("ordinals must build");
        let vectors = docs
            .keys()
            .enumerate()
            .map(|(value, id)| {
                let value = u16::try_from(value).expect("fixture index fits u16");
                (
                    ordinals.ordinal(id).expect("ordinal must exist"),
                    QuantizedVector::encode(
                        vec![
                            f32::from(value),
                            f32::from(value % 7),
                            f32::from(value % 11),
                            f32::from(value % 13),
                        ],
                        QuantizeType::Undefined,
                    )
                    .expect("vector must encode"),
                )
            })
            .collect();
        (ordinals, vectors)
    }

    #[test]
    fn pq_generation_is_deterministic_and_exhaustive_when_requested() {
        let (ordinals, vectors) = fixture();
        let first = DiskannIndex::build(&vectors, &ordinals, 4, 12, 48, 2, 1.2, MetricType::L2)
            .expect("DiskANN must build");
        let second = DiskannIndex::build(&vectors, &ordinals, 4, 12, 48, 2, 1.2, MetricType::L2)
            .expect("DiskANN must rebuild");
        assert_eq!(first.graph, second.graph);
        assert_eq!(first.pq, second.pq);
        assert!(first.validates(&vectors, 4, 12, 48, 2, 1.2));
        let candidates = first
            .candidates(
                &vectors,
                &ordinals,
                &[42.0, 0.0, 9.0, 3.0],
                Some(vectors.len()),
                10,
                MetricType::L2,
            )
            .expect("ADC query must succeed");
        assert_eq!(candidates.len(), vectors.len() as u64);
    }
}
