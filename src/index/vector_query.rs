//! ANN candidate traversal for immutable bases and incremental overlays.

use super::diskann_index::DiskannIndex;
use super::hnsw::{HnswFilter, HnswIndex};
use super::ivf::{scaled_candidate_limit, IvfIndex};
use super::vamana::VamanaIndex;
use super::{
    bitmap_count_to_usize, candidate_set_is_sufficient, optional_f32_query_parameter,
    optional_positive_query_parameter, proportional_candidate_limit, AnnOrdinals, AnnSearchContext,
    VectorIndex,
};
use roaring::RoaringTreemap;

impl VectorIndex {
    pub(super) fn hnsw_candidates(
        &self,
        hnsw: &HnswIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<RoaringTreemap> {
        let requested_ef = optional_positive_query_parameter(search.query, "ef");
        let limit = hnsw.candidate_limit(requested_ef, search.topk, search.eligible_count);
        let base = if let Some(allowed) = search.allowed {
            let base_eligible_count = self.base_eligible_vector_count(allowed);
            let traversal_limit =
                proportional_candidate_limit(limit, self.base.vectors.len(), base_eligible_count);
            if traversal_limit >= search.eligible_count {
                return None;
            }
            hnsw.filtered_candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                limit,
                traversal_limit,
                search.metric,
                HnswFilter {
                    allowed,
                    excluded: &self.tombstones,
                    eligible_count: base_eligible_count,
                },
            )
        } else {
            let base_ef = limit
                .saturating_add(bitmap_count_to_usize(self.tombstones.len()))
                .min(self.base.vectors.len());
            hnsw.candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                Some(base_ef),
                search.topk,
                search.metric,
            )
        };
        let merged = self.merge_candidates(
            base,
            search.vector,
            Some(limit),
            search.metric,
            search.allowed,
            search.ordinals,
        );
        candidate_set_is_sufficient(&merged, search).then_some(merged)
    }

    pub(super) fn ivf_candidates(
        &self,
        ivf: &IvfIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<RoaringTreemap> {
        let requested_nprobe = optional_positive_query_parameter(search.query, "nprobe");
        let base = search.allowed.map_or_else(
            || ivf.candidates(search.vector, requested_nprobe),
            |allowed| {
                ivf.filtered_candidates(
                    search.vector,
                    requested_nprobe,
                    search.topk,
                    allowed,
                    &self.tombstones,
                )
            },
        );
        let merged = self.merge_candidates(
            base,
            search.vector,
            None,
            search.metric,
            search.allowed,
            search.ordinals,
        );
        let Some(scale_factor) = optional_f32_query_parameter(search.query, "scale_factor") else {
            return candidate_set_is_sufficient(&merged, search).then_some(merged);
        };
        let limit = scaled_candidate_limit(search.topk, scale_factor, search.eligible_count);
        let limited = self.limit_candidates(
            &merged,
            search.vector,
            limit,
            search.metric,
            search.ordinals,
        );
        candidate_set_is_sufficient(&limited, search).then_some(limited)
    }

    pub(super) fn vamana_candidates(
        &self,
        vamana: &VamanaIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<AnnOrdinals> {
        let requested_list_size = optional_positive_query_parameter(search.query, "list_size");
        let limit = vamana.candidate_limit(requested_list_size, search.topk, search.eligible_count);
        let (base, diskann_sector_reads) = match search.allowed {
            Some(allowed) => self.vamana_filtered_base(vamana, search, allowed, limit)?,
            None => self.vamana_unfiltered_base(vamana, search, limit),
        };
        let merged = self.merge_candidates(
            base,
            search.vector,
            Some(limit),
            search.metric,
            search.allowed,
            search.ordinals,
        );
        candidate_set_is_sufficient(&merged, search).then_some(AnnOrdinals {
            ids: merged,
            diskann_sector_reads,
        })
    }

    pub(super) fn diskann_candidates(
        &self,
        diskann: &DiskannIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<AnnOrdinals> {
        let requested_list_size = optional_positive_query_parameter(search.query, "list_size");
        let limit = diskann.graph().candidate_limit(
            requested_list_size,
            search.topk,
            search.eligible_count,
        );
        let (base, diskann_sector_reads) = match search.allowed {
            Some(allowed) => self.diskann_filtered_base(diskann, search, allowed, limit)?,
            None => self.diskann_unfiltered_base(diskann, search, limit)?,
        };
        let merged = self.merge_candidates(
            base,
            search.vector,
            Some(limit),
            search.metric,
            search.allowed,
            search.ordinals,
        );
        candidate_set_is_sufficient(&merged, search).then_some(AnnOrdinals {
            ids: merged,
            diskann_sector_reads,
        })
    }

    fn diskann_filtered_base(
        &self,
        diskann: &DiskannIndex,
        search: &AnnSearchContext<'_>,
        allowed: &RoaringTreemap,
        limit: usize,
    ) -> Option<(RoaringTreemap, u64)> {
        let base_eligible_count = self.base_eligible_vector_count(allowed);
        let traversal_limit =
            proportional_candidate_limit(limit, self.base.vectors.len(), base_eligible_count);
        if traversal_limit >= search.eligible_count {
            return None;
        }
        if let Some(reader) = &self.base.diskann {
            if let Ok(result) = reader.filtered_candidates(
                search.vector,
                limit,
                traversal_limit,
                search.metric,
                allowed,
                &self.tombstones,
                search.ordinals,
            ) {
                return Some((result.candidates, result.sector_reads));
            }
        }
        diskann
            .filtered_candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                limit,
                traversal_limit,
                search.metric,
                allowed,
                &self.tombstones,
            )
            .ok()
            .map(|candidates| (candidates, 0))
    }

    fn diskann_unfiltered_base(
        &self,
        diskann: &DiskannIndex,
        search: &AnnSearchContext<'_>,
        limit: usize,
    ) -> Option<(RoaringTreemap, u64)> {
        let base_list_size = limit
            .saturating_add(bitmap_count_to_usize(self.tombstones.len()))
            .min(self.base.vectors.len());
        if base_list_size < self.base.vectors.len() {
            if let Some(reader) = &self.base.diskann {
                if let Ok(result) = reader.candidates(
                    search.vector,
                    base_list_size,
                    search.metric,
                    search.ordinals,
                ) {
                    return Some((result.candidates, result.sector_reads));
                }
            }
        }
        diskann
            .candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                Some(base_list_size),
                search.topk,
                search.metric,
            )
            .ok()
            .map(|candidates| (candidates, 0))
    }

    fn vamana_filtered_base(
        &self,
        vamana: &VamanaIndex,
        search: &AnnSearchContext<'_>,
        allowed: &RoaringTreemap,
        limit: usize,
    ) -> Option<(RoaringTreemap, u64)> {
        let base_eligible_count = self.base_eligible_vector_count(allowed);
        let traversal_limit =
            proportional_candidate_limit(limit, self.base.vectors.len(), base_eligible_count);
        if traversal_limit >= search.eligible_count {
            return None;
        }
        if let Some(reader) = &self.base.diskann {
            if let Ok(result) = reader.filtered_candidates(
                search.vector,
                limit,
                traversal_limit,
                search.metric,
                allowed,
                &self.tombstones,
                search.ordinals,
            ) {
                return Some((result.candidates, result.sector_reads));
            }
        }
        Some((
            vamana.filtered_candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                limit,
                traversal_limit,
                search.metric,
                allowed,
                &self.tombstones,
            ),
            0,
        ))
    }

    fn vamana_unfiltered_base(
        &self,
        vamana: &VamanaIndex,
        search: &AnnSearchContext<'_>,
        limit: usize,
    ) -> (RoaringTreemap, u64) {
        let base_list_size = limit
            .saturating_add(bitmap_count_to_usize(self.tombstones.len()))
            .min(self.base.vectors.len());
        if base_list_size < self.base.vectors.len() {
            if let Some(reader) = &self.base.diskann {
                if let Ok(result) = reader.candidates(
                    search.vector,
                    base_list_size,
                    search.metric,
                    search.ordinals,
                ) {
                    return (result.candidates, result.sector_reads);
                }
            }
        }
        (
            vamana.candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                Some(base_list_size),
                search.topk,
                search.metric,
            ),
            0,
        )
    }
}
