//! ANN candidate traversal for immutable bases and incremental overlays.

use super::diskann_index::DiskannIndex;
use super::hnsw::{HnswFilter, HnswIndex};
use super::ivf::{scaled_candidate_limit, IvfIndex};
use super::rabitq_index::{HnswRabitqIndex, IvfRabitqIndex};
use super::vamana::VamanaIndex;
use super::{
    bitmap_count_to_usize, candidate_set_is_sufficient, optional_f32_query_parameter,
    optional_positive_query_parameter, proportional_candidate_limit, AnnOrdinals, AnnSearchContext,
    VectorIndex,
};
use crate::config::IoBackend;
use crate::index::quantization::{dense_query_norm_fast, QuantizedVector};
use crate::types::MetricType;
use roaring::RoaringTreemap;

fn pack_dense_f32(
    vectors: &super::ordinal_map::OrdinalMap<QuantizedVector>,
) -> Option<super::DenseF32Base> {
    if vectors.is_empty() {
        return None;
    }
    let mut dimension = None;
    for vector in vectors.values() {
        let QuantizedVector::F32(values) = vector else {
            return None;
        };
        if values.is_empty() {
            return None;
        }
        match dimension {
            None => dimension = Some(values.len()),
            Some(expected) if expected != values.len() => return None,
            Some(_) => {}
        }
    }
    let dimension = dimension?;
    let len = vectors.slot_count().checked_mul(dimension)?;
    let mut packed = vec![0.0_f32; len];
    for (ordinal, vector) in vectors.iter() {
        let QuantizedVector::F32(values) = vector else {
            return None;
        };
        let index = usize::try_from(ordinal).ok()?;
        let start = index.checked_mul(dimension)?;
        let destination = packed.get_mut(start..start.saturating_add(dimension))?;
        if destination.len() != values.len() {
            return None;
        }
        destination.copy_from_slice(values);
    }
    Some(super::DenseF32Base {
        dimension,
        values: packed,
    })
}

impl VectorIndex {
    pub(super) fn hnsw_candidates(
        &self,
        hnsw: &HnswIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<RoaringTreemap> {
        let requested_ef = optional_positive_query_parameter(search.query, "ef");
        let limit = hnsw.candidate_limit(requested_ef, search.topk, search.eligible_count);
        let candidate_norms =
            (search.metric == MetricType::Cosine).then(|| self.cosine_candidate_norms());
        let packed = self.packed_f32();
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
                candidate_norms,
                packed,
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
                candidate_norms,
                packed,
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

    fn cosine_candidate_norms(&self) -> &[f32] {
        self.base.cosine_norms.get_or_init(|| {
            let mut norms = vec![f32::NAN; self.base.vectors.slot_count()];
            for (ordinal, vector) in self.base.vectors.iter() {
                let QuantizedVector::F32(values) = vector else {
                    continue;
                };
                let Ok(index) = usize::try_from(ordinal) else {
                    continue;
                };
                if let Some(slot) = norms.get_mut(index) {
                    *slot = dense_query_norm_fast(values);
                }
            }
            norms
        })
    }

    fn packed_f32(&self) -> Option<(usize, &[f32])> {
        self.base
            .dense_f32
            .get_or_init(|| pack_dense_f32(&self.base.vectors))
            .as_ref()
            .map(|packed| (packed.dimension, packed.values.as_slice()))
    }

    pub(super) fn hnsw_rabitq_candidates(
        &self,
        hnsw: &HnswRabitqIndex,
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
                search.ordinals,
                search.vector,
                limit,
                traversal_limit,
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
            hnsw.candidates(search.ordinals, search.vector, Some(base_ef), search.topk)
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

    pub(super) fn ivf_rabitq_candidates(
        &self,
        ivf: &IvfRabitqIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<RoaringTreemap> {
        let requested_nprobe = optional_positive_query_parameter(search.query, "nprobe");
        let scale_factor =
            optional_f32_query_parameter(search.query, "scale_factor").unwrap_or(4.0);
        let limit = scaled_candidate_limit(search.topk, scale_factor, search.eligible_count);
        let base_limit = if search.allowed.is_some() {
            limit
        } else {
            limit
                .saturating_add(bitmap_count_to_usize(self.tombstones.len()))
                .min(self.base.vectors.len())
        };
        let base = search.allowed.map_or_else(
            || ivf.candidates(search.vector, requested_nprobe, base_limit, search.ordinals),
            |allowed| {
                ivf.filtered_candidates(
                    search.vector,
                    requested_nprobe,
                    base_limit,
                    base_limit,
                    allowed,
                    &self.tombstones,
                    search.ordinals,
                )
            },
        );
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

    pub(super) fn vamana_candidates(
        &self,
        vamana: &VamanaIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<AnnOrdinals> {
        let requested_list_size = optional_positive_query_parameter(search.query, "list_size");
        let limit = vamana.candidate_limit(requested_list_size, search.topk, search.eligible_count);
        let (base, diskann_sector_reads, diskann_io_backend) = match search.allowed {
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
            diskann_io_backend,
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
        let (base, diskann_sector_reads, diskann_io_backend) = match search.allowed {
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
            diskann_io_backend,
        })
    }

    fn diskann_filtered_base(
        &self,
        diskann: &DiskannIndex,
        search: &AnnSearchContext<'_>,
        allowed: &RoaringTreemap,
        limit: usize,
    ) -> Option<(RoaringTreemap, u64, Option<IoBackend>)> {
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
                return Some((
                    result.candidates,
                    result.sector_reads,
                    Some(result.io_backend),
                ));
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
            .map(|candidates| (candidates, 0, None))
    }

    fn diskann_unfiltered_base(
        &self,
        diskann: &DiskannIndex,
        search: &AnnSearchContext<'_>,
        limit: usize,
    ) -> Option<(RoaringTreemap, u64, Option<IoBackend>)> {
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
                    return Some((
                        result.candidates,
                        result.sector_reads,
                        Some(result.io_backend),
                    ));
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
            .map(|candidates| (candidates, 0, None))
    }

    fn vamana_filtered_base(
        &self,
        vamana: &VamanaIndex,
        search: &AnnSearchContext<'_>,
        allowed: &RoaringTreemap,
        limit: usize,
    ) -> Option<(RoaringTreemap, u64, Option<IoBackend>)> {
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
                return Some((
                    result.candidates,
                    result.sector_reads,
                    Some(result.io_backend),
                ));
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
            None,
        ))
    }

    fn vamana_unfiltered_base(
        &self,
        vamana: &VamanaIndex,
        search: &AnnSearchContext<'_>,
        limit: usize,
    ) -> (RoaringTreemap, u64, Option<IoBackend>) {
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
                    return (
                        result.candidates,
                        result.sector_reads,
                        Some(result.io_backend),
                    );
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
            None,
        )
    }

    /// Unquantized coordinates for one ordinal.
    ///
    /// Delta vectors shadow the base. Quantized or removed ordinals return
    /// `None` so the caller keeps the document score.
    pub(super) fn unquantized_f32(&self, ordinal: u64) -> Option<&[f32]> {
        if let Some(vector) = self.delta.get(&ordinal) {
            return match vector {
                QuantizedVector::F32(values) => Some(values.as_slice()),
                _ => None,
            };
        }
        if self.tombstones.contains(ordinal) {
            return None;
        }
        let (dimension, values) = self.packed_f32()?;
        let index = usize::try_from(ordinal).ok()?;
        let start = index.checked_mul(dimension)?;
        let end = start.checked_add(dimension)?;
        values.get(start..end)
    }
}
