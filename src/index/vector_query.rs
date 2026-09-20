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
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const FLAT_PREFETCH_AHEAD: usize = 8;

#[inline]
fn flat_score_f64(
    query: &[f64],
    coordinates: &[f64],
    metric: MetricType,
    query_norm: f64,
    candidate_norm: Option<f64>,
) -> f64 {
    let score = match (metric, candidate_norm) {
        (MetricType::Cosine, Some(candidate_norm)) => {
            if query_norm == 0.0 || candidate_norm == 0.0 {
                0.0
            } else {
                crate::score_f64::dot_f64(query, coordinates) / (query_norm * candidate_norm)
            }
        }
        (MetricType::Cosine, None) => {
            let mut dot = 0.0_f64;
            let mut norm = 0.0_f64;
            for (left, right) in query.iter().zip(coordinates) {
                dot += *left * *right;
                norm += *right * *right;
            }
            if query_norm == 0.0 || norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * norm.sqrt())
            }
        }
        (MetricType::L2, _) => -crate::score_f64::l2sq_f64(query, coordinates),
        (MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined, _) => {
            crate::score_f64::dot_f64(query, coordinates)
        }
    };
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

#[inline]
fn flat_score(
    query: &[f64],
    coordinates: &[f32],
    metric: MetricType,
    query_norm: f64,
    candidate_norm: Option<f64>,
) -> f64 {
    let score = match (metric, candidate_norm) {
        (MetricType::Cosine, Some(candidate_norm)) => {
            if query_norm == 0.0 || candidate_norm == 0.0 {
                0.0
            } else {
                crate::score_f64::dot_f64_f32(query, coordinates) / (query_norm * candidate_norm)
            }
        }
        (MetricType::Cosine, None) => {
            let (dot, candidate_norm_sq) =
                crate::score_f64::cosine_parts_f64_f32(query, coordinates);
            if query_norm == 0.0 || candidate_norm_sq == 0.0 {
                0.0
            } else {
                dot / (query_norm * candidate_norm_sq.sqrt())
            }
        }
        (MetricType::L2, _) => -crate::score_f64::l2sq_f64_f32(query, coordinates),
        (MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined, _) => {
            crate::score_f64::dot_f64_f32(query, coordinates)
        }
    };
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

#[inline]
fn retain_flat_rank(ranked: &mut BinaryHeap<FlatRank>, candidate: FlatRank, limit: usize) {
    if !candidate.exact_score.is_finite() {
        return;
    }
    if ranked.len() < limit {
        ranked.push(candidate);
        return;
    }
    if ranked
        .peek()
        .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
    {
        ranked.pop();
        ranked.push(candidate);
    }
}

#[inline]
fn prefetch_f64_at(values: &[f64], index: usize) {
    if index >= values.len() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                values.as_ptr().add(index).cast::<i8>(),
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            let ptr = values.as_ptr().add(index);
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (values, index);
    }
}

#[inline]
fn prefetch_f32_at(values: &[f32], index: usize) {
    if index >= values.len() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                values.as_ptr().add(index).cast::<i8>(),
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            let ptr = values.as_ptr().add(index);
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (values, index);
    }
}

struct FlatRank {
    exact_score: f64,
    ordinal: u64,
}

impl PartialEq for FlatRank {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for FlatRank {}

impl PartialOrd for FlatRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FlatRank {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; keep the worst retained score at the root.
        other
            .exact_score
            .total_cmp(&self.exact_score)
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

fn pack_dense_f64(
    vectors: &super::ordinal_map::OrdinalMap<QuantizedVector>,
) -> Option<super::DenseF64Base> {
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
    let mut packed = vec![0.0_f64; len];
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
        for (slot, value) in destination.iter_mut().zip(values.iter()) {
            *slot = f64::from(*value);
        }
    }
    Some(super::DenseF64Base {
        dimension,
        values: packed,
    })
}

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
    /// Exact Flat top-k over packed promoted `f64` coordinates when the base
    /// has no overlay; otherwise falls back to packed `f32` with on-the-fly
    /// promotion. Both paths preserve the authoritative left-to-right `f64`
    /// score contract.
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn flat_candidates(&self, search: &AnnSearchContext<'_>) -> Option<RoaringTreemap> {
        let mut live = &self.base.vector_ordinals - &self.tombstones;
        live |= &self.delta_ordinals;
        if let Some(allowed) = search.allowed {
            live &= allowed;
        }
        if live.is_empty() || search.topk == 0 {
            return Some(RoaringTreemap::new());
        }
        let limit = search.topk.min(bitmap_count_to_usize(live.len()));
        let query_f64: Vec<f64> = search
            .vector
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        let query_norm = if search.metric == MetricType::Cosine {
            crate::score_f64::norm_sq_f32(search.vector).sqrt()
        } else {
            0.0
        };
        let cosine_norms = (search.metric == MetricType::Cosine).then(|| self.exact_cosine_norms());
        let overlay_empty = self.tombstones.is_empty()
            && self.delta_ordinals.is_empty()
            && search.allowed.is_none();
        if overlay_empty {
            if let Some(ranked) = self.flat_scan_packed_f64(
                &live,
                &query_f64,
                search.metric,
                query_norm,
                cosine_norms,
                limit,
            ) {
                return Some(ranked);
            }
        }
        Some(self.flat_scan_live(
            &live,
            &query_f64,
            search.metric,
            query_norm,
            cosine_norms,
            limit,
        ))
    }

    fn flat_scan_packed_f64(
        &self,
        live: &RoaringTreemap,
        query_f64: &[f64],
        metric: MetricType,
        query_norm: f64,
        cosine_norms: Option<&[f64]>,
        limit: usize,
    ) -> Option<RoaringTreemap> {
        let (dimension, values) = self.packed_f64()?;
        let slots = values.len() / dimension.max(1);
        let mut ranked = BinaryHeap::with_capacity(limit);
        for slot in 0..slots {
            let start = slot.saturating_mul(dimension);
            let end = start.saturating_add(dimension);
            let Some(coordinates) = values.get(start..end) else {
                continue;
            };
            if let Some(ahead) = slot
                .checked_add(FLAT_PREFETCH_AHEAD)
                .filter(|value| *value < slots)
            {
                prefetch_f64_at(values, ahead.saturating_mul(dimension));
            }
            let ordinal = u64::try_from(slot).unwrap_or(u64::MAX);
            if !live.contains(ordinal) {
                continue;
            }
            let candidate_norm = cosine_norms.map(|norms| {
                usize::try_from(ordinal)
                    .ok()
                    .and_then(|index| norms.get(index).copied())
                    .filter(|value| value.is_finite())
                    .unwrap_or_else(|| {
                        coordinates
                            .iter()
                            .map(|value| value * value)
                            .sum::<f64>()
                            .sqrt()
                    })
            });
            let exact_score =
                flat_score_f64(query_f64, coordinates, metric, query_norm, candidate_norm);
            retain_flat_rank(
                &mut ranked,
                FlatRank {
                    exact_score,
                    ordinal,
                },
                limit,
            );
        }
        Some(
            ranked
                .into_iter()
                .map(|candidate| candidate.ordinal)
                .collect(),
        )
    }

    fn flat_scan_live(
        &self,
        live: &RoaringTreemap,
        query_f64: &[f64],
        metric: MetricType,
        query_norm: f64,
        cosine_norms: Option<&[f64]>,
        limit: usize,
    ) -> RoaringTreemap {
        let packed_f32 = self.packed_f32();
        let ordinals: Vec<u64> = live.iter().collect();
        let mut ranked = BinaryHeap::with_capacity(limit);
        for (index, &ordinal) in ordinals.iter().enumerate() {
            if let Some((dimension, values)) = packed_f32 {
                if let Some(ahead) = ordinals
                    .get(index.saturating_add(FLAT_PREFETCH_AHEAD))
                    .and_then(|value| usize::try_from(*value).ok())
                {
                    prefetch_f32_at(values, ahead.saturating_mul(dimension));
                }
            }
            let Some(coordinates) = self.unquantized_f32(ordinal) else {
                continue;
            };
            let exact_score = flat_score(
                query_f64,
                coordinates,
                metric,
                query_norm,
                cosine_norms.map(|norms| self.exact_cosine_norm_at(ordinal, coordinates, norms)),
            );
            retain_flat_rank(
                &mut ranked,
                FlatRank {
                    exact_score,
                    ordinal,
                },
                limit,
            );
        }
        ranked
            .into_iter()
            .map(|candidate| candidate.ordinal)
            .collect()
    }

    fn exact_cosine_norms(&self) -> &[f64] {
        self.base.exact_cosine_norms.get_or_init(|| {
            let mut norms = vec![f64::NAN; self.base.vectors.slot_count()];
            for (ordinal, vector) in self.base.vectors.iter() {
                let QuantizedVector::F32(values) = vector else {
                    continue;
                };
                let Ok(index) = usize::try_from(ordinal) else {
                    continue;
                };
                if let Some(slot) = norms.get_mut(index) {
                    *slot = crate::score_f64::norm_sq_f32(values).sqrt();
                }
            }
            norms
        })
    }

    fn exact_cosine_norm_at(&self, ordinal: u64, coordinates: &[f32], norms: &[f64]) -> f64 {
        if self.delta.contains_key(&ordinal) {
            return crate::score_f64::norm_sq_f32(coordinates).sqrt();
        }
        usize::try_from(ordinal)
            .ok()
            .and_then(|index| norms.get(index).copied())
            .filter(|value| value.is_finite())
            .unwrap_or_else(|| crate::score_f64::norm_sq_f32(coordinates).sqrt())
    }

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

    fn packed_f64(&self) -> Option<(usize, &[f64])> {
        self.base
            .dense_f64
            .get_or_init(|| pack_dense_f64(&self.base.vectors))
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
