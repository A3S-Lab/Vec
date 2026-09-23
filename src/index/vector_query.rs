//! ANN candidate traversal for immutable bases and incremental overlays.

use super::diskann_index::DiskannIndex;
use super::hnsw::{HnswFilter, HnswIndex};
use super::ivf::{scaled_candidate_limit, IvfIndex};
use super::ordinals::OrdinalTable;
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
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const FLAT_PREFETCH_AHEAD: usize = 8;

#[inline]
fn live_is_dense(live: &RoaringTreemap, slots: usize) -> bool {
    let Ok(slots) = u64::try_from(slots) else {
        return false;
    };
    if slots == 0 {
        return live.is_empty();
    }
    live.len() == slots && live.min() == Some(0) && live.max() == Some(slots - 1)
}

#[cfg(test)]
mod live_density_tests {
    use super::live_is_dense;
    use roaring::RoaringTreemap;

    #[test]
    fn live_density_helper_covers_empty_sparse_and_dense_shapes() {
        let empty = RoaringTreemap::new();
        assert!(live_is_dense(&empty, 0));
        assert!(!live_is_dense(&empty, 3));

        let dense: RoaringTreemap = (0..4).collect();
        assert!(live_is_dense(&dense, 4));
        assert!(!live_is_dense(&dense, 5));

        let sparse: RoaringTreemap = [0_u64, 2].into_iter().collect();
        assert!(!live_is_dense(&sparse, 3));
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
fn search_vector_f32(query_f64: &[f64]) -> Vec<f32> {
    query_f64.iter().map(|value| *value as f32).collect()
}

#[allow(clippy::too_many_arguments)]
fn flat_cosine_scan_serial<'a>(
    live: &RoaringTreemap,
    dense: bool,
    dimension: usize,
    values: &[f32],
    inv_norms: &[f64],
    query_f64: &[f64],
    query_f32: &[f32],
    query_norm: f64,
    limit: usize,
    slots: usize,
    id_of: &(impl Fn(u64) -> Option<&'a str> + Sync),
) -> RoaringTreemap {
    let ranked = scan_cosine_slots(
        live, dense, dimension, values, inv_norms, query_f64, query_f32, query_norm, limit, 0,
        slots, id_of,
    );
    ranked
        .into_iter()
        .map(|candidate| candidate.ordinal)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn scan_cosine_slots<'a>(
    live: &RoaringTreemap,
    dense: bool,
    dimension: usize,
    values: &[f32],
    inv_norms: &[f64],
    query_f64: &[f64],
    query_f32: &[f32],
    query_norm: f64,
    limit: usize,
    start_slot: usize,
    end_slot: usize,
    id_of: &(impl Fn(u64) -> Option<&'a str> + Sync),
) -> Vec<FlatRank<'a>> {
    let relative_error = crate::score_f64::f32_cosine_score_error(dimension, query_norm);
    let absolute_error = crate::score_f64::f32_dot_absolute_error(dimension);
    let mut ranked: BinaryHeap<FlatRank<'_>> = BinaryHeap::with_capacity(limit);
    for slot in start_slot..end_slot {
        let ordinal = u64::try_from(slot).unwrap_or(u64::MAX);
        if !dense && !live.contains(ordinal) {
            continue;
        }
        let start = slot.saturating_mul(dimension);
        let end = start.saturating_add(dimension);
        let Some(coordinates) = values.get(start..end) else {
            continue;
        };
        if let Some(ahead) = slot
            .checked_add(FLAT_PREFETCH_AHEAD)
            .filter(|value| *value < end_slot)
        {
            prefetch_f32_at(values, ahead.saturating_mul(dimension));
        }
        let Some(inv_norm) = inv_norms.get(slot).copied() else {
            continue;
        };
        let approximate =
            f64::from(crate::score_f64::dot_f32_approx(query_f32, coordinates)) * inv_norm;
        if approximate.is_finite() && ranked.len() >= limit {
            if let Some(worst) = ranked.peek() {
                let allowance = relative_error + absolute_error * inv_norm;
                if approximate + allowance < worst.exact_score {
                    continue;
                }
            }
        }
        let exact = crate::score_f64::dot_f64_f32(query_f64, coordinates) * inv_norm;
        let Some(id) = id_of(ordinal) else {
            continue;
        };
        retain_flat_rank(
            &mut ranked,
            FlatRank {
                exact_score: exact,
                id,
                ordinal,
            },
            limit,
        );
    }
    ranked.into_vec()
}

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
fn retain_flat_rank<'a>(
    ranked: &mut BinaryHeap<FlatRank<'a>>,
    candidate: FlatRank<'a>,
    limit: usize,
) {
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

struct FlatRank<'a> {
    exact_score: f64,
    /// Primary key. Score ties keep the ascending key, matching the document
    /// collector. Ordinal is only the last key when two keys compare equal.
    id: &'a str,
    ordinal: u64,
}

impl PartialEq for FlatRank<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for FlatRank<'_> {}

impl PartialOrd for FlatRank<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FlatRank<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; keep the worst retained score at the root.
        // The same order as `RankedId`: higher score, then smaller primary key.
        other
            .exact_score
            .total_cmp(&self.exact_score)
            .then_with(|| self.id.cmp(other.id))
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
        let cosine_inv_norms =
            (search.metric == MetricType::Cosine).then(|| self.exact_cosine_inv_norms());
        let overlay_empty = self.tombstones.is_empty()
            && self.delta_ordinals.is_empty()
            && search.allowed.is_none();
        if overlay_empty && search.metric == MetricType::Cosine {
            if let Some(ranked) = self.flat_scan_packed_cosine(
                &live,
                &query_f64,
                query_norm,
                cosine_inv_norms,
                limit,
                search.ordinals,
            ) {
                return Some(ranked);
            }
        }
        if overlay_empty && search.metric != MetricType::Cosine {
            if let Some(ranked) = self.flat_scan_packed_f64(
                &live,
                &query_f64,
                search.metric,
                query_norm,
                limit,
                search.ordinals,
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
            search.ordinals,
        ))
    }

    /// Cosine Flat scan over packed promoted `f64` with `dot * inv_norm`
    /// ranking. Chunks in parallel when Rayon has more than one worker; the
    /// 1-worker harness stays serial. Winners are re-scored with the public
    /// `f64` formula in the query engine.
    fn flat_scan_packed_cosine(
        &self,
        live: &RoaringTreemap,
        query_f64: &[f64],
        query_norm: f64,
        cosine_inv_norms: Option<&[f64]>,
        limit: usize,
        ordinals: &OrdinalTable,
    ) -> Option<RoaringTreemap> {
        // A zero-norm cosine query scores 0 against every finite vector. The
        // scan still has to publish top-k; an empty bitmap drops those hits.
        let (dimension, values) = self.packed_f32()?;
        let inv_norms = cosine_inv_norms?;
        let slots = values.len() / dimension.max(1);
        let dense = live_is_dense(live, slots);
        let threads = rayon::current_num_threads().max(1);
        let id_of = |ordinal| ordinals.id(ordinal);
        if threads == 1 || slots < 4_096 {
            return Some(flat_cosine_scan_serial(
                live,
                dense,
                dimension,
                values,
                inv_norms,
                query_f64,
                &search_vector_f32(query_f64),
                query_norm,
                limit,
                slots,
                &id_of,
            ));
        }
        let query_f32 = search_vector_f32(query_f64);
        let chunk = slots.div_ceil(threads);
        let partial: Vec<Vec<FlatRank<'_>>> = (0..threads)
            .into_par_iter()
            .map(|worker| {
                let start_slot = worker.saturating_mul(chunk);
                let end_slot = start_slot.saturating_add(chunk).min(slots);
                scan_cosine_slots(
                    live, dense, dimension, values, inv_norms, query_f64, &query_f32, query_norm,
                    limit, start_slot, end_slot, &id_of,
                )
            })
            .collect();
        let mut ranked = BinaryHeap::with_capacity(limit);
        for candidate in partial.into_iter().flatten() {
            retain_flat_rank(&mut ranked, candidate, limit);
        }
        Some(
            ranked
                .into_iter()
                .map(|candidate| candidate.ordinal)
                .collect(),
        )
    }

    fn flat_scan_packed_f64(
        &self,
        live: &RoaringTreemap,
        query_f64: &[f64],
        metric: MetricType,
        query_norm: f64,
        limit: usize,
        ordinals: &OrdinalTable,
    ) -> Option<RoaringTreemap> {
        let (dimension, values) = self.packed_f64()?;
        let slots = values.len() / dimension.max(1);
        let dense = live_is_dense(live, slots);
        let mut ranked = BinaryHeap::with_capacity(limit);
        for slot in 0..slots {
            let ordinal = u64::try_from(slot).unwrap_or(u64::MAX);
            if !dense && !live.contains(ordinal) {
                continue;
            }
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
            let exact_score = flat_score_f64(query_f64, coordinates, metric, query_norm, None);
            let Some(id) = ordinals.id(ordinal) else {
                continue;
            };
            retain_flat_rank(
                &mut ranked,
                FlatRank {
                    exact_score,
                    id,
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

    #[allow(clippy::too_many_arguments)]
    fn flat_scan_live(
        &self,
        live: &RoaringTreemap,
        query_f64: &[f64],
        metric: MetricType,
        query_norm: f64,
        cosine_norms: Option<&[f64]>,
        limit: usize,
        names: &OrdinalTable,
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
            let Some(id) = names.id(ordinal) else {
                continue;
            };
            retain_flat_rank(
                &mut ranked,
                FlatRank {
                    exact_score,
                    id,
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

    fn exact_cosine_inv_norms(&self) -> &[f64] {
        self.base.exact_cosine_inv_norms.get_or_init(|| {
            self.exact_cosine_norms()
                .iter()
                .map(|norm| {
                    if norm.is_finite() && *norm != 0.0 {
                        1.0 / *norm
                    } else {
                        0.0
                    }
                })
                .collect()
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

#[cfg(test)]
#[allow(
    clippy::bool_assert_comparison,
    clippy::float_cmp,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::index::ordinal_map::OrdinalMap;
    use crate::index::quantization::QuantizedVector;
    use crate::types::MetricType;
    use roaring::RoaringTreemap;

    #[test]
    fn live_is_dense_oracle() {
        let mut live = RoaringTreemap::new();
        assert!(live_is_dense(&live, 0));
        assert!(!live_is_dense(&live, 3));
        live.insert(0);
        live.insert(1);
        live.insert(2);
        assert!(live_is_dense(&live, 3));
        live.remove(1);
        assert!(!live_is_dense(&live, 3));
    }

    #[test]
    fn flat_score_f64_metrics_match_independent_oracle() {
        let query = [1.0_f64, 0.0, 0.0];
        let candidate = [0.6_f64, 0.8, 0.0];
        let query_norm = 1.0;
        let cand_norm = 1.0;

        let cosine = flat_score_f64(
            &query,
            &candidate,
            MetricType::Cosine,
            query_norm,
            Some(cand_norm),
        );
        assert!((cosine - 0.6).abs() < 1e-12);

        let cosine_zero =
            flat_score_f64(&query, &candidate, MetricType::Cosine, 0.0, Some(cand_norm));
        assert_eq!(cosine_zero, 0.0);

        let cosine_none = flat_score_f64(&query, &candidate, MetricType::Cosine, query_norm, None);
        assert!((cosine_none - 0.6).abs() < 1e-12);

        let zero_cand = [0.0_f64, 0.0, 0.0];
        assert_eq!(
            flat_score_f64(&query, &zero_cand, MetricType::Cosine, query_norm, None),
            0.0
        );

        let l2 = flat_score_f64(&query, &candidate, MetricType::L2, 0.0, None);
        let expected_l2 = -((1.0_f64 - 0.6).powi(2) + 0.8_f64.powi(2));
        assert!((l2 - expected_l2).abs() < 1e-12);

        let ip = flat_score_f64(&query, &candidate, MetricType::Ip, 0.0, None);
        assert!((ip - 0.6).abs() < 1e-12);
        let mips = flat_score_f64(&query, &candidate, MetricType::MipsL2, 0.0, None);
        assert_eq!(mips.to_bits(), ip.to_bits());
    }

    #[test]
    fn flat_score_f32_metrics_match_independent_oracle() {
        let query = [1.0_f64, 0.0];
        let candidate = [0.0_f32, 1.0];
        let query_norm = 1.0;

        let cosine = flat_score(
            &query,
            &candidate,
            MetricType::Cosine,
            query_norm,
            Some(1.0),
        );
        assert_eq!(cosine, 0.0);

        let cosine_none = flat_score(&query, &candidate, MetricType::Cosine, query_norm, None);
        assert_eq!(cosine_none, 0.0);

        assert_eq!(
            flat_score(&query, &candidate, MetricType::Cosine, 0.0, Some(1.0)),
            0.0
        );
        assert_eq!(
            flat_score(&query, &[0.0, 0.0], MetricType::Cosine, query_norm, None),
            0.0
        );

        let l2 = flat_score(&query, &candidate, MetricType::L2, 0.0, None);
        assert!((l2 - (-2.0)).abs() < 1e-12);
        let ip = flat_score(&query, &candidate, MetricType::Ip, 0.0, None);
        assert_eq!(ip, 0.0);
    }

    #[test]
    fn retain_flat_rank_keeps_topk_by_score_then_ordinal() {
        let mut ranked = BinaryHeap::new();
        retain_flat_rank(
            &mut ranked,
            FlatRank {
                exact_score: f64::NAN,
                id: "",
                ordinal: 9,
            },
            2,
        );
        assert!(ranked.is_empty());

        retain_flat_rank(
            &mut ranked,
            FlatRank {
                exact_score: 1.0,
                id: "",
                ordinal: 1,
            },
            2,
        );
        retain_flat_rank(
            &mut ranked,
            FlatRank {
                exact_score: 3.0,
                id: "",
                ordinal: 3,
            },
            2,
        );
        retain_flat_rank(
            &mut ranked,
            FlatRank {
                exact_score: 2.0,
                id: "",
                ordinal: 2,
            },
            2,
        );
        assert_eq!(ranked.len(), 2);
        let mut ids: Vec<_> = ranked.into_iter().map(|c| c.ordinal).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn pack_dense_rejects_empty_and_non_f32() {
        let empty: OrdinalMap<QuantizedVector> = OrdinalMap::default();
        assert!(pack_dense_f64(&empty).is_none());
        assert!(pack_dense_f32(&empty).is_none());

        let mut mixed = OrdinalMap::default();
        mixed.insert(0, QuantizedVector::F32(vec![1.0, 0.0]));
        mixed.insert(1, QuantizedVector::Fp16(vec![0]));
        assert!(pack_dense_f64(&mixed).is_none());
        assert!(pack_dense_f32(&mixed).is_none());

        let mut jagged = OrdinalMap::default();
        jagged.insert(0, QuantizedVector::F32(vec![1.0]));
        jagged.insert(1, QuantizedVector::F32(vec![1.0, 0.0]));
        assert!(pack_dense_f64(&jagged).is_none());
        assert!(pack_dense_f32(&jagged).is_none());

        let mut empty_vec = OrdinalMap::default();
        empty_vec.insert(0, QuantizedVector::F32(vec![]));
        assert!(pack_dense_f64(&empty_vec).is_none());
        assert!(pack_dense_f32(&empty_vec).is_none());

        let mut ok = OrdinalMap::default();
        ok.insert(0, QuantizedVector::F32(vec![1.0, 2.0]));
        ok.insert(2, QuantizedVector::F32(vec![3.0, 4.0]));
        let packed = pack_dense_f64(&ok).expect("pack f64");
        assert_eq!(packed.dimension, 2);
        assert_eq!(packed.values.len(), 6);
        let packed_f32 = pack_dense_f32(&ok).expect("pack f32");
        assert_eq!(packed_f32.values[4], 3.0);
    }

    #[test]
    fn flat_cosine_scan_serial_respects_sparse_live_and_topk() {
        let values = vec![1.0_f32, 0.0, 0.0, 1.0, 0.7, 0.3];
        let inv_norms = vec![1.0, 1.0, 1.0];
        let query = [1.0_f64, 0.0];
        let query_f32 = [1.0_f32, 0.0];
        let mut live = RoaringTreemap::new();
        live.insert(0);
        live.insert(2);
        let ranked = flat_cosine_scan_serial(
            &live,
            false,
            2,
            &values,
            &inv_norms,
            &query,
            &query_f32,
            1.0,
            1,
            3,
            &|_| Some(""),
        );
        assert_eq!(ranked.len(), 1);
        assert!(ranked.contains(0));
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::needless_range_loop
    )]
    fn flat_cosine_rejector_matches_exact_topk() {
        let dimension = 17_usize;
        let slots = 180_usize;
        let limit = 7_usize;
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut bits = || {
            state = state
                .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                .wrapping_add(0x94D0_49BB_1331_11EB);
            state
        };
        let mut values = Vec::with_capacity(slots * dimension);
        let mut inv_norms = Vec::with_capacity(slots);
        for slot in 0..slots {
            let mut norm_sq = 0.0_f64;
            for axis in 0..dimension {
                let mixed = bits();
                let value = if slot == 3 {
                    0.0
                } else if slot % 11 == 0 {
                    f32::from_bits(((mixed as u32) & 0x0000_0fff) | 0x0000_0001)
                } else if axis == 0 && slot + 1 < slots && slot % 9 == 0 {
                    1.0
                } else {
                    let unit = (mixed >> 11) as f64 / 9_007_199_254_740_992.0;
                    (unit * 2.0 - 1.0) as f32
                };
                values.push(value);
                let wide = f64::from(value);
                norm_sq += wide * wide;
            }
            let norm = norm_sq.sqrt();
            inv_norms.push(if norm == 0.0 { 0.0 } else { 1.0 / norm });
        }
        // One-ulp neighbor of slot 0 so the boundary is tighter than a random gap.
        if slots > 1 {
            let neighbor = dimension;
            let (head, tail) = values.split_at_mut(neighbor);
            tail[..dimension].copy_from_slice(&head[..dimension]);
            let nudged = values[neighbor].next_up();
            values[neighbor] = nudged;
            let norm_sq: f64 = values[neighbor..neighbor + dimension]
                .iter()
                .map(|value| {
                    let wide = f64::from(*value);
                    wide * wide
                })
                .sum();
            let norm = norm_sq.sqrt();
            inv_norms[1] = if norm == 0.0 { 0.0 } else { 1.0 / norm };
        }
        let query_f32: Vec<f32> = (0..dimension)
            .map(|_| {
                let unit = (bits() >> 11) as f64 / 9_007_199_254_740_992.0;
                (unit * 2.0 - 1.0) as f32
            })
            .collect();
        let query_f64: Vec<f64> = query_f32.iter().copied().map(f64::from).collect();
        let query_norm = query_f64
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let live: RoaringTreemap = (0..u64::try_from(slots).expect("slots")).collect();
        let selected = flat_cosine_scan_serial(
            &live,
            true,
            dimension,
            &values,
            &inv_norms,
            &query_f64,
            &query_f32,
            query_norm,
            limit,
            slots,
            &|_| Some(""),
        );
        let mut exact = std::collections::BinaryHeap::new();
        for slot in 0..slots {
            let start = slot * dimension;
            let coordinates = &values[start..start + dimension];
            retain_flat_rank(
                &mut exact,
                FlatRank {
                    exact_score: crate::score_f64::dot_f64_f32(&query_f64, coordinates)
                        * inv_norms[slot],
                    id: "",
                    ordinal: u64::try_from(slot).expect("slot"),
                },
                limit,
            );
        }
        let mut expected: Vec<u64> = exact.into_iter().map(|rank| rank.ordinal).collect();
        expected.sort_unstable();
        let mut actual: Vec<u64> = selected.iter().collect();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn prefetch_helpers_tolerate_out_of_range() {
        prefetch_f64_at(&[], 0);
        prefetch_f64_at(&[1.0], 5);
        prefetch_f32_at(&[], 0);
        prefetch_f32_at(&[1.0], 5);
        prefetch_f64_at(&[1.0, 2.0], 0);
        prefetch_f32_at(&[1.0, 2.0], 0);
    }

    #[test]
    fn flat_rank_ord_tie_breaks_on_primary_key_then_ordinal() {
        let earlier_key = FlatRank {
            exact_score: 0.0,
            id: "b",
            ordinal: 2,
        };
        let earlier_insert = FlatRank {
            exact_score: 0.0,
            id: "c",
            ordinal: 0,
        };
        assert_eq!(earlier_key.cmp(&earlier_insert), Ordering::Less);
        let same_key_low_ordinal = FlatRank {
            exact_score: 1.0,
            id: "a",
            ordinal: 1,
        };
        let same_key_high_ordinal = FlatRank {
            exact_score: 1.0,
            id: "a",
            ordinal: 2,
        };
        assert_eq!(
            same_key_low_ordinal.cmp(&same_key_high_ordinal),
            Ordering::Less
        );
        assert_eq!(
            same_key_low_ordinal.partial_cmp(&same_key_high_ordinal),
            Some(Ordering::Less)
        );
        assert!(!same_key_low_ordinal.eq(&same_key_high_ordinal));
    }
}
