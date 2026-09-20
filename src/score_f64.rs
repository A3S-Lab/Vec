//! Authoritative `f64` dense scoring kernels.
//!
//! Public Flat scores and exact ANN re-ranking promote stored `f32` coordinates
//! to `f64` and accumulate left-to-right. These kernels preserve that arithmetic
//! (same add order, same IEEE results) while using platform SIMD to convert and
//! multiply pairs of lanes. Horizontal reduction still folds into a scalar in
//! index order so differential fixtures against the document path stay
//! bit-identical.

use crate::types::MetricType;

/// Squared L2 over `f32` coordinates promoted to `f64`.
#[inline]
pub(crate) fn l2sq_f32(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    dispatch_f32(a, b, Kernel::L2)
}

/// Inner product over `f32` coordinates promoted to `f64`.
#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    dispatch_f32(a, b, Kernel::Dot)
}

/// `Σ v[i]²` after promoting each `f32` lane to `f64`.
#[inline]
pub(crate) fn norm_sq_f32(values: &[f32]) -> f64 {
    dispatch_f32(values, values, Kernel::NormSq)
}

/// Squared L2 for an already-promoted `f64` query against stored `f32`.
#[inline]
pub(crate) fn l2sq_f64_f32(query: &[f64], candidate: &[f32]) -> f64 {
    debug_assert_eq!(query.len(), candidate.len());
    dispatch_f64_f32(query, candidate, Kernel::L2)
}

/// Inner product for an already-promoted `f64` query against stored `f32`.
#[inline]
pub(crate) fn dot_f64_f32(query: &[f64], candidate: &[f32]) -> f64 {
    debug_assert_eq!(query.len(), candidate.len());
    dispatch_f64_f32(query, candidate, Kernel::Dot)
}

/// Cosine: returns `(dot, candidate_norm_sq)` in one left-to-right pass.
#[inline]
pub(crate) fn cosine_parts_f32(query: &[f32], candidate: &[f32]) -> (f64, f64) {
    debug_assert_eq!(query.len(), candidate.len());
    cosine_parts_f32_impl(query, candidate)
}

/// Cosine parts for `f64` query × `f32` candidate.
#[inline]
pub(crate) fn cosine_parts_f64_f32(query: &[f64], candidate: &[f32]) -> (f64, f64) {
    debug_assert_eq!(query.len(), candidate.len());
    cosine_parts_f64_f32_impl(query, candidate)
}

/// Scores unquantized `f32` vectors with the authoritative `f64` contract.
#[inline]
pub(crate) fn score_f32(
    query: &[f32],
    candidate: &[f32],
    metric: MetricType,
    query_norm: f64,
) -> f64 {
    if query.len() != candidate.len() {
        return f64::NEG_INFINITY;
    }
    match metric {
        MetricType::L2 => -l2sq_f32(query, candidate),
        MetricType::Cosine => {
            let (dot, candidate_norm_sq) = cosine_parts_f32(query, candidate);
            if query_norm == 0.0 || candidate_norm_sq == 0.0 {
                0.0
            } else {
                dot / (query_norm * candidate_norm_sq.sqrt())
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => dot_f32(query, candidate),
    }
}

/// Scores a stored `f32` document against an already-promoted `f64` query.
#[inline]
pub(crate) fn score_f64_f32(
    query: &[f64],
    candidate: &[f32],
    metric: MetricType,
    query_norm: f64,
) -> f64 {
    if query.len() != candidate.len() {
        return f64::NEG_INFINITY;
    }
    match metric {
        MetricType::L2 => -l2sq_f64_f32(query, candidate),
        MetricType::Cosine => {
            let (dot, candidate_norm_sq) = cosine_parts_f64_f32(query, candidate);
            if query_norm == 0.0 || candidate_norm_sq == 0.0 {
                0.0
            } else {
                dot / (query_norm * candidate_norm_sq.sqrt())
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => {
            dot_f64_f32(query, candidate)
        }
    }
}

#[derive(Clone, Copy)]
enum Kernel {
    L2,
    Dot,
    NormSq,
}

#[inline]
fn dispatch_f32(a: &[f32], b: &[f32], kernel: Kernel) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: slices share length; NEON helpers only read `len` lanes.
        #[allow(unsafe_code)]
        return unsafe { f32_neon(a, b, kernel) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: slices share length; SSE2 is baseline on `x86_64`.
        #[allow(unsafe_code)]
        return unsafe { f32_sse2(a, b, kernel) };
    }
    #[allow(unreachable_code)]
    f32_scalar(a, b, kernel)
}

#[inline]
fn dispatch_f64_f32(query: &[f64], candidate: &[f32], kernel: Kernel) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        return unsafe { f64_f32_neon(query, candidate, kernel) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        return unsafe { f64_f32_sse2(query, candidate, kernel) };
    }
    #[allow(unreachable_code)]
    f64_f32_scalar(query, candidate, kernel)
}

#[inline]
fn f32_scalar(a: &[f32], b: &[f32], kernel: Kernel) -> f64 {
    let mut sum = 0.0_f64;
    match kernel {
        Kernel::L2 => {
            for (left, right) in a.iter().zip(b) {
                let difference = f64::from(*left) - f64::from(*right);
                sum += difference * difference;
            }
        }
        Kernel::Dot => {
            for (left, right) in a.iter().zip(b) {
                sum += f64::from(*left) * f64::from(*right);
            }
        }
        Kernel::NormSq => {
            for value in a {
                let wide = f64::from(*value);
                sum += wide * wide;
            }
        }
    }
    sum
}

#[inline]
fn f64_f32_scalar(query: &[f64], candidate: &[f32], kernel: Kernel) -> f64 {
    let mut sum = 0.0_f64;
    match kernel {
        Kernel::L2 => {
            for (left, right) in query.iter().zip(candidate) {
                let difference = *left - f64::from(*right);
                sum += difference * difference;
            }
        }
        Kernel::Dot => {
            for (left, right) in query.iter().zip(candidate) {
                sum += *left * f64::from(*right);
            }
        }
        Kernel::NormSq => {
            for value in candidate {
                let wide = f64::from(*value);
                sum += wide * wide;
            }
        }
    }
    sum
}

#[inline]
fn cosine_parts_f32_impl(query: &[f32], candidate: &[f32]) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        return unsafe { cosine_parts_f32_neon(query, candidate) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        return unsafe { cosine_parts_f32_sse2(query, candidate) };
    }
    #[allow(unreachable_code)]
    {
        let mut dot = 0.0_f64;
        let mut norm = 0.0_f64;
        for (left, right) in query.iter().zip(candidate) {
            let left = f64::from(*left);
            let right = f64::from(*right);
            dot += left * right;
            norm += right * right;
        }
        (dot, norm)
    }
}

#[inline]
fn cosine_parts_f64_f32_impl(query: &[f64], candidate: &[f32]) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        return unsafe { cosine_parts_f64_f32_neon(query, candidate) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        return unsafe { cosine_parts_f64_f32_sse2(query, candidate) };
    }
    #[allow(unreachable_code)]
    {
        let mut dot = 0.0_f64;
        let mut norm = 0.0_f64;
        for (left, right) in query.iter().zip(candidate) {
            let right = f64::from(*right);
            dot += *left * right;
            norm += right * right;
        }
        (dot, norm)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn f32_neon(a: &[f32], b: &[f32], kernel: Kernel) -> f64 {
    use std::arch::aarch64::{vcvt_f64_f32, vgetq_lane_f64, vld1_f32, vmulq_f64, vsubq_f64};
    let n = a.len();
    let mut sum = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vld1_f32(a.as_ptr().add(index));
        let right = match kernel {
            Kernel::NormSq => left,
            _ => vld1_f32(b.as_ptr().add(index)),
        };
        let left = vcvt_f64_f32(left);
        let right = vcvt_f64_f32(right);
        let lanes = match kernel {
            Kernel::L2 => {
                let difference = vsubq_f64(left, right);
                vmulq_f64(difference, difference)
            }
            Kernel::Dot | Kernel::NormSq => vmulq_f64(left, right),
        };
        sum += vgetq_lane_f64(lanes, 0);
        sum += vgetq_lane_f64(lanes, 1);
        index += 2;
    }
    while index < n {
        match kernel {
            Kernel::L2 => {
                let difference = f64::from(a[index]) - f64::from(b[index]);
                sum += difference * difference;
            }
            Kernel::Dot => sum += f64::from(a[index]) * f64::from(b[index]),
            Kernel::NormSq => {
                let wide = f64::from(a[index]);
                sum += wide * wide;
            }
        }
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn f64_f32_neon(query: &[f64], candidate: &[f32], kernel: Kernel) -> f64 {
    use std::arch::aarch64::{
        vcvt_f64_f32, vgetq_lane_f64, vld1_f32, vld1q_f64, vmulq_f64, vsubq_f64,
    };
    let n = query.len();
    let mut sum = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vld1q_f64(query.as_ptr().add(index));
        let right = vcvt_f64_f32(vld1_f32(candidate.as_ptr().add(index)));
        let lanes = match kernel {
            Kernel::L2 => {
                let difference = vsubq_f64(left, right);
                vmulq_f64(difference, difference)
            }
            Kernel::Dot => vmulq_f64(left, right),
            Kernel::NormSq => vmulq_f64(right, right),
        };
        sum += vgetq_lane_f64(lanes, 0);
        sum += vgetq_lane_f64(lanes, 1);
        index += 2;
    }
    while index < n {
        match kernel {
            Kernel::L2 => {
                let difference = query[index] - f64::from(candidate[index]);
                sum += difference * difference;
            }
            Kernel::Dot => sum += query[index] * f64::from(candidate[index]),
            Kernel::NormSq => {
                let wide = f64::from(candidate[index]);
                sum += wide * wide;
            }
        }
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn cosine_parts_f32_neon(query: &[f32], candidate: &[f32]) -> (f64, f64) {
    use std::arch::aarch64::{vcvt_f64_f32, vgetq_lane_f64, vld1_f32, vmulq_f64};
    let n = query.len();
    let mut dot = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vcvt_f64_f32(vld1_f32(query.as_ptr().add(index)));
        let right = vcvt_f64_f32(vld1_f32(candidate.as_ptr().add(index)));
        let products = vmulq_f64(left, right);
        let squares = vmulq_f64(right, right);
        // Keep the scalar left-to-right contract: lane0 then lane1 for both
        // accumulators before the next pair.
        dot += vgetq_lane_f64(products, 0);
        norm += vgetq_lane_f64(squares, 0);
        dot += vgetq_lane_f64(products, 1);
        norm += vgetq_lane_f64(squares, 1);
        index += 2;
    }
    while index < n {
        let left = f64::from(query[index]);
        let right = f64::from(candidate[index]);
        dot += left * right;
        norm += right * right;
        index += 1;
    }
    (dot, norm)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn cosine_parts_f64_f32_neon(query: &[f64], candidate: &[f32]) -> (f64, f64) {
    use std::arch::aarch64::{vcvt_f64_f32, vgetq_lane_f64, vld1_f32, vld1q_f64, vmulq_f64};
    let n = query.len();
    let mut dot = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vld1q_f64(query.as_ptr().add(index));
        let right = vcvt_f64_f32(vld1_f32(candidate.as_ptr().add(index)));
        let products = vmulq_f64(left, right);
        let squares = vmulq_f64(right, right);
        dot += vgetq_lane_f64(products, 0);
        norm += vgetq_lane_f64(squares, 0);
        dot += vgetq_lane_f64(products, 1);
        norm += vgetq_lane_f64(squares, 1);
        index += 2;
    }
    while index < n {
        let right = f64::from(candidate[index]);
        dot += query[index] * right;
        norm += right * right;
        index += 1;
    }
    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn f32_sse2(a: &[f32], b: &[f32], kernel: Kernel) -> f64 {
    use std::arch::x86_64::{
        _mm_castsi128_ps, _mm_cvtps_pd, _mm_cvtsd_f64, _mm_loadl_epi64, _mm_mul_pd, _mm_sub_pd,
        _mm_unpackhi_pd,
    };
    let n = a.len();
    let mut sum = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(
            a.as_ptr().add(index).cast(),
        )));
        let right = match kernel {
            Kernel::NormSq => left,
            _ => _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(
                b.as_ptr().add(index).cast(),
            ))),
        };
        let lanes = match kernel {
            Kernel::L2 => {
                let difference = _mm_sub_pd(left, right);
                _mm_mul_pd(difference, difference)
            }
            Kernel::Dot | Kernel::NormSq => _mm_mul_pd(left, right),
        };
        sum += _mm_cvtsd_f64(lanes);
        sum += _mm_cvtsd_f64(_mm_unpackhi_pd(lanes, lanes));
        index += 2;
    }
    while index < n {
        match kernel {
            Kernel::L2 => {
                let difference = f64::from(a[index]) - f64::from(b[index]);
                sum += difference * difference;
            }
            Kernel::Dot => sum += f64::from(a[index]) * f64::from(b[index]),
            Kernel::NormSq => {
                let wide = f64::from(a[index]);
                sum += wide * wide;
            }
        }
        index += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn f64_f32_sse2(query: &[f64], candidate: &[f32], kernel: Kernel) -> f64 {
    use std::arch::x86_64::{
        _mm_castsi128_ps, _mm_cvtps_pd, _mm_cvtsd_f64, _mm_loadl_epi64, _mm_loadu_pd, _mm_mul_pd,
        _mm_sub_pd, _mm_unpackhi_pd,
    };
    let n = query.len();
    let mut sum = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = _mm_loadu_pd(query.as_ptr().add(index));
        let right = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(
            candidate.as_ptr().add(index).cast(),
        )));
        let lanes = match kernel {
            Kernel::L2 => {
                let difference = _mm_sub_pd(left, right);
                _mm_mul_pd(difference, difference)
            }
            Kernel::Dot => _mm_mul_pd(left, right),
            Kernel::NormSq => _mm_mul_pd(right, right),
        };
        sum += _mm_cvtsd_f64(lanes);
        sum += _mm_cvtsd_f64(_mm_unpackhi_pd(lanes, lanes));
        index += 2;
    }
    while index < n {
        match kernel {
            Kernel::L2 => {
                let difference = query[index] - f64::from(candidate[index]);
                sum += difference * difference;
            }
            Kernel::Dot => sum += query[index] * f64::from(candidate[index]),
            Kernel::NormSq => {
                let wide = f64::from(candidate[index]);
                sum += wide * wide;
            }
        }
        index += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn cosine_parts_f32_sse2(query: &[f32], candidate: &[f32]) -> (f64, f64) {
    use std::arch::x86_64::{
        _mm_castsi128_ps, _mm_cvtps_pd, _mm_cvtsd_f64, _mm_loadl_epi64, _mm_mul_pd, _mm_unpackhi_pd,
    };
    let n = query.len();
    let mut dot = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(
            query.as_ptr().add(index).cast(),
        )));
        let right = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(
            candidate.as_ptr().add(index).cast(),
        )));
        let products = _mm_mul_pd(left, right);
        let squares = _mm_mul_pd(right, right);
        dot += _mm_cvtsd_f64(products);
        norm += _mm_cvtsd_f64(squares);
        dot += _mm_cvtsd_f64(_mm_unpackhi_pd(products, products));
        norm += _mm_cvtsd_f64(_mm_unpackhi_pd(squares, squares));
        index += 2;
    }
    while index < n {
        let left = f64::from(query[index]);
        let right = f64::from(candidate[index]);
        dot += left * right;
        norm += right * right;
        index += 1;
    }
    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn cosine_parts_f64_f32_sse2(query: &[f64], candidate: &[f32]) -> (f64, f64) {
    use std::arch::x86_64::{
        _mm_castsi128_ps, _mm_cvtps_pd, _mm_cvtsd_f64, _mm_loadl_epi64, _mm_loadu_pd, _mm_mul_pd,
        _mm_unpackhi_pd,
    };
    let n = query.len();
    let mut dot = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = _mm_loadu_pd(query.as_ptr().add(index));
        let right = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(
            candidate.as_ptr().add(index).cast(),
        )));
        let products = _mm_mul_pd(left, right);
        let squares = _mm_mul_pd(right, right);
        dot += _mm_cvtsd_f64(products);
        norm += _mm_cvtsd_f64(squares);
        dot += _mm_cvtsd_f64(_mm_unpackhi_pd(products, products));
        norm += _mm_cvtsd_f64(_mm_unpackhi_pd(squares, squares));
        index += 2;
    }
    while index < n {
        let right = f64::from(candidate[index]);
        dot += query[index] * right;
        norm += right * right;
        index += 1;
    }
    (dot, norm)
}

#[cfg(test)]
mod tests {
    use super::{
        cosine_parts_f32, cosine_parts_f64_f32, dot_f32, dot_f64_f32, l2sq_f32, l2sq_f64_f32,
        norm_sq_f32, score_f32, score_f64_f32,
    };
    use crate::types::MetricType;

    fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum()
    }

    fn scalar_l2sq_f32(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum()
    }

    #[test]
    fn f32_kernels_match_left_to_right_scalar_bits() {
        let query = [
            0.25_f32, -0.5, 0.75, 0.125, -1.0, 0.375, 0.625, -0.875, 0.0625,
        ];
        let candidate = [-0.75_f32, -0.25, 0.5, 1.0, 0.125, 0.25, -0.5, 0.75, -0.3125];
        assert_eq!(
            dot_f32(&query, &candidate).to_bits(),
            scalar_dot_f32(&query, &candidate).to_bits()
        );
        assert_eq!(
            l2sq_f32(&query, &candidate).to_bits(),
            scalar_l2sq_f32(&query, &candidate).to_bits()
        );
        assert_eq!(
            norm_sq_f32(&candidate).to_bits(),
            candidate
                .iter()
                .map(|value| {
                    let wide = f64::from(*value);
                    wide * wide
                })
                .sum::<f64>()
                .to_bits()
        );
        let (dot, norm) = cosine_parts_f32(&query, &candidate);
        let mut expected_dot = 0.0_f64;
        let mut expected_norm = 0.0_f64;
        for (left, right) in query.iter().zip(&candidate) {
            let left = f64::from(*left);
            let right = f64::from(*right);
            expected_dot += left * right;
            expected_norm += right * right;
        }
        assert_eq!(dot.to_bits(), expected_dot.to_bits());
        assert_eq!(norm.to_bits(), expected_norm.to_bits());
    }

    #[test]
    fn f64_f32_kernels_match_document_promotion_bits() {
        let query: Vec<f64> = [0.25_f32, -0.5, 0.75, 0.125, -1.0]
            .into_iter()
            .map(f64::from)
            .collect();
        let candidate = [-0.75_f32, -0.25, 0.5, 1.0, 0.125];
        let expected_dot: f64 = query
            .iter()
            .zip(&candidate)
            .map(|(left, right)| *left * f64::from(*right))
            .sum();
        assert_eq!(
            dot_f64_f32(&query, &candidate).to_bits(),
            expected_dot.to_bits()
        );
        let expected_l2: f64 = query
            .iter()
            .zip(&candidate)
            .map(|(left, right)| {
                let difference = *left - f64::from(*right);
                difference * difference
            })
            .sum();
        assert_eq!(
            l2sq_f64_f32(&query, &candidate).to_bits(),
            expected_l2.to_bits()
        );
        let (dot, norm) = cosine_parts_f64_f32(&query, &candidate);
        let mut expected_dot = 0.0_f64;
        let mut expected_norm = 0.0_f64;
        for (left, right) in query.iter().zip(&candidate) {
            let right = f64::from(*right);
            expected_dot += *left * right;
            expected_norm += right * right;
        }
        assert_eq!(dot.to_bits(), expected_dot.to_bits());
        assert_eq!(norm.to_bits(), expected_norm.to_bits());
    }

    #[test]
    fn score_helpers_cover_all_metrics() {
        let query = [0.5_f32, -0.25, 0.125];
        let candidate = [0.25_f32, 0.5, -0.75];
        let query_norm = norm_sq_f32(&query).sqrt();
        for metric in [
            MetricType::L2,
            MetricType::Ip,
            MetricType::Cosine,
            MetricType::MipsL2,
        ] {
            let from_f32 = score_f32(&query, &candidate, metric, query_norm);
            let query_f64: Vec<f64> = query.iter().copied().map(f64::from).collect();
            let from_doc = score_f64_f32(&query_f64, &candidate, metric, query_norm);
            assert_eq!(from_f32.to_bits(), from_doc.to_bits(), "{metric:?}");
            assert!(from_f32.is_finite());
        }
    }
}
