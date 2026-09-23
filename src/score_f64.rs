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

/// Four independent copies of [`dot_f32`].
///
/// Each accumulator still folds lane 0 then lane 1, so every result matches
/// the scalar left-to-right product. The four chains are only interleaved so
/// the `f64` add latency of one neighbor does not stall the next.
#[inline]
pub(crate) fn dot_f32_x4(query: &[f32], candidates: [&[f32]; 4]) -> [f64; 4] {
    debug_assert!(candidates
        .iter()
        .all(|candidate| candidate.len() == query.len()));
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            return dot_f32_x4_neon(query, candidates);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            return dot_f32_x4_sse2(query, candidates);
        }
    }
    #[allow(unreachable_code)]
    [
        dot_f32(query, candidates[0]),
        dot_f32(query, candidates[1]),
        dot_f32(query, candidates[2]),
        dot_f32(query, candidates[3]),
    ]
}

/// Eight independent copies of [`dot_f32`], with the same lane order as [`dot_f32_x4`].
#[inline]
#[allow(unreachable_code)]
pub(crate) fn dot_f32_x8(query: &[f32], candidates: [&[f32]; 8]) -> [f64; 8] {
    debug_assert!(candidates
        .iter()
        .all(|candidate| candidate.len() == query.len()));
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            return dot_f32_x8_neon(query, candidates);
        }
    }
    let mut sums = [0.0_f64; 8];
    for (lane, candidate) in candidates.iter().enumerate() {
        sums[lane] = dot_f32(query, candidate);
    }
    sums
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

/// Inner product over already-promoted `f64` coordinates.
#[inline]
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    dispatch_f64(a, b, Kernel::Dot)
}

/// Squared L2 over already-promoted `f64` coordinates.
#[inline]
pub(crate) fn l2sq_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    dispatch_f64(a, b, Kernel::L2)
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
fn dispatch_f64(a: &[f64], b: &[f64], kernel: Kernel) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        return unsafe { f64_neon(a, b, kernel) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        return unsafe { f64_sse2(a, b, kernel) };
    }
    #[allow(unreachable_code)]
    f64_scalar(a, b, kernel)
}

/// Fast `f32` dot used only to reject Flat candidates. The exact `f64` dot
/// still decides membership. A finite result below the retained worst score
/// by [`f32_cosine_score_error`] plus [`f32_dot_absolute_error`] cannot enter
/// the exact top-k. Non-finite results are not a rejection.
#[inline]
pub(crate) fn dot_f32_approx(query: &[f32], candidate: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: both slices are read only up to the shared prefix length.
        #[allow(unsafe_code)]
        unsafe {
            return dot_f32_neon(query, candidate);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            return dot_f32_sse2(query, candidate);
        }
    }
    #[allow(unreachable_code)]
    dot_f32_scalar(query, candidate)
}

/// Relative allowance on `|approx_score - exact_score|` for cosine rank keys
/// `dot * inv_norm`.
///
/// Each `f32` product and addition rounds with unit roundoff no larger than
/// `f32::EPSILON` (twice IEEE roundoff). Fewer than `2n` of those steps touch
/// one dot of `n` terms, so while `2n*u < 1/2`:
/// `|dot_f32 - dot_f64| <= γ_{2n} * Σ|q_i c_i|` and Cauchy–Schwarz gives
/// `|dot_f32 - dot_f64| / ||c|| <= γ_{2n} * ||q||`.
/// The extra `f64` multiply by `inv_norm` is covered by one `f64` epsilon.
/// Callers still have to exact-check a non-finite `f32` dot: overflow is
/// outside this model. Underflow is [`f32_dot_absolute_error`].
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn f32_cosine_score_error(dimension: usize, query_norm: f64) -> f64 {
    if !query_norm.is_finite() {
        return f64::INFINITY;
    }
    let terms = dimension.max(1) as f64;
    let unit = f64::from(f32::EPSILON);
    let scaled = 2.0 * terms * unit;
    if scaled >= 0.5 {
        return f64::INFINITY;
    }
    let gamma = scaled / (1.0 - scaled);
    gamma * query_norm * (1.0 + f64::EPSILON) + query_norm * f64::EPSILON
}

/// Absolute dot error from subnormal rounding, in dot units. Divide by the
/// candidate norm before comparing cosine rank keys.
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn f32_dot_absolute_error(dimension: usize) -> f64 {
    let terms = dimension.max(1) as f64;
    // Half an ulp at the `f32` subnormal boundary is `2^{-150}`.
    let half_subnormal_ulp = f64::from(f32::MIN_POSITIVE) * 2.0_f64.powi(-24);
    2.0 * terms * half_subnormal_ulp
}

#[inline]
fn dot_f32_scalar(query: &[f32], candidate: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (left, right) in query.iter().zip(candidate) {
        sum += *left * *right;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn dot_f32_neon(query: &[f32], candidate: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vgetq_lane_f32, vld1q_f32};
    let n = query.len().min(candidate.len());
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 16 <= n {
        let left0 = vld1q_f32(query.as_ptr().add(index));
        let right0 = vld1q_f32(candidate.as_ptr().add(index));
        let left1 = vld1q_f32(query.as_ptr().add(index + 4));
        let right1 = vld1q_f32(candidate.as_ptr().add(index + 4));
        let left2 = vld1q_f32(query.as_ptr().add(index + 8));
        let right2 = vld1q_f32(candidate.as_ptr().add(index + 8));
        let left3 = vld1q_f32(query.as_ptr().add(index + 12));
        let right3 = vld1q_f32(candidate.as_ptr().add(index + 12));
        acc0 = vfmaq_f32(acc0, left0, right0);
        acc1 = vfmaq_f32(acc1, left1, right1);
        acc2 = vfmaq_f32(acc2, left2, right2);
        acc3 = vfmaq_f32(acc3, left3, right3);
        index += 16;
    }
    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    while index + 4 <= n {
        let left = vld1q_f32(query.as_ptr().add(index));
        let right = vld1q_f32(candidate.as_ptr().add(index));
        acc = vfmaq_f32(acc, left, right);
        index += 4;
    }
    let mut sum = vgetq_lane_f32(acc, 0)
        + vgetq_lane_f32(acc, 1)
        + vgetq_lane_f32(acc, 2)
        + vgetq_lane_f32(acc, 3);
    while index < n {
        sum += *query.get_unchecked(index) * *candidate.get_unchecked(index);
        index += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn dot_f32_sse2(query: &[f32], candidate: &[f32]) -> f32 {
    use std::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_storeu_ps};
    let n = query.len().min(candidate.len());
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut index = 0usize;
    while index + 16 <= n {
        let left0 = _mm_loadu_ps(query.as_ptr().add(index));
        let right0 = _mm_loadu_ps(candidate.as_ptr().add(index));
        let left1 = _mm_loadu_ps(query.as_ptr().add(index + 4));
        let right1 = _mm_loadu_ps(candidate.as_ptr().add(index + 4));
        let left2 = _mm_loadu_ps(query.as_ptr().add(index + 8));
        let right2 = _mm_loadu_ps(candidate.as_ptr().add(index + 8));
        let left3 = _mm_loadu_ps(query.as_ptr().add(index + 12));
        let right3 = _mm_loadu_ps(candidate.as_ptr().add(index + 12));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(left0, right0));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(left1, right1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(left2, right2));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(left3, right3));
        index += 16;
    }
    let mut acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    while index + 4 <= n {
        let left = _mm_loadu_ps(query.as_ptr().add(index));
        let right = _mm_loadu_ps(candidate.as_ptr().add(index));
        acc = _mm_add_ps(acc, _mm_mul_ps(left, right));
        index += 4;
    }
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while index < n {
        sum += *query.get_unchecked(index) * *candidate.get_unchecked(index);
        index += 1;
    }
    sum
}

#[inline]
fn f64_scalar(a: &[f64], b: &[f64], kernel: Kernel) -> f64 {
    let mut sum = 0.0_f64;
    match kernel {
        Kernel::L2 => {
            for (left, right) in a.iter().zip(b) {
                let difference = *left - *right;
                sum += difference * difference;
            }
        }
        Kernel::Dot => {
            for (left, right) in a.iter().zip(b) {
                sum += *left * *right;
            }
        }
        Kernel::NormSq => {
            for value in a {
                sum += *value * *value;
            }
        }
    }
    sum
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
unsafe fn f64_neon(a: &[f64], b: &[f64], kernel: Kernel) -> f64 {
    use std::arch::aarch64::{vgetq_lane_f64, vld1q_f64, vmulq_f64, vsubq_f64};
    let n = a.len();
    let mut sum = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vld1q_f64(a.as_ptr().add(index));
        let right = match kernel {
            Kernel::NormSq => left,
            _ => vld1q_f64(b.as_ptr().add(index)),
        };
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
                let difference = a[index] - b[index];
                sum += difference * difference;
            }
            Kernel::Dot => sum += a[index] * b[index],
            Kernel::NormSq => sum += a[index] * a[index],
        }
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn dot_f32_x4_neon(query: &[f32], candidates: [&[f32]; 4]) -> [f64; 4] {
    use std::arch::aarch64::{vcvt_f64_f32, vgetq_lane_f64, vld1_f32, vmulq_f64};
    let n = query.len();
    let mut sums = [0.0_f64; 4];
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vcvt_f64_f32(vld1_f32(query.as_ptr().add(index)));
        let product0 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[0].as_ptr().add(index))),
        );
        let product1 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[1].as_ptr().add(index))),
        );
        let product2 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[2].as_ptr().add(index))),
        );
        let product3 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[3].as_ptr().add(index))),
        );
        sums[0] += vgetq_lane_f64(product0, 0);
        sums[1] += vgetq_lane_f64(product1, 0);
        sums[2] += vgetq_lane_f64(product2, 0);
        sums[3] += vgetq_lane_f64(product3, 0);
        sums[0] += vgetq_lane_f64(product0, 1);
        sums[1] += vgetq_lane_f64(product1, 1);
        sums[2] += vgetq_lane_f64(product2, 1);
        sums[3] += vgetq_lane_f64(product3, 1);
        index += 2;
    }
    while index < n {
        let left = f64::from(*query.get_unchecked(index));
        for lane in 0..4 {
            let right = f64::from(*candidates[lane].get_unchecked(index));
            sums[lane] += left * right;
        }
        index += 1;
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn dot_f32_x8_neon(query: &[f32], candidates: [&[f32]; 8]) -> [f64; 8] {
    use std::arch::aarch64::{vcvt_f64_f32, vgetq_lane_f64, vld1_f32, vmulq_f64};
    let n = query.len();
    let mut sum0 = 0.0_f64;
    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut sum3 = 0.0_f64;
    let mut sum4 = 0.0_f64;
    let mut sum5 = 0.0_f64;
    let mut sum6 = 0.0_f64;
    let mut sum7 = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = vcvt_f64_f32(vld1_f32(query.as_ptr().add(index)));
        let product0 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[0].as_ptr().add(index))),
        );
        let product1 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[1].as_ptr().add(index))),
        );
        let product2 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[2].as_ptr().add(index))),
        );
        let product3 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[3].as_ptr().add(index))),
        );
        let product4 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[4].as_ptr().add(index))),
        );
        let product5 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[5].as_ptr().add(index))),
        );
        let product6 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[6].as_ptr().add(index))),
        );
        let product7 = vmulq_f64(
            left,
            vcvt_f64_f32(vld1_f32(candidates[7].as_ptr().add(index))),
        );
        sum0 += vgetq_lane_f64(product0, 0);
        sum1 += vgetq_lane_f64(product1, 0);
        sum2 += vgetq_lane_f64(product2, 0);
        sum3 += vgetq_lane_f64(product3, 0);
        sum4 += vgetq_lane_f64(product4, 0);
        sum5 += vgetq_lane_f64(product5, 0);
        sum6 += vgetq_lane_f64(product6, 0);
        sum7 += vgetq_lane_f64(product7, 0);
        sum0 += vgetq_lane_f64(product0, 1);
        sum1 += vgetq_lane_f64(product1, 1);
        sum2 += vgetq_lane_f64(product2, 1);
        sum3 += vgetq_lane_f64(product3, 1);
        sum4 += vgetq_lane_f64(product4, 1);
        sum5 += vgetq_lane_f64(product5, 1);
        sum6 += vgetq_lane_f64(product6, 1);
        sum7 += vgetq_lane_f64(product7, 1);
        index += 2;
    }
    while index < n {
        let left = f64::from(*query.get_unchecked(index));
        sum0 += left * f64::from(*candidates[0].get_unchecked(index));
        sum1 += left * f64::from(*candidates[1].get_unchecked(index));
        sum2 += left * f64::from(*candidates[2].get_unchecked(index));
        sum3 += left * f64::from(*candidates[3].get_unchecked(index));
        sum4 += left * f64::from(*candidates[4].get_unchecked(index));
        sum5 += left * f64::from(*candidates[5].get_unchecked(index));
        sum6 += left * f64::from(*candidates[6].get_unchecked(index));
        sum7 += left * f64::from(*candidates[7].get_unchecked(index));
        index += 1;
    }
    [sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn dot_f32_x4_sse2(query: &[f32], candidates: [&[f32]; 4]) -> [f64; 4] {
    [
        dot_f32(query, candidates[0]),
        dot_f32(query, candidates[1]),
        dot_f32(query, candidates[2]),
        dot_f32(query, candidates[3]),
    ]
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
unsafe fn f64_sse2(a: &[f64], b: &[f64], kernel: Kernel) -> f64 {
    use std::arch::x86_64::{_mm_cvtsd_f64, _mm_loadu_pd, _mm_mul_pd, _mm_sub_pd, _mm_unpackhi_pd};
    let n = a.len();
    let mut sum = 0.0_f64;
    let mut index = 0usize;
    while index + 2 <= n {
        let left = _mm_loadu_pd(a.as_ptr().add(index));
        let right = match kernel {
            Kernel::NormSq => left,
            _ => _mm_loadu_pd(b.as_ptr().add(index)),
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
                let difference = a[index] - b[index];
                sum += difference * difference;
            }
            Kernel::Dot => sum += a[index] * b[index],
            Kernel::NormSq => sum += a[index] * a[index],
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
#[allow(clippy::float_cmp)]
mod tests {
    use super::{
        cosine_parts_f32, cosine_parts_f64_f32, dispatch_f32, dot_f32, dot_f64, dot_f64_f32,
        f32_scalar, f64_f32_scalar, f64_scalar, l2sq_f32, l2sq_f64, l2sq_f64_f32, norm_sq_f32,
        score_f32, score_f64_f32, Kernel,
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

    #[test]
    fn portable_scalar_kernels_match_dispatch_bits() {
        let a32 = [0.25_f32, -0.5, 0.75, 0.125, -1.0];
        let b32 = [-0.75_f32, -0.25, 0.5, 1.0, 0.125];
        assert_eq!(
            f32_scalar(&a32, &b32, Kernel::Dot).to_bits(),
            dot_f32(&a32, &b32).to_bits()
        );
        assert_eq!(
            f32_scalar(&a32, &b32, Kernel::L2).to_bits(),
            l2sq_f32(&a32, &b32).to_bits()
        );
        assert_eq!(
            f32_scalar(&a32, &a32, Kernel::NormSq).to_bits(),
            norm_sq_f32(&a32).to_bits()
        );

        let a64: Vec<f64> = a32.iter().copied().map(f64::from).collect();
        let b64: Vec<f64> = b32.iter().copied().map(f64::from).collect();
        assert_eq!(
            f64_scalar(&a64, &b64, Kernel::Dot).to_bits(),
            dot_f64(&a64, &b64).to_bits()
        );
        assert_eq!(
            f64_scalar(&a64, &b64, Kernel::L2).to_bits(),
            l2sq_f64(&a64, &b64).to_bits()
        );
        assert_eq!(
            f64_scalar(&a64, &a64, Kernel::NormSq).to_bits(),
            a64.iter().map(|v| v * v).sum::<f64>().to_bits()
        );

        assert_eq!(
            f64_f32_scalar(&a64, &b32, Kernel::Dot).to_bits(),
            dot_f64_f32(&a64, &b32).to_bits()
        );
        assert_eq!(
            f64_f32_scalar(&a64, &b32, Kernel::L2).to_bits(),
            l2sq_f64_f32(&a64, &b32).to_bits()
        );
        assert_eq!(
            f64_f32_scalar(&a64, &b32, Kernel::NormSq).to_bits(),
            b32.iter()
                .map(|v| {
                    let wide = f64::from(*v);
                    wide * wide
                })
                .sum::<f64>()
                .to_bits()
        );
    }

    #[test]
    fn score_helpers_reject_mismatched_dimensions() {
        let query = [1.0_f32, 0.0];
        let candidate = [1.0_f32];
        assert_eq!(
            score_f32(&query, &candidate, MetricType::L2, 1.0),
            f64::NEG_INFINITY
        );
        let query_f64 = [1.0_f64, 0.0];
        assert_eq!(
            score_f64_f32(&query_f64, &candidate, MetricType::Ip, 1.0),
            f64::NEG_INFINITY
        );
        assert_eq!(score_f32(&query, &[0.0, 0.0], MetricType::Cosine, 0.0), 0.0);
    }

    #[test]
    fn enterprise_ga_portable_and_simd_kernels_match() {
        let left = [
            0.25_f32, -1.5, 2.0, 0.5, 3.25, -0.125, 8.0, 1.0, 0.0625, -4.0,
        ];
        let right = [1.0_f32, 0.0, -2.0, 4.0, 0.5, 0.25, -8.0, 2.0, 0.5, 0.125];
        for kernel in [Kernel::L2, Kernel::Dot, Kernel::NormSq] {
            assert_eq!(
                f32_scalar(&left, &right, kernel).to_bits(),
                dispatch_f32(&left, &right, kernel).to_bits()
            );
        }
    }

    #[test]
    fn dot_f32_x4_matches_scalar_left_to_right() {
        use super::dot_f32_x4;
        let query = [0.25_f32, -1.5, 2.0, 0.5, 3.25, -0.125, 8.0, 1.0, 0.0625];
        let candidates = [
            [1.0_f32, 0.0, -2.0, 4.0, 0.5, 0.25, -8.0, 2.0, 0.5],
            [-0.5_f32, 0.25, 1.0, -1.0, 0.0, 3.0, 0.125, -0.25, 2.0],
            [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [4.0_f32, 4.0, 4.0, -4.0, 0.5, -0.5, 0.25, 0.125, -1.0],
        ];
        let wide = dot_f32_x4(
            &query,
            [
                &candidates[0],
                &candidates[1],
                &candidates[2],
                &candidates[3],
            ],
        );
        for (lane, candidate) in candidates.iter().enumerate() {
            assert_eq!(wide[lane].to_bits(), dot_f32(&query, candidate).to_bits());
            assert_eq!(
                wide[lane].to_bits(),
                super::cosine_parts_f32(&query, candidate).0.to_bits()
            );
        }
        let odd = &query[..8];
        let odd_candidates = [
            &candidates[0][..8],
            &candidates[1][..8],
            &candidates[2][..8],
            &candidates[3][..8],
        ];
        let wide_odd = dot_f32_x4(odd, odd_candidates);
        for (lane, candidate) in odd_candidates.iter().enumerate() {
            assert_eq!(wide_odd[lane].to_bits(), dot_f32(odd, candidate).to_bits());
        }
    }

    #[test]
    fn dot_f32_x8_matches_scalar_left_to_right() {
        use super::dot_f32_x8;
        let query = [0.25_f32, -1.5, 2.0, 0.5, 3.25, -0.125, 8.0, 1.0, 0.0625];
        let candidates = [
            [1.0_f32, 0.0, -2.0, 4.0, 0.5, 0.25, -8.0, 2.0, 0.5],
            [-0.5_f32, 0.25, 1.0, -1.0, 0.0, 3.0, 0.125, -0.25, 2.0],
            [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [4.0_f32, 4.0, 4.0, -4.0, 0.5, -0.5, 0.25, 0.125, -1.0],
            [0.5_f32, -0.5, 0.25, 0.125, 1.0, -2.0, 0.0, 3.0, -0.25],
            [-8.0_f32, 0.125, 2.0, -0.5, 0.5, 0.25, 1.0, -1.0, 4.0],
            [0.0625_f32, 8.0, -0.125, 3.25, 0.5, 2.0, -1.5, 0.25, 1.0],
            [-1.0_f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ];
        let wide = dot_f32_x8(
            &query,
            [
                &candidates[0],
                &candidates[1],
                &candidates[2],
                &candidates[3],
                &candidates[4],
                &candidates[5],
                &candidates[6],
                &candidates[7],
            ],
        );
        for (lane, candidate) in candidates.iter().enumerate() {
            assert_eq!(wide[lane].to_bits(), dot_f32(&query, candidate).to_bits());
            assert_eq!(
                wide[lane].to_bits(),
                super::cosine_parts_f32(&query, candidate).0.to_bits()
            );
        }
        let odd = &query[..8];
        let odd_candidates = [
            &candidates[0][..8],
            &candidates[1][..8],
            &candidates[2][..8],
            &candidates[3][..8],
            &candidates[4][..8],
            &candidates[5][..8],
            &candidates[6][..8],
            &candidates[7][..8],
        ];
        let wide_odd = dot_f32_x8(odd, odd_candidates);
        for (lane, candidate) in odd_candidates.iter().enumerate() {
            assert_eq!(wide_odd[lane].to_bits(), dot_f32(odd, candidate).to_bits());
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn f32_rejector_allowance_covers_the_approx_kernel() {
        use super::{dot_f32_approx, f32_cosine_score_error, f32_dot_absolute_error};

        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut bits = || {
            state = state
                .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                .wrapping_add(0x94D0_49BB_1331_11EB);
            state
        };
        let mut sample = |kind: u8| -> f32 {
            let mixed = bits();
            match kind {
                0 => ((mixed >> 11) as f64 / 9_007_199_254_740_992.0 * 2.0 - 1.0) as f32,
                1 => {
                    let unit = (mixed >> 11) as f64 / 9_007_199_254_740_992.0;
                    (unit * 2.0e10 - 1.0e10) as f32
                }
                2 => f32::from_bits(((mixed as u32) & 0x0000_ffff) | 0x0000_0001),
                3 => f32::from_bits(0x7f7f_ffff),
                _ => 0.0,
            }
        };
        for dimension in [1_usize, 3, 4, 7, 17, 32, 128, 130] {
            for kind in 0..5_u8 {
                for _ in 0..12 {
                    let query: Vec<f32> = (0..dimension).map(|_| sample(kind)).collect();
                    let mut candidate: Vec<f32> =
                        (0..dimension).map(|_| sample(kind % 4)).collect();
                    if dimension > 1 {
                        candidate[1] = query[0];
                    }
                    let query_f64: Vec<f64> = query.iter().copied().map(f64::from).collect();
                    let approximate = f64::from(dot_f32_approx(&query, &candidate));
                    if !approximate.is_finite() {
                        continue;
                    }
                    let exact = dot_f64_f32(&query_f64, &candidate);
                    let candidate_norm = candidate
                        .iter()
                        .map(|value| {
                            let wide = f64::from(*value);
                            wide * wide
                        })
                        .sum::<f64>()
                        .sqrt();
                    if candidate_norm == 0.0 || !candidate_norm.is_finite() {
                        continue;
                    }
                    let query_norm = query_f64
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                        .sqrt();
                    let inverse_norm = 1.0 / candidate_norm;
                    let delta = (approximate - exact).abs() * inverse_norm;
                    let allowance = f32_cosine_score_error(dimension, query_norm)
                        + f32_dot_absolute_error(dimension) * inverse_norm;
                    assert!(
                        delta <= allowance,
                        "dim {dimension} kind {kind} delta {delta} allowance {allowance} approx {approximate} exact {exact}"
                    );
                }
            }
        }
    }
}
