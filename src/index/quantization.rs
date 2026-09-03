//! Index-only scalar quantization.

use crate::error::{Error, Result};
use crate::types::{MetricType, QuantizeType};

/// Approximate vector representation owned by a derived index. Authoritative
/// document vectors remain unchanged and are used for final ranking.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) enum QuantizedVector {
    F32(Vec<f32>),
    Fp16(Vec<u16>),
    Int8 {
        codes: Vec<i8>,
        scale: f32,
    },
    Int4 {
        nibbles: Vec<u8>,
        scale: f32,
        dimension: usize,
    },
}

impl QuantizedVector {
    pub(super) fn encode(values: Vec<f32>, quantize: QuantizeType) -> Result<Self> {
        if values.is_empty() || !values.iter().all(|value| value.is_finite()) {
            return Err(Error::invalid_argument(
                "ANN index vectors must be non-empty and finite",
            ));
        }
        match quantize {
            QuantizeType::Undefined => Ok(Self::F32(values)),
            QuantizeType::Fp16 => {
                if values.iter().any(|value| value.abs() > 65_504.0) {
                    return Err(Error::invalid_argument(
                        "FP16 index quantization requires coordinates in the finite FP16 range",
                    ));
                }
                Ok(Self::Fp16(zvec_core::engine::simd::encode_fp16(&values)))
            }
            QuantizeType::Int8 => {
                let (codes, scale) = zvec_core::engine::simd::quantize_i8(&values);
                Ok(Self::Int8 { codes, scale })
            }
            QuantizeType::Int4 => {
                let dimension = values.len();
                let (nibbles, scale) = zvec_core::engine::simd::quantize_i4(&values);
                Ok(Self::Int4 {
                    nibbles,
                    scale,
                    dimension,
                })
            }
            QuantizeType::Rabitq | QuantizeType::Pq => Err(Error::not_supported(format!(
                "{quantize:?} ANN quantization is not implemented"
            ))),
        }
    }

    pub(super) fn decode(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values.clone(),
            Self::Fp16(values) => zvec_core::engine::simd::decode_fp16(values),
            Self::Int8 { codes, scale } => zvec_core::engine::simd::dequantize_i8(codes, *scale),
            Self::Int4 {
                nibbles,
                scale,
                dimension,
            } => zvec_core::engine::simd::dequantize_i4(nibbles, *scale, *dimension),
        }
    }

    pub(super) fn encoded_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.len() * std::mem::size_of::<f32>(),
            Self::Fp16(values) => values.len() * std::mem::size_of::<u16>(),
            Self::Int8 { codes, .. } => codes.len() + std::mem::size_of::<f32>(),
            Self::Int4 { nibbles, .. } => nibbles.len() + std::mem::size_of::<f32>(),
        }
    }

    pub(super) fn validates(&self, dimension: usize) -> bool {
        match self {
            Self::F32(values) => {
                values.len() == dimension && values.iter().all(|value| value.is_finite())
            }
            Self::Fp16(values) => {
                values.len() == dimension
                    && values
                        .iter()
                        .all(|value| zvec_core::engine::simd::f16_to_f32(*value).is_finite())
            }
            Self::Int8 { codes, scale } => {
                codes.len() == dimension && scale.is_finite() && *scale >= 0.0
            }
            Self::Int4 {
                nibbles,
                scale,
                dimension: encoded_dimension,
            } => {
                *encoded_dimension == dimension
                    && nibbles.len() == dimension.saturating_add(1) / 2
                    && scale.is_finite()
                    && *scale >= 0.0
            }
        }
    }
}

#[cfg(test)]
pub(super) fn score(query: &[f32], candidate: &QuantizedVector, metric: MetricType) -> f64 {
    let query_norm = if metric == MetricType::Cosine {
        dense_query_norm(query)
    } else {
        0.0
    };
    score_with_query_norm(query, candidate, metric, query_norm)
}

pub(super) fn score_with_query_norm(
    query: &[f32],
    candidate: &QuantizedVector,
    metric: MetricType,
    query_norm: f64,
) -> f64 {
    match candidate {
        QuantizedVector::F32(values) => {
            score_dense_with_query_norm(query, values, metric, query_norm)
        }
        QuantizedVector::Fp16(values) => score_iter(
            query,
            values.len(),
            values
                .iter()
                .map(|value| zvec_core::engine::simd::f16_to_f32(*value)),
            metric,
            query_norm,
        ),
        QuantizedVector::Int8 { codes, scale } => score_iter(
            query,
            codes.len(),
            codes.iter().map(|code| f32::from(*code) * *scale),
            metric,
            query_norm,
        ),
        QuantizedVector::Int4 {
            nibbles,
            scale,
            dimension,
        } => score_iter(
            query,
            *dimension,
            nibbles
                .iter()
                .flat_map(|byte| {
                    [
                        decoded_nibble(*byte) * *scale,
                        decoded_nibble(*byte >> 4) * *scale,
                    ]
                })
                .take(*dimension),
            metric,
            query_norm,
        ),
    }
}

/// Fast f32 scoring for the unquantized ANN representation.
#[inline]
pub(super) fn score_dense_fast(
    query: &[f32],
    candidate: &[f32],
    metric: MetricType,
    query_norm: f32,
) -> f32 {
    if query.len() != candidate.len() {
        return f32::NEG_INFINITY;
    }
    match metric {
        MetricType::L2 => -zvec_core::engine::simd::l2sq(query, candidate),
        MetricType::Cosine => {
            let candidate_norm = zvec_core::engine::simd::dot(candidate, candidate).sqrt();
            let dot = zvec_core::engine::simd::dot(query, candidate);
            if !query_norm.is_finite()
                || !candidate_norm.is_finite()
                || !dot.is_finite()
                || query_norm == 0.0
                || candidate_norm == 0.0
            {
                f32::NAN
            } else {
                dot / (query_norm * candidate_norm)
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => {
            zvec_core::engine::simd::dot(query, candidate)
        }
    }
}

/// Dispatches an ANN score to the SIMD f32 path when the index stores raw
/// f32 coordinates and otherwise preserves the representation-aware scorer.
/// The second norm is kept in f64 for encoded variants so this optimization
/// does not alter their existing ranking arithmetic.
#[inline]
pub(super) fn score_ann(
    query: &[f32],
    candidate: &QuantizedVector,
    metric: MetricType,
    query_norm_f32: f32,
    query_norm_f64: f64,
) -> f64 {
    match candidate {
        QuantizedVector::F32(values) => {
            let fast = score_dense_fast(query, values, metric, query_norm_f32);
            if fast.is_finite() {
                f64::from(fast)
            } else {
                score_dense_with_query_norm(query, values, metric, query_norm_f64)
            }
        }
        _ => score_with_query_norm(query, candidate, metric, query_norm_f64),
    }
}

#[inline]
pub(super) fn dense_query_norm_fast(query: &[f32]) -> f32 {
    zvec_core::engine::simd::dot(query, query).sqrt()
}

fn score_iter(
    query: &[f32],
    dimension: usize,
    candidate: impl Iterator<Item = f32>,
    metric: MetricType,
    query_norm: f64,
) -> f64 {
    if query.len() != dimension {
        return f64::NEG_INFINITY;
    }
    match metric {
        MetricType::L2 => -query
            .iter()
            .copied()
            .zip(candidate)
            .map(|(left, right)| {
                let difference = f64::from(left) - f64::from(right);
                difference * difference
            })
            .sum::<f64>(),
        MetricType::Cosine => {
            let (dot, candidate_norm) = query.iter().copied().zip(candidate).fold(
                (0.0, 0.0),
                |(dot, candidate_norm), (left, right)| {
                    let left = f64::from(left);
                    let right = f64::from(right);
                    (dot + left * right, candidate_norm + right * right)
                },
            );
            if query_norm == 0.0 || candidate_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * candidate_norm.sqrt())
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => query
            .iter()
            .copied()
            .zip(candidate)
            .map(|(left, right)| f64::from(left) * f64::from(right))
            .sum(),
    }
}

fn decoded_nibble(value: u8) -> f32 {
    let value = i16::from(value & 0x0f);
    if value > 7 {
        f32::from(value - 16)
    } else {
        f32::from(value)
    }
}

pub(super) fn dense_query_norm(query: &[f32]) -> f64 {
    query
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
pub(super) fn score_dense(query: &[f32], candidate: &[f32], metric: MetricType) -> f64 {
    let query_norm = if metric == MetricType::Cosine {
        dense_query_norm(query)
    } else {
        0.0
    };
    score_dense_with_query_norm(query, candidate, metric, query_norm)
}

pub(super) fn score_dense_with_query_norm(
    query: &[f32],
    candidate: &[f32],
    metric: MetricType,
    query_norm: f64,
) -> f64 {
    if query.len() != candidate.len() {
        return f64::NEG_INFINITY;
    }
    match metric {
        MetricType::L2 => -query
            .iter()
            .zip(candidate)
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum::<f64>(),
        MetricType::Cosine => {
            let dot = query
                .iter()
                .zip(candidate)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>();
            let candidate_norm = candidate
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                .sqrt();
            if query_norm == 0.0 || candidate_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm * candidate_norm)
            }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => query
            .iter()
            .zip(candidate)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum::<f64>(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dense_query_norm, dense_query_norm_fast, score, score_ann, score_dense, QuantizedVector,
    };
    use crate::types::{MetricType, QuantizeType};

    #[test]
    fn quantized_encodings_reduce_the_index_payload() {
        let source = vec![0.25; 33];
        let raw = QuantizedVector::encode(source.clone(), QuantizeType::Undefined)
            .expect("raw encoding must succeed");
        for quantize in [QuantizeType::Fp16, QuantizeType::Int8, QuantizeType::Int4] {
            let encoded = QuantizedVector::encode(source.clone(), quantize)
                .expect("quantized encoding must succeed");
            assert!(encoded.encoded_bytes() < raw.encoded_bytes());
            assert_eq!(encoded.decode().len(), source.len());
        }
    }

    #[test]
    fn quantized_scoring_matches_the_decoded_reference_without_materializing_it() {
        let source = vec![-0.75, -0.25, 0.5, 1.0, 0.125];
        let query = [0.25, -0.5, 0.75, 0.125, -1.0];
        for quantize in [
            QuantizeType::Undefined,
            QuantizeType::Fp16,
            QuantizeType::Int8,
            QuantizeType::Int4,
        ] {
            let encoded = QuantizedVector::encode(source.clone(), quantize)
                .expect("quantization must succeed");
            let decoded = encoded.decode();
            for metric in [MetricType::L2, MetricType::Ip, MetricType::Cosine] {
                let actual = score(&query, &encoded, metric);
                let expected = score_dense(&query, &decoded, metric);
                assert!(
                    (actual - expected).abs() <= f64::EPSILON,
                    "quantize={quantize:?} metric={metric:?} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn simd_dense_scoring_tracks_the_authoritative_f64_reference() {
        let query = [0.25, -0.5, 0.75, 0.125, -1.0, 0.375, 0.625, -0.875];
        let candidate = [-0.75, -0.25, 0.5, 1.0, 0.125, 0.25, -0.5, 0.75];
        let query_norm_f32 = dense_query_norm_fast(&query);
        let query_norm_f64 = dense_query_norm(&query);
        let encoded = QuantizedVector::F32(candidate.to_vec());
        for metric in [MetricType::L2, MetricType::Ip, MetricType::Cosine] {
            let fast = score_ann(&query, &encoded, metric, query_norm_f32, query_norm_f64);
            let exact = score_dense(&query, &candidate, metric);
            assert!((fast - exact).abs() < 1.0e-5, "metric={metric:?}");
        }
    }

    #[test]
    fn simd_scoring_falls_back_when_f32_accumulators_overflow() {
        let query = [f32::MAX, f32::MAX];
        let candidate = QuantizedVector::F32(vec![f32::MAX, f32::MAX]);
        let actual = score_ann(
            &query,
            &candidate,
            MetricType::Cosine,
            dense_query_norm_fast(&query),
            dense_query_norm(&query),
        );
        let expected = score(&query, &candidate, MetricType::Cosine);
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < f64::EPSILON);
    }
}
