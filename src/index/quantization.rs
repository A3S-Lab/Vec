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
                Ok(Self::Fp16(encode_fp16_bits(&values)?))
            }
            QuantizeType::Int8 => {
                let (codes, scale) = quantize_i8(&values);
                Ok(Self::Int8 { codes, scale })
            }
            QuantizeType::Int4 => {
                let dimension = values.len();
                let (nibbles, scale) = quantize_i4(&values);
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
            Self::Fp16(values) => values
                .iter()
                .copied()
                .map(crate::doc::fp16_to_f32)
                .collect(),
            Self::Int8 { codes, scale } => {
                codes.iter().map(|code| f32::from(*code) * *scale).collect()
            }
            Self::Int4 {
                nibbles,
                scale,
                dimension,
            } => nibbles
                .iter()
                .flat_map(|byte| {
                    [
                        decoded_nibble(*byte) * *scale,
                        decoded_nibble(*byte >> 4) * *scale,
                    ]
                })
                .take(*dimension)
                .collect(),
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
                        .all(|value| crate::doc::fp16_to_f32(*value).is_finite())
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
            values.iter().map(|value| crate::doc::fp16_to_f32(*value)),
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
    candidate_norm: Option<f32>,
) -> f32 {
    if query.len() != candidate.len() {
        return f32::NEG_INFINITY;
    }
    match metric {
        MetricType::L2 => -f32_from_f64(crate::score_f64::l2sq_f32(query, candidate)),
        MetricType::Cosine => {
            let candidate_norm = candidate_norm
                .filter(|value| value.is_finite())
                .unwrap_or_else(|| f32_from_f64(crate::score_f64::norm_sq_f32(candidate).sqrt()));
            // Graph construction still selects neighbors with `f64` dots.
            // Query navigation only needs a fast order; exact re-rank restores
            // the public `f64` score.
            let dot = crate::score_f64::dot_f32_approx(query, candidate);
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
            f32_from_f64(crate::score_f64::dot_f32(query, candidate))
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
    candidate_norm: Option<f32>,
) -> f64 {
    match candidate {
        QuantizedVector::F32(values) => {
            let fast = score_dense_fast(query, values, metric, query_norm_f32, candidate_norm);
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
    f32_from_f64(crate::score_f64::norm_sq_f32(query).sqrt())
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
    crate::score_f64::norm_sq_f32(query).sqrt()
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

pub(super) fn dense_candidate_norm(candidate: &[f32]) -> f64 {
    dense_query_norm(candidate)
}

pub(super) fn score_dense_cosine(
    query: &[f32],
    candidate: &[f32],
    query_norm: f64,
    candidate_norm: f64,
) -> f64 {
    if query.len() != candidate.len() {
        return f64::NEG_INFINITY;
    }
    // Prefer the fused SIMD pass when the caller did not supply a precomputed
    // candidate norm. When a norm is provided (graph edge caches), keep the
    // historical divide so cached construction scores stay unchanged.
    if candidate_norm.is_finite() && candidate_norm > 0.0 {
        // The cached norm makes the squared-norm half of `cosine_parts_f32`
        // unused. `dot_f32` folds the same left-to-right products.
        let dot = crate::score_f64::dot_f32(query, candidate);
        if query_norm == 0.0 {
            0.0
        } else {
            dot / (query_norm * candidate_norm)
        }
    } else {
        crate::score_f64::score_f32(query, candidate, MetricType::Cosine, query_norm)
    }
}

pub(super) fn score_dense_with_query_norm(
    query: &[f32],
    candidate: &[f32],
    metric: MetricType,
    query_norm: f64,
) -> f64 {
    crate::score_f64::score_f32(query, candidate, metric, query_norm)
}

fn f32_from_f64(value: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        value as f32
    }
}

fn encode_fp16_bits(values: &[f32]) -> Result<Vec<u16>> {
    values
        .iter()
        .copied()
        .map(crate::doc::f32_to_fp16)
        .collect()
}

/// Symmetric INT8 codes. `scale` is `max(|v|) / 127`, or `1` for an all-zero vector.
fn quantize_i8(values: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inverse = 1.0 / scale;
    let codes = values
        .iter()
        .map(|value| {
            #[allow(clippy::cast_possible_truncation)]
            {
                (value * inverse).round().clamp(-127.0, 127.0) as i8
            }
        })
        .collect();
    (codes, scale)
}

/// Symmetric INT4 codes packed two values per byte, range `-7..=7`.
fn quantize_i4(values: &[f32]) -> (Vec<u8>, f32) {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
    let inverse = 1.0 / scale;
    let mut nibbles = Vec::with_capacity(values.len().div_ceil(2));
    let mut index = 0;
    while index + 1 < values.len() {
        let low = clamped_nibble(values[index] * inverse);
        let high = clamped_nibble(values[index + 1] * inverse);
        nibbles.push(low | (high << 4));
        index += 2;
    }
    if index < values.len() {
        nibbles.push(clamped_nibble(values[index] * inverse));
    }
    (nibbles, scale)
}

fn clamped_nibble(value: f32) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (value.round().clamp(-7.0, 7.0) as i8 & 0x0f) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dense_candidate_norm, dense_query_norm, dense_query_norm_fast, score, score_ann,
        score_dense, score_dense_cosine, score_dense_with_query_norm, QuantizedVector,
    };
    use crate::doc::VectorValue;
    use crate::types::{MetricType, QuantizeType};

    #[test]
    fn fp16_int8_and_int4_decode_to_finite_coordinates() {
        let values = vec![0.0_f32, 1.0, -2.5, 3.25];
        for quantize in [QuantizeType::Fp16, QuantizeType::Int8, QuantizeType::Int4] {
            let encoded = QuantizedVector::encode(values.clone(), quantize).expect("encode");
            let decoded = encoded.decode();
            assert_eq!(decoded.len(), values.len());
            assert!(
                decoded.iter().all(|value| value.is_finite()),
                "{quantize:?} decoded {decoded:?}"
            );
            assert!(score(&values, &encoded, MetricType::Cosine).is_finite());
        }
    }

    #[test]
    fn unquantized_f32_score_matches_the_document_promotion() {
        let query = [0.25_f32, -0.5, 0.75, 0.125];
        let stored = [0.5_f32, 0.25, -0.125, 1.0];
        let document = VectorValue::Fp32(stored.to_vec());
        let query_f64: Vec<f64> = query.iter().copied().map(f64::from).collect();
        let norm = query_f64
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let from_document = document
            .dense_score(&query_f64, norm, MetricType::Cosine)
            .expect("fp32 document must score");
        let from_index = score_dense_with_query_norm(&query, &stored, MetricType::Cosine, norm);
        assert_eq!(from_document.to_bits(), from_index.to_bits());
    }

    #[test]
    fn cached_cosine_norm_matches_the_live_f64_score() {
        let query = [1.0_f32, -2.0, 0.5, 0.0];
        let candidate = [0.25_f32, 4.0, -1.0, 3.5];
        let query_norm = dense_query_norm(&query);
        let live = score_dense_with_query_norm(&query, &candidate, MetricType::Cosine, query_norm);
        let cached = score_dense_cosine(
            &query,
            &candidate,
            query_norm,
            dense_candidate_norm(&candidate),
        );
        assert_eq!(live.to_bits(), cached.to_bits());
    }

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
            let fast = score_ann(
                &query,
                &encoded,
                metric,
                query_norm_f32,
                query_norm_f64,
                None,
            );
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
            None,
        );
        let expected = score(&query, &candidate, MetricType::Cosine);
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn precomputed_cosine_norm_matches_the_live_simd_norm() {
        let query = [0.25, -0.5, 0.75, 0.125, -1.0, 0.375, 0.625, -0.875];
        let candidate = [-0.75, -0.25, 0.5, 1.0, 0.125, 0.25, -0.5, 0.75];
        let encoded = QuantizedVector::F32(candidate.to_vec());
        let query_norm_f32 = dense_query_norm_fast(&query);
        let query_norm_f64 = dense_query_norm(&query);
        let stored = dense_query_norm_fast(&candidate);
        let live = score_ann(
            &query,
            &encoded,
            MetricType::Cosine,
            query_norm_f32,
            query_norm_f64,
            None,
        );
        let cached = score_ann(
            &query,
            &encoded,
            MetricType::Cosine,
            query_norm_f32,
            query_norm_f64,
            Some(stored),
        );
        assert_eq!(live.to_bits(), cached.to_bits());
    }
}
