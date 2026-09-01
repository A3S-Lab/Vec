//! Portable multi-bit `RaBitQ` encoding and unbiased score estimation.

mod rotation;

use self::rotation::RabitqRotation;
use super::ivf::{squared_l2, train_centroids};
use super::ordinal_map::OrdinalMap;
use super::quantization::QuantizedVector;
use crate::error::{Error, Result};
use crate::types::MetricType;

const KMEANS_ITERATIONS: usize = 5;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct RabitqCode {
    center: u32,
    packed: Vec<u8>,
    scale: f64,
    l2_add: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct RabitqQuantizer {
    metric: MetricType,
    total_bits: usize,
    requested_clusters: usize,
    sample_count: usize,
    rotation: RabitqRotation,
    centroids: Vec<Vec<f32>>,
    codes: OrdinalMap<RabitqCode>,
}

pub(super) struct PreparedRabitqQuery {
    rotated: Vec<f64>,
    center_terms: Vec<f64>,
}

impl RabitqQuantizer {
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        total_bits: usize,
        requested_clusters: usize,
        sample_count: usize,
        metric: MetricType,
    ) -> Result<Self> {
        validate_options(dimension, total_bits, requested_clusters, metric)?;
        let rotation = RabitqRotation::new(dimension)?;
        if vectors.is_empty() {
            return Ok(Self {
                metric,
                total_bits,
                requested_clusters,
                sample_count,
                rotation,
                centroids: Vec::new(),
                codes: OrdinalMap::default(),
            });
        }

        let items: Vec<(u64, Vec<f32>)> = vectors
            .iter()
            .map(|(ordinal, vector)| (ordinal, metric_vector(&vector.decode(), metric)))
            .collect();
        let cluster_count = requested_clusters.min(items.len()).max(1);
        let training_count = if sample_count == 0 {
            items.len()
        } else {
            sample_count.max(cluster_count)
        };
        let training = training_sample(&items, training_count);
        let centroids = train_centroids(&training, cluster_count, KMEANS_ITERATIONS);
        let rotated_centroids: Vec<Vec<f64>> = centroids
            .iter()
            .map(|centroid| {
                rotation.rotate(centroid).ok_or_else(|| {
                    Error::internal("RaBitQ centroid rotation produced invalid values")
                })
            })
            .collect::<Result<_>>()?;
        let codes = items
            .iter()
            .map(|(ordinal, vector)| {
                let center = nearest_centroid(vector, &centroids);
                let code = encode_code(
                    vector,
                    center,
                    &centroids[center],
                    &rotated_centroids[center],
                    total_bits,
                    metric,
                    &rotation,
                )?;
                Ok((*ordinal, code))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            metric,
            total_bits,
            requested_clusters,
            sample_count,
            rotation,
            centroids,
            codes,
        })
    }

    pub(super) fn prepare_query(&self, query: &[f32]) -> Option<PreparedRabitqQuery> {
        let query = metric_vector(query, self.metric);
        let rotated = self.rotation.rotate(&query)?;
        let center_terms = self
            .centroids
            .iter()
            .map(|centroid| match self.metric {
                MetricType::L2 => squared_l2(&query, centroid),
                MetricType::Ip | MetricType::Cosine => dot(&query, centroid),
                _ => f64::NAN,
            })
            .collect();
        Some(PreparedRabitqQuery {
            rotated,
            center_terms,
        })
    }

    pub(super) fn score(&self, query: &PreparedRabitqQuery, ordinal: u64) -> Option<f64> {
        let code = self.codes.get(ordinal)?;
        let center = usize::try_from(code.center).ok()?;
        let center_term = *query.center_terms.get(center)?;
        let quantized_dot = packed_dot(
            &code.packed,
            self.total_bits,
            self.rotation.padded_dimension(),
            &query.rotated,
        )?;
        let scale = code.scale;
        match self.metric {
            MetricType::L2 => Some(-(code.l2_add + center_term - 2.0 * scale * quantized_dot)),
            MetricType::Ip | MetricType::Cosine => Some(center_term + scale * quantized_dot),
            _ => None,
        }
    }

    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        total_bits: usize,
        requested_clusters: usize,
        sample_count: usize,
        metric: MetricType,
    ) -> bool {
        let expected_centroids = requested_clusters.min(vectors.len());
        self.metric == metric
            && self.total_bits == total_bits
            && self.requested_clusters == requested_clusters
            && self.sample_count == sample_count
            && self.rotation.validates(dimension)
            && self.centroids.len() == expected_centroids
            && self.centroids.iter().all(|centroid| {
                centroid.len() == dimension && centroid.iter().all(|value| value.is_finite())
            })
            && self.codes.validates(vectors.slot_count())
            && self.codes.keys().eq(vectors.keys())
            && self.codes.values().all(|code| {
                usize::try_from(code.center)
                    .ok()
                    .is_some_and(|center| center < self.centroids.len())
                    && code.packed.len()
                        == packed_bytes(self.rotation.padded_dimension(), total_bits)
                    && code.scale.is_finite()
                    && code.scale >= 0.0
                    && code.l2_add.is_finite()
            })
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        let centroids = self.centroids.iter().fold(0_usize, |total, centroid| {
            total.saturating_add(centroid.len().saturating_mul(std::mem::size_of::<f32>()))
        });
        self.codes.values().fold(
            centroids.saturating_add(RabitqRotation::estimated_payload_bytes()),
            |total, code| {
                total
                    .saturating_add(std::mem::size_of::<RabitqCode>())
                    .saturating_add(code.packed.len())
            },
        )
    }
}

fn validate_options(
    dimension: usize,
    total_bits: usize,
    requested_clusters: usize,
    metric: MetricType,
) -> Result<()> {
    if dimension == 0 || requested_clusters == 0 || !(1..=9).contains(&total_bits) {
        return Err(Error::invalid_argument(
            "RaBitQ requires a positive dimension and cluster count with total_bits in 1..=9",
        ));
    }
    if !matches!(metric, MetricType::L2 | MetricType::Ip | MetricType::Cosine) {
        return Err(Error::not_supported(
            "RaBitQ supports L2, inner-product, and cosine metrics",
        ));
    }
    Ok(())
}

fn training_sample(items: &[(u64, Vec<f32>)], requested: usize) -> Vec<(u64, Vec<f32>)> {
    if requested == 0 || requested >= items.len() {
        return items.to_vec();
    }
    if requested == 1 {
        return vec![items[0].clone()];
    }
    (0..requested)
        .map(|position| {
            let index = position.saturating_mul(items.len().saturating_sub(1))
                / requested.saturating_sub(1);
            items[index].clone()
        })
        .collect()
}

fn metric_vector(vector: &[f32], metric: MetricType) -> Vec<f32> {
    if metric != MetricType::Cosine {
        return vector.to_vec();
    }
    let norm = dot(vector, vector).sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return vec![0.0; vector.len()];
    }
    vector
        .iter()
        .map(|value| f64_to_f32(f64::from(*value) / norm))
        .collect()
}

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|left, right| {
            squared_l2(vector, left.1)
                .total_cmp(&squared_l2(vector, right.1))
                .then_with(|| left.0.cmp(&right.0))
        })
        .map_or(0, |(index, _)| index)
}

#[allow(clippy::too_many_arguments)]
fn encode_code(
    vector: &[f32],
    center: usize,
    centroid: &[f32],
    rotated_centroid: &[f64],
    total_bits: usize,
    metric: MetricType,
    rotation: &RabitqRotation,
) -> Result<RabitqCode> {
    let center = u32::try_from(center)
        .map_err(|_| Error::resource_exhausted("RaBitQ cluster count exceeds u32"))?;
    let residual: Vec<f32> = vector
        .iter()
        .zip(centroid)
        .map(|(value, center)| *value - *center)
        .collect();
    let rotated = rotation
        .rotate(&residual)
        .ok_or_else(|| Error::internal("RaBitQ residual rotation produced invalid values"))?;
    let norm_squared = dot_f64(&rotated, &rotated);
    if norm_squared <= f64::EPSILON {
        return Ok(RabitqCode {
            center,
            packed: vec![0; packed_bytes(rotation.padded_dimension(), total_bits)],
            scale: 0.0,
            l2_add: 0.0,
        });
    }
    let norm = norm_squared.sqrt();
    let ex_bits = total_bits - 1;
    let ex_levels = (1_u16 << ex_bits) - 1;
    let normalized_abs: Vec<f64> = rotated.iter().map(|value| value.abs() / norm).collect();
    let rescale = optimal_rescale(&normalized_abs, ex_levels);
    let sign_bit = 1_u16 << ex_bits;
    let values: Vec<u16> = rotated
        .iter()
        .zip(&normalized_abs)
        .map(|(value, magnitude)| {
            let extra = quantized_magnitude(*magnitude, rescale, ex_levels);
            if *value > 0.0 {
                sign_bit + extra
            } else {
                ex_levels - extra
            }
        })
        .collect();
    let alignment = centered_dot(&values, total_bits, &rotated);
    if !alignment.is_finite() || alignment <= 0.0 {
        return Err(Error::internal(
            "RaBitQ code is not aligned with its source residual",
        ));
    }
    let scale = norm_squared / alignment;
    let l2_add = if metric == MetricType::L2 {
        norm_squared + 2.0 * scale * centered_dot(&values, total_bits, rotated_centroid)
    } else {
        0.0
    };
    Ok(RabitqCode {
        center,
        packed: pack_values(&values, total_bits),
        scale,
        l2_add,
    })
}

fn optimal_rescale(magnitudes: &[f64], levels: u16) -> f64 {
    if levels == 0 {
        return 0.0;
    }
    let maximum = magnitudes.iter().copied().fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return 0.0;
    }
    let mut rescale = (f64::from(levels) + 0.5) / maximum;
    for _ in 0..6 {
        let (numerator, denominator) =
            magnitudes
                .iter()
                .fold((0.0_f64, 0.0_f64), |(numerator, denominator), magnitude| {
                    let level = f64::from(quantized_magnitude(*magnitude, rescale, levels)) + 0.5;
                    (numerator + level * *magnitude, denominator + level * level)
                });
        if numerator <= 0.0 || denominator <= 0.0 {
            break;
        }
        rescale = denominator / numerator;
    }
    rescale
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantized_magnitude(magnitude: f64, rescale: f64, levels: u16) -> u16 {
    ((rescale * magnitude + 1e-5).floor() as u16).min(levels)
}

fn centered_dot(values: &[u16], total_bits: usize, vector: &[f64]) -> f64 {
    let center = (f64::from((1_u16 << total_bits) - 1)) / 2.0;
    values
        .iter()
        .zip(vector)
        .map(|(value, coordinate)| (f64::from(*value) - center) * *coordinate)
        .sum()
}

fn packed_dot(packed: &[u8], total_bits: usize, dimension: usize, vector: &[f64]) -> Option<f64> {
    if packed.len() != packed_bytes(dimension, total_bits) || vector.len() != dimension {
        return None;
    }
    let center = (f64::from((1_u16 << total_bits) - 1)) / 2.0;
    let mask = (1_u32 << total_bits) - 1;
    let mut buffer = 0_u32;
    let mut buffered_bits = 0_usize;
    let mut byte = 0_usize;
    let mut result = 0.0;
    for coordinate in vector {
        while buffered_bits < total_bits {
            buffer |= u32::from(*packed.get(byte)?) << buffered_bits;
            byte = byte.saturating_add(1);
            buffered_bits = buffered_bits.saturating_add(8);
        }
        let value = buffer & mask;
        buffer >>= total_bits;
        buffered_bits -= total_bits;
        result += (f64::from(value) - center) * *coordinate;
    }
    Some(result)
}

fn pack_values(values: &[u16], total_bits: usize) -> Vec<u8> {
    let mut packed = vec![0_u8; packed_bytes(values.len(), total_bits)];
    for (index, value) in values.iter().copied().enumerate() {
        let bit_offset = index.saturating_mul(total_bits);
        for bit in 0..total_bits {
            if value & (1_u16 << bit) != 0 {
                let offset = bit_offset.saturating_add(bit);
                packed[offset / 8] |= 1_u8 << (offset % 8);
            }
        }
    }
    packed
}

#[cfg(test)]
fn unpack_value(packed: &[u8], index: usize, total_bits: usize) -> u16 {
    let bit_offset = index.saturating_mul(total_bits);
    (0..total_bits).fold(0_u16, |value, bit| {
        let offset = bit_offset.saturating_add(bit);
        value | (u16::from((packed[offset / 8] >> (offset % 8)) & 1) << bit)
    })
}

fn packed_bytes(dimension: usize, total_bits: usize) -> usize {
    dimension.saturating_mul(total_bits).saturating_add(7) / 8
}

fn dot(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::NAN;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return f64::NAN;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[allow(clippy::cast_possible_truncation)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests;
