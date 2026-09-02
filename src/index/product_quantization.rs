//! Deterministic product-quantizer training and asymmetric distance tables.

use super::ordinal_map::OrdinalMap;
use super::quantization::QuantizedVector;
use crate::error::{Error, Result};
use crate::types::MetricType;

const MAX_CENTROIDS: usize = 256;
const TRAINING_ITERATIONS: usize = 8;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct ProductCodebook {
    dimension: usize,
    chunk_offsets: Vec<usize>,
    centroids: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct ProductQuantizer {
    codebook: ProductCodebook,
    codes: OrdinalMap<Vec<u8>>,
}

#[derive(Debug)]
pub(super) struct AdcTable {
    distances: Vec<Vec<f64>>,
    similarities: Vec<Vec<f64>>,
    centroid_norms_squared: Vec<Vec<f64>>,
    query_norm_squared: f64,
    metric: MetricType,
}

impl ProductQuantizer {
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        chunk_count: usize,
    ) -> Result<Self> {
        validate_shape(dimension, chunk_count)?;
        let decoded: Vec<(u64, Vec<f32>)> = vectors
            .iter()
            .map(|(ordinal, vector)| (ordinal, vector.decode()))
            .collect();
        if decoded.iter().any(|(_, vector)| {
            vector.len() != dimension || vector.iter().any(|value| !value.is_finite())
        }) {
            return Err(Error::internal(
                "PQ training received an invalid base vector",
            ));
        }

        let chunk_offsets = balanced_offsets(dimension, chunk_count);
        let centroid_count = decoded.len().min(MAX_CENTROIDS);
        let centroids = chunk_offsets
            .windows(2)
            .map(|range| {
                let items: Vec<_> = decoded
                    .iter()
                    .map(|(ordinal, vector)| (*ordinal, vector[range[0]..range[1]].to_vec()))
                    .collect();
                train_chunk(&items, centroid_count)
            })
            .collect();
        let codebook = ProductCodebook {
            dimension,
            chunk_offsets,
            centroids,
        };
        let codes = decoded
            .iter()
            .map(|(ordinal, vector)| codebook.encode(vector).map(|code| (*ordinal, code)))
            .collect::<Result<_>>()?;
        Ok(Self { codebook, codes })
    }

    pub(super) fn codebook(&self) -> &ProductCodebook {
        &self.codebook
    }

    pub(super) fn code(&self, ordinal: u64) -> Option<&[u8]> {
        self.codes.get(ordinal).map(Vec::as_slice)
    }

    pub(super) fn table(&self, query: &[f32], metric: MetricType) -> Result<AdcTable> {
        self.codebook.table(query, metric)
    }

    pub(super) fn score(&self, table: &AdcTable, ordinal: u64) -> Option<f64> {
        self.code(ordinal)
            .and_then(|code| self.codebook.score(table, code))
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        let code_bytes = self
            .codes
            .values()
            .map(Vec::len)
            .sum::<usize>()
            .saturating_add(self.codes.slot_count());
        self.codebook
            .estimated_payload_bytes()
            .saturating_add(code_bytes)
    }

    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        chunk_count: usize,
    ) -> bool {
        if !self
            .codebook
            .validates(dimension, chunk_count, vectors.len())
            || !self.codes.validates(vectors.slot_count())
            || self.codes.keys().ne(vectors.keys())
        {
            return false;
        }
        vectors.iter().all(|(ordinal, vector)| {
            let decoded = vector.decode();
            self.codebook
                .encode(&decoded)
                .ok()
                .as_deref()
                .is_some_and(|expected| self.code(ordinal) == Some(expected))
        })
    }
}

impl ProductCodebook {
    pub(super) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(super) fn chunk_count(&self) -> usize {
        self.centroids.len()
    }

    pub(super) fn centroid_count(&self) -> usize {
        self.centroids.first().map_or(0, Vec::len)
    }

    pub(super) fn flattened_centroids(&self) -> impl Iterator<Item = f32> + '_ {
        self.centroids.iter().flatten().flatten().copied()
    }

    pub(super) fn table(&self, query: &[f32], metric: MetricType) -> Result<AdcTable> {
        if query.len() != self.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid_argument(
                "PQ query dimension or values are invalid",
            ));
        }
        let mut distances = Vec::with_capacity(self.centroids.len());
        let mut similarities = Vec::with_capacity(self.centroids.len());
        let mut centroid_norms_squared = Vec::with_capacity(self.centroids.len());
        for (range, centroids) in self.chunk_offsets.windows(2).zip(&self.centroids) {
            let query_chunk = &query[range[0]..range[1]];
            distances.push(
                centroids
                    .iter()
                    .map(|centroid| squared_l2(query_chunk, centroid))
                    .collect(),
            );
            similarities.push(
                centroids
                    .iter()
                    .map(|centroid| dot(query_chunk, centroid))
                    .collect(),
            );
            centroid_norms_squared.push(
                centroids
                    .iter()
                    .map(|centroid| dot(centroid, centroid))
                    .collect(),
            );
        }
        Ok(AdcTable {
            distances,
            similarities,
            centroid_norms_squared,
            query_norm_squared: dot(query, query),
            metric,
        })
    }

    pub(super) fn score(&self, table: &AdcTable, code: &[u8]) -> Option<f64> {
        if code.len() != self.chunk_count() {
            return None;
        }
        table.score(code)
    }
}

impl AdcTable {
    pub(super) fn score(&self, code: &[u8]) -> Option<f64> {
        if code.len() != self.distances.len() {
            return None;
        }
        match self.metric {
            MetricType::L2 => {
                let distance =
                    code.iter()
                        .enumerate()
                        .try_fold(0.0_f64, |total, (chunk, centroid)| {
                            self.distances
                                .get(chunk)?
                                .get(usize::from(*centroid))
                                .map(|distance| total + distance)
                        })?;
                Some(-distance)
            }
            MetricType::Cosine => {
                let (similarity, candidate_norm_squared) = code.iter().enumerate().try_fold(
                    (0.0_f64, 0.0_f64),
                    |(similarity, norm), (chunk, centroid)| {
                        Some((
                            similarity
                                + self.similarities.get(chunk)?.get(usize::from(*centroid))?,
                            norm + self
                                .centroid_norms_squared
                                .get(chunk)?
                                .get(usize::from(*centroid))?,
                        ))
                    },
                )?;
                if self.query_norm_squared == 0.0 || candidate_norm_squared == 0.0 {
                    Some(0.0)
                } else {
                    Some(
                        similarity
                            / (self.query_norm_squared.sqrt() * candidate_norm_squared.sqrt()),
                    )
                }
            }
            MetricType::Ip | MetricType::MipsL2 | MetricType::Undefined => code
                .iter()
                .enumerate()
                .try_fold(0.0_f64, |total, (chunk, centroid)| {
                    self.similarities
                        .get(chunk)?
                        .get(usize::from(*centroid))
                        .map(|similarity| total + similarity)
                }),
        }
    }
}

fn dot(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::NEG_INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

impl ProductCodebook {
    pub(super) fn validates(
        &self,
        dimension: usize,
        chunk_count: usize,
        vector_count: usize,
    ) -> bool {
        let centroid_count = vector_count.min(MAX_CENTROIDS);
        self.dimension == dimension
            && self.chunk_offsets == balanced_offsets(dimension, chunk_count)
            && self.centroids.len() == chunk_count
            && self.centroids.iter().all(|chunk| {
                chunk.len() == centroid_count
                    && chunk.iter().all(|centroid| {
                        !centroid.is_empty() && centroid.iter().all(|value| value.is_finite())
                    })
            })
            && self
                .chunk_offsets
                .windows(2)
                .zip(&self.centroids)
                .all(|(range, chunk)| {
                    chunk
                        .iter()
                        .all(|centroid| centroid.len() == range[1] - range[0])
                })
    }

    fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.dimension || vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::internal("PQ cannot encode an invalid vector"));
        }
        self.chunk_offsets
            .windows(2)
            .zip(&self.centroids)
            .map(|(range, centroids)| {
                let index = nearest_centroid(&vector[range[0]..range[1]], centroids)
                    .ok_or_else(|| Error::internal("PQ codebook has no centroid"))?;
                u8::try_from(index)
                    .map_err(|_| Error::resource_exhausted("PQ centroid index exceeds one byte"))
            })
            .collect()
    }

    fn estimated_payload_bytes(&self) -> usize {
        self.flattened_centroids()
            .count()
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(
                self.chunk_offsets
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }
}

fn validate_shape(dimension: usize, chunk_count: usize) -> Result<()> {
    if dimension == 0 || chunk_count == 0 || chunk_count > dimension {
        return Err(Error::invalid_argument(
            "PQ chunk count must be between one and the vector dimension",
        ));
    }
    Ok(())
}

fn balanced_offsets(dimension: usize, chunk_count: usize) -> Vec<usize> {
    (0..=chunk_count)
        .map(|chunk| chunk.saturating_mul(dimension) / chunk_count.max(1))
        .collect()
}

fn train_chunk(items: &[(u64, Vec<f32>)], centroid_count: usize) -> Vec<Vec<f32>> {
    if centroid_count == 0 {
        return Vec::new();
    }
    let mut centroids = diverse_seeds(items, centroid_count);
    for _ in 0..TRAINING_ITERATIONS {
        let assignments = assign(items, &centroids);
        centroids = recompute(items, &assignments, &centroids);
    }
    centroids
}

fn diverse_seeds(items: &[(u64, Vec<f32>)], count: usize) -> Vec<Vec<f32>> {
    let mut selected = vec![0_usize];
    while selected.len() < count {
        let next = items
            .iter()
            .enumerate()
            .filter(|(index, _)| !selected.contains(index))
            .max_by(
                |(left_index, (left_ordinal, left)), (right_index, (right_ordinal, right))| {
                    let left_distance = minimum_seed_distance(left, items, &selected);
                    let right_distance = minimum_seed_distance(right, items, &selected);
                    left_distance
                        .total_cmp(&right_distance)
                        .then_with(|| right_ordinal.cmp(left_ordinal))
                        .then_with(|| right_index.cmp(left_index))
                },
            )
            .map(|(index, _)| index);
        let Some(next) = next else {
            break;
        };
        selected.push(next);
    }
    selected
        .into_iter()
        .filter_map(|index| items.get(index).map(|(_, vector)| vector.clone()))
        .collect()
}

fn minimum_seed_distance(vector: &[f32], items: &[(u64, Vec<f32>)], selected: &[usize]) -> f64 {
    selected
        .iter()
        .filter_map(|index| items.get(*index))
        .map(|(_, seed)| squared_l2(vector, seed))
        .fold(f64::INFINITY, f64::min)
}

fn assign(items: &[(u64, Vec<f32>)], centroids: &[Vec<f32>]) -> Vec<usize> {
    items
        .iter()
        .map(|(_, vector)| nearest_centroid(vector, centroids).unwrap_or(0))
        .collect()
}

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> Option<usize> {
    centroids
        .iter()
        .enumerate()
        .min_by(|left, right| {
            squared_l2(vector, left.1)
                .total_cmp(&squared_l2(vector, right.1))
                .then_with(|| left.0.cmp(&right.0))
        })
        .map(|(index, _)| index)
}

fn recompute(
    items: &[(u64, Vec<f32>)],
    assignments: &[usize],
    previous: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let dimension = previous.first().map_or(0, Vec::len);
    let mut sums = vec![vec![0.0_f64; dimension]; previous.len()];
    let mut counts = vec![0_usize; previous.len()];
    for ((_, vector), centroid) in items.iter().zip(assignments) {
        let Some(count) = counts.get_mut(*centroid) else {
            continue;
        };
        *count = count.saturating_add(1);
        let Some(sum) = sums.get_mut(*centroid) else {
            continue;
        };
        for (total, value) in sum.iter_mut().zip(vector) {
            *total += f64::from(*value);
        }
    }
    sums.into_iter()
        .enumerate()
        .filter_map(|(index, sum)| {
            let previous = previous.get(index)?;
            let count = *counts.get(index)?;
            if count == 0 {
                return Some(previous.clone());
            }
            let divisor = usize_to_f64(count);
            Some(
                sum.into_iter()
                    .map(|value| f64_to_f32(value / divisor))
                    .collect(),
            )
        })
        .collect()
}

fn squared_l2(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = f64::from(*left) - f64::from(*right);
            delta * delta
        })
        .sum()
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
    use super::ProductQuantizer;
    use crate::index::ordinal_map::OrdinalMap;
    use crate::index::quantization::QuantizedVector;
    use crate::types::QuantizeType;

    fn vectors() -> OrdinalMap<QuantizedVector> {
        (0_u16..32)
            .map(|value| {
                let group = f32::from(value / 8) * 100.0;
                (
                    u64::from(value),
                    QuantizedVector::encode(
                        vec![group, f32::from(value % 8), group + 1.0, 1.0, group + 2.0],
                        QuantizeType::Undefined,
                    )
                    .expect("fixture vector must encode"),
                )
            })
            .collect()
    }

    #[test]
    fn training_codes_and_adc_tables_are_deterministic() {
        let vectors = vectors();
        let first = ProductQuantizer::build(&vectors, 5, 3).expect("PQ must train");
        let second = ProductQuantizer::build(&vectors, 5, 3).expect("PQ must retrain");
        assert_eq!(first, second);
        assert_eq!(first.codebook.chunk_offsets, vec![0, 1, 3, 5]);
        assert_eq!(first.codebook.centroid_count(), 32);
        assert!(first.validates(&vectors, 5, 3));

        let query = vectors
            .get(17)
            .expect("fixture ordinal must exist")
            .decode();
        let table = first
            .table(&query, crate::types::MetricType::L2)
            .expect("ADC table must build");
        let exact = first.score(&table, 17).expect("fixture code must score");
        let distant = first.score(&table, 1).expect("fixture code must score");
        assert!(exact > distant);
    }

    #[test]
    fn adc_tables_support_similarity_metrics_with_finite_scores() {
        let vectors = vectors();
        let quantizer = ProductQuantizer::build(&vectors, 5, 3).expect("PQ must train");
        let query = vectors
            .get(17)
            .expect("fixture ordinal must exist")
            .decode();
        for metric in [
            crate::types::MetricType::Ip,
            crate::types::MetricType::Cosine,
            crate::types::MetricType::MipsL2,
        ] {
            let table = quantizer
                .table(&query, metric)
                .expect("similarity ADC table must build");
            let score = quantizer
                .score(&table, 17)
                .expect("similarity ADC score must exist");
            assert!(score.is_finite(), "metric={metric:?} score={score}");
        }
    }

    #[test]
    fn empty_training_keeps_a_valid_generation_shape() {
        let vectors = OrdinalMap::default();
        let quantizer = ProductQuantizer::build(&vectors, 7, 4).expect("empty PQ must build");
        assert_eq!(quantizer.codebook.centroid_count(), 0);
        assert!(quantizer.validates(&vectors, 7, 4));
        let table = quantizer
            .table(&[0.0; 7], crate::types::MetricType::L2)
            .expect("empty ADC table must build");
        assert!(quantizer.codebook.score(&table, &[0; 4]).is_none());
    }

    #[test]
    fn invalid_chunk_shapes_fail_explicitly() {
        let vectors = vectors();
        assert!(ProductQuantizer::build(&vectors, 5, 0).is_err());
        assert!(ProductQuantizer::build(&vectors, 5, 6).is_err());
    }
}
