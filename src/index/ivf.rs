//! Deterministic IVF training, ordinal posting lists, and centroid probing.

use super::ordinal_map::OrdinalMap;
use super::quantization::QuantizedVector;
use roaring::RoaringTreemap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct IvfIndex {
    centroids: Vec<Vec<f32>>,
    postings: Vec<RoaringTreemap>,
}

impl IvfIndex {
    pub(super) fn build(
        vectors: &OrdinalMap<QuantizedVector>,
        n_list: usize,
        n_iters: usize,
        use_soar: bool,
    ) -> Self {
        if vectors.is_empty() {
            return Self {
                centroids: Vec::new(),
                postings: Vec::new(),
            };
        }
        let decoded: Vec<(u64, Vec<f32>)> = vectors
            .iter()
            .map(|(ordinal, vector)| (ordinal, vector.decode()))
            .collect();
        let count = n_list.min(decoded.len()).max(1);
        let iterations = if n_iters == 0 { 3 } else { n_iters };
        let centroids = train_centroids(&decoded, count, iterations);
        let assignments = assign(&decoded, &centroids);
        let mut postings = vec![RoaringTreemap::new(); centroids.len()];
        for ((ordinal, vector), primary) in decoded.iter().zip(assignments) {
            postings[primary].insert(*ordinal);
            if let Some(secondary) = use_soar
                .then(|| soar_secondary(vector, primary, &centroids))
                .flatten()
            {
                postings[secondary].insert(*ordinal);
            }
        }
        Self {
            centroids,
            postings,
        }
    }

    pub(super) fn candidates(
        &self,
        query: &[f32],
        requested_nprobe: Option<usize>,
    ) -> RoaringTreemap {
        if self.centroids.is_empty() {
            return RoaringTreemap::new();
        }
        let nprobe = self.probe_count(requested_nprobe);
        let mut candidates = RoaringTreemap::new();
        for centroid in self.ranked_probes(query).into_iter().take(nprobe) {
            candidates |= &self.postings[centroid];
        }
        candidates
    }

    /// Intersects posting bitmaps before unioning them, then extends the probe
    /// window until enough eligible candidates are available for top-k.
    pub(super) fn filtered_candidates(
        &self,
        query: &[f32],
        requested_nprobe: Option<usize>,
        minimum_candidates: usize,
        allowed: &RoaringTreemap,
        excluded: &RoaringTreemap,
    ) -> RoaringTreemap {
        if self.centroids.is_empty() || allowed.is_empty() {
            return RoaringTreemap::new();
        }
        let initial_probe_count = self.probe_count(requested_nprobe);
        let minimum_candidates = u64::try_from(minimum_candidates).unwrap_or(u64::MAX);
        let mut candidates = RoaringTreemap::new();
        for (rank, centroid) in self.ranked_probes(query).into_iter().enumerate() {
            if rank >= initial_probe_count && candidates.len() >= minimum_candidates {
                break;
            }
            let mut posting_candidates = &self.postings[centroid] & allowed;
            posting_candidates -= excluded;
            candidates |= posting_candidates;
        }
        candidates
    }

    fn probe_count(&self, requested_nprobe: Option<usize>) -> usize {
        let default_nprobe = integer_sqrt_ceil(self.centroids.len());
        requested_nprobe
            .unwrap_or(default_nprobe)
            .max(1)
            .min(self.centroids.len())
    }

    fn ranked_probes(&self, query: &[f32]) -> Vec<usize> {
        let mut probes: Vec<(usize, f64)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(index, centroid)| (index, squared_l2(query, centroid)))
            .collect();
        probes.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        probes.into_iter().map(|(index, _)| index).collect()
    }

    #[cfg(test)]
    pub(super) fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    pub(super) fn estimated_payload_bytes(&self) -> usize {
        let centroids = self
            .centroids
            .iter()
            .map(|centroid| centroid.len().saturating_mul(std::mem::size_of::<f32>()))
            .sum::<usize>();
        let postings = self
            .postings
            .iter()
            .map(RoaringTreemap::serialized_size)
            .sum::<usize>();
        centroids.saturating_add(postings)
    }

    pub(super) fn validates(
        &self,
        vectors: &OrdinalMap<QuantizedVector>,
        dimension: usize,
        n_list: usize,
        use_soar: bool,
    ) -> bool {
        if vectors.is_empty() {
            return self.centroids.is_empty() && self.postings.is_empty();
        }
        let expected_centroids = n_list.min(vectors.len()).max(1);
        if self.centroids.len() != expected_centroids
            || self.postings.len() != expected_centroids
            || self.centroids.iter().any(|centroid| {
                centroid.len() != dimension || centroid.iter().any(|value| !value.is_finite())
            })
        {
            return false;
        }
        let expected_assignments = u8::from(use_soar && expected_centroids > 1) + 1;
        let mut assignments = vec![0_u8; vectors.slot_count()];
        for posting in &self.postings {
            for ordinal in posting {
                let Ok(index) = usize::try_from(ordinal) else {
                    return false;
                };
                let Some(count) = assignments.get_mut(index) else {
                    return false;
                };
                if !vectors.contains_key(ordinal) {
                    return false;
                }
                *count = count.saturating_add(1);
                if *count > expected_assignments {
                    return false;
                }
            }
        }
        vectors.keys().all(|ordinal| {
            usize::try_from(ordinal)
                .ok()
                .and_then(|index| assignments.get(index))
                == Some(&expected_assignments)
        })
    }
}

pub(super) fn train_centroids(
    items: &[(u64, Vec<f32>)],
    count: usize,
    iterations: usize,
) -> Vec<Vec<f32>> {
    if items.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut centroids = diverse_seeds(items, count.min(items.len()));
    for _ in 0..iterations {
        let assignments = assign(items, &centroids);
        centroids = recompute(items, &assignments, &centroids);
    }
    centroids
}

fn diverse_seeds(items: &[(u64, Vec<f32>)], count: usize) -> Vec<Vec<f32>> {
    let mut selected = vec![0];
    while selected.len() < count {
        let next = items
            .iter()
            .enumerate()
            .filter(|(index, _)| !selected.contains(index))
            .max_by(
                |(left_index, (left_ordinal, left)), (right_index, (right_ordinal, right))| {
                    let left_distance = selected
                        .iter()
                        .map(|seed| squared_l2(left, &items[*seed].1))
                        .fold(f64::INFINITY, f64::min);
                    let right_distance = selected
                        .iter()
                        .map(|seed| squared_l2(right, &items[*seed].1))
                        .fold(f64::INFINITY, f64::min);
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
        .map(|index| items[index].1.clone())
        .collect()
}

fn assign(items: &[(u64, Vec<f32>)], centroids: &[Vec<f32>]) -> Vec<usize> {
    items
        .iter()
        .map(|(_, vector)| {
            centroids
                .iter()
                .enumerate()
                .min_by(|left, right| {
                    squared_l2(vector, left.1)
                        .total_cmp(&squared_l2(vector, right.1))
                        .then_with(|| left.0.cmp(&right.0))
                })
                .map_or(0, |(index, _)| index)
        })
        .collect()
}

fn soar_secondary(vector: &[f32], primary: usize, centroids: &[Vec<f32>]) -> Option<usize> {
    let primary_centroid = centroids.get(primary)?;
    centroids
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != primary)
        .min_by(|left, right| {
            soar_loss(vector, primary_centroid, left.1)
                .total_cmp(&soar_loss(vector, primary_centroid, right.1))
                .then_with(|| left.0.cmp(&right.0))
        })
        .map(|(index, _)| index)
}

/// SOAR's secondary-assignment objective with the paper's lambda of one:
/// squared secondary residual plus its squared projection onto the primary
/// residual. A zero primary residual has no defined direction and therefore no
/// projection penalty.
fn soar_loss(vector: &[f32], primary: &[f32], secondary: &[f32]) -> f64 {
    if vector.len() != primary.len() || vector.len() != secondary.len() {
        return f64::INFINITY;
    }
    let (mut primary_norm, mut secondary_norm, mut residual_dot) = (0.0, 0.0, 0.0);
    for ((value, primary), secondary) in vector.iter().zip(primary).zip(secondary) {
        let primary_residual = f64::from(*value) - f64::from(*primary);
        let secondary_residual = f64::from(*value) - f64::from(*secondary);
        primary_norm += primary_residual * primary_residual;
        secondary_norm += secondary_residual * secondary_residual;
        residual_dot += primary_residual * secondary_residual;
    }
    let projection_penalty = if primary_norm > 0.0 {
        residual_dot * residual_dot / primary_norm
    } else {
        0.0
    };
    secondary_norm + projection_penalty
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
        counts[*centroid] += 1;
        for (sum, value) in sums[*centroid].iter_mut().zip(vector) {
            *sum += f64::from(*value);
        }
    }
    sums.into_iter()
        .enumerate()
        .map(|(index, sum)| {
            if counts[index] == 0 {
                return previous[index].clone();
            }
            let divisor = count_to_f64(counts[index]);
            sum.into_iter()
                .map(|value| f64_to_f32(value / divisor))
                .collect()
        })
        .collect()
}

pub(super) fn squared_l2(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = f64::from(*left) - f64::from(*right);
            difference * difference
        })
        .sum()
}

fn integer_sqrt_ceil(value: usize) -> usize {
    let mut root = 1_usize;
    while root.saturating_mul(root) < value {
        root = root.saturating_add(1);
    }
    root
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(super) fn scaled_candidate_limit(topk: usize, scale_factor: f32, available: usize) -> usize {
    (((topk as f64) * f64::from(scale_factor)).ceil() as usize)
        .max(topk)
        .min(available)
}

#[allow(clippy::cast_precision_loss)]
fn count_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
    use super::{assign, soar_loss, IvfIndex};
    use crate::index::ordinal_map::OrdinalMap;
    use crate::index::quantization::QuantizedVector;
    use crate::types::QuantizeType;
    use roaring::RoaringTreemap;

    fn vectors() -> OrdinalMap<QuantizedVector> {
        (0_u64..50)
            .map(|ordinal| {
                let coordinate = f32::from(u16::try_from(ordinal).expect("fixture fits u16"));
                (
                    ordinal,
                    QuantizedVector::encode(
                        vec![
                            coordinate,
                            f32::from(u8::try_from(ordinal % 5).expect("remainder fits u8")),
                        ],
                        QuantizeType::Undefined,
                    )
                    .expect("encoding must succeed"),
                )
            })
            .collect()
    }

    #[test]
    fn training_is_deterministic_and_exhaustive_probes_return_every_ordinal() {
        let vectors = vectors();
        let first = IvfIndex::build(&vectors, 7, 5, false);
        let second = IvfIndex::build(&vectors, 7, 5, false);
        assert_eq!(first.centroids, second.centroids);
        assert_eq!(first.postings, second.postings);
        assert_eq!(first.centroid_count(), 7);
        let candidates = first.candidates(&[20.0, 0.0], Some(7));
        assert_eq!(candidates.len(), vectors.len() as u64);
    }

    #[test]
    fn filtered_probe_window_expands_until_topk_is_available() {
        let vectors = vectors();
        let index = IvfIndex::build(&vectors, 7, 5, false);
        let allowed: RoaringTreemap = (35_u64..50).collect();
        let candidates =
            index.filtered_candidates(&[0.0, 0.0], Some(1), 5, &allowed, &RoaringTreemap::new());
        assert!(candidates.len() >= 5);
        assert!(candidates.iter().all(|ordinal| allowed.contains(ordinal)));
    }

    #[test]
    fn soar_assignments_are_deterministic_exhaustive_and_minimize_the_objective() {
        let vectors = vectors();
        let first = IvfIndex::build(&vectors, 7, 5, true);
        let second = IvfIndex::build(&vectors, 7, 5, true);
        assert_eq!(first.centroids, second.centroids);
        assert_eq!(first.postings, second.postings);
        assert!(first.validates(&vectors, 2, 7, true));
        assert!(!first.validates(&vectors, 2, 7, false));

        let decoded: Vec<_> = vectors
            .iter()
            .map(|(ordinal, vector)| (ordinal, vector.decode()))
            .collect();
        let primary_assignments = assign(&decoded, &first.centroids);
        for ((ordinal, vector), primary) in decoded.iter().zip(primary_assignments) {
            let assigned: Vec<_> = first
                .postings
                .iter()
                .enumerate()
                .filter_map(|(index, posting)| posting.contains(*ordinal).then_some(index))
                .collect();
            assert_eq!(assigned.len(), 2);
            assert!(assigned.contains(&primary));
            let secondary = assigned
                .into_iter()
                .find(|index| *index != primary)
                .expect("SOAR must select a distinct secondary centroid");
            let expected = first
                .centroids
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != primary)
                .min_by(|left, right| {
                    soar_loss(vector, &first.centroids[primary], left.1)
                        .total_cmp(&soar_loss(vector, &first.centroids[primary], right.1))
                        .then_with(|| left.0.cmp(&right.0))
                })
                .map(|(index, _)| index)
                .expect("fixture has multiple centroids");
            assert_eq!(secondary, expected);
        }

        assert_eq!(
            first.candidates(&[20.0, 0.0], Some(7)).len(),
            vectors.len() as u64
        );
    }

    #[test]
    fn filtered_probe_expansion_counts_unique_soar_candidates() {
        let index = IvfIndex {
            centroids: vec![vec![0.0], vec![1.0], vec![2.0]],
            postings: vec![
                [0_u64, 1].into_iter().collect(),
                [0_u64, 1].into_iter().collect(),
                [2_u64, 3, 4].into_iter().collect(),
            ],
        };
        let allowed: RoaringTreemap = (0_u64..5).collect();
        let candidates =
            index.filtered_candidates(&[0.0], Some(1), 4, &allowed, &RoaringTreemap::new());
        assert_eq!(candidates.len(), 5);
    }
}
