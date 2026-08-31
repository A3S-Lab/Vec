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
        let mut centroids = diverse_seeds(&decoded, count);
        let iterations = if n_iters == 0 { 3 } else { n_iters };
        for _ in 0..iterations {
            let assignments = assign(&decoded, &centroids);
            centroids = recompute(&decoded, &assignments, &centroids);
        }
        let assignments = assign(&decoded, &centroids);
        let mut postings = vec![RoaringTreemap::new(); centroids.len()];
        for ((ordinal, _), centroid) in decoded.iter().zip(assignments) {
            postings[centroid].insert(*ordinal);
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
        let probes = self.ranked_probes(query);
        let mut candidates_by_rank = Vec::with_capacity(probes.len());
        for centroid in probes {
            let mut candidates = &self.postings[centroid] & allowed;
            candidates -= excluded;
            candidates_by_rank.push(candidates);
        }

        let mut probe_count = self.probe_count(requested_nprobe);
        let mut candidate_count = candidates_by_rank
            .iter()
            .take(probe_count)
            .map(RoaringTreemap::len)
            .sum::<u64>();
        let minimum_candidates = u64::try_from(minimum_candidates).unwrap_or(u64::MAX);
        while candidate_count < minimum_candidates && probe_count < candidates_by_rank.len() {
            candidate_count = candidate_count.saturating_add(candidates_by_rank[probe_count].len());
            probe_count += 1;
        }
        let mut candidates = RoaringTreemap::new();
        for bitmap in candidates_by_rank.into_iter().take(probe_count) {
            candidates |= bitmap;
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
        let mut assigned = RoaringTreemap::new();
        for posting in &self.postings {
            if posting.intersection_len(&assigned) != 0 {
                return false;
            }
            assigned |= posting;
        }
        assigned.iter().eq(vectors.keys())
    }
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

fn squared_l2(left: &[f32], right: &[f32]) -> f64 {
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
    use super::IvfIndex;
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
        let first = IvfIndex::build(&vectors, 7, 5);
        let second = IvfIndex::build(&vectors, 7, 5);
        assert_eq!(first.centroids, second.centroids);
        assert_eq!(first.postings, second.postings);
        assert_eq!(first.centroid_count(), 7);
        let candidates = first.candidates(&[20.0, 0.0], Some(7));
        assert_eq!(candidates.len(), vectors.len() as u64);
    }

    #[test]
    fn filtered_probe_window_expands_until_topk_is_available() {
        let vectors = vectors();
        let index = IvfIndex::build(&vectors, 7, 5);
        let allowed: RoaringTreemap = (35_u64..50).collect();
        let candidates =
            index.filtered_candidates(&[0.0, 0.0], Some(1), 5, &allowed, &RoaringTreemap::new());
        assert!(candidates.len() >= 5);
        assert!(candidates.iter().all(|ordinal| allowed.contains(ordinal)));
    }
}
