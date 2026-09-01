//! Deterministic portable random rotation for `RaBitQ`.

use crate::error::{Error, Result};

const ROTATION_ROUNDS: u64 = 4;
const ROTATION_SEED: u64 = 0xa3_5e_c9_71_4b_17_d2_0f;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct RabitqRotation {
    dimension: usize,
    padded_dimension: usize,
    seed: u64,
}

impl RabitqRotation {
    pub(super) fn new(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::invalid_argument(
                "RaBitQ vector dimension must be positive",
            ));
        }
        let padded_dimension = dimension.checked_next_power_of_two().ok_or_else(|| {
            Error::resource_exhausted("RaBitQ padded dimension exceeds this platform")
        })?;
        Ok(Self {
            dimension,
            padded_dimension,
            seed: ROTATION_SEED ^ u64::try_from(dimension).unwrap_or(u64::MAX),
        })
    }

    pub(super) fn padded_dimension(&self) -> usize {
        self.padded_dimension
    }

    pub(super) fn rotate(&self, input: &[f32]) -> Option<Vec<f64>> {
        if input.len() != self.dimension || input.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut output = vec![0.0; self.padded_dimension];
        for (output, input) in output.iter_mut().zip(input) {
            *output = f64::from(*input);
        }
        for round in 0..ROTATION_ROUNDS {
            for (index, value) in output.iter_mut().enumerate() {
                if random_sign(self.seed, round, index) < 0.0 {
                    *value = -*value;
                }
            }
            normalized_hadamard(&mut output);
        }
        Some(output)
    }

    pub(super) fn validates(&self, dimension: usize) -> bool {
        Self::new(dimension).ok().as_ref() == Some(self)
    }

    pub(super) fn estimated_payload_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

fn random_sign(seed: u64, round: u64, index: usize) -> f64 {
    let index = u64::try_from(index).unwrap_or(u64::MAX);
    let value = splitmix64(
        seed ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    );
    if value & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[allow(clippy::cast_precision_loss)]
fn normalized_hadamard(values: &mut [f64]) {
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(half.saturating_mul(2)) {
            let (left, right) = block.split_at_mut(half);
            for (left, right) in left.iter_mut().zip(right) {
                let sum = *left + *right;
                let difference = *left - *right;
                *left = sum;
                *right = difference;
            }
        }
        half = half.saturating_mul(2);
    }
    let scale = 1.0 / (values.len() as f64).sqrt();
    for value in values {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::RabitqRotation;

    fn dot_f32(left: &[f32], right: &[f32]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum()
    }

    fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }

    #[test]
    fn deterministic_rotation_preserves_inner_products_with_padding() {
        let rotation = RabitqRotation::new(5).expect("rotation must build");
        let left = [1.0, -2.0, 3.0, 0.5, -0.25];
        let right = [-0.5, 1.0, 2.0, -3.0, 0.75];
        let rotated_left = rotation.rotate(&left).expect("left vector must rotate");
        let rotated_right = rotation.rotate(&right).expect("right vector must rotate");
        assert_eq!(rotation.padded_dimension(), 8);
        assert_eq!(rotated_left, rotation.rotate(&left).unwrap());
        assert!((dot_f32(&left, &right) - dot_f64(&rotated_left, &rotated_right)).abs() < 1e-10);
    }
}
