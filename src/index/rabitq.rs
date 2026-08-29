//! Portable RaBitQ-style binary refinement helpers.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct RaBitQCode {
    pub bits: Vec<u8>,
    pub dimension: usize,
    pub scale: f32,
    pub offset: f32,
}

pub fn encode(vector: &[f32], bits_per_dimension: u8) -> Result<RaBitQCode> {
    if vector.is_empty() || !matches!(bits_per_dimension, 1 | 2 | 4 | 8) {
        return Err(Error::invalid_argument(
            "RaBitQ requires a non-empty vector and 1, 2, 4, or 8 bits",
        ));
    }
    let max = vector.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = if max == 0.0 {
        1.0
    } else {
        max / ((1_u32 << bits_per_dimension.min(8)) - 1) as f32
    };
    let bits = vector
        .iter()
        .map(|v| ((*v / scale).round().clamp(0.0, 255.0)) as u8)
        .collect();
    Ok(RaBitQCode {
        bits,
        dimension: vector.len(),
        scale,
        offset: 0.0,
    })
}

pub fn decode(code: &RaBitQCode) -> Vec<f32> {
    code.bits
        .iter()
        .map(|v| *v as f32 * code.scale + code.offset)
        .collect()
}
