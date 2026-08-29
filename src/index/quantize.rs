//! Scalar vector codecs used by compact indexes and snapshots.

use crate::error::{Error, Result};

pub fn quantize_int8(vector: &[f32]) -> Result<(Vec<i8>, f32)> {
    if vector.is_empty() || !vector.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid_argument(
            "INT8 quantization requires finite values",
        ));
    }
    let max = vector.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    Ok((
        vector
            .iter()
            .map(|v| (*v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect(),
        scale,
    ))
}

pub fn dequantize_int8(codes: &[i8], scale: f32) -> Vec<f32> {
    codes.iter().map(|v| *v as f32 * scale).collect()
}

pub fn quantize_int4(vector: &[f32]) -> Result<(Vec<u8>, f32)> {
    if vector.is_empty() || !vector.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid_argument(
            "INT4 quantization requires finite values",
        ));
    }
    let max = vector.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = if max == 0.0 { 1.0 } else { max / 7.0 };
    let mut out = Vec::with_capacity(vector.len().div_ceil(2));
    for chunk in vector.chunks(2) {
        let lo = ((chunk[0] / scale).round().clamp(-8.0, 7.0) as i8 & 0x0f) as u8;
        let hi = chunk.get(1).map_or(0, |v| {
            ((*v / scale).round().clamp(-8.0, 7.0) as i8 & 0x0f) as u8
        });
        out.push(lo | (hi << 4));
    }
    Ok((out, scale))
}

pub fn dequantize_int4(codes: &[u8], scale: f32, dimension: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dimension);
    for byte in codes {
        let lo = (byte & 0x0f) as i8;
        out.push((if lo >= 8 { lo - 16 } else { lo }) as f32 * scale);
        if out.len() < dimension {
            let hi = ((byte >> 4) & 0x0f) as i8;
            out.push((if hi >= 8 { hi - 16 } else { hi }) as f32 * scale);
        }
    }
    out
}
