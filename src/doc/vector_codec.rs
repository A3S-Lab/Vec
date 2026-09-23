use super::VectorValue;
use crate::error::{Error, Result};
use std::collections::BTreeSet;

pub(super) fn validate_vector(value: &VectorValue) -> Result<()> {
    match value {
        VectorValue::Binary32(values) => validate_binary(values, 4, "Binary32"),
        VectorValue::Binary64(values) => validate_binary(values, 8, "Binary64"),
        VectorValue::Fp16(values) => {
            validate_nonempty(values)?;
            if values.iter().any(|bits| !fp16_is_finite(*bits)) {
                return Err(Error::invalid_argument("FP16 vector values must be finite"));
            }
            Ok(())
        }
        VectorValue::Fp32(values) => {
            validate_nonempty(values)?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::invalid_argument("vector values must be finite"));
            }
            Ok(())
        }
        VectorValue::Fp64(values) => {
            validate_nonempty(values)?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::invalid_argument("vector values must be finite"));
            }
            Ok(())
        }
        VectorValue::Int4(values) => {
            validate_nonempty(values)?;
            if values.iter().any(|value| !(-8..=7).contains(value)) {
                return Err(Error::invalid_argument(
                    "signed INT4 vector values must be in -8..=7",
                ));
            }
            Ok(())
        }
        VectorValue::Int8(values) => validate_nonempty(values),
        VectorValue::Int16(values) => validate_nonempty(values),
        VectorValue::SparseFp16 { indices, values } => {
            validate_sparse_indices(indices, values.len())?;
            if values.iter().any(|bits| !fp16_is_finite(*bits)) {
                return Err(Error::invalid_argument(
                    "sparse FP16 vector values must be finite",
                ));
            }
            Ok(())
        }
        VectorValue::SparseFp32 { indices, values } => {
            validate_sparse_indices(indices, values.len())?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::invalid_argument(
                    "sparse vector values must be finite",
                ));
            }
            Ok(())
        }
    }
}

fn validate_nonempty<T>(values: &[T]) -> Result<()> {
    if values.is_empty() {
        Err(Error::invalid_argument("vector values must not be empty"))
    } else {
        Ok(())
    }
}

fn validate_binary(values: &[u8], bytes_per_chunk: usize, name: &str) -> Result<()> {
    validate_nonempty(values)?;
    if values.len() % bytes_per_chunk != 0 {
        return Err(Error::invalid_argument(format!(
            "{name} payload must contain complete {}-bit chunks",
            bytes_per_chunk * 8
        )));
    }
    Ok(())
}

fn validate_sparse_indices(indices: &[u32], value_count: usize) -> Result<()> {
    if indices.is_empty() || indices.len() != value_count {
        return Err(Error::invalid_argument(
            "sparse vector indices and values must have equal non-zero length",
        ));
    }
    let mut seen = BTreeSet::new();
    if indices.iter().any(|index| !seen.insert(*index)) {
        return Err(Error::invalid_argument(
            "sparse vector contains a duplicate index",
        ));
    }
    Ok(())
}

pub(super) fn encode_fp16(values: &[f32]) -> Result<Vec<u16>> {
    if values.is_empty() {
        return Err(Error::invalid_argument("vector values must not be empty"));
    }
    values.iter().copied().map(f32_to_fp16).collect()
}

pub(crate) fn f32_to_fp16(value: f32) -> Result<u16> {
    if !value.is_finite() {
        return Err(Error::invalid_argument("FP16 source values must be finite"));
    }
    if value.abs() > 65_504.0 {
        return Err(Error::invalid_argument(
            "FP16 source value exceeds the finite range",
        ));
    }

    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent_bits = u8::try_from((bits >> 23) & 0xff)
        .map_err(|_| Error::internal("convert the FP32 exponent to FP16"))?;
    let exponent = i32::from(exponent_bits) - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return u16::try_from(sign)
                .map_err(|_| Error::internal("convert the FP16 signed zero"));
        }
        let significand = mantissa | 0x0080_0000;
        let shift = u32::try_from(14 - exponent)
            .map_err(|_| Error::internal("convert the FP16 subnormal shift"))?;
        let rounded = round_shift_right(significand, shift);
        return u16::try_from(sign | rounded)
            .map_err(|_| Error::internal("encode the FP16 subnormal value"));
    }

    let rounded_mantissa = round_shift_right(mantissa, 13);
    let exponent_bits =
        u32::try_from(exponent).map_err(|_| Error::internal("convert the FP16 normal exponent"))?;
    let mut encoded = (exponent_bits << 10) | (rounded_mantissa & 0x03ff);
    if rounded_mantissa == 0x0400 {
        encoded = u32::try_from(exponent + 1)
            .map_err(|_| Error::internal("carry the rounded FP16 exponent"))?
            << 10;
    }
    if encoded >= 0x7c00 {
        return Err(Error::invalid_argument(
            "FP16 source value rounds outside the finite range",
        ));
    }
    u16::try_from(sign | encoded).map_err(|_| Error::internal("encode the FP16 normal value"))
}

fn round_shift_right(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let mask = (1_u32 << shift) - 1;
    let remainder = value & mask;
    let halfway = 1_u32 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1))
}

pub(super) fn fp16_is_finite(bits: u16) -> bool {
    bits & 0x7c00 != 0x7c00
}

pub(crate) fn fp16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x03ff);
    let raw = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut fraction = frac;
            let mut exponent: u32 = 127 - 14;
            while fraction & 0x400 == 0 {
                fraction <<= 1;
                exponent = exponent.saturating_sub(1);
            }
            sign | (exponent << 23) | ((fraction & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((u32::from(exp) + 112) << 23) | (frac << 13)
    };
    f32::from_bits(raw)
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn f64_to_f32(value: f64) -> Option<f32> {
    (value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX))
        .then_some(value as f32)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{
        encode_fp16, f32_to_fp16, f64_to_f32, fp16_is_finite, fp16_to_f32, validate_vector,
    };
    use crate::doc::VectorValue;

    #[test]
    fn every_finite_fp16_bit_pattern_round_trips() {
        for bits in u16::MIN..=u16::MAX {
            if !fp16_is_finite(bits) {
                continue;
            }
            let value = fp16_to_f32(bits);
            assert!(value.is_finite(), "bits={bits:#06x}");
            let encoded = f32_to_fp16(value).expect("decoded FP16 value must encode");
            assert_eq!(encoded, bits, "value={value}, bits={bits:#06x}");
        }
    }

    #[test]
    fn validate_vector_rejects_empty_nonfinite_and_sparse_shape_errors() {
        assert!(validate_vector(&VectorValue::Fp32(vec![])).is_err());
        assert!(validate_vector(&VectorValue::Fp32(vec![f32::NAN])).is_err());
        assert!(validate_vector(&VectorValue::Fp64(vec![f64::INFINITY])).is_err());
        assert!(validate_vector(&VectorValue::Fp16(vec![0x7c00])).is_err());
        assert!(validate_vector(&VectorValue::Int4(vec![8])).is_err());
        assert!(validate_vector(&VectorValue::Binary32(vec![1, 2])).is_err());
        assert!(validate_vector(&VectorValue::Binary64(vec![1; 4])).is_err());
        assert!(validate_vector(&VectorValue::SparseFp32 {
            indices: vec![],
            values: vec![],
        })
        .is_err());
        assert!(validate_vector(&VectorValue::SparseFp32 {
            indices: vec![0, 0],
            values: vec![1.0, 2.0],
        })
        .is_err());
        assert!(validate_vector(&VectorValue::SparseFp16 {
            indices: vec![0],
            values: vec![0x7c00],
        })
        .is_err());
        assert!(validate_vector(&VectorValue::SparseFp32 {
            indices: vec![0],
            values: vec![f32::NAN],
        })
        .is_err());
        assert!(validate_vector(&VectorValue::Fp32(vec![1.0])).is_ok());
        assert!(validate_vector(&VectorValue::Binary32(vec![0; 4])).is_ok());
    }

    #[test]
    fn fp16_encode_and_f64_promotion_cover_range_guards() {
        assert!(encode_fp16(&[]).is_err());
        assert!(encode_fp16(&[f32::NAN]).is_err());
        assert!(encode_fp16(&[70_000.0]).is_err());
        assert!(encode_fp16(&[1.0]).is_ok());
        assert_eq!(f64_to_f32(1.5), Some(1.5));
        assert_eq!(f64_to_f32(f64::NAN), None);
        assert_eq!(f64_to_f32(f64::INFINITY), None);
        assert_eq!(f64_to_f32(f64::from(f32::MAX) * 2.0), None);
        // Subnormal / signed-zero FP16 decode arms.
        assert_eq!(fp16_to_f32(0x0000), 0.0);
        assert!(fp16_to_f32(0x0001).is_finite());
        assert!(!fp16_is_finite(0x7c00));
        assert!(!fp16_is_finite(0xfc00));
    }
}
