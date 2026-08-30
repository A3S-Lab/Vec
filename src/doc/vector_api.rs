//! Native vector conversions and typed document accessors.

use super::vector_codec::{encode_fp16, f64_to_f32, fp16_to_f32, validate_vector};
use super::{type_error, Doc, VectorValue};
use crate::error::Result;
use crate::types::DataType;
use std::collections::BTreeMap;

impl VectorValue {
    /// Encodes finite f32 coordinates as IEEE 754 half-precision bits.
    pub fn encode_fp16(values: &[f32]) -> Result<Self> {
        Ok(Self::Fp16(encode_fp16(values)?))
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Self::Binary32(_) => DataType::VectorBinary32,
            Self::Binary64(_) => DataType::VectorBinary64,
            Self::Fp16(_) => DataType::VectorFp16,
            Self::Fp32(_) => DataType::VectorFp32,
            Self::Fp64(_) => DataType::VectorFp64,
            Self::Int4(_) => DataType::VectorInt4,
            Self::Int8(_) => DataType::VectorInt8,
            Self::Int16(_) => DataType::VectorInt16,
            Self::SparseFp16 { .. } => DataType::SparseVectorFp16,
            Self::SparseFp32 { .. } => DataType::SparseVectorFp32,
        }
    }

    #[allow(clippy::match_same_arms)]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Binary32(v) | Self::Binary64(v) => v.len().saturating_mul(8),
            Self::Fp16(v) => v.len(),
            Self::Fp32(v) => v.len(),
            Self::Fp64(v) => v.len(),
            Self::Int4(v) | Self::Int8(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::SparseFp16 { indices, .. } | Self::SparseFp32 { indices, .. } => indices
                .iter()
                .max()
                .map_or(0, |v| (*v as usize).saturating_add(1)),
        }
    }

    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::SparseFp16 { .. } | Self::SparseFp32 { .. })
    }

    /// Converts numeric dense forms to f32 for adapters that require it.
    ///
    /// FP64 coordinates may be narrowed. The exact collection executor uses
    /// [`Self::to_dense_f64`] instead.
    pub fn to_dense_f32(&self) -> Option<Vec<f32>> {
        match self {
            Self::Fp16(values) => Some(values.iter().map(|v| fp16_to_f32(*v)).collect()),
            Self::Fp32(values) => Some(values.clone()),
            Self::Fp64(values) => values.iter().copied().map(f64_to_f32).collect(),
            Self::Int4(values) | Self::Int8(values) => {
                Some(values.iter().map(|v| f32::from(*v)).collect())
            }
            Self::Int16(values) => Some(values.iter().map(|v| f32::from(*v)).collect()),
            _ => None,
        }
    }

    /// Decodes all numeric dense forms without narrowing FP64 coordinates.
    pub fn to_dense_f64(&self) -> Option<Vec<f64>> {
        match self {
            Self::Fp16(values) => Some(
                values
                    .iter()
                    .map(|value| f64::from(fp16_to_f32(*value)))
                    .collect(),
            ),
            Self::Fp32(values) => Some(values.iter().map(|value| f64::from(*value)).collect()),
            Self::Fp64(values) => Some(values.clone()),
            Self::Int4(values) | Self::Int8(values) => {
                Some(values.iter().map(|value| f64::from(*value)).collect())
            }
            Self::Int16(values) => Some(values.iter().map(|value| f64::from(*value)).collect()),
            _ => None,
        }
    }

    pub fn to_sparse_f64(&self) -> Option<BTreeMap<u32, f64>> {
        match self {
            Self::SparseFp16 { indices, values } => {
                if indices.len() != values.len() {
                    return None;
                }
                Some(
                    indices
                        .iter()
                        .copied()
                        .zip(values.iter().map(|value| f64::from(fp16_to_f32(*value))))
                        .collect(),
                )
            }
            Self::SparseFp32 { indices, values } => {
                if indices.len() != values.len() {
                    return None;
                }
                Some(
                    indices
                        .iter()
                        .copied()
                        .zip(values.iter().map(|value| f64::from(*value)))
                        .collect(),
                )
            }
            _ => None,
        }
    }

    pub(crate) fn to_core(&self) -> Option<zvec_core::model::StoredVector> {
        use zvec_core::model::StoredVector;
        match self {
            Self::Binary32(_) | Self::Binary64(_) => None,
            Self::Fp32(v) => Some(StoredVector::Dense(v.clone())),
            Self::Fp64(v) => Some(StoredVector::Dense(
                v.iter().copied().map(f64_to_f32).collect::<Option<_>>()?,
            )),
            Self::Fp16(v) => Some(StoredVector::DenseFp16 { data: v.clone() }),
            Self::Int4(v) | Self::Int8(v) => Some(StoredVector::Dense(
                v.iter().map(|value| f32::from(*value)).collect(),
            )),
            Self::Int16(v) => Some(StoredVector::Dense(
                v.iter().map(|value| f32::from(*value)).collect(),
            )),
            Self::SparseFp16 { indices, values } => {
                let map = indices
                    .iter()
                    .copied()
                    .zip(values.iter().map(|value| f64::from(fp16_to_f32(*value))))
                    .map(|(index, value)| (index.to_string(), value))
                    .collect();
                Some(StoredVector::Sparse(map))
            }
            Self::SparseFp32 { indices, values } => {
                let map = indices
                    .iter()
                    .copied()
                    .zip(values.iter().map(|value| f64::from(*value)))
                    .map(|(index, value)| (index.to_string(), value))
                    .collect();
                Some(StoredVector::Sparse(map))
            }
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_vector(self)
    }
}

impl Doc {
    pub fn add_vector_f32(&mut self, name: &str, vector: &[f32]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Fp32(vector.to_vec()))
    }

    pub fn add_vector_f64(&mut self, name: &str, vector: &[f64]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Fp64(vector.to_vec()))
    }

    pub fn add_vector_i8(&mut self, name: &str, vector: &[i8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Int8(vector.to_vec()))
    }

    pub fn add_vector_i16(&mut self, name: &str, vector: &[i16]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Int16(vector.to_vec()))
    }

    pub fn add_vector_fp16(&mut self, name: &str, vector: &[u16]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Fp16(vector.to_vec()))
    }

    pub fn add_vector_fp16_f32(&mut self, name: &str, vector: &[f32]) -> Result<()> {
        self.set_vector_value(name, VectorValue::encode_fp16(vector)?)
    }

    pub fn add_vector_i4(&mut self, name: &str, vector: &[i8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Int4(vector.to_vec()))
    }

    pub fn add_vector_binary32(&mut self, name: &str, vector: &[u8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Binary32(vector.to_vec()))
    }

    pub fn add_vector_binary64(&mut self, name: &str, vector: &[u8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Binary64(vector.to_vec()))
    }

    pub fn add_sparse_vector(&mut self, name: &str, indices: &[u32], values: &[f32]) -> Result<()> {
        self.set_vector_value(
            name,
            VectorValue::SparseFp32 {
                indices: indices.to_vec(),
                values: values.to_vec(),
            },
        )
    }

    pub fn add_sparse_vector_f32(
        &mut self,
        name: &str,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        self.add_sparse_vector(name, indices, values)
    }

    pub fn add_sparse_vector_fp16(
        &mut self,
        name: &str,
        indices: &[u32],
        values: &[u16],
    ) -> Result<()> {
        self.set_vector_value(
            name,
            VectorValue::SparseFp16 {
                indices: indices.to_vec(),
                values: values.to_vec(),
            },
        )
    }

    pub fn add_sparse_vector_fp16_f32(
        &mut self,
        name: &str,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        self.set_vector_value(
            name,
            VectorValue::SparseFp16 {
                indices: indices.to_vec(),
                values: encode_fp16(values)?,
            },
        )
    }

    pub fn get_vector_f32(&self, name: &str) -> Result<Option<Vec<f32>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Fp32(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorFp32)),
        }
    }

    pub fn get_vector_f64(&self, name: &str) -> Result<Option<Vec<f64>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Fp64(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorFp64)),
        }
    }

    pub fn get_vector_fp16(&self, name: &str) -> Result<Option<Vec<u16>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Fp16(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorFp16)),
        }
    }

    pub fn get_vector_i4(&self, name: &str) -> Result<Option<Vec<i8>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Int4(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorInt4)),
        }
    }

    pub fn get_vector_i8(&self, name: &str) -> Result<Option<Vec<i8>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Int8(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorInt8)),
        }
    }

    pub fn get_vector_i16(&self, name: &str) -> Result<Option<Vec<i16>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Int16(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorInt16)),
        }
    }

    pub fn get_vector_binary32(&self, name: &str) -> Result<Option<Vec<u8>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Binary32(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorBinary32)),
        }
    }

    pub fn get_vector_binary64(&self, name: &str) -> Result<Option<Vec<u8>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Binary64(values)) => Ok(Some(values.clone())),
            Some(_) => Err(type_error(name, DataType::VectorBinary64)),
        }
    }

    pub fn get_sparse_vector_f32(&self, name: &str) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::SparseFp32 { indices, values }) => {
                Ok(Some((indices.clone(), values.clone())))
            }
            Some(_) => Err(type_error(name, DataType::SparseVectorFp32)),
        }
    }

    pub fn get_sparse_vector_fp16(&self, name: &str) -> Result<Option<(Vec<u32>, Vec<u16>)>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::SparseFp16 { indices, values }) => {
                Ok(Some((indices.clone(), values.clone())))
            }
            Some(_) => Err(type_error(name, DataType::SparseVectorFp16)),
        }
    }
}
