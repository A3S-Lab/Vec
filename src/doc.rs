//! Typed documents and lossless JSON representation.

use crate::error::{Error, Result};
use crate::types::DataType;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::BTreeMap;

/// Scalar and array field values accepted by a collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum FieldValue {
    Null,
    Binary(Vec<u8>),
    String(String),
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Uint32(u32),
    Uint64(u64),
    Float(f32),
    Double(f64),
    ArrayBinary(Vec<Vec<u8>>),
    ArrayString(Vec<String>),
    ArrayBool(Vec<bool>),
    ArrayInt32(Vec<i32>),
    ArrayInt64(Vec<i64>),
    ArrayUint32(Vec<u32>),
    ArrayUint64(Vec<u64>),
    ArrayFloat(Vec<f32>),
    ArrayDouble(Vec<f64>),
    /// Escape hatch for adapters that need a JSON scalar/array while retaining
    /// the field in the document.  Schema validation still rejects values
    /// whose declared type is incompatible.
    Json(Value),
}

impl FieldValue {
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Null => DataType::Undefined,
            Self::Binary(_) => DataType::Binary,
            Self::String(_) => DataType::String,
            Self::Bool(_) => DataType::Bool,
            Self::Int32(_) => DataType::Int32,
            Self::Int64(_) => DataType::Int64,
            Self::Uint32(_) => DataType::Uint32,
            Self::Uint64(_) => DataType::Uint64,
            Self::Float(_) => DataType::Float,
            Self::Double(_) => DataType::Double,
            Self::ArrayBinary(_) => DataType::ArrayBinary,
            Self::ArrayString(_) => DataType::ArrayString,
            Self::ArrayBool(_) => DataType::ArrayBool,
            Self::ArrayInt32(_) => DataType::ArrayInt32,
            Self::ArrayInt64(_) => DataType::ArrayInt64,
            Self::ArrayUint32(_) => DataType::ArrayUint32,
            Self::ArrayUint64(_) => DataType::ArrayUint64,
            Self::ArrayFloat(_) => DataType::ArrayFloat,
            Self::ArrayDouble(_) => DataType::ArrayDouble,
            Self::Json(_) => DataType::Undefined,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Binary(bytes) => Value::String(base64_encode(bytes)),
            Self::String(value) => Value::String(value.clone()),
            Self::Bool(value) => Value::Bool(*value),
            Self::Int32(value) => Value::Number((*value).into()),
            Self::Int64(value) => Value::Number((*value).into()),
            Self::Uint32(value) => Value::Number((*value).into()),
            Self::Uint64(value) => Value::Number((*value).into()),
            Self::Float(value) => number_from_f64(*value as f64),
            Self::Double(value) => number_from_f64(*value),
            Self::ArrayBinary(values) => Value::Array(
                values
                    .iter()
                    .map(|v| Value::String(base64_encode(v)))
                    .collect(),
            ),
            Self::ArrayString(values) => {
                Value::Array(values.iter().cloned().map(Value::String).collect())
            }
            Self::ArrayBool(values) => Value::Array(values.iter().copied().map(Value::Bool).collect()),
            Self::ArrayInt32(values) => Value::Array(
                values
                    .iter()
                    .map(|v| Value::Number((*v).into()))
                    .collect(),
            ),
            Self::ArrayInt64(values) => Value::Array(
                values
                    .iter()
                    .map(|v| Value::Number((*v).into()))
                    .collect(),
            ),
            Self::ArrayUint32(values) => Value::Array(
                values
                    .iter()
                    .map(|v| Value::Number((*v).into()))
                    .collect(),
            ),
            Self::ArrayUint64(values) => Value::Array(
                values
                    .iter()
                    .map(|v| Value::Number((*v).into()))
                    .collect(),
            ),
            Self::ArrayFloat(values) => {
                Value::Array(values.iter().map(|v| number_from_f64(*v as f64)).collect())
            }
            Self::ArrayDouble(values) => {
                Value::Array(values.iter().map(|v| number_from_f64(*v)).collect())
            }
            Self::Json(value) => value.clone(),
        }
    }

    pub(crate) fn from_json(value: &Value, data_type: Option<DataType>) -> Self {
        match data_type.unwrap_or(DataType::Undefined) {
            DataType::Binary => value
                .as_str()
                .map(|s| FieldValue::Binary(base64_decode(s)))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::String => value
                .as_str()
                .map(|s| FieldValue::String(s.to_string()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Bool => value
                .as_bool()
                .map(FieldValue::Bool)
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Int32 => value
                .as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .map(FieldValue::Int32)
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Int64 => value
                .as_i64()
                .map(FieldValue::Int64)
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Uint32 => value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .map(FieldValue::Uint32)
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Uint64 => value
                .as_u64()
                .map(FieldValue::Uint64)
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Float => value
                .as_f64()
                .map(|v| FieldValue::Float(v as f32))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::Double => value
                .as_f64()
                .map(FieldValue::Double)
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayString => value
                .as_array()
                .map(|v| FieldValue::ArrayString(v.iter().filter_map(Value::as_str).map(str::to_string).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayBool => value
                .as_array()
                .map(|v| FieldValue::ArrayBool(v.iter().filter_map(Value::as_bool).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayInt32 => value
                .as_array()
                .map(|v| FieldValue::ArrayInt32(v.iter().filter_map(Value::as_i64).filter_map(|x| i32::try_from(x).ok()).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayInt64 => value
                .as_array()
                .map(|v| FieldValue::ArrayInt64(v.iter().filter_map(Value::as_i64).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayUint32 => value
                .as_array()
                .map(|v| FieldValue::ArrayUint32(v.iter().filter_map(Value::as_u64).filter_map(|x| u32::try_from(x).ok()).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayUint64 => value
                .as_array()
                .map(|v| FieldValue::ArrayUint64(v.iter().filter_map(Value::as_u64).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayFloat => value
                .as_array()
                .map(|v| FieldValue::ArrayFloat(v.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayDouble => value
                .as_array()
                .map(|v| FieldValue::ArrayDouble(v.iter().filter_map(Value::as_f64).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            DataType::ArrayBinary => value
                .as_array()
                .map(|v| FieldValue::ArrayBinary(v.iter().filter_map(Value::as_str).map(base64_decode).collect()))
                .unwrap_or_else(|| FieldValue::Json(value.clone())),
            _ if value.is_null() => FieldValue::Null,
            _ => FieldValue::Json(value.clone()),
        }
    }
}

/// Dense and sparse vector payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum VectorValue {
    Binary32(Vec<u8>),
    Binary64(Vec<u8>),
    Fp16(Vec<u16>),
    Fp32(Vec<f32>),
    Fp64(Vec<f64>),
    Int4(Vec<i8>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    SparseFp16 { indices: Vec<u32>, values: Vec<f32> },
    SparseFp32 { indices: Vec<u32>, values: Vec<f32> },
}

impl VectorValue {
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

    pub fn dimension(&self) -> usize {
        match self {
            Self::Binary32(v) => v.len().saturating_mul(8),
            Self::Binary64(v) => v.len().saturating_mul(8),
            Self::Fp16(v) => v.len(),
            Self::Fp32(v) => v.len(),
            Self::Fp64(v) => v.len(),
            Self::Int4(v) => v.len(),
            Self::Int8(v) => v.len(),
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

    /// Decodes all numeric dense forms to f32 for the portable scoring path.
    pub fn to_dense_f32(&self) -> Option<Vec<f32>> {
        match self {
            Self::Fp16(values) => Some(values.iter().map(|v| fp16_to_f32(*v)).collect()),
            Self::Fp32(values) => Some(values.clone()),
            Self::Fp64(values) => Some(values.iter().map(|v| *v as f32).collect()),
            Self::Int4(values) => Some(values.iter().map(|v| *v as f32).collect()),
            Self::Int8(values) => Some(values.iter().map(|v| *v as f32).collect()),
            Self::Int16(values) => Some(values.iter().map(|v| *v as f32).collect()),
            _ => None,
        }
    }

    pub fn to_sparse_f64(&self) -> Option<BTreeMap<u32, f64>> {
        match self {
            Self::SparseFp16 { indices, values } | Self::SparseFp32 { indices, values } => {
                if indices.len() != values.len() {
                    return None;
                }
                Some(
                    indices
                        .iter()
                        .copied()
                        .zip(values.iter().map(|v| *v as f64))
                        .collect(),
                )
            }
            _ => None,
        }
    }

    pub(crate) fn to_core(&self) -> zvec_core::model::StoredVector {
        use zvec_core::model::StoredVector;
        match self {
            Self::Fp32(v) => StoredVector::Dense(v.clone()),
            Self::Fp64(v) => StoredVector::Dense(v.iter().map(|x| *x as f32).collect()),
            Self::Fp16(v) => StoredVector::Dense(v.iter().map(|x| fp16_to_f32(*x)).collect()),
            Self::Int4(v) => StoredVector::Dense(v.iter().map(|x| *x as f32).collect()),
            Self::Int8(v) => StoredVector::Dense(v.iter().map(|x| *x as f32).collect()),
            Self::Int16(v) => StoredVector::Dense(v.iter().map(|x| *x as f32).collect()),
            Self::Binary32(v) | Self::Binary64(v) => {
                StoredVector::Dense(v.iter().map(|x| *x as f32).collect())
            }
            Self::SparseFp16 { indices, values } | Self::SparseFp32 { indices, values } => {
                let map = indices
                    .iter()
                    .copied()
                    .zip(values.iter().map(|v| *v as f64))
                    .map(|(i, v)| (i.to_string(), v))
                    .collect();
                StoredVector::Sparse(map)
            }
        }
    }

    pub(crate) fn from_core(value: &zvec_core::model::StoredVector) -> Self {
        use zvec_core::model::StoredVector;
        match value {
            StoredVector::Dense(v) => Self::Fp32(v.clone()),
            StoredVector::DenseInt8 { codes, scale } => {
                Self::Fp32(codes.iter().map(|v| *v as f32 * *scale).collect())
            }
            StoredVector::DenseFp16 { data } => Self::Fp16(data.clone()),
            StoredVector::DenseInt4 { nibbles, scale, dim } => {
                let mut values = Vec::with_capacity(*dim);
                for byte in nibbles {
                    let lo = (byte & 0x0f) as i8;
                    let hi = ((byte >> 4) & 0x0f) as i8;
                    values.push(if lo >= 8 { lo - 16 } else { lo });
                    if values.len() < *dim {
                        values.push(if hi >= 8 { hi - 16 } else { hi });
                    }
                }
                Self::Fp32(values.into_iter().map(|v| v as f32 * *scale).collect())
            }
            StoredVector::Sparse(map) => {
                let mut pairs: Vec<(u32, f32)> = map
                    .iter()
                    .filter_map(|(i, v)| i.parse::<u32>().ok().map(|i| (i, *v as f32)))
                    .collect();
                pairs.sort_by_key(|(i, _)| *i);
                let (indices, values) = pairs.into_iter().unzip();
                Self::SparseFp32 { indices, values }
            }
        }
    }
}

/// A typed document.  `BTreeMap` gives deterministic snapshots and tie-breaks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    pk: Option<String>,
    #[serde(default)]
    score: f32,
    #[serde(default)]
    doc_id: Option<u64>,
    #[serde(default)]
    fields: BTreeMap<String, FieldValue>,
    #[serde(default)]
    vectors: BTreeMap<String, VectorValue>,
}

impl Default for Doc {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            pk: None,
            score: 0.0,
            doc_id: None,
            fields: BTreeMap::new(),
            vectors: BTreeMap::new(),
        })
    }
}

impl Doc {
    pub fn new() -> Result<Self> {
        Ok(Self {
            pk: None,
            score: 0.0,
            doc_id: None,
            fields: BTreeMap::new(),
            vectors: BTreeMap::new(),
        })
    }

    pub fn with_pk(pk: impl Into<String>) -> Result<Self> {
        let mut doc = Self::new()?;
        doc.set_pk(&pk.into());
        Ok(doc)
    }

    /// Sets the primary key.  Empty keys are rejected at the collection write
    /// boundary so this method remains source-compatible with zvec's `()` API.
    pub fn set_pk(&mut self, pk: &str) {
        self.pk = Some(pk.to_string());
    }

    pub fn get_pk(&self) -> Option<&str> {
        self.pk.as_deref()
    }

    pub fn get_score(&self) -> f32 {
        self.score
    }

    pub fn score(&self) -> f32 {
        self.score
    }

    pub fn set_score(&mut self, score: f32) -> Result<()> {
        if !score.is_finite() {
            return Err(Error::invalid_argument("score must be finite"));
        }
        self.score = score;
        Ok(())
    }

    pub fn doc_id(&self) -> Option<u64> {
        self.doc_id
    }

    pub fn field_count(&self) -> usize {
        self.fields.len() + self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty() && self.vectors.is_empty() && self.pk.is_none()
    }

    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name) || self.vectors.contains_key(name)
    }

    pub fn is_field_null(&self, name: &str) -> bool {
        matches!(self.fields.get(name), Some(FieldValue::Null))
    }

    pub fn field(&self, name: &str) -> Option<&FieldValue> {
        self.fields.get(name)
    }

    pub fn vector(&self, name: &str) -> Option<&VectorValue> {
        self.vectors.get(name)
    }

    pub fn fields(&self) -> &BTreeMap<String, FieldValue> {
        &self.fields
    }

    pub fn vectors(&self) -> &BTreeMap<String, VectorValue> {
        &self.vectors
    }

    pub fn set_field_value(&mut self, name: &str, value: FieldValue) -> Result<()> {
        validate_name(name)?;
        validate_field_finite(&value)?;
        self.fields.insert(name.to_string(), value);
        Ok(())
    }

    pub fn set_vector_value(&mut self, name: &str, value: VectorValue) -> Result<()> {
        validate_name(name)?;
        validate_vector(&value)?;
        self.vectors.insert(name.to_string(), value);
        Ok(())
    }

    pub fn add_string(&mut self, name: &str, value: &str) -> Result<()> {
        self.set_field_value(name, FieldValue::String(value.to_string()))
    }

    pub fn add_bool(&mut self, name: &str, value: bool) -> Result<()> {
        self.set_field_value(name, FieldValue::Bool(value))
    }

    pub fn add_i32(&mut self, name: &str, value: i32) -> Result<()> {
        self.set_field_value(name, FieldValue::Int32(value))
    }

    pub fn add_i64(&mut self, name: &str, value: i64) -> Result<()> {
        self.set_field_value(name, FieldValue::Int64(value))
    }

    pub fn add_u32(&mut self, name: &str, value: u32) -> Result<()> {
        self.set_field_value(name, FieldValue::Uint32(value))
    }

    pub fn add_u64(&mut self, name: &str, value: u64) -> Result<()> {
        self.set_field_value(name, FieldValue::Uint64(value))
    }

    pub fn add_f32(&mut self, name: &str, value: f32) -> Result<()> {
        self.set_field_value(name, FieldValue::Float(value))
    }

    pub fn add_f64(&mut self, name: &str, value: f64) -> Result<()> {
        self.set_field_value(name, FieldValue::Double(value))
    }

    pub fn add_binary(&mut self, name: &str, value: &[u8]) -> Result<()> {
        self.set_field_value(name, FieldValue::Binary(value.to_vec()))
    }

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

    pub fn add_vector_i4(&mut self, name: &str, vector: &[i8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Int4(vector.to_vec()))
    }

    pub fn add_vector_binary32(&mut self, name: &str, vector: &[u8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Binary32(vector.to_vec()))
    }

    pub fn add_vector_binary64(&mut self, name: &str, vector: &[u8]) -> Result<()> {
        self.set_vector_value(name, VectorValue::Binary64(vector.to_vec()))
    }

    pub fn add_sparse_vector(
        &mut self,
        name: &str,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        if indices.len() != values.len() || indices.is_empty() {
            return Err(Error::invalid_argument(
                "sparse vector indices and values must have equal non-zero length",
            ));
        }
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

    pub fn add_array_binary(&mut self, name: &str, values: &[Vec<u8>]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayBinary(values.to_vec()))
    }

    pub fn add_array_string(&mut self, name: &str, values: &[&str]) -> Result<()> {
        self.set_field_value(
            name,
            FieldValue::ArrayString(values.iter().map(|v| (*v).to_string()).collect()),
        )
    }

    pub fn add_array_i32(&mut self, name: &str, values: &[i32]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayInt32(values.to_vec()))
    }

    pub fn add_array_i64(&mut self, name: &str, values: &[i64]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayInt64(values.to_vec()))
    }

    pub fn add_array_u32(&mut self, name: &str, values: &[u32]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayUint32(values.to_vec()))
    }

    pub fn add_array_u64(&mut self, name: &str, values: &[u64]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayUint64(values.to_vec()))
    }

    pub fn add_array_f32(&mut self, name: &str, values: &[f32]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayFloat(values.to_vec()))
    }

    pub fn add_array_f64(&mut self, name: &str, values: &[f64]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayDouble(values.to_vec()))
    }

    pub fn add_array_bool(&mut self, name: &str, values: &[bool]) -> Result<()> {
        self.set_field_value(name, FieldValue::ArrayBool(values.to_vec()))
    }

    pub fn set_field_null(&mut self, name: &str) -> Result<()> {
        self.set_field_value(name, FieldValue::Null)
    }

    pub fn remove_field(&mut self, name: &str) -> Result<()> {
        validate_name(name)?;
        self.fields.remove(name);
        self.vectors.remove(name);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.fields.clear();
        self.vectors.clear();
        self.score = 0.0;
    }

    pub fn get_string(&self, name: &str) -> Result<Option<String>> {
        Ok(match self.fields.get(name) {
            Some(FieldValue::String(v)) => Some(v.clone()),
            Some(FieldValue::Null) | None => None,
            Some(_) => return Err(type_error(name, DataType::String)),
        })
    }

    pub fn get_bool(&self, name: &str) -> Result<Option<bool>> {
        self.get_scalar(name, |v| match v { FieldValue::Bool(x) => Some(*x), _ => None }, DataType::Bool)
    }

    pub fn get_i32(&self, name: &str) -> Result<Option<i32>> {
        self.get_scalar(name, |v| match v { FieldValue::Int32(x) => Some(*x), _ => None }, DataType::Int32)
    }

    pub fn get_i64(&self, name: &str) -> Result<Option<i64>> {
        self.get_scalar(name, |v| match v { FieldValue::Int64(x) => Some(*x), _ => None }, DataType::Int64)
    }

    pub fn get_u32(&self, name: &str) -> Result<Option<u32>> {
        self.get_scalar(name, |v| match v { FieldValue::Uint32(x) => Some(*x), _ => None }, DataType::Uint32)
    }

    pub fn get_u64(&self, name: &str) -> Result<Option<u64>> {
        self.get_scalar(name, |v| match v { FieldValue::Uint64(x) => Some(*x), _ => None }, DataType::Uint64)
    }

    pub fn get_f32(&self, name: &str) -> Result<Option<f32>> {
        self.get_scalar(name, |v| match v { FieldValue::Float(x) => Some(*x), _ => None }, DataType::Float)
    }

    pub fn get_f64(&self, name: &str) -> Result<Option<f64>> {
        self.get_scalar(name, |v| match v { FieldValue::Double(x) => Some(*x), _ => None }, DataType::Double)
    }

    pub fn get_binary(&self, name: &str) -> Result<Option<Vec<u8>>> {
        self.get_scalar(name, |v| match v { FieldValue::Binary(x) => Some(x.clone()), _ => None }, DataType::Binary)
    }

    pub fn get_vector_f32(&self, name: &str) -> Result<Option<Vec<f32>>> {
        self.get_vector_dense(name, DataType::VectorFp32)
    }

    pub fn get_vector_f64(&self, name: &str) -> Result<Option<Vec<f64>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Fp64(v)) => Ok(Some(v.clone())),
            Some(_) => Err(type_error(name, DataType::VectorFp64)),
        }
    }

    pub fn get_vector_i8(&self, name: &str) -> Result<Option<Vec<i8>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Int8(v)) => Ok(Some(v.clone())),
            Some(_) => Err(type_error(name, DataType::VectorInt8)),
        }
    }

    pub fn get_vector_i16(&self, name: &str) -> Result<Option<Vec<i16>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Int16(v)) => Ok(Some(v.clone())),
            Some(_) => Err(type_error(name, DataType::VectorInt16)),
        }
    }

    pub fn get_array_i32(&self, name: &str) -> Result<Option<Vec<i32>>> { self.get_array(name, |v| match v { FieldValue::ArrayInt32(x) => Some(x.clone()), _ => None }, DataType::ArrayInt32) }
    pub fn get_array_i64(&self, name: &str) -> Result<Option<Vec<i64>>> { self.get_array(name, |v| match v { FieldValue::ArrayInt64(x) => Some(x.clone()), _ => None }, DataType::ArrayInt64) }
    pub fn get_array_u32(&self, name: &str) -> Result<Option<Vec<u32>>> { self.get_array(name, |v| match v { FieldValue::ArrayUint32(x) => Some(x.clone()), _ => None }, DataType::ArrayUint32) }
    pub fn get_array_u64(&self, name: &str) -> Result<Option<Vec<u64>>> { self.get_array(name, |v| match v { FieldValue::ArrayUint64(x) => Some(x.clone()), _ => None }, DataType::ArrayUint64) }
    pub fn get_array_f32(&self, name: &str) -> Result<Option<Vec<f32>>> { self.get_array(name, |v| match v { FieldValue::ArrayFloat(x) => Some(x.clone()), _ => None }, DataType::ArrayFloat) }
    pub fn get_array_f64(&self, name: &str) -> Result<Option<Vec<f64>>> { self.get_array(name, |v| match v { FieldValue::ArrayDouble(x) => Some(x.clone()), _ => None }, DataType::ArrayDouble) }
    pub fn get_array_bool(&self, name: &str) -> Result<Option<Vec<bool>>> { self.get_array(name, |v| match v { FieldValue::ArrayBool(x) => Some(x.clone()), _ => None }, DataType::ArrayBool) }

    /// Returns a projection suitable for query/fetch output.
    pub fn project(&self, output_fields: Option<&[String]>, include_vector: bool) -> Self {
        let mut out = self.clone();
        if let Some(fields) = output_fields {
            let wanted: std::collections::BTreeSet<&str> = fields.iter().map(String::as_str).collect();
            out.fields.retain(|k, _| wanted.contains(k.as_str()));
            if include_vector {
                out.vectors.retain(|k, _| wanted.contains(k.as_str()));
            } else {
                out.vectors.clear();
            }
        } else if !include_vector {
            out.vectors.clear();
        }
        out
    }

    pub(crate) fn to_core(&self) -> zvec_core::model::Doc {
        let vectors = self
            .vectors
            .iter()
            .map(|(name, value)| (name.clone(), value.to_core()))
            .collect();
        let fields = self
            .fields
            .iter()
            .map(|(name, value)| (name.clone(), value.to_json()))
            .collect();
        zvec_core::model::Doc::new(
            self.pk.clone().unwrap_or_default(),
            Some(self.score as f64),
            vectors,
            fields,
        )
    }

    pub(crate) fn from_core(doc: &zvec_core::model::Doc) -> Self {
        let fields = doc
            .fields
            .iter()
            .map(|(name, value)| (name.clone(), FieldValue::Json(value.clone())))
            .collect();
        let vectors = doc
            .vectors
            .iter()
            .map(|(name, value)| (name.clone(), VectorValue::from_core(value)))
            .collect();
        Self {
            pk: Some(doc.id.clone()),
            score: doc.score.unwrap_or(0.0) as f32,
            doc_id: None,
            fields,
            vectors,
        }
    }

    pub(crate) fn scalar_json(&self, name: &str) -> Option<Value> {
        self.fields.get(name).map(FieldValue::to_json)
    }

    fn get_scalar<T, F>(&self, name: &str, f: F, expected: DataType) -> Result<Option<T>>
    where
        F: FnOnce(&FieldValue) -> Option<T>,
    {
        match self.fields.get(name) {
            None | Some(FieldValue::Null) => Ok(None),
            Some(value) => f(value)
                .map(Some)
                .ok_or_else(|| type_error(name, expected)),
        }
    }

    fn get_array<T, F>(&self, name: &str, f: F, expected: DataType) -> Result<Option<T>>
    where
        F: FnOnce(&FieldValue) -> Option<T>,
    {
        self.get_scalar(name, f, expected)
    }

    fn get_vector_dense(&self, name: &str, expected: DataType) -> Result<Option<Vec<f32>>> {
        match self.vectors.get(name) {
            None => Ok(None),
            Some(VectorValue::Fp32(v)) => Ok(Some(v.clone())),
            Some(_) => Err(type_error(name, expected)),
        }
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains('\0') {
        return Err(Error::invalid_argument(
            "field name must be non-empty and contain no NUL byte",
        ));
    }
    Ok(())
}

fn validate_field_finite(value: &FieldValue) -> Result<()> {
    let finite = match value {
        FieldValue::Float(v) => v.is_finite(),
        FieldValue::Double(v) => v.is_finite(),
        FieldValue::ArrayFloat(v) => v.iter().all(|x| x.is_finite()),
        FieldValue::ArrayDouble(v) => v.iter().all(|x| x.is_finite()),
        _ => true,
    };
    finite.then_some(()).ok_or_else(|| Error::invalid_argument("floating-point field values must be finite"))
}

fn validate_vector(value: &VectorValue) -> Result<()> {
    match value {
        VectorValue::SparseFp16 { indices, values }
        | VectorValue::SparseFp32 { indices, values } => {
            if indices.len() != values.len() || indices.is_empty() {
                return Err(Error::invalid_argument("sparse vector indices and values must have equal non-zero length"));
            }
            if !values.iter().all(|v| v.is_finite()) {
                return Err(Error::invalid_argument("sparse vector values must be finite"));
            }
        }
        VectorValue::Fp32(values) => {
            if !values.iter().all(|v| v.is_finite()) { return Err(Error::invalid_argument("vector values must be finite")); }
        }
        VectorValue::Fp64(values) => {
            if !values.iter().all(|v| v.is_finite()) { return Err(Error::invalid_argument("vector values must be finite")); }
        }
        _ => {}
    }
    Ok(())
}

fn type_error(name: &str, expected: DataType) -> Error {
    Error::invalid_argument(format!("field '{name}' is not of type {expected}"))
}

fn number_from_f64(value: f64) -> Value {
    Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
}

// Base64 is implemented locally to avoid adding a mandatory codec dependency
// to this small crate.  The alphabet and padding follow RFC 4648.
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        out.push(TABLE[((a >> 2) & 63) as usize] as char);
        out.push(TABLE[(((a << 4) | (b >> 4)) & 63) as usize] as char);
        if chunk.len() > 1 { out.push(TABLE[(((b << 2) | (c >> 6)) & 63) as usize] as char); } else { out.push('='); }
        if chunk.len() > 2 { out.push(TABLE[(c & 63) as usize] as char); } else { out.push('='); }
    }
    out
}

fn base64_decode(value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u8;
    for byte in value.bytes() {
        let digit = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => continue,
        } as u32;
        buf = (buf << 6) | digit;
        bits = bits.saturating_add(6);
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    out
}

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;
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
        sign | (((exp as u32 + 112) & 0xff) << 23) | (frac << 13)
    };
    f32::from_bits(raw)
}
