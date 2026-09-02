//! Typed documents and lossless JSON representation.

mod vector_api;
mod vector_codec;

use crate::error::{Error, Result};
use crate::types::DataType;
use im::OrdMap;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

use vector_codec::validate_vector;

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
            Self::Null | Self::Json(_) => DataType::Undefined,
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
            Self::Float(value) => number_from_f64(f64::from(*value)),
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
            Self::ArrayBool(values) => {
                Value::Array(values.iter().copied().map(Value::Bool).collect())
            }
            Self::ArrayInt32(values) => {
                Value::Array(values.iter().map(|v| Value::Number((*v).into())).collect())
            }
            Self::ArrayInt64(values) => {
                Value::Array(values.iter().map(|v| Value::Number((*v).into())).collect())
            }
            Self::ArrayUint32(values) => {
                Value::Array(values.iter().map(|v| Value::Number((*v).into())).collect())
            }
            Self::ArrayUint64(values) => {
                Value::Array(values.iter().map(|v| Value::Number((*v).into())).collect())
            }
            Self::ArrayFloat(values) => Value::Array(
                values
                    .iter()
                    .map(|v| number_from_f64(f64::from(*v)))
                    .collect(),
            ),
            Self::ArrayDouble(values) => {
                Value::Array(values.iter().map(|v| number_from_f64(*v)).collect())
            }
            Self::Json(value) => value.clone(),
        }
    }
}

/// Dense and sparse vector payloads.
///
/// FP16 values are raw IEEE 754 half-precision bits. INT4/INT8/INT16 values
/// are authoritative integer coordinates, not scale-bearing index
/// quantization. Binary values are packed bytes in 32- or 64-bit chunks.
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
    SparseFp16 { indices: Vec<u32>, values: Vec<u16> },
    SparseFp32 { indices: Vec<u32>, values: Vec<f32> },
}

/// A typed document.  `BTreeMap` gives deterministic snapshots and tie-breaks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    pk: Option<String>,
    #[serde(default)]
    score: f32,
    #[serde(default)]
    #[serde(rename = "doc_id")]
    internal_id: Option<u64>,
    #[serde(default)]
    fields: BTreeMap<String, FieldValue>,
    #[serde(default)]
    vectors: BTreeMap<String, VectorValue>,
}

/// Collection snapshots use a persistent ordered tree and share immutable
/// documents across generations. A write copy-on-writes only the tree path and
/// document it changes; public APIs and persistence still use owned `Doc`s.
pub(crate) type DocumentMap = OrdMap<String, Arc<Doc>>;

impl Default for Doc {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            pk: None,
            score: 0.0,
            internal_id: None,
            fields: BTreeMap::new(),
            vectors: BTreeMap::new(),
        })
    }
}

impl Doc {
    pub(crate) fn from_persisted_parts(
        pk: Option<String>,
        score: f32,
        internal_id: Option<u64>,
        fields: BTreeMap<String, FieldValue>,
        vectors: BTreeMap<String, VectorValue>,
    ) -> Self {
        Self {
            pk,
            score,
            internal_id,
            fields,
            vectors,
        }
    }

    pub fn new() -> Result<Self> {
        Ok(Self {
            pk: None,
            score: 0.0,
            internal_id: None,
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

    /// Returns the generation-local ordinal exposed by a query that requested
    /// document IDs. Ordinary input documents and queries without that option
    /// return `None`.
    pub fn doc_id(&self) -> Option<u64> {
        self.internal_id
    }

    pub(crate) fn set_internal_id(&mut self, doc_id: Option<u64>) {
        self.internal_id = doc_id;
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
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Bool(x) => Some(*x),
                _ => None,
            },
            DataType::Bool,
        )
    }

    pub fn get_i32(&self, name: &str) -> Result<Option<i32>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Int32(x) => Some(*x),
                _ => None,
            },
            DataType::Int32,
        )
    }

    pub fn get_i64(&self, name: &str) -> Result<Option<i64>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Int64(x) => Some(*x),
                _ => None,
            },
            DataType::Int64,
        )
    }

    pub fn get_u32(&self, name: &str) -> Result<Option<u32>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Uint32(x) => Some(*x),
                _ => None,
            },
            DataType::Uint32,
        )
    }

    pub fn get_u64(&self, name: &str) -> Result<Option<u64>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Uint64(x) => Some(*x),
                _ => None,
            },
            DataType::Uint64,
        )
    }

    pub fn get_f32(&self, name: &str) -> Result<Option<f32>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Float(x) => Some(*x),
                _ => None,
            },
            DataType::Float,
        )
    }

    pub fn get_f64(&self, name: &str) -> Result<Option<f64>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Double(x) => Some(*x),
                _ => None,
            },
            DataType::Double,
        )
    }

    pub fn get_binary(&self, name: &str) -> Result<Option<Vec<u8>>> {
        self.get_scalar(
            name,
            |v| match v {
                FieldValue::Binary(x) => Some(x.clone()),
                _ => None,
            },
            DataType::Binary,
        )
    }

    pub fn get_array_i32(&self, name: &str) -> Result<Option<Vec<i32>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayInt32(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayInt32,
        )
    }
    pub fn get_array_i64(&self, name: &str) -> Result<Option<Vec<i64>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayInt64(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayInt64,
        )
    }
    pub fn get_array_u32(&self, name: &str) -> Result<Option<Vec<u32>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayUint32(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayUint32,
        )
    }
    pub fn get_array_u64(&self, name: &str) -> Result<Option<Vec<u64>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayUint64(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayUint64,
        )
    }
    pub fn get_array_f32(&self, name: &str) -> Result<Option<Vec<f32>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayFloat(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayFloat,
        )
    }
    pub fn get_array_f64(&self, name: &str) -> Result<Option<Vec<f64>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayDouble(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayDouble,
        )
    }
    pub fn get_array_bool(&self, name: &str) -> Result<Option<Vec<bool>>> {
        self.get_array(
            name,
            |v| match v {
                FieldValue::ArrayBool(x) => Some(x.clone()),
                _ => None,
            },
            DataType::ArrayBool,
        )
    }

    /// Returns a projection suitable for query/fetch output.
    pub fn project(&self, output_fields: Option<&[String]>, include_vector: bool) -> Self {
        let mut out = self.clone();
        if let Some(fields) = output_fields {
            let wanted: std::collections::BTreeSet<&str> =
                fields.iter().map(String::as_str).collect();
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
            .filter_map(|(name, value)| value.to_core().map(|value| (name.clone(), value)))
            .collect();
        let fields = self
            .fields
            .iter()
            .map(|(name, value)| (name.clone(), value.to_json()))
            .collect();
        zvec_core::model::Doc::new(
            self.pk.clone().unwrap_or_default(),
            Some(f64::from(self.score)),
            vectors,
            fields,
        )
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
            Some(value) => f(value).map(Some).ok_or_else(|| type_error(name, expected)),
        }
    }

    fn get_array<T, F>(&self, name: &str, f: F, expected: DataType) -> Result<Option<T>>
    where
        F: FnOnce(&FieldValue) -> Option<T>,
    {
        self.get_scalar(name, f, expected)
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
    finite
        .then_some(())
        .ok_or_else(|| Error::invalid_argument("floating-point field values must be finite"))
}

fn type_error(name: &str, expected: DataType) -> Error {
    Error::invalid_argument(format!("field '{name}' is not of type {expected}"))
}

fn number_from_f64(value: f64) -> Value {
    Number::from_f64(value).map_or(Value::Null, Value::Number)
}

// Base64 is implemented locally to avoid adding a mandatory codec dependency
// to this small crate.  The alphabet and padding follow RFC 4648.
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = u32::from(chunk[0]);
        let b = u32::from(chunk.get(1).copied().unwrap_or(0));
        let c = u32::from(chunk.get(2).copied().unwrap_or(0));
        out.push(TABLE[((a >> 2) & 63) as usize] as char);
        out.push(TABLE[(((a << 4) | (b >> 4)) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b << 2) | (c >> 6)) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(c & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
