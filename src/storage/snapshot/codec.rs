//! Compact binary snapshot representation.

use crate::doc::{Doc, FieldValue, VectorValue};
use crate::schema::CollectionSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct BinarySnapshot {
    format_version: u32,
    generation: u64,
    revision: u64,
    schema: CollectionSchema,
    docs: Vec<BinaryDoc>,
}

impl BinarySnapshot {
    pub(super) fn new(
        format_version: u32,
        generation: u64,
        revision: u64,
        schema: &CollectionSchema,
        docs: &[Doc],
    ) -> Self {
        Self {
            format_version,
            generation,
            revision,
            schema: schema.clone(),
            docs: docs.iter().map(BinaryDoc::from).collect(),
        }
    }

    pub(super) fn into_parts(self) -> (u32, u64, u64, CollectionSchema, Vec<Doc>) {
        (
            self.format_version,
            self.generation,
            self.revision,
            self.schema,
            self.docs.into_iter().map(Doc::from).collect(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinaryDoc {
    pk: Option<String>,
    score: f32,
    internal_id: Option<u64>,
    fields: BTreeMap<String, BinaryFieldValue>,
    vectors: BTreeMap<String, BinaryVectorValue>,
}

impl From<&Doc> for BinaryDoc {
    fn from(doc: &Doc) -> Self {
        Self {
            pk: doc.get_pk().map(str::to_owned),
            score: doc.get_score(),
            internal_id: doc.doc_id(),
            fields: doc
                .fields()
                .iter()
                .map(|(name, value)| (name.clone(), BinaryFieldValue::from(value)))
                .collect(),
            vectors: doc
                .vectors()
                .iter()
                .map(|(name, value)| (name.clone(), BinaryVectorValue::from(value)))
                .collect(),
        }
    }
}

impl From<BinaryDoc> for Doc {
    fn from(doc: BinaryDoc) -> Self {
        Self::from_persisted_parts(
            doc.pk,
            doc.score,
            doc.internal_id,
            doc.fields
                .into_iter()
                .map(|(name, value)| (name, FieldValue::from(value)))
                .collect(),
            doc.vectors
                .into_iter()
                .map(|(name, value)| (name, VectorValue::from(value)))
                .collect(),
        )
    }
}

// The public document enums use an adjacently tagged representation to keep
// their JSON stable. Compact MessagePack structs omit field names, which makes
// a unit variant such as `Null` ambiguous in that representation. These
// snapshot-only enums use Serde's native binary enum encoding instead. Variant
// declaration order is part of format 4; changing it requires a format bump.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum BinaryFieldValue {
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
    Json(Value),
}

impl From<&FieldValue> for BinaryFieldValue {
    fn from(value: &FieldValue) -> Self {
        match value {
            FieldValue::Null => Self::Null,
            FieldValue::Binary(value) => Self::Binary(value.clone()),
            FieldValue::String(value) => Self::String(value.clone()),
            FieldValue::Bool(value) => Self::Bool(*value),
            FieldValue::Int32(value) => Self::Int32(*value),
            FieldValue::Int64(value) => Self::Int64(*value),
            FieldValue::Uint32(value) => Self::Uint32(*value),
            FieldValue::Uint64(value) => Self::Uint64(*value),
            FieldValue::Float(value) => Self::Float(*value),
            FieldValue::Double(value) => Self::Double(*value),
            FieldValue::ArrayBinary(value) => Self::ArrayBinary(value.clone()),
            FieldValue::ArrayString(value) => Self::ArrayString(value.clone()),
            FieldValue::ArrayBool(value) => Self::ArrayBool(value.clone()),
            FieldValue::ArrayInt32(value) => Self::ArrayInt32(value.clone()),
            FieldValue::ArrayInt64(value) => Self::ArrayInt64(value.clone()),
            FieldValue::ArrayUint32(value) => Self::ArrayUint32(value.clone()),
            FieldValue::ArrayUint64(value) => Self::ArrayUint64(value.clone()),
            FieldValue::ArrayFloat(value) => Self::ArrayFloat(value.clone()),
            FieldValue::ArrayDouble(value) => Self::ArrayDouble(value.clone()),
            FieldValue::Json(value) => Self::Json(value.clone()),
        }
    }
}

impl From<BinaryFieldValue> for FieldValue {
    fn from(value: BinaryFieldValue) -> Self {
        match value {
            BinaryFieldValue::Null => Self::Null,
            BinaryFieldValue::Binary(value) => Self::Binary(value),
            BinaryFieldValue::String(value) => Self::String(value),
            BinaryFieldValue::Bool(value) => Self::Bool(value),
            BinaryFieldValue::Int32(value) => Self::Int32(value),
            BinaryFieldValue::Int64(value) => Self::Int64(value),
            BinaryFieldValue::Uint32(value) => Self::Uint32(value),
            BinaryFieldValue::Uint64(value) => Self::Uint64(value),
            BinaryFieldValue::Float(value) => Self::Float(value),
            BinaryFieldValue::Double(value) => Self::Double(value),
            BinaryFieldValue::ArrayBinary(value) => Self::ArrayBinary(value),
            BinaryFieldValue::ArrayString(value) => Self::ArrayString(value),
            BinaryFieldValue::ArrayBool(value) => Self::ArrayBool(value),
            BinaryFieldValue::ArrayInt32(value) => Self::ArrayInt32(value),
            BinaryFieldValue::ArrayInt64(value) => Self::ArrayInt64(value),
            BinaryFieldValue::ArrayUint32(value) => Self::ArrayUint32(value),
            BinaryFieldValue::ArrayUint64(value) => Self::ArrayUint64(value),
            BinaryFieldValue::ArrayFloat(value) => Self::ArrayFloat(value),
            BinaryFieldValue::ArrayDouble(value) => Self::ArrayDouble(value),
            BinaryFieldValue::Json(value) => Self::Json(value),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum BinaryVectorValue {
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

impl From<&VectorValue> for BinaryVectorValue {
    fn from(value: &VectorValue) -> Self {
        match value {
            VectorValue::Binary32(value) => Self::Binary32(value.clone()),
            VectorValue::Binary64(value) => Self::Binary64(value.clone()),
            VectorValue::Fp16(value) => Self::Fp16(value.clone()),
            VectorValue::Fp32(value) => Self::Fp32(value.clone()),
            VectorValue::Fp64(value) => Self::Fp64(value.clone()),
            VectorValue::Int4(value) => Self::Int4(value.clone()),
            VectorValue::Int8(value) => Self::Int8(value.clone()),
            VectorValue::Int16(value) => Self::Int16(value.clone()),
            VectorValue::SparseFp16 { indices, values } => Self::SparseFp16 {
                indices: indices.clone(),
                values: values.clone(),
            },
            VectorValue::SparseFp32 { indices, values } => Self::SparseFp32 {
                indices: indices.clone(),
                values: values.clone(),
            },
        }
    }
}

impl From<BinaryVectorValue> for VectorValue {
    fn from(value: BinaryVectorValue) -> Self {
        match value {
            BinaryVectorValue::Binary32(value) => Self::Binary32(value),
            BinaryVectorValue::Binary64(value) => Self::Binary64(value),
            BinaryVectorValue::Fp16(value) => Self::Fp16(value),
            BinaryVectorValue::Fp32(value) => Self::Fp32(value),
            BinaryVectorValue::Fp64(value) => Self::Fp64(value),
            BinaryVectorValue::Int4(value) => Self::Int4(value),
            BinaryVectorValue::Int8(value) => Self::Int8(value),
            BinaryVectorValue::Int16(value) => Self::Int16(value),
            BinaryVectorValue::SparseFp16 { indices, values } => {
                Self::SparseFp16 { indices, values }
            }
            BinaryVectorValue::SparseFp32 { indices, values } => {
                Self::SparseFp32 { indices, values }
            }
        }
    }
}
