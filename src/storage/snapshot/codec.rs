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

pub(super) fn encode_binary_snapshot(
    format_version: u32,
    generation: u64,
    revision: u64,
    schema: &CollectionSchema,
    docs: &(impl super::SnapshotDocs + ?Sized),
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let mut encoded = Vec::with_capacity(docs.len());
    docs.for_each(&mut |doc| encoded.push(BorrowedDoc::from_doc(doc)));
    let snapshot = BorrowedSnapshot {
        format_version,
        generation,
        revision,
        schema,
        docs: &encoded,
    };
    rmp_serde::to_vec(&snapshot)
}

#[derive(Serialize)]
struct BorrowedSnapshot<'a> {
    format_version: u32,
    generation: u64,
    revision: u64,
    schema: &'a CollectionSchema,
    docs: &'a [BorrowedDoc<'a>],
}

#[derive(Serialize)]
struct BorrowedDoc<'a> {
    pk: Option<&'a str>,
    score: f32,
    internal_id: Option<u64>,
    fields: BTreeMap<&'a str, BorrowedField<'a>>,
    vectors: BTreeMap<&'a str, BorrowedVector<'a>>,
}

impl<'a> BorrowedDoc<'a> {
    fn from_doc(doc: &'a crate::doc::Doc) -> Self {
        Self {
            pk: doc.get_pk(),
            score: doc.get_score(),
            internal_id: doc.doc_id(),
            fields: doc
                .fields()
                .iter()
                .map(|(name, value)| (name.as_str(), BorrowedField::from(value)))
                .collect(),
            vectors: doc
                .vectors()
                .iter()
                .map(|(name, value)| (name.as_str(), BorrowedVector::from(value)))
                .collect(),
        }
    }
}

#[derive(Serialize)]
enum BorrowedField<'a> {
    Null,
    Binary(&'a [u8]),
    String(&'a str),
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Uint32(u32),
    Uint64(u64),
    Float(f32),
    Double(f64),
    ArrayBinary(&'a [Vec<u8>]),
    ArrayString(&'a [String]),
    ArrayBool(&'a [bool]),
    ArrayInt32(&'a [i32]),
    ArrayInt64(&'a [i64]),
    ArrayUint32(&'a [u32]),
    ArrayUint64(&'a [u64]),
    ArrayFloat(&'a [f32]),
    ArrayDouble(&'a [f64]),
    Json(&'a Value),
}

impl<'a> From<&'a FieldValue> for BorrowedField<'a> {
    fn from(value: &'a FieldValue) -> Self {
        match value {
            FieldValue::Null => Self::Null,
            FieldValue::Binary(value) => Self::Binary(value),
            FieldValue::String(value) => Self::String(value),
            FieldValue::Bool(value) => Self::Bool(*value),
            FieldValue::Int32(value) => Self::Int32(*value),
            FieldValue::Int64(value) => Self::Int64(*value),
            FieldValue::Uint32(value) => Self::Uint32(*value),
            FieldValue::Uint64(value) => Self::Uint64(*value),
            FieldValue::Float(value) => Self::Float(*value),
            FieldValue::Double(value) => Self::Double(*value),
            FieldValue::ArrayBinary(value) => Self::ArrayBinary(value),
            FieldValue::ArrayString(value) => Self::ArrayString(value),
            FieldValue::ArrayBool(value) => Self::ArrayBool(value),
            FieldValue::ArrayInt32(value) => Self::ArrayInt32(value),
            FieldValue::ArrayInt64(value) => Self::ArrayInt64(value),
            FieldValue::ArrayUint32(value) => Self::ArrayUint32(value),
            FieldValue::ArrayUint64(value) => Self::ArrayUint64(value),
            FieldValue::ArrayFloat(value) => Self::ArrayFloat(value),
            FieldValue::ArrayDouble(value) => Self::ArrayDouble(value),
            FieldValue::Json(value) => Self::Json(value),
        }
    }
}

#[derive(Serialize)]
enum BorrowedVector<'a> {
    Binary32(&'a [u8]),
    Binary64(&'a [u8]),
    Fp16(&'a [u16]),
    Fp32(&'a [f32]),
    Fp64(&'a [f64]),
    Int4(&'a [i8]),
    Int8(&'a [i8]),
    Int16(&'a [i16]),
    SparseFp16 {
        indices: &'a [u32],
        values: &'a [u16],
    },
    SparseFp32 {
        indices: &'a [u32],
        values: &'a [f32],
    },
}

impl<'a> From<&'a VectorValue> for BorrowedVector<'a> {
    fn from(value: &'a VectorValue) -> Self {
        match value {
            VectorValue::Binary32(value) => Self::Binary32(value),
            VectorValue::Binary64(value) => Self::Binary64(value),
            VectorValue::Fp16(value) => Self::Fp16(value),
            VectorValue::Fp32(value) => Self::Fp32(value),
            VectorValue::Fp64(value) => Self::Fp64(value),
            VectorValue::Int4(value) => Self::Int4(value),
            VectorValue::Int8(value) => Self::Int8(value),
            VectorValue::Int16(value) => Self::Int16(value),
            VectorValue::SparseFp16 { indices, values } => Self::SparseFp16 { indices, values },
            VectorValue::SparseFp32 { indices, values } => Self::SparseFp32 { indices, values },
        }
    }
}

impl BinarySnapshot {
    #[cfg(test)]
    pub(super) fn new(
        format_version: u32,
        generation: u64,
        revision: u64,
        schema: &CollectionSchema,
        docs: &(impl super::SnapshotDocs + ?Sized),
    ) -> Self {
        let mut encoded = Vec::with_capacity(docs.len());
        docs.for_each(&mut |doc| encoded.push(BinaryDoc::from(doc)));
        Self {
            format_version,
            generation,
            revision,
            schema: schema.clone(),
            docs: encoded,
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

/// Format 5 stores only the documents that changed since `base_generation`.
/// Unchanged bodies remain in the base snapshot file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DeltaSnapshot {
    format_version: u32,
    generation: u64,
    revision: u64,
    schema: CollectionSchema,
    base_generation: u64,
    base_checksum: u32,
    removed: Vec<String>,
    upserted: Vec<BinaryDoc>,
}

impl DeltaSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        format_version: u32,
        generation: u64,
        revision: u64,
        schema: &CollectionSchema,
        base_generation: u64,
        base_checksum: u32,
        removed: Vec<String>,
        upserted: &[Doc],
    ) -> Self {
        Self {
            format_version,
            generation,
            revision,
            schema: schema.clone(),
            base_generation,
            base_checksum,
            removed,
            upserted: upserted.iter().map(BinaryDoc::from).collect(),
        }
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        u32,
        u64,
        u64,
        CollectionSchema,
        u64,
        u32,
        Vec<String>,
        Vec<Doc>,
    ) {
        (
            self.format_version,
            self.generation,
            self.revision,
            self.schema,
            self.base_generation,
            self.base_checksum,
            self.removed,
            self.upserted.into_iter().map(Doc::from).collect(),
        )
    }

    pub(super) fn base_generation(&self) -> u64 {
        self.base_generation
    }

    pub(super) fn base_checksum(&self) -> u32 {
        self.base_checksum
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

#[cfg(test)]
mod encode_tests {
    use super::{encode_binary_snapshot, BinarySnapshot};
    use crate::doc::Doc;
    use crate::schema::{CollectionSchema, FieldSchema};
    use crate::types::DataType;

    #[test]
    fn borrowed_encoder_matches_owned_snapshot_bytes() {
        let schema = CollectionSchema::builder("snapshot-bytes")
            .add_field(FieldSchema::new("body", DataType::String, true, 0).expect("body field"))
            .add_field(
                FieldSchema::new("embedding", DataType::VectorFp32, false, 4)
                    .expect("vector field"),
            )
            .build()
            .expect("schema");
        let mut with_field = Doc::with_pk("doc-a").expect("pk");
        with_field.add_string("body", "red").expect("field");
        with_field
            .add_vector_f32("embedding", &[1.0, -2.0, 0.5, 4.0])
            .expect("vector");
        let mut vector_only = Doc::with_pk("doc-b").expect("pk");
        vector_only
            .add_vector_f32("embedding", &[0.0, 1.0, 0.0, 0.0])
            .expect("vector");
        let docs = [with_field, vector_only];
        let owned = rmp_serde::to_vec(&BinarySnapshot::new(4, 7, 9, &schema, &docs))
            .expect("owned snapshot");
        let borrowed = encode_binary_snapshot(4, 7, 9, &schema, &docs).expect("borrowed snapshot");
        assert_eq!(borrowed, owned);
        assert_eq!(borrowed.first().copied(), Some(0x95));
        assert_eq!(borrowed.get(1).copied(), Some(0x04));
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

#[cfg(test)]
mod tests {
    use super::BinarySnapshot;
    use crate::doc::{Doc, FieldValue, VectorValue};
    use crate::schema::{CollectionSchema, FieldSchema};
    use crate::types::DataType;
    use serde_json::json;

    fn empty_schema() -> CollectionSchema {
        CollectionSchema::builder("codec-roundtrip")
            .add_field(FieldSchema::new("title", DataType::String, false, 0).expect("field"))
            .build()
            .expect("schema")
    }

    #[test]
    fn binary_snapshot_roundtrips_every_field_and_vector_variant() {
        let mut doc = Doc::with_pk("pk").expect("pk");
        doc.set_score(0.5).expect("score");
        doc.set_field_null("null").expect("null");
        doc.add_binary("bin", &[1, 2]).expect("bin");
        doc.add_string("s", "x").expect("s");
        doc.add_bool("b", true).expect("b");
        doc.add_i32("i32", -1).expect("i32");
        doc.add_i64("i64", -2).expect("i64");
        doc.add_u32("u32", 3).expect("u32");
        doc.add_u64("u64", 4).expect("u64");
        doc.add_f32("f32", 1.5).expect("f32");
        doc.add_f64("f64", 2.5).expect("f64");
        doc.add_array_binary("ab", &[vec![9]]).expect("ab");
        doc.add_array_string("as", &["a"]).expect("as");
        doc.add_array_bool("abo", &[true]).expect("abo");
        doc.add_array_i32("ai32", &[1]).expect("ai32");
        doc.add_array_i64("ai64", &[2]).expect("ai64");
        doc.add_array_u32("au32", &[3]).expect("au32");
        doc.add_array_u64("au64", &[4]).expect("au64");
        doc.add_array_f32("af32", &[1.25]).expect("af32");
        doc.add_array_f64("af64", &[2.5]).expect("af64");
        doc.set_field_value("json", FieldValue::Json(json!({"k": 1})))
            .expect("json");

        doc.set_vector_value("v_bin32", VectorValue::Binary32(vec![0xff; 4]))
            .expect("v_bin32");
        doc.set_vector_value("v_bin64", VectorValue::Binary64(vec![0xaa; 8]))
            .expect("v_bin64");
        doc.set_vector_value("v_fp16", VectorValue::Fp16(vec![0x3c00]))
            .expect("v_fp16");
        doc.set_vector_value("v_fp32", VectorValue::Fp32(vec![1.0, 0.0]))
            .expect("v_fp32");
        doc.set_vector_value("v_fp64", VectorValue::Fp64(vec![1.0, 0.0]))
            .expect("v_fp64");
        doc.set_vector_value("v_i4", VectorValue::Int4(vec![1, -1]))
            .expect("v_i4");
        doc.set_vector_value("v_i8", VectorValue::Int8(vec![2, -2]))
            .expect("v_i8");
        doc.set_vector_value("v_i16", VectorValue::Int16(vec![3, -3]))
            .expect("v_i16");
        doc.set_vector_value(
            "v_s16",
            VectorValue::SparseFp16 {
                indices: vec![0, 2],
                values: vec![0x3c00, 0x4000],
            },
        )
        .expect("v_s16");
        doc.set_vector_value(
            "v_s32",
            VectorValue::SparseFp32 {
                indices: vec![1],
                values: vec![0.5],
            },
        )
        .expect("v_s32");

        let snapshot = BinarySnapshot::new(4, 1, 2, &empty_schema(), &[doc.clone()]);
        let (format, generation, revision, _schema, docs) = snapshot.into_parts();
        assert_eq!((format, generation, revision), (4, 1, 2));
        assert_eq!(docs.len(), 1);
        let restored = &docs[0];
        assert_eq!(restored.get_pk(), Some("pk"));
        assert!((restored.get_score() - 0.5).abs() < f32::EPSILON);
        assert_eq!(restored.field("s"), Some(&FieldValue::String("x".into())));
        assert_eq!(
            restored.field("json"),
            Some(&FieldValue::Json(json!({"k": 1})))
        );
        assert_eq!(
            restored.vector("v_fp32"),
            Some(&VectorValue::Fp32(vec![1.0, 0.0]))
        );
        assert_eq!(
            restored.vector("v_s32"),
            Some(&VectorValue::SparseFp32 {
                indices: vec![1],
                values: vec![0.5],
            })
        );
        assert_eq!(restored.fields().len(), doc.fields().len());
        assert_eq!(restored.vectors().len(), doc.vectors().len());
    }
}
