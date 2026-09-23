//! Length-delimited, checksummed write-ahead log.

use super::fault::{FaultInjector, FaultPoint};
use super::manifest::sync_directory;
use crate::doc::Doc;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"A3VW";
const VERSION: u16 = 5;
const JSON_VERSION: u16 = 4;
const MIN_READABLE_VERSION: u16 = 3;
const HEADER_LEN: usize = 4 + 2 + 4 + 4;
const MAX_WAL_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[cfg(test)]
pub(crate) use crate::storage_ceilings::DEFAULT_WAL_REPLAY_BYTES as MAX_WAL_REPLAY_BYTES;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalRecord {
    pub revision: u64,
    pub operation_id: u64,
    pub operation: WalOperation,
}

impl WalRecord {
    pub fn new(revision: u64, operation: WalOperation) -> Result<Self> {
        if revision == 0 {
            return Err(Error::invalid_argument("WAL revision must be positive"));
        }
        Ok(Self {
            revision,
            operation_id: revision,
            operation,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOperation {
    Insert {
        docs: Vec<Doc>,
    },
    Update {
        docs: Vec<Doc>,
    },
    Upsert {
        docs: Vec<Doc>,
    },
    Delete {
        ids: Vec<String>,
    },
    Schema {
        schema: crate::schema::CollectionSchema,
        docs: Vec<Doc>,
    },
    /// Persists a schema revision when the document set is unchanged.
    ///
    /// Index creation and removal only mutate the schema. Keeping that
    /// operation separate from [`WalOperation::Schema`] avoids serializing
    /// the complete document set into one WAL frame (which is bounded by
    /// `MAX_WAL_FRAME_BYTES`).
    SchemaOnly {
        schema: crate::schema::CollectionSchema,
    },
}

pub fn segment_path(root: &Path, sequence: u64) -> PathBuf {
    root.join("wal").join(format!("wal-{sequence:020}.bin"))
}

#[cfg(test)]
pub fn append(
    root: &Path,
    sequence: u64,
    committed_bytes: u64,
    record: &WalRecord,
    sync: bool,
) -> Result<u64> {
    append_with_faults(
        root,
        sequence,
        committed_bytes,
        record,
        sync,
        &FaultInjector::default(),
    )
}

pub(super) fn append_with_faults(
    root: &Path,
    sequence: u64,
    committed_bytes: u64,
    record: &WalRecord,
    sync: bool,
    faults: &FaultInjector,
) -> Result<u64> {
    fs::create_dir_all(root.join("wal"))
        .map_err(|e| Error::internal(format!("create WAL directory: {e}")))?;
    let payload = encode_record(record)?;
    if payload.len() > MAX_WAL_FRAME_BYTES {
        return Err(Error::resource_exhausted(format!(
            "WAL record exceeds the {MAX_WAL_FRAME_BYTES}-byte frame limit"
        )));
    }
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| Error::resource_exhausted("WAL record exceeds 4 GiB"))?;
    let checksum = crc32fast::hash(&payload);
    let mut header = [0_u8; HEADER_LEN];
    header[..4].copy_from_slice(MAGIC);
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
    header[6..10].copy_from_slice(&payload_len.to_le_bytes());
    header[10..14].copy_from_slice(&checksum.to_le_bytes());

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(segment_path(root, sequence))
        .map_err(|e| Error::internal(format!("open WAL segment: {e}")))?;
    let actual_bytes = file
        .metadata()
        .map_err(|e| Error::internal(format!("read WAL segment metadata: {e}")))?
        .len();
    if actual_bytes < committed_bytes {
        return Err(Error::internal(format!(
            "WAL segment is shorter than the committed boundary: expected at least {committed_bytes} bytes, got {actual_bytes}"
        )));
    }
    file.set_len(committed_bytes)
        .and_then(|()| file.seek(SeekFrom::Start(committed_bytes)).map(|_| ()))
        .map_err(|e| Error::internal(format!("prepare WAL append boundary: {e}")))?;
    faults.hit(FaultPoint::WalPrepared)?;
    file.write_all(&header)
        .map_err(|e| Error::internal(format!("append WAL record: {e}")))?;
    faults.hit(FaultPoint::WalHeaderWritten)?;
    file.write_all(&payload)
        .map_err(|e| Error::internal(format!("append WAL record: {e}")))?;
    faults.hit(FaultPoint::WalPayloadWritten)?;
    if sync {
        // Park before the durability sync, while the caller still has not
        // published the new revision. The fail-once point stays after sync_all.
        faults.stall(FaultPoint::WalSynced);
        file.sync_all()
            .map_err(|e| Error::internal(format!("sync WAL segment: {e}")))?;
        faults.hit(FaultPoint::WalSynced)?;
    }
    u64::try_from(HEADER_LEN + payload.len())
        .map_err(|_| Error::resource_exhausted("WAL frame size exceeds this platform"))
}

/// Replays committed records in the inclusive sequence range. Bytes beyond
/// `active_committed_bytes` in the active segment are an uncommitted tail and
/// are deliberately ignored.
pub fn replay(
    root: &Path,
    first: u64,
    last: u64,
    active_committed_bytes: u64,
    max_wal_replay_bytes: u64,
) -> Result<Vec<WalRecord>> {
    let mut records = Vec::new();
    let mut total_bytes = 0_u64;
    if first > last {
        return Ok(records);
    }
    for sequence in first..=last {
        let path = segment_path(root, sequence);
        if !path.exists() {
            continue;
        }
        let file =
            File::open(&path).map_err(|e| Error::internal(format!("open WAL segment: {e}")))?;
        let actual_bytes = file
            .metadata()
            .map_err(|e| Error::internal(format!("read WAL segment metadata: {e}")))?
            .len();
        let committed_bytes = if sequence == last {
            active_committed_bytes
        } else {
            actual_bytes
        };
        if actual_bytes < committed_bytes {
            return Err(Error::internal(format!(
                "WAL segment is shorter than the committed boundary: expected {committed_bytes} bytes, got {actual_bytes}"
            )));
        }
        total_bytes = total_bytes.saturating_add(committed_bytes);
        if total_bytes > max_wal_replay_bytes {
            return Err(Error::resource_exhausted(format!(
                "WAL replay exceeds the {max_wal_replay_bytes}-byte recovery limit"
            )));
        }
        let capacity = usize::try_from(committed_bytes)
            .map_err(|_| Error::resource_exhausted("WAL segment is too large for this platform"))?;
        let mut bytes = Vec::with_capacity(capacity);
        file.take(committed_bytes)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::internal(format!("read WAL segment: {e}")))?;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let remaining = bytes.len() - offset;
            if remaining < HEADER_LEN {
                return Err(Error::internal(
                    "truncated WAL header inside committed data",
                ));
            }
            if &bytes[offset..offset + 4] != MAGIC {
                return Err(Error::internal("WAL magic mismatch"));
            }
            let version = u16::from_le_bytes([bytes[offset + 4], bytes[offset + 5]]);
            if !(MIN_READABLE_VERSION..=VERSION).contains(&version) {
                return Err(Error::new(
                    crate::error::ErrorCode::NotSupported,
                    format!("unsupported WAL frame version {version}"),
                ));
            }
            let len = u32::from_le_bytes([
                bytes[offset + 6],
                bytes[offset + 7],
                bytes[offset + 8],
                bytes[offset + 9],
            ]) as usize;
            if len > MAX_WAL_FRAME_BYTES {
                return Err(Error::resource_exhausted(format!(
                    "WAL frame exceeds the {MAX_WAL_FRAME_BYTES}-byte recovery limit"
                )));
            }
            let expected_crc = u32::from_le_bytes([
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
                bytes[offset + 13],
            ]);
            let frame_end = offset.saturating_add(HEADER_LEN).saturating_add(len);
            if frame_end > bytes.len() {
                return Err(Error::internal(
                    "truncated WAL payload inside committed data",
                ));
            }
            let payload = &bytes[offset + HEADER_LEN..frame_end];
            if crc32fast::hash(payload) != expected_crc {
                return Err(Error::internal("WAL checksum mismatch"));
            }
            let record = decode_record(version, payload)?;
            if version < JSON_VERSION && matches!(record.operation, WalOperation::SchemaOnly { .. })
            {
                return Err(Error::new(
                    crate::error::ErrorCode::NotSupported,
                    "schema-only WAL operations require frame version 4",
                ));
            }
            validate_record(&record)?;
            records.push(record);
            offset = frame_end;
        }
    }
    Ok(records)
}

fn encode_record(record: &WalRecord) -> Result<Vec<u8>> {
    rmp_serde::to_vec(&PackedRecord::from(record))
        .map_err(|error| Error::internal(format!("serialize WAL record: {error}")))
}

fn decode_record(version: u16, payload: &[u8]) -> Result<WalRecord> {
    if version >= VERSION {
        let packed: PackedRecord = rmp_serde::from_slice(payload)
            .map_err(|error| Error::internal(format!("decode WAL record: {error}")))?;
        Ok(packed.into_record())
    } else {
        serde_json::from_slice(payload)
            .map_err(|error| Error::internal(format!("decode WAL record: {error}")))
    }
}

#[cfg(test)]
pub fn legacy_json_from_frame(version: u16, payload: &[u8]) -> Result<Vec<u8>> {
    let record = decode_record(version, payload)?;
    serde_json::to_vec(&record)
        .map_err(|error| Error::internal(format!("encode legacy WAL: {error}")))
}

/// `MessagePack` form of a WAL record.
///
/// `Doc` keeps its JSON-adjacent field tags for version-4 replay. Those tags
/// do not round-trip through `MessagePack`, so version 5 stores externally
/// tagged copies and rebuilds the document on read.
#[derive(Serialize, Deserialize)]
struct PackedRecord {
    revision: u64,
    operation_id: u64,
    operation: PackedOperation,
}

#[derive(Serialize, Deserialize)]
enum PackedOperation {
    Insert {
        docs: Vec<PackedDoc>,
    },
    Update {
        docs: Vec<PackedDoc>,
    },
    Upsert {
        docs: Vec<PackedDoc>,
    },
    Delete {
        ids: Vec<String>,
    },
    Schema {
        schema: crate::schema::CollectionSchema,
        docs: Vec<PackedDoc>,
    },
    SchemaOnly {
        schema: crate::schema::CollectionSchema,
    },
}

#[derive(Serialize, Deserialize)]
struct PackedDoc {
    pk: Option<String>,
    score: f32,
    doc_id: Option<u64>,
    fields: BTreeMap<String, PackedField>,
    vectors: BTreeMap<String, PackedVector>,
}

#[derive(Serialize, Deserialize)]
enum PackedField {
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
    Json(serde_json::Value),
}

#[derive(Serialize, Deserialize)]
enum PackedVector {
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

impl From<&WalRecord> for PackedRecord {
    fn from(record: &WalRecord) -> Self {
        Self {
            revision: record.revision,
            operation_id: record.operation_id,
            operation: PackedOperation::from(&record.operation),
        }
    }
}

impl PackedRecord {
    fn into_record(self) -> WalRecord {
        WalRecord {
            revision: self.revision,
            operation_id: self.operation_id,
            operation: self.operation.into_operation(),
        }
    }
}

impl From<&WalOperation> for PackedOperation {
    fn from(operation: &WalOperation) -> Self {
        match operation {
            WalOperation::Insert { docs } => Self::Insert {
                docs: docs.iter().map(PackedDoc::from).collect(),
            },
            WalOperation::Update { docs } => Self::Update {
                docs: docs.iter().map(PackedDoc::from).collect(),
            },
            WalOperation::Upsert { docs } => Self::Upsert {
                docs: docs.iter().map(PackedDoc::from).collect(),
            },
            WalOperation::Delete { ids } => Self::Delete { ids: ids.clone() },
            WalOperation::Schema { schema, docs } => Self::Schema {
                schema: schema.clone(),
                docs: docs.iter().map(PackedDoc::from).collect(),
            },
            WalOperation::SchemaOnly { schema } => Self::SchemaOnly {
                schema: schema.clone(),
            },
        }
    }
}

impl PackedOperation {
    fn into_operation(self) -> WalOperation {
        match self {
            Self::Insert { docs } => WalOperation::Insert {
                docs: docs.into_iter().map(PackedDoc::into_doc).collect(),
            },
            Self::Update { docs } => WalOperation::Update {
                docs: docs.into_iter().map(PackedDoc::into_doc).collect(),
            },
            Self::Upsert { docs } => WalOperation::Upsert {
                docs: docs.into_iter().map(PackedDoc::into_doc).collect(),
            },
            Self::Delete { ids } => WalOperation::Delete { ids },
            Self::Schema { schema, docs } => WalOperation::Schema {
                schema,
                docs: docs.into_iter().map(PackedDoc::into_doc).collect(),
            },
            Self::SchemaOnly { schema } => WalOperation::SchemaOnly { schema },
        }
    }
}

impl From<&crate::doc::Doc> for PackedDoc {
    fn from(doc: &crate::doc::Doc) -> Self {
        Self {
            pk: doc.get_pk().map(str::to_string),
            score: doc.get_score(),
            doc_id: doc.doc_id(),
            fields: doc
                .fields()
                .iter()
                .map(|(name, value)| (name.clone(), PackedField::from(value)))
                .collect(),
            vectors: doc
                .vectors()
                .iter()
                .map(|(name, value)| (name.clone(), PackedVector::from(value)))
                .collect(),
        }
    }
}

impl PackedDoc {
    fn into_doc(self) -> crate::doc::Doc {
        crate::doc::Doc::from_persisted_parts(
            self.pk,
            self.score,
            self.doc_id,
            self.fields
                .into_iter()
                .map(|(name, value)| (name, value.into_field()))
                .collect(),
            self.vectors
                .into_iter()
                .map(|(name, value)| (name, value.into_vector()))
                .collect(),
        )
    }
}

impl From<&crate::doc::FieldValue> for PackedField {
    fn from(value: &crate::doc::FieldValue) -> Self {
        use crate::doc::FieldValue;
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

impl PackedField {
    fn into_field(self) -> crate::doc::FieldValue {
        use crate::doc::FieldValue;
        match self {
            Self::Null => FieldValue::Null,
            Self::Binary(value) => FieldValue::Binary(value),
            Self::String(value) => FieldValue::String(value),
            Self::Bool(value) => FieldValue::Bool(value),
            Self::Int32(value) => FieldValue::Int32(value),
            Self::Int64(value) => FieldValue::Int64(value),
            Self::Uint32(value) => FieldValue::Uint32(value),
            Self::Uint64(value) => FieldValue::Uint64(value),
            Self::Float(value) => FieldValue::Float(value),
            Self::Double(value) => FieldValue::Double(value),
            Self::ArrayBinary(value) => FieldValue::ArrayBinary(value),
            Self::ArrayString(value) => FieldValue::ArrayString(value),
            Self::ArrayBool(value) => FieldValue::ArrayBool(value),
            Self::ArrayInt32(value) => FieldValue::ArrayInt32(value),
            Self::ArrayInt64(value) => FieldValue::ArrayInt64(value),
            Self::ArrayUint32(value) => FieldValue::ArrayUint32(value),
            Self::ArrayUint64(value) => FieldValue::ArrayUint64(value),
            Self::ArrayFloat(value) => FieldValue::ArrayFloat(value),
            Self::ArrayDouble(value) => FieldValue::ArrayDouble(value),
            Self::Json(value) => FieldValue::Json(value),
        }
    }
}

impl From<&crate::doc::VectorValue> for PackedVector {
    fn from(value: &crate::doc::VectorValue) -> Self {
        use crate::doc::VectorValue;
        match value {
            VectorValue::Binary32(value) => Self::Binary32(value.clone()),
            VectorValue::Binary64(value) => Self::Binary64(value.clone()),
            VectorValue::Fp16(value) => Self::Fp16(value.clone()),
            VectorValue::Fp64(value) => Self::Fp64(value.clone()),
            VectorValue::Fp32(value) => Self::Fp32(value.clone()),
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

impl PackedVector {
    fn into_vector(self) -> crate::doc::VectorValue {
        use crate::doc::VectorValue;
        match self {
            Self::Binary32(value) => VectorValue::Binary32(value),
            Self::Binary64(value) => VectorValue::Binary64(value),
            Self::Fp16(value) => VectorValue::Fp16(value),
            Self::Fp32(value) => VectorValue::Fp32(value),
            Self::Fp64(value) => VectorValue::Fp64(value),
            Self::Int4(value) => VectorValue::Int4(value),
            Self::Int8(value) => VectorValue::Int8(value),
            Self::Int16(value) => VectorValue::Int16(value),
            Self::SparseFp16 { indices, values } => VectorValue::SparseFp16 { indices, values },
            Self::SparseFp32 { indices, values } => VectorValue::SparseFp32 { indices, values },
        }
    }
}

fn validate_record(record: &WalRecord) -> Result<()> {
    if record.revision == 0 {
        return Err(Error::internal("WAL record revision must be positive"));
    }
    if record.operation_id != record.revision {
        return Err(Error::internal(
            "WAL operation identity does not match its revision",
        ));
    }
    Ok(())
}

pub(super) fn prune_with_faults(root: &Path, through: u64, faults: &FaultInjector) -> Result<()> {
    let directory = root.join("wal");
    if !directory.exists() {
        return Ok(());
    }
    let mut removed = false;
    for entry in
        fs::read_dir(&directory).map_err(|e| Error::internal(format!("read WAL directory: {e}")))?
    {
        let entry = entry.map_err(|e| Error::internal(format!("read WAL entry: {e}")))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(number) = name
            .strip_prefix("wal-")
            .and_then(|v| v.strip_suffix(".bin"))
        else {
            continue;
        };
        if number.parse::<u64>().is_ok_and(|seq| seq <= through) {
            faults.hit(FaultPoint::WalPruneBeforeRemove)?;
            fs::remove_file(entry.path())
                .map_err(|e| Error::internal(format!("prune WAL segment: {e}")))?;
            removed = true;
            faults.hit(FaultPoint::WalPruneAfterRemove)?;
        }
    }
    if removed {
        sync_directory(&directory)?;
        faults.hit(FaultPoint::WalPruneDirectorySynced)?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::{
        append, prune_with_faults, replay, segment_path, validate_record, WalOperation, WalRecord,
        HEADER_LEN, MAGIC, MAX_WAL_REPLAY_BYTES, VERSION,
    };
    use crate::doc::Doc;
    use crate::error::ErrorCode;
    use crate::storage::fault::FaultInjector;
    use std::fs;
    use tempfile::tempdir;

    fn insert_record(revision: u64) -> WalRecord {
        let doc = Doc::with_pk(format!("doc-{revision}")).expect("pk");
        WalRecord::new(revision, WalOperation::Insert { docs: vec![doc] }).expect("record")
    }

    #[test]
    fn wal_record_and_replay_reject_corrupt_and_inconsistent_frames() {
        assert_eq!(
            WalRecord::new(0, WalOperation::Delete { ids: vec![] })
                .expect_err("revision 0")
                .code,
            ErrorCode::InvalidArgument
        );

        let temporary = tempdir().expect("temp");
        let root = temporary.path();
        let record = insert_record(1);
        let written = append(root, 1, 0, &record, true).expect("append");
        assert!(written > HEADER_LEN as u64);
        let replayed = replay(root, 1, 1, written, MAX_WAL_REPLAY_BYTES).expect("replay");
        assert_eq!(replayed, vec![record.clone()]);
        assert!(replay(root, 2, 1, 0, MAX_WAL_REPLAY_BYTES)
            .expect("empty range")
            .is_empty());

        let mut bad = record.clone();
        bad.operation_id = 99;
        assert!(validate_record(&bad).is_err());
        bad.revision = 0;
        bad.operation_id = 0;
        assert!(validate_record(&bad).is_err());

        let path = segment_path(root, 2);
        fs::create_dir_all(path.parent().expect("wal dir")).expect("mkdir");
        fs::write(&path, b"XXXX").expect("short");
        assert!(replay(root, 2, 2, 4, MAX_WAL_REPLAY_BYTES)
            .expect_err("truncated")
            .message
            .contains("truncated"));

        let mut frame = Vec::new();
        frame.extend_from_slice(b"BAD!");
        frame.extend_from_slice(&VERSION.to_le_bytes());
        frame.extend_from_slice(&4u32.to_le_bytes());
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.extend_from_slice(b"dead");
        fs::write(&path, &frame).expect("magic");
        assert!(replay(root, 2, 2, frame.len() as u64, MAX_WAL_REPLAY_BYTES)
            .expect_err("magic")
            .message
            .contains("magic"));

        let payload = serde_json::to_vec(&record).expect("json");
        let mut good = Vec::new();
        good.extend_from_slice(MAGIC);
        good.extend_from_slice(&VERSION.to_le_bytes());
        good.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        good.extend_from_slice(&0u32.to_le_bytes()); // wrong crc
        good.extend_from_slice(&payload);
        fs::write(&path, &good).expect("crc");
        assert!(replay(root, 2, 2, good.len() as u64, MAX_WAL_REPLAY_BYTES)
            .expect_err("checksum")
            .message
            .contains("checksum"));

        prune_with_faults(root, 1, &FaultInjector::default()).expect("prune");
        assert!(!segment_path(root, 1).exists());
        prune_with_faults(root, 1, &FaultInjector::default()).expect("idempotent");
    }

    #[test]
    fn append_rejects_segment_shorter_than_committed_boundary() {
        let temporary = tempdir().expect("temp");
        let root = temporary.path();
        let record = insert_record(1);
        let written = append(root, 1, 0, &record, false).expect("append");
        let error =
            append(root, 1, written + 8, &insert_record(2), false).expect_err("committed boundary");
        assert_eq!(error.code, ErrorCode::InternalError);
        assert!(error.message.contains("shorter than the committed"));
    }
}
