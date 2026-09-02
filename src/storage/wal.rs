//! Length-delimited, checksummed write-ahead log.

use super::fault::{FaultInjector, FaultPoint};
use super::manifest::sync_directory;
use crate::doc::Doc;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"A3VW";
const VERSION: u16 = 4;
const MIN_READABLE_VERSION: u16 = 3;
const HEADER_LEN: usize = 4 + 2 + 4 + 4;
const MAX_WAL_FRAME_BYTES: usize = 64 * 1024 * 1024;
const MAX_WAL_REPLAY_BYTES: u64 = 512 * 1024 * 1024;

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
    let payload = serde_json::to_vec(record)
        .map_err(|e| Error::internal(format!("serialize WAL record: {e}")))?;
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
        if total_bytes > MAX_WAL_REPLAY_BYTES {
            return Err(Error::resource_exhausted(format!(
                "WAL replay exceeds the {MAX_WAL_REPLAY_BYTES}-byte recovery limit"
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
            let record: WalRecord = serde_json::from_slice(payload)
                .map_err(|e| Error::internal(format!("decode WAL record: {e}")))?;
            if version < VERSION && matches!(record.operation, WalOperation::SchemaOnly { .. }) {
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
