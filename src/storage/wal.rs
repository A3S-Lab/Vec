//! Length-delimited, checksummed write-ahead log.

use crate::doc::Doc;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"A3VW";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 4 + 2 + 4 + 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalRecord {
    Insert { docs: Vec<Doc> },
    Update { docs: Vec<Doc> },
    Upsert { docs: Vec<Doc> },
    Delete { ids: Vec<String> },
    Schema { schema: crate::schema::CollectionSchema },
}

impl WalRecord {
    pub fn operation_name(&self) -> &'static str {
        match self {
            Self::Insert { .. } => "insert",
            Self::Update { .. } => "update",
            Self::Upsert { .. } => "upsert",
            Self::Delete { .. } => "delete",
            Self::Schema { .. } => "schema",
        }
    }
}

pub fn segment_path(root: &Path, sequence: u64) -> PathBuf {
    root.join("wal").join(format!("wal-{sequence:020}.bin"))
}

pub fn append(root: &Path, sequence: u64, record: &WalRecord, sync: bool) -> Result<u64> {
    fs::create_dir_all(root.join("wal"))
        .map_err(|e| Error::internal(format!("create WAL directory: {e}")))?;
    let payload = serde_json::to_vec(record)
        .map_err(|e| Error::internal(format!("serialize WAL record: {e}")))?;
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
        .append(true)
        .open(segment_path(root, sequence))
        .map_err(|e| Error::internal(format!("open WAL segment: {e}")))?;
    file.write_all(&header)
        .and_then(|_| file.write_all(&payload))
        .map_err(|e| Error::internal(format!("append WAL record: {e}")))?;
    if sync {
        file.sync_all()
            .map_err(|e| Error::internal(format!("sync WAL segment: {e}")))?;
    } else {
        file.sync_data()
            .map_err(|e| Error::internal(format!("sync WAL data: {e}")))?;
    }
    Ok((HEADER_LEN + payload.len()) as u64)
}

/// Replays all records in the inclusive sequence range. A truncated final
/// frame is treated as a power-loss tail; an invalid complete frame is
/// reported as corruption.
pub fn replay(root: &Path, first: u64, last: u64) -> Result<Vec<WalRecord>> {
    let mut records = Vec::new();
    if first > last {
        return Ok(records);
    }
    for sequence in first..=last {
        let path = segment_path(root, sequence);
        if !path.exists() {
            continue;
        }
        let mut file = File::open(&path)
            .map_err(|e| Error::internal(format!("open WAL segment: {e}")))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| Error::internal(format!("read WAL segment: {e}")))?;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let remaining = bytes.len() - offset;
            if remaining < HEADER_LEN {
                if sequence == last {
                    break;
                }
                return Err(Error::internal("truncated WAL header in completed segment"));
            }
            if &bytes[offset..offset + 4] != MAGIC {
                return Err(Error::internal("WAL magic mismatch"));
            }
            let version = u16::from_le_bytes([bytes[offset + 4], bytes[offset + 5]]);
            if version != VERSION {
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
            let expected_crc = u32::from_le_bytes([
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
                bytes[offset + 13],
            ]);
            let frame_end = offset.saturating_add(HEADER_LEN).saturating_add(len);
            if frame_end > bytes.len() {
                if sequence == last {
                    break;
                }
                return Err(Error::internal("truncated WAL payload in completed segment"));
            }
            let payload = &bytes[offset + HEADER_LEN..frame_end];
            if crc32fast::hash(payload) != expected_crc {
                return Err(Error::internal("WAL checksum mismatch"));
            }
            let record = serde_json::from_slice(payload)
                .map_err(|e| Error::internal(format!("decode WAL record: {e}")))?;
            records.push(record);
            offset = frame_end;
        }
    }
    Ok(records)
}

pub fn prune(root: &Path, through: u64) -> Result<()> {
    let directory = root.join("wal");
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)
        .map_err(|e| Error::internal(format!("read WAL directory: {e}")))?
    {
        let entry = entry.map_err(|e| Error::internal(format!("read WAL entry: {e}")))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(number) = name.strip_prefix("wal-").and_then(|v| v.strip_suffix(".bin")) else { continue };
        if number.parse::<u64>().ok().is_some_and(|seq| seq <= through) {
            fs::remove_file(entry.path())
                .map_err(|e| Error::internal(format!("prune WAL segment: {e}")))?;
        }
    }
    Ok(())
}
