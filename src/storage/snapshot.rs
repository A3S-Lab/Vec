//! Atomic document/schema snapshots.

use super::manifest::{atomic_write, checksum, Manifest};
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

const SNAPSHOT_FORMAT_VERSION: u32 = 3;
pub(super) const MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snapshot {
    format_version: u32,
    generation: u64,
    revision: u64,
    schema: CollectionSchema,
    docs: Vec<Doc>,
}

pub fn write(
    root: &Path,
    schema: &CollectionSchema,
    docs: &[Doc],
    generation: u64,
    revision: u64,
    sync: bool,
) -> Result<u32> {
    if generation == 0 {
        return Err(Error::invalid_argument(
            "snapshot generation must be positive",
        ));
    }
    let snapshot = Snapshot {
        format_version: SNAPSHOT_FORMAT_VERSION,
        generation,
        revision,
        schema: schema.clone(),
        docs: docs.to_vec(),
    };
    let bytes = serde_json::to_vec(&snapshot)
        .map_err(|e| Error::internal(format!("serialize document snapshot: {e}")))?;
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if byte_len > MAX_SNAPSHOT_BYTES {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte storage limit"
        )));
    }
    let digest = checksum(&bytes);
    atomic_write(root, &relative_path(generation), &bytes, sync)?;
    Ok(digest)
}

pub fn read(root: &Path, manifest: &Manifest) -> Result<(CollectionSchema, Vec<Doc>)> {
    let path = root.join(relative_path(manifest.generation));
    let metadata = fs::metadata(&path)
        .map_err(|e| Error::internal(format!("read document snapshot metadata: {e}")))?;
    if metadata.len() > MAX_SNAPSHOT_BYTES {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte recovery limit"
        )));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        Error::resource_exhausted("document snapshot is too large for this platform")
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(&path)
        .map_err(|e| Error::internal(format!("open document snapshot: {e}")))?
        .take(MAX_SNAPSHOT_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| Error::internal(format!("read document snapshot: {e}")))?;
    let actual = checksum(&bytes);
    if manifest.docs_checksum != actual {
        return Err(Error::internal(format!(
            "document snapshot checksum mismatch: expected {}, got {}",
            manifest.docs_checksum, actual
        )));
    }
    let snapshot: Snapshot = serde_json::from_slice(&bytes)
        .map_err(|e| Error::internal(format!("parse document snapshot: {e}")))?;
    if snapshot.format_version != SNAPSHOT_FORMAT_VERSION {
        return Err(Error::new(
            crate::error::ErrorCode::NotSupported,
            format!(
                "unsupported document snapshot format version {}",
                snapshot.format_version
            ),
        ));
    }
    if snapshot.generation != manifest.generation {
        return Err(Error::internal(
            "snapshot generation does not match manifest",
        ));
    }
    if snapshot.revision != manifest.checkpoint_revision {
        return Err(Error::internal(
            "snapshot revision does not match manifest checkpoint revision",
        ));
    }
    if snapshot.schema.name != manifest.collection_name {
        return Err(Error::internal(
            "snapshot collection name does not match manifest",
        ));
    }
    if snapshot.schema.digest() != manifest.schema_digest {
        return Err(Error::internal("schema digest does not match manifest"));
    }
    Ok((snapshot.schema, snapshot.docs))
}

pub fn prune(root: &Path, keep_generation: u64) -> Result<()> {
    let directory = root.join("segments");
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(&directory)
        .map_err(|e| Error::internal(format!("read snapshot directory: {e}")))?
    {
        let entry = entry.map_err(|e| Error::internal(format!("read snapshot entry: {e}")))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(number) = name
            .strip_prefix("snapshot-")
            .and_then(|value| value.strip_suffix(".json"))
        else {
            continue;
        };
        if number
            .parse::<u64>()
            .is_ok_and(|generation| generation != keep_generation)
        {
            fs::remove_file(entry.path())
                .map_err(|e| Error::internal(format!("prune document snapshot: {e}")))?;
        }
    }
    Ok(())
}

fn relative_path(generation: u64) -> PathBuf {
    Path::new("segments").join(format!("snapshot-{generation:020}.json"))
}
