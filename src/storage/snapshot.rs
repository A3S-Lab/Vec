//! Atomic document/schema snapshots.

mod codec;

use super::fault::{FaultInjector, FaultPoint};
use super::manifest::{
    atomic_write_with_faults, checksum, sync_directory, AtomicWriteKind, Manifest,
};
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use codec::BinarySnapshot;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

const LEGACY_SNAPSHOT_FORMAT_VERSION: u32 = 3;
const SNAPSHOT_FORMAT_VERSION: u32 = 4;
pub(super) const MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySnapshot {
    format_version: u32,
    generation: u64,
    revision: u64,
    schema: CollectionSchema,
    docs: Vec<Doc>,
}

#[cfg(test)]
pub fn write(
    root: &Path,
    schema: &CollectionSchema,
    docs: &[Doc],
    generation: u64,
    revision: u64,
    sync: bool,
) -> Result<u32> {
    write_with_faults(
        root,
        schema,
        docs,
        generation,
        revision,
        sync,
        &FaultInjector::default(),
    )
}

pub(super) fn write_with_faults(
    root: &Path,
    schema: &CollectionSchema,
    docs: &[Doc],
    generation: u64,
    revision: u64,
    sync: bool,
    faults: &FaultInjector,
) -> Result<u32> {
    if generation == 0 {
        return Err(Error::invalid_argument(
            "snapshot generation must be positive",
        ));
    }
    let snapshot = BinarySnapshot::new(SNAPSHOT_FORMAT_VERSION, generation, revision, schema, docs);
    let bytes = rmp_serde::to_vec(&snapshot)
        .map_err(|error| Error::internal(format!("serialize document snapshot: {error}")))?;
    write_bytes(
        root,
        &binary_relative_path(generation),
        &bytes,
        sync,
        faults,
    )
}

#[cfg(test)]
pub(super) fn write_legacy(
    root: &Path,
    schema: &CollectionSchema,
    docs: &[Doc],
    generation: u64,
    revision: u64,
) -> Result<u32> {
    let snapshot = LegacySnapshot {
        format_version: LEGACY_SNAPSHOT_FORMAT_VERSION,
        generation,
        revision,
        schema: schema.clone(),
        docs: docs.to_vec(),
    };
    let bytes = serde_json::to_vec(&snapshot)
        .map_err(|error| Error::internal(format!("serialize legacy snapshot: {error}")))?;
    write_bytes(
        root,
        &legacy_relative_path(generation),
        &bytes,
        true,
        &FaultInjector::default(),
    )
}

fn write_bytes(
    root: &Path,
    relative_path: &Path,
    bytes: &[u8],
    sync: bool,
    faults: &FaultInjector,
) -> Result<u32> {
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if byte_len > MAX_SNAPSHOT_BYTES {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte storage limit"
        )));
    }
    let digest = checksum(bytes);
    atomic_write_with_faults(
        root,
        relative_path,
        bytes,
        sync,
        AtomicWriteKind::Snapshot,
        faults,
    )?;
    Ok(digest)
}

pub fn read(root: &Path, manifest: &Manifest) -> Result<(CollectionSchema, Vec<Doc>)> {
    match manifest.format_version {
        LEGACY_SNAPSHOT_FORMAT_VERSION => read_legacy(root, manifest),
        SNAPSHOT_FORMAT_VERSION => read_binary(root, manifest),
        version => Err(Error::not_supported(format!(
            "unsupported document snapshot format version {version}"
        ))),
    }
}

fn read_binary(root: &Path, manifest: &Manifest) -> Result<(CollectionSchema, Vec<Doc>)> {
    let bytes = read_bytes(root, &binary_relative_path(manifest.generation), manifest)?;
    let mut decoder = rmp_serde::Deserializer::new(Cursor::new(bytes.as_slice()));
    let snapshot = BinarySnapshot::deserialize(&mut decoder)
        .map_err(|error| Error::internal(format!("parse binary document snapshot: {error}")))?;
    let encoded_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if decoder.position() != encoded_len {
        return Err(Error::internal(
            "parse binary document snapshot: trailing payload",
        ));
    }
    let (format_version, generation, revision, schema, docs) = snapshot.into_parts();
    if format_version != SNAPSHOT_FORMAT_VERSION {
        return Err(Error::not_supported(format!(
            "unsupported document snapshot format version {format_version}"
        )));
    }
    validate_metadata(manifest, generation, revision, &schema)?;
    Ok((schema, docs))
}

fn read_legacy(root: &Path, manifest: &Manifest) -> Result<(CollectionSchema, Vec<Doc>)> {
    let bytes = read_bytes(root, &legacy_relative_path(manifest.generation), manifest)?;
    let snapshot: LegacySnapshot = serde_json::from_slice(&bytes)
        .map_err(|error| Error::internal(format!("parse legacy document snapshot: {error}")))?;
    if snapshot.format_version != LEGACY_SNAPSHOT_FORMAT_VERSION {
        return Err(Error::not_supported(format!(
            "unsupported document snapshot format version {}",
            snapshot.format_version
        )));
    }
    validate_metadata(
        manifest,
        snapshot.generation,
        snapshot.revision,
        &snapshot.schema,
    )?;
    Ok((snapshot.schema, snapshot.docs))
}

fn read_bytes(root: &Path, relative_path: &Path, manifest: &Manifest) -> Result<Vec<u8>> {
    let path = root.join(relative_path);
    let metadata = fs::metadata(&path)
        .map_err(|error| Error::internal(format!("read document snapshot metadata: {error}")))?;
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
        .map_err(|error| Error::internal(format!("open document snapshot: {error}")))?
        .take(MAX_SNAPSHOT_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| Error::internal(format!("read document snapshot: {error}")))?;
    let actual_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if actual_len > MAX_SNAPSHOT_BYTES {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte recovery limit"
        )));
    }
    let actual = checksum(&bytes);
    if manifest.docs_checksum != actual {
        return Err(Error::internal(format!(
            "document snapshot checksum mismatch: expected {}, got {actual}",
            manifest.docs_checksum
        )));
    }
    Ok(bytes)
}

fn validate_metadata(
    manifest: &Manifest,
    generation: u64,
    revision: u64,
    schema: &CollectionSchema,
) -> Result<()> {
    if generation != manifest.generation {
        return Err(Error::internal(
            "snapshot generation does not match manifest",
        ));
    }
    if revision != manifest.checkpoint_revision {
        return Err(Error::internal(
            "snapshot revision does not match manifest checkpoint revision",
        ));
    }
    if schema.name != manifest.collection_name {
        return Err(Error::internal(
            "snapshot collection name does not match manifest",
        ));
    }
    if schema.digest() != manifest.schema_digest {
        return Err(Error::internal("schema digest does not match manifest"));
    }
    Ok(())
}

pub(super) fn prune_with_faults(
    root: &Path,
    keep_generation: u64,
    faults: &FaultInjector,
) -> Result<()> {
    let directory = root.join("segments");
    if !directory.exists() {
        return Ok(());
    }
    let mut removed = false;
    for entry in fs::read_dir(&directory)
        .map_err(|error| Error::internal(format!("read snapshot directory: {error}")))?
    {
        let entry =
            entry.map_err(|error| Error::internal(format!("read snapshot entry: {error}")))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(generation) = snapshot_generation(name) else {
            continue;
        };
        if generation != keep_generation {
            faults.hit(FaultPoint::SnapshotPruneBeforeRemove)?;
            fs::remove_file(entry.path())
                .map_err(|error| Error::internal(format!("prune document snapshot: {error}")))?;
            removed = true;
            faults.hit(FaultPoint::SnapshotPruneAfterRemove)?;
        }
    }
    if removed {
        sync_directory(&directory)?;
        faults.hit(FaultPoint::SnapshotPruneDirectorySynced)?;
    }
    Ok(())
}

fn snapshot_generation(name: &str) -> Option<u64> {
    let encoded = name.strip_prefix("snapshot-")?;
    encoded
        .strip_suffix(".bin")
        .or_else(|| encoded.strip_suffix(".json"))?
        .parse()
        .ok()
}

pub(super) fn binary_relative_path(generation: u64) -> PathBuf {
    Path::new("segments").join(format!("snapshot-{generation:020}.bin"))
}

pub(super) fn legacy_relative_path(generation: u64) -> PathBuf {
    Path::new("segments").join(format!("snapshot-{generation:020}.json"))
}
