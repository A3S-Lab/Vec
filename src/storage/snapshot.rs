//! Atomic document/schema snapshots.

mod codec;

use super::fault::{FaultInjector, FaultPoint};
use super::manifest::{
    atomic_write_with_faults, checksum, sync_directory, AtomicWriteKind, Manifest,
};
use crate::doc::{Doc, DocumentMap};
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use codec::{BinarySnapshot, DeltaSnapshot};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

/// A document set snapshot encoding can walk without owning a second body.
pub(crate) trait SnapshotDocs {
    fn len(&self) -> usize;
    fn for_each<'a>(&'a self, visit: &mut dyn FnMut(&'a Doc));
}

impl SnapshotDocs for [Doc] {
    fn len(&self) -> usize {
        <[Doc]>::len(self)
    }

    fn for_each<'a>(&'a self, visit: &mut dyn FnMut(&'a Doc)) {
        for doc in self {
            visit(doc);
        }
    }
}

impl<const N: usize> SnapshotDocs for [Doc; N] {
    fn len(&self) -> usize {
        N
    }

    fn for_each<'a>(&'a self, visit: &mut dyn FnMut(&'a Doc)) {
        for doc in self {
            visit(doc);
        }
    }
}

impl SnapshotDocs for Vec<Doc> {
    fn len(&self) -> usize {
        self.len()
    }

    fn for_each<'a>(&'a self, visit: &mut dyn FnMut(&'a Doc)) {
        self.as_slice().for_each(visit);
    }
}

impl SnapshotDocs for DocumentMap {
    fn len(&self) -> usize {
        self.len()
    }

    fn for_each<'a>(&'a self, visit: &mut dyn FnMut(&'a Doc)) {
        for doc in self.values() {
            visit(doc.as_ref());
        }
    }
}

const LEGACY_SNAPSHOT_FORMAT_VERSION: u32 = 3;
const SNAPSHOT_FORMAT_VERSION: u32 = 4;
const DELTA_SNAPSHOT_FORMAT_VERSION: u32 = 5;

#[cfg(test)]
pub(crate) use crate::storage_ceilings::DEFAULT_SNAPSHOT_BYTES as MAX_SNAPSHOT_BYTES;

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
        MAX_SNAPSHOT_BYTES,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_with_faults(
    root: &Path,
    schema: &CollectionSchema,
    docs: &(impl SnapshotDocs + ?Sized),
    generation: u64,
    revision: u64,
    sync: bool,
    faults: &FaultInjector,
    max_snapshot_bytes: u64,
) -> Result<u32> {
    if generation == 0 {
        return Err(Error::invalid_argument(
            "snapshot generation must be positive",
        ));
    }
    let bytes =
        codec::encode_binary_snapshot(SNAPSHOT_FORMAT_VERSION, generation, revision, schema, docs)
            .map_err(|error| Error::internal(format!("serialize document snapshot: {error}")))?;
    write_bytes(
        root,
        &binary_relative_path(generation),
        &bytes,
        sync,
        faults,
        max_snapshot_bytes,
    )
}

/// Writes a format-5 delta against a non-empty format-4-or-newer base.
///
/// `previous` is the logical document set of `base_generation`. Only removed
/// primary keys and upserted documents are stored; unchanged bodies stay in
/// the base file.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_delta_with_faults(
    root: &Path,
    schema: &CollectionSchema,
    docs: &(impl SnapshotDocs + ?Sized),
    previous: &[Doc],
    generation: u64,
    revision: u64,
    base_generation: u64,
    base_checksum: u32,
    sync: bool,
    faults: &FaultInjector,
    max_snapshot_bytes: u64,
) -> Result<u32> {
    if generation == 0 || base_generation == 0 || base_generation == generation {
        return Err(Error::invalid_argument(
            "snapshot delta requires a distinct positive base generation",
        ));
    }
    let (removed, upserted) = document_delta(previous, docs);
    let snapshot = DeltaSnapshot::new(
        DELTA_SNAPSHOT_FORMAT_VERSION,
        generation,
        revision,
        schema,
        base_generation,
        base_checksum,
        removed,
        &upserted,
    );
    let bytes = rmp_serde::to_vec(&snapshot)
        .map_err(|error| Error::internal(format!("serialize document snapshot delta: {error}")))?;
    write_bytes(
        root,
        &binary_relative_path(generation),
        &bytes,
        sync,
        faults,
        max_snapshot_bytes,
    )
}

pub(super) fn documents_unchanged(previous: &[Doc], next: &(impl SnapshotDocs + ?Sized)) -> bool {
    if previous.len() != next.len() {
        return false;
    }
    let (removed, upserted) = document_delta(previous, next);
    removed.is_empty() && upserted.is_empty()
}

fn document_delta(
    previous: &[Doc],
    next: &(impl SnapshotDocs + ?Sized),
) -> (Vec<String>, Vec<Doc>) {
    let mut previous_by_pk = BTreeMap::<&str, &Doc>::new();
    for doc in previous {
        if let Some(pk) = doc.get_pk() {
            previous_by_pk.insert(pk, doc);
        }
    }
    let mut next_by_pk = BTreeMap::<&str, &Doc>::new();
    next.for_each(&mut |doc| {
        if let Some(pk) = doc.get_pk() {
            next_by_pk.insert(pk, doc);
        }
    });
    let mut removed = Vec::new();
    for pk in previous_by_pk.keys() {
        if !next_by_pk.contains_key(pk) {
            removed.push((*pk).to_string());
        }
    }
    let mut upserted = Vec::new();
    for (pk, doc) in &next_by_pk {
        match previous_by_pk.get(pk) {
            Some(existing) if *existing == *doc => {}
            _ => upserted.push((*doc).clone()),
        }
    }
    (removed, upserted)
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
        MAX_SNAPSHOT_BYTES,
    )
}

fn write_bytes(
    root: &Path,
    relative_path: &Path,
    bytes: &[u8],
    sync: bool,
    faults: &FaultInjector,
    max_snapshot_bytes: u64,
) -> Result<u32> {
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if byte_len > max_snapshot_bytes {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {max_snapshot_bytes}-byte storage limit"
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

pub fn read(
    root: &Path,
    manifest: &Manifest,
    max_snapshot_bytes: u64,
) -> Result<(CollectionSchema, Vec<Doc>)> {
    match manifest.format_version {
        LEGACY_SNAPSHOT_FORMAT_VERSION => read_legacy(root, manifest, max_snapshot_bytes),
        SNAPSHOT_FORMAT_VERSION => read_binary(root, manifest, max_snapshot_bytes),
        version => Err(Error::not_supported(format!(
            "unsupported document snapshot format version {version}"
        ))),
    }
}

fn read_binary(
    root: &Path,
    manifest: &Manifest,
    max_snapshot_bytes: u64,
) -> Result<(CollectionSchema, Vec<Doc>)> {
    let bytes = read_bytes(
        root,
        &binary_relative_path(manifest.generation),
        manifest,
        max_snapshot_bytes,
    )?;
    let mut seen = HashSet::new();
    resolve_snapshot(
        root,
        &bytes,
        manifest.generation,
        Some(manifest),
        max_snapshot_bytes,
        &mut seen,
    )
}

fn resolve_snapshot(
    root: &Path,
    bytes: &[u8],
    expected_generation: u64,
    manifest: Option<&Manifest>,
    max_snapshot_bytes: u64,
    seen: &mut HashSet<u64>,
) -> Result<(CollectionSchema, Vec<Doc>)> {
    let format_version = peek_format_version(bytes)?;
    match format_version {
        SNAPSHOT_FORMAT_VERSION => {
            let snapshot = decode_full(bytes)?;
            let (version, generation, revision, schema, docs) = snapshot.into_parts();
            if version != SNAPSHOT_FORMAT_VERSION {
                return Err(Error::not_supported(format!(
                    "unsupported document snapshot format version {version}"
                )));
            }
            if generation != expected_generation {
                return Err(Error::internal(
                    "document snapshot generation does not match its file",
                ));
            }
            if !seen.insert(generation) {
                return Err(Error::internal("document snapshot delta cycle"));
            }
            if let Some(manifest) = manifest {
                validate_metadata(manifest, generation, revision, &schema)?;
            }
            Ok((schema, docs))
        }
        DELTA_SNAPSHOT_FORMAT_VERSION => {
            let snapshot = decode_delta(bytes)?;
            let (
                version,
                generation,
                revision,
                schema,
                base_generation,
                base_checksum,
                removed,
                upserted,
            ) = snapshot.into_parts();
            if version != DELTA_SNAPSHOT_FORMAT_VERSION {
                return Err(Error::not_supported(format!(
                    "unsupported document snapshot format version {version}"
                )));
            }
            if generation != expected_generation {
                return Err(Error::internal(
                    "document snapshot generation does not match its file",
                ));
            }
            if !seen.insert(generation) {
                return Err(Error::internal("document snapshot delta cycle"));
            }
            if let Some(manifest) = manifest {
                validate_metadata(manifest, generation, revision, &schema)?;
            }
            if base_generation == 0 || base_generation == generation {
                return Err(Error::internal(
                    "document snapshot delta base generation is invalid",
                ));
            }
            let base_bytes =
                read_generation_bytes(root, base_generation, base_checksum, max_snapshot_bytes)?;
            let (base_schema, base_docs) = resolve_snapshot(
                root,
                &base_bytes,
                base_generation,
                None,
                max_snapshot_bytes,
                seen,
            )?;
            if base_schema.digest() != schema.digest() {
                return Err(Error::internal(
                    "snapshot delta base schema does not match the tip",
                ));
            }
            let docs = apply_delta(base_docs, &removed, upserted)?;
            Ok((schema, docs))
        }
        version => Err(Error::not_supported(format!(
            "unsupported document snapshot format version {version}"
        ))),
    }
}

fn decode_full(bytes: &[u8]) -> Result<BinarySnapshot> {
    let mut decoder = rmp_serde::Deserializer::new(Cursor::new(bytes));
    let snapshot = BinarySnapshot::deserialize(&mut decoder)
        .map_err(|error| Error::internal(format!("parse binary document snapshot: {error}")))?;
    reject_trailing(bytes, decoder.position())?;
    Ok(snapshot)
}

fn decode_delta(bytes: &[u8]) -> Result<DeltaSnapshot> {
    let mut decoder = rmp_serde::Deserializer::new(Cursor::new(bytes));
    let snapshot = DeltaSnapshot::deserialize(&mut decoder)
        .map_err(|error| Error::internal(format!("parse document snapshot delta: {error}")))?;
    reject_trailing(bytes, decoder.position())?;
    Ok(snapshot)
}

fn reject_trailing(bytes: &[u8], position: u64) -> Result<()> {
    let encoded_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if position != encoded_len {
        return Err(Error::internal(
            "parse binary document snapshot: trailing payload",
        ));
    }
    Ok(())
}

fn peek_format_version(bytes: &[u8]) -> Result<u32> {
    let mut cursor = Cursor::new(bytes);
    rmp::decode::read_array_len(&mut cursor)
        .map_err(|error| Error::internal(format!("parse document snapshot header: {error}")))?;
    rmp::decode::read_int(&mut cursor)
        .map_err(|error| Error::internal(format!("parse document snapshot version: {error}")))
}

fn apply_delta(base_docs: Vec<Doc>, removed: &[String], upserted: Vec<Doc>) -> Result<Vec<Doc>> {
    let mut merged = BTreeMap::<String, Doc>::new();
    for doc in base_docs {
        let pk = doc
            .get_pk()
            .ok_or_else(|| Error::internal("snapshot document has no primary key"))?;
        merged.insert(pk.to_string(), doc);
    }
    for pk in removed {
        merged.remove(pk);
    }
    for doc in upserted {
        let pk = doc
            .get_pk()
            .ok_or_else(|| Error::internal("snapshot document has no primary key"))?;
        merged.insert(pk.to_string(), doc);
    }
    Ok(merged.into_values().collect())
}

fn read_legacy(
    root: &Path,
    manifest: &Manifest,
    max_snapshot_bytes: u64,
) -> Result<(CollectionSchema, Vec<Doc>)> {
    let bytes = read_bytes(
        root,
        &legacy_relative_path(manifest.generation),
        manifest,
        max_snapshot_bytes,
    )?;
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

fn read_bytes(
    root: &Path,
    relative_path: &Path,
    manifest: &Manifest,
    max_snapshot_bytes: u64,
) -> Result<Vec<u8>> {
    let path = root.join(relative_path);
    let metadata = fs::metadata(&path)
        .map_err(|error| Error::internal(format!("read document snapshot metadata: {error}")))?;
    if metadata.len() > max_snapshot_bytes {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {max_snapshot_bytes}-byte recovery limit"
        )));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        Error::resource_exhausted("document snapshot is too large for this platform")
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(&path)
        .map_err(|error| Error::internal(format!("open document snapshot: {error}")))?
        .take(max_snapshot_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| Error::internal(format!("read document snapshot: {error}")))?;
    let actual_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if actual_len > max_snapshot_bytes {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {max_snapshot_bytes}-byte recovery limit"
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

fn read_generation_bytes(
    root: &Path,
    generation: u64,
    expected_checksum: u32,
    max_snapshot_bytes: u64,
) -> Result<Vec<u8>> {
    let path = root.join(binary_relative_path(generation));
    let metadata = fs::metadata(&path)
        .map_err(|error| Error::internal(format!("read document snapshot metadata: {error}")))?;
    if metadata.len() > max_snapshot_bytes {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {max_snapshot_bytes}-byte recovery limit"
        )));
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| Error::internal(format!("open document snapshot: {error}")))?
        .take(max_snapshot_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| Error::internal(format!("read document snapshot: {error}")))?;
    let actual_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("document snapshot exceeds u64 bytes"))?;
    if actual_len > max_snapshot_bytes {
        return Err(Error::resource_exhausted(format!(
            "document snapshot exceeds the {max_snapshot_bytes}-byte recovery limit"
        )));
    }
    let actual = checksum(&bytes);
    if actual != expected_checksum {
        return Err(Error::internal(format!(
            "document snapshot checksum mismatch: expected {expected_checksum}, got {actual}"
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
    keep_checksum: u32,
    max_snapshot_bytes: u64,
    faults: &FaultInjector,
) -> Result<()> {
    let directory = root.join("segments");
    if !directory.exists() {
        return Ok(());
    }
    let keep = retained_generations(root, keep_generation, keep_checksum, max_snapshot_bytes)?;
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
        if !keep.contains(&generation) {
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

fn retained_generations(
    root: &Path,
    keep_generation: u64,
    keep_checksum: u32,
    max_snapshot_bytes: u64,
) -> Result<BTreeSet<u64>> {
    let mut keep = BTreeSet::new();
    let mut generation = keep_generation;
    let mut expected_checksum = keep_checksum;
    loop {
        if !keep.insert(generation) {
            return Err(Error::internal("document snapshot delta cycle"));
        }
        let bytes = read_generation_bytes(root, generation, expected_checksum, max_snapshot_bytes)?;
        match peek_format_version(&bytes)? {
            SNAPSHOT_FORMAT_VERSION => return Ok(keep),
            DELTA_SNAPSHOT_FORMAT_VERSION => {
                let delta = decode_delta(&bytes)?;
                generation = delta.base_generation();
                expected_checksum = delta.base_checksum();
            }
            version => {
                return Err(Error::not_supported(format!(
                    "unsupported document snapshot format version {version}"
                )))
            }
        }
    }
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

#[cfg(test)]
#[allow(clippy::needless_borrows_for_generic_args)]
mod tests {
    use super::{
        binary_relative_path, prune_with_faults, read, snapshot_generation, validate_metadata,
        write_with_faults, MAX_SNAPSHOT_BYTES, SNAPSHOT_FORMAT_VERSION,
    };
    use crate::error::ErrorCode;
    use crate::schema::{CollectionSchema, FieldSchema};
    use crate::storage::fault::FaultInjector;
    use crate::storage::manifest::Manifest;
    use crate::types::DataType;
    use std::fs;
    use tempfile::tempdir;

    fn fixture_schema() -> CollectionSchema {
        CollectionSchema::builder("fixture")
            .add_field(FieldSchema::new("tag", DataType::String, false, 0).expect("field"))
            .build()
            .expect("schema")
    }

    #[test]
    fn unsupported_snapshot_format_and_generation_parsing_fail_closed() {
        let temporary = tempdir().expect("temp");
        let mut manifest = Manifest::new("fixture", "digest");
        manifest.generation = 1;
        manifest.format_version = SNAPSHOT_FORMAT_VERSION + 9;
        assert!(read(temporary.path(), &manifest, MAX_SNAPSHOT_BYTES)
            .expect_err("unsupported")
            .message
            .contains("unsupported"));

        assert_eq!(
            snapshot_generation("snapshot-00000000000000000007.bin"),
            Some(7)
        );
        assert_eq!(
            snapshot_generation("snapshot-00000000000000000008.json"),
            Some(8)
        );
        assert_eq!(snapshot_generation("wal-1.bin"), None);
        assert_eq!(snapshot_generation("snapshot-not-a-number.bin"), None);
    }

    #[test]
    fn snapshot_write_read_and_metadata_validation_fail_closed() {
        let temporary = tempdir().expect("temp");
        let root = temporary.path();
        let schema = fixture_schema();
        assert_eq!(
            write_with_faults(
                root,
                &schema,
                &[],
                0,
                1,
                true,
                &FaultInjector::default(),
                MAX_SNAPSHOT_BYTES
            )
            .expect_err("generation 0")
            .code,
            ErrorCode::InvalidArgument
        );

        let digest = write_with_faults(
            root,
            &schema,
            &[],
            1,
            1,
            true,
            &FaultInjector::default(),
            MAX_SNAPSHOT_BYTES,
        )
        .expect("write");
        let mut manifest = Manifest::new("fixture", &schema.digest());
        manifest.generation = 1;
        manifest.checkpoint_revision = 1;
        manifest.format_version = SNAPSHOT_FORMAT_VERSION;
        manifest.docs_checksum = digest;
        let (loaded_schema, docs) = read(root, &manifest, MAX_SNAPSHOT_BYTES).expect("read");
        assert_eq!(loaded_schema.name, "fixture");
        assert!(docs.is_empty());

        manifest.docs_checksum ^= 1;
        assert!(read(root, &manifest, MAX_SNAPSHOT_BYTES)
            .expect_err("checksum")
            .message
            .contains("checksum"));

        let mut bad = Manifest::new("other", &schema.digest());
        bad.generation = 2;
        bad.checkpoint_revision = 1;
        assert!(validate_metadata(&bad, 1, 1, &schema)
            .expect_err("generation")
            .message
            .contains("generation"));
        bad.generation = 1;
        bad.checkpoint_revision = 9;
        assert!(validate_metadata(&bad, 1, 1, &schema)
            .expect_err("revision")
            .message
            .contains("revision"));
        bad.checkpoint_revision = 1;
        bad.collection_name = "other".into();
        assert!(validate_metadata(&bad, 1, 1, &schema)
            .expect_err("name")
            .message
            .contains("collection name"));
        bad.collection_name = "fixture".into();
        bad.schema_digest = "wrong".into();
        assert!(validate_metadata(&bad, 1, 1, &schema)
            .expect_err("digest")
            .message
            .contains("schema digest"));

        // Second generation so prune removes the first artifact.
        let second = write_with_faults(
            root,
            &schema,
            &[],
            2,
            2,
            true,
            &FaultInjector::default(),
            MAX_SNAPSHOT_BYTES,
        )
        .expect("second");
        prune_with_faults(
            root,
            2,
            second,
            MAX_SNAPSHOT_BYTES,
            &FaultInjector::default(),
        )
        .expect("prune");
        assert!(!root.join(binary_relative_path(1)).exists());
        assert!(root.join(binary_relative_path(2)).exists());
        // Non-snapshot junk in segments is ignored.
        fs::write(root.join("segments").join("notes.txt"), b"keep").expect("junk");
        prune_with_faults(
            root,
            2,
            second,
            MAX_SNAPSHOT_BYTES,
            &FaultInjector::default(),
        )
        .expect("ignore junk");
        assert!(root.join("segments").join("notes.txt").exists());
    }
}
