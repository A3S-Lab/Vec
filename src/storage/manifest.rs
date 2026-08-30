//! Versioned manifest and atomic JSON metadata writes.

use super::fault::{FaultInjector, FaultPoint};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::fs::File;

pub const FORMAT_VERSION: u32 = 3;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub(super) enum AtomicWriteKind {
    Manifest,
    Snapshot,
}

impl AtomicWriteKind {
    const fn written(self) -> FaultPoint {
        match self {
            Self::Manifest => FaultPoint::ManifestWritten,
            Self::Snapshot => FaultPoint::SnapshotWritten,
        }
    }

    const fn synced(self) -> FaultPoint {
        match self {
            Self::Manifest => FaultPoint::ManifestSynced,
            Self::Snapshot => FaultPoint::SnapshotSynced,
        }
    }

    const fn renamed(self) -> FaultPoint {
        match self {
            Self::Manifest => FaultPoint::ManifestRenamed,
            Self::Snapshot => FaultPoint::SnapshotRenamed,
        }
    }

    const fn directory_synced(self) -> FaultPoint {
        match self {
            Self::Manifest => FaultPoint::ManifestDirectorySynced,
            Self::Snapshot => FaultPoint::SnapshotDirectorySynced,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub collection_name: String,
    pub schema_digest: String,
    pub generation: u64,
    pub revision: u64,
    pub checkpoint_revision: u64,
    pub wal_active_seq: u64,
    pub wal_checkpoint_seq: u64,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
    pub docs_checksum: u32,
}

impl Manifest {
    pub fn new(collection_name: impl Into<String>, schema_digest: impl Into<String>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            collection_name: collection_name.into(),
            schema_digest: schema_digest.into(),
            generation: 0,
            revision: 0,
            checkpoint_revision: 0,
            wal_active_seq: 0,
            wal_checkpoint_seq: 0,
            wal_ops_since_checkpoint: 0,
            wal_bytes_since_checkpoint: 0,
            docs_checksum: 0,
        }
    }
}

pub fn read(path: &Path) -> Result<Manifest> {
    let manifest_path = path.join("manifest.json");
    let metadata = fs::metadata(&manifest_path)
        .map_err(|e| Error::internal(format!("read manifest metadata: {e}")))?;
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(Error::resource_exhausted(format!(
            "manifest exceeds the {MAX_MANIFEST_BYTES}-byte recovery limit"
        )));
    }
    let raw =
        fs::read(&manifest_path).map_err(|e| Error::internal(format!("read manifest: {e}")))?;
    let manifest: Manifest = serde_json::from_slice(&raw)
        .map_err(|e| Error::internal(format!("parse manifest: {e}")))?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(Error::new(
            crate::error::ErrorCode::NotSupported,
            format!(
                "unsupported a3s-vec format version {}",
                manifest.format_version
            ),
        ));
    }
    if manifest.generation == 0 {
        return Err(Error::internal("manifest generation must be positive"));
    }
    if manifest.checkpoint_revision > manifest.revision {
        return Err(Error::internal(
            "manifest checkpoint revision exceeds its current revision",
        ));
    }
    if manifest.wal_checkpoint_seq >= manifest.wal_active_seq {
        return Err(Error::internal(
            "manifest active WAL sequence must follow its checkpoint sequence",
        ));
    }
    Ok(manifest)
}

pub(super) fn write_with_faults(
    path: &Path,
    manifest: &Manifest,
    sync: bool,
    faults: &FaultInjector,
) -> Result<()> {
    let raw = serde_json::to_vec_pretty(manifest)
        .map_err(|e| Error::internal(format!("serialize manifest: {e}")))?;
    let byte_len = u64::try_from(raw.len())
        .map_err(|_| Error::resource_exhausted("manifest exceeds u64 bytes"))?;
    if byte_len > MAX_MANIFEST_BYTES {
        return Err(Error::resource_exhausted(format!(
            "manifest exceeds the {MAX_MANIFEST_BYTES}-byte storage limit"
        )));
    }
    atomic_write_with_faults(
        path,
        Path::new("manifest.json"),
        &raw,
        sync,
        AtomicWriteKind::Manifest,
        faults,
    )
}

#[cfg(test)]
pub fn atomic_write(root: &Path, relative: &Path, bytes: &[u8], sync: bool) -> Result<()> {
    atomic_write_with_faults(
        root,
        relative,
        bytes,
        sync,
        AtomicWriteKind::Manifest,
        &FaultInjector::default(),
    )
}

pub(super) fn atomic_write_with_faults(
    root: &Path,
    relative: &Path,
    bytes: &[u8],
    sync: bool,
    kind: AtomicWriteKind,
    faults: &FaultInjector,
) -> Result<()> {
    let target = root.join(relative);
    let parent = target
        .parent()
        .ok_or_else(|| Error::internal("storage target has no parent directory"))?;
    fs::create_dir_all(parent)
        .map_err(|e| Error::internal(format!("create storage directory: {e}")))?;
    let file_name = target
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("data");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{stamp}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|e| Error::internal(format!("create temporary data file: {e}")))?;
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(Error::internal(format!("write data file: {error}")));
    }
    // An injected failure deliberately leaves the temporary file behind: a
    // real process crash would not execute cleanup either, and recovery must
    // ignore unpublished candidates.
    faults.hit(kind.written())?;
    if sync {
        if let Err(error) = file.sync_all() {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(Error::internal(format!("sync data file: {error}")));
        }
        faults.hit(kind.synced())?;
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, &target) {
        let _ = fs::remove_file(&temporary);
        return Err(Error::internal(format!("publish data file: {error}")));
    }
    faults.hit(kind.renamed())?;
    if sync {
        sync_directory(parent)?;
        faults.hit(kind.directory_synced())?;
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn sync_directory(path: &Path) -> Result<()> {
    let directory =
        File::open(path).map_err(|e| Error::internal(format!("open storage directory: {e}")))?;
    directory
        .sync_all()
        .map_err(|e| Error::internal(format!("sync storage directory: {e}")))
}

#[cfg(windows)]
pub(super) fn sync_directory(_path: &Path) -> Result<()> {
    // Rust's Windows `File::sync_all` calls `FlushFileBuffers`, which requires
    // a handle with GENERIC_WRITE access. Directories cannot be opened that
    // way through the safe standard-library API, so attempting to flush a
    // FILE_FLAG_BACKUP_SEMANTICS handle fails with ERROR_ACCESS_DENIED. The
    // temporary file itself has already been synced before the same-volume
    // atomic replacement. Do not turn a successful publication into a false
    // durability error by issuing an unsupported directory flush.
    Ok(())
}

pub fn checksum(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

#[cfg(test)]
mod tests {
    use super::atomic_write;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn atomic_write_replaces_an_existing_file_without_leaking_a_temporary() {
        let temporary = tempdir().expect("temporary directory must be available");
        let target = Path::new("state.json");

        atomic_write(temporary.path(), target, b"first", true)
            .expect("first atomic write must succeed");
        atomic_write(temporary.path(), target, b"second", true)
            .expect("replacement atomic write must succeed");

        assert_eq!(
            fs::read(temporary.path().join(target)).expect("published file must be readable"),
            b"second"
        );
        let entries: Vec<_> = fs::read_dir(temporary.path())
            .expect("temporary directory must be readable")
            .map(|entry| entry.expect("directory entry must be readable").file_name())
            .collect();
        assert_eq!(entries, [target.as_os_str()]);
    }
}
