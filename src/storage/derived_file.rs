//! Bounded, atomic storage shared by non-authoritative index artifacts.

use super::manifest::sync_directory;
use crate::config::IoBackend;
use crate::error::{Error, Result};
use memmap2::{Mmap, MmapMut};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub(crate) struct PositionedFile {
    file: Arc<File>,
    length: u64,
    label: String,
}

#[derive(Clone, Debug)]
pub(crate) enum RandomAccessReader {
    Positioned(PositionedFile),
    Mmap { bytes: Arc<Mmap>, label: Arc<str> },
}

impl PositionedFile {
    pub(crate) fn len(&self) -> u64 {
        self.length
    }

    pub(crate) fn read_exact_at(&self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        read_exact_at(&self.file, offset, bytes)
            .map_err(|error| Error::internal(format!("read {}: {error}", self.label)))
    }

    pub(crate) fn read_all(&self) -> Result<Vec<u8>> {
        let length = usize::try_from(self.length).map_err(|_| {
            Error::resource_exhausted(format!("{} is too large for this platform", self.label))
        })?;
        let mut bytes = vec![0_u8; length];
        self.read_exact_at(0, &mut bytes)?;
        Ok(bytes)
    }

    pub(crate) fn into_random_access(
        self,
        backend: IoBackend,
        validated_bytes: &[u8],
    ) -> Result<RandomAccessReader> {
        match backend {
            IoBackend::Positioned => Ok(RandomAccessReader::Positioned(self)),
            IoBackend::Mmap => {
                if validated_bytes.is_empty() {
                    return Err(Error::internal(format!("cannot mmap empty {}", self.label)));
                }
                let mut mapping = MmapMut::map_anon(validated_bytes.len()).map_err(|error| {
                    Error::resource_exhausted(format!(
                        "allocate mmap snapshot for {}: {error}",
                        self.label
                    ))
                })?;
                mapping.copy_from_slice(validated_bytes);
                let mapping = mapping.make_read_only().map_err(|error| {
                    Error::internal(format!(
                        "make mmap snapshot read-only for {}: {error}",
                        self.label
                    ))
                })?;
                Ok(RandomAccessReader::Mmap {
                    bytes: Arc::new(mapping),
                    label: Arc::from(self.label),
                })
            }
        }
    }
}

impl RandomAccessReader {
    pub(crate) fn len(&self) -> u64 {
        match self {
            Self::Positioned(file) => file.len(),
            Self::Mmap { bytes, .. } => u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        }
    }

    pub(crate) fn io_backend(&self) -> IoBackend {
        match self {
            Self::Positioned(_) => IoBackend::Positioned,
            Self::Mmap { .. } => IoBackend::Mmap,
        }
    }

    pub(crate) fn read_exact_at(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        match self {
            Self::Positioned(file) => file.read_exact_at(offset, output),
            Self::Mmap { bytes, label } => {
                let start = usize::try_from(offset).map_err(|_| {
                    Error::resource_exhausted(format!("read {label}: offset exceeds usize"))
                })?;
                let end = start.checked_add(output.len()).ok_or_else(|| {
                    Error::resource_exhausted(format!("read {label}: byte range overflow"))
                })?;
                let source = bytes.get(start..end).ok_or_else(|| {
                    Error::internal(format!(
                        "read {label}: derived index artifact ended before its declared length"
                    ))
                })?;
                output.copy_from_slice(source);
                Ok(())
            }
        }
    }
}

pub(super) fn open(
    root: &Path,
    relative_path: &Path,
    maximum_bytes: u64,
    label: &str,
) -> Result<Option<PositionedFile>> {
    let path = root.join(relative_path);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Error::internal(format!("open {label}: {error}"))),
    };
    let length = file
        .metadata()
        .map_err(|error| Error::internal(format!("read {label} metadata: {error}")))?
        .len();
    if length > maximum_bytes {
        return Err(Error::resource_exhausted(format!(
            "{label} exceeds the {maximum_bytes}-byte recovery limit"
        )));
    }
    Ok(Some(PositionedFile {
        file: Arc::new(file),
        length,
        label: label.to_string(),
    }))
}

pub(super) fn read(
    root: &Path,
    relative_path: &Path,
    maximum_bytes: u64,
    label: &str,
) -> Result<Option<Vec<u8>>> {
    open(root, relative_path, maximum_bytes, label)?
        .map(|file| file.read_all())
        .transpose()
}

pub(super) fn write(
    root: &Path,
    relative_path: &Path,
    bytes: &[u8],
    maximum_bytes: u64,
    label: &str,
    sync: bool,
) -> Result<()> {
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted(format!("{label} exceeds u64 bytes")))?;
    if byte_len > maximum_bytes {
        return Err(Error::resource_exhausted(format!(
            "{label} exceeds the {maximum_bytes}-byte storage limit"
        )));
    }
    let target = root.join(relative_path);
    let parent = target
        .parent()
        .ok_or_else(|| Error::internal(format!("{label} has no parent directory")))?;
    fs::create_dir_all(parent)
        .map_err(|error| Error::internal(format!("create {label} directory: {error}")))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::internal(format!("{label} has no UTF-8 file name")))?;
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{stamp}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| Error::internal(format!("create {label}: {error}")))?;
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(Error::internal(format!("write {label}: {error}")));
    }
    if sync {
        if let Err(error) = file.sync_all() {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(Error::internal(format!("sync {label}: {error}")));
        }
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, &target) {
        let _ = fs::remove_file(&temporary);
        return Err(Error::internal(format!("publish {label}: {error}")));
    }
    if sync {
        sync_directory(parent)?;
    }
    Ok(())
}

fn read_exact_at(file: &File, mut offset: u64, mut bytes: &mut [u8]) -> std::io::Result<()> {
    while !bytes.is_empty() {
        let read = read_at(file, bytes, offset)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "derived index artifact ended before its declared length",
            ));
        }
        offset = offset
            .checked_add(u64::try_from(read).unwrap_or(u64::MAX))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "file offset overflow")
            })?;
        bytes = &mut bytes[read..];
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(bytes, offset)
}

#[cfg(windows)]
fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(bytes, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::io::{Read, Seek, SeekFrom};
    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    clone.read(bytes)
}

#[cfg(test)]
mod tests {
    use super::{open, read, read_exact_at};
    use crate::{ErrorCode, IoBackend};
    use std::fs::{self, File};
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn positioned_reads_are_exact_and_report_truncation() {
        let temporary = tempdir().expect("temporary directory must be available");
        let path = temporary.path().join("positioned.bin");
        fs::write(&path, b"0123456789").expect("fixture must write");
        let file = File::open(path).expect("fixture must open");
        let mut selected = [0_u8; 4];
        read_exact_at(&file, 3, &mut selected).expect("positioned read must succeed");
        assert_eq!(&selected, b"3456");
        let mut truncated = [0_u8; 4];
        let error = read_exact_at(&file, 8, &mut truncated).expect_err("short read must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn metadata_bound_rejects_oversized_artifacts_before_allocation() {
        let temporary = tempdir().expect("temporary directory must be available");
        fs::write(temporary.path().join("oversized.bin"), b"12345").expect("fixture must write");
        let error = read(
            temporary.path(),
            Path::new("oversized.bin"),
            4,
            "test artifact",
        )
        .expect_err("oversized artifact must fail");
        assert_eq!(error.code, ErrorCode::ResourceExhausted);
    }

    #[test]
    fn anonymous_mmap_snapshot_is_immutable_and_bounds_checked() {
        let temporary = tempdir().expect("temporary directory must be available");
        let relative = Path::new("mapped.bin");
        let path = temporary.path().join(relative);
        fs::write(&path, b"0123456789").expect("fixture must write");
        let file = open(temporary.path(), relative, 10, "mapped fixture")
            .expect("fixture must open")
            .expect("fixture must exist");
        let bytes = file.read_all().expect("fixture must be readable");
        let reader = file
            .into_random_access(IoBackend::Mmap, &bytes)
            .expect("mmap snapshot must build");
        fs::write(&path, b"x").expect("source file must truncate");

        let mut selected = [0_u8; 4];
        reader
            .read_exact_at(3, &mut selected)
            .expect("snapshot read must succeed");
        assert_eq!(&selected, b"3456");
        assert_eq!(reader.io_backend(), IoBackend::Mmap);
        let error = reader
            .read_exact_at(8, &mut selected)
            .expect_err("out-of-range snapshot read must fail");
        assert_eq!(error.code, ErrorCode::InternalError);
    }
}
