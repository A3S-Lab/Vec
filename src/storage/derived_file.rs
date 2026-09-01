//! Bounded, atomic storage shared by non-authoritative index artifacts.

use super::manifest::sync_directory;
use crate::error::{Error, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn read(
    root: &Path,
    relative_path: &Path,
    maximum_bytes: u64,
    label: &str,
) -> Result<Option<Vec<u8>>> {
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
    let length = usize::try_from(length).map_err(|_| {
        Error::resource_exhausted(format!("{label} is too large for this platform"))
    })?;
    let mut bytes = vec![0_u8; length];
    read_exact_at(&file, 0, &mut bytes)
        .map_err(|error| Error::internal(format!("read {label}: {error}")))?;
    Ok(Some(bytes))
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
    use super::{read, read_exact_at};
    use crate::ErrorCode;
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
}
