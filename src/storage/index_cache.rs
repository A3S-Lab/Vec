//! Atomic storage for the optional derived-index cache.

use super::manifest::sync_directory;
use crate::error::{Error, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_INDEX_CACHE_BYTES: u64 = 512 * 1024 * 1024 + 4_096;

pub(super) fn read(root: &Path) -> Result<Option<Vec<u8>>> {
    let path = root.join(relative_path());
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(Error::internal(format!(
                "read derived index cache metadata: {error}"
            )))
        }
    };
    if metadata.len() > MAX_INDEX_CACHE_BYTES {
        return Err(Error::resource_exhausted(format!(
            "derived index cache exceeds the {MAX_INDEX_CACHE_BYTES}-byte recovery limit"
        )));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        Error::resource_exhausted("derived index cache is too large for this platform")
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(&path)
        .map_err(|error| Error::internal(format!("open derived index cache: {error}")))?
        .take(MAX_INDEX_CACHE_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| Error::internal(format!("read derived index cache: {error}")))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_INDEX_CACHE_BYTES {
        return Err(Error::resource_exhausted(format!(
            "derived index cache exceeds the {MAX_INDEX_CACHE_BYTES}-byte recovery limit"
        )));
    }
    Ok(Some(bytes))
}

pub(super) fn write(root: &Path, bytes: &[u8], sync: bool) -> Result<()> {
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| Error::resource_exhausted("derived index cache exceeds u64 bytes"))?;
    if byte_len > MAX_INDEX_CACHE_BYTES {
        return Err(Error::resource_exhausted(format!(
            "derived index cache exceeds the {MAX_INDEX_CACHE_BYTES}-byte storage limit"
        )));
    }
    let target = root.join(relative_path());
    let parent = target
        .parent()
        .ok_or_else(|| Error::internal("derived index cache has no parent directory"))?;
    fs::create_dir_all(parent)
        .map_err(|error| Error::internal(format!("create index cache directory: {error}")))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let temporary = parent.join(format!(
        ".index-cache.bin.tmp-{}-{stamp}",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| Error::internal(format!("create derived index cache: {error}")))?;
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(Error::internal(format!(
            "write derived index cache: {error}"
        )));
    }
    if sync {
        if let Err(error) = file.sync_all() {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(Error::internal(format!(
                "sync derived index cache: {error}"
            )));
        }
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, &target) {
        let _ = fs::remove_file(&temporary);
        return Err(Error::internal(format!(
            "publish derived index cache: {error}"
        )));
    }
    if sync {
        sync_directory(parent)?;
    }
    Ok(())
}

fn relative_path() -> PathBuf {
    Path::new("indexes").join("index-cache.bin")
}

#[cfg(test)]
mod tests {
    use super::{read, write};
    use tempfile::tempdir;

    #[test]
    fn cache_bytes_are_atomically_replaced() {
        let temporary = tempdir().expect("temporary directory must be available");
        assert!(read(temporary.path())
            .expect("missing cache must be readable")
            .is_none());
        write(temporary.path(), b"first", false).expect("cache must write");
        assert_eq!(
            read(temporary.path()).expect("cache must read"),
            Some(b"first".to_vec())
        );
        write(temporary.path(), b"second", false).expect("cache must replace");
        assert_eq!(
            read(temporary.path()).expect("cache must read"),
            Some(b"second".to_vec())
        );
    }
}
