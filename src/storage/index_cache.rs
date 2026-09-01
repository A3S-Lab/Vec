//! Atomic storage for the optional derived-index cache.

use super::derived_file;
use crate::error::Result;
use std::path::{Path, PathBuf};

const MAX_INDEX_CACHE_BYTES: u64 = 512 * 1024 * 1024 + 4_096;

pub(super) fn read(root: &Path) -> Result<Option<Vec<u8>>> {
    derived_file::read(
        root,
        &relative_path(),
        MAX_INDEX_CACHE_BYTES,
        "derived index cache",
    )
}

pub(super) fn write(root: &Path, bytes: &[u8], sync: bool) -> Result<()> {
    derived_file::write(
        root,
        &relative_path(),
        bytes,
        MAX_INDEX_CACHE_BYTES,
        "derived index cache",
        sync,
    )
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
