//! Atomic storage for the optional derived-index cache.

use super::derived_file;
use crate::error::Result;
use std::path::{Path, PathBuf};

pub(super) fn read(root: &Path, max_index_cache_file_bytes: u64) -> Result<Option<Vec<u8>>> {
    derived_file::read(
        root,
        &relative_path(),
        max_index_cache_file_bytes,
        "derived index cache",
    )
}

pub(super) fn write(
    root: &Path,
    bytes: &[u8],
    sync: bool,
    max_index_cache_file_bytes: u64,
) -> Result<()> {
    derived_file::write(
        root,
        &relative_path(),
        bytes,
        max_index_cache_file_bytes,
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
    use crate::storage_ceilings::StorageCeilings;
    use tempfile::tempdir;

    #[test]
    fn cache_bytes_are_atomically_replaced() {
        let temporary = tempdir().expect("temporary directory must be available");
        let max = StorageCeilings::default().max_index_cache_file_bytes();
        assert!(read(temporary.path(), max)
            .expect("missing cache must be readable")
            .is_none());
        write(temporary.path(), b"first", false, max).expect("cache must write");
        assert_eq!(
            read(temporary.path(), max).expect("cache must read"),
            Some(b"first".to_vec())
        );
        write(temporary.path(), b"second", false, max).expect("cache must replace");
        assert_eq!(
            read(temporary.path(), max).expect("cache must read"),
            Some(b"second".to_vec())
        );
    }
}
