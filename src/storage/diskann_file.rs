//! Atomic storage for the optional native `DiskANN` sector sidecar.

use super::derived_file::{self, PositionedFile};
use crate::error::Result;
use std::path::{Path, PathBuf};

pub(super) fn open(root: &Path, max_diskann_file_bytes: u64) -> Result<Option<PositionedFile>> {
    derived_file::open(
        root,
        &relative_path(),
        max_diskann_file_bytes,
        "DiskANN sector sidecar",
    )
}

pub(super) fn write(
    root: &Path,
    bytes: &[u8],
    sync: bool,
    max_diskann_file_bytes: u64,
) -> Result<()> {
    derived_file::write(
        root,
        &relative_path(),
        bytes,
        max_diskann_file_bytes,
        "DiskANN sector sidecar",
        sync,
    )
}

fn relative_path() -> PathBuf {
    Path::new("indexes").join("diskann-graph.bin")
}
