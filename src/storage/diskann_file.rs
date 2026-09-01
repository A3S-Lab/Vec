//! Atomic storage for the optional native `DiskANN` sector sidecar.

use super::derived_file::{self, PositionedFile};
use crate::error::Result;
use std::path::{Path, PathBuf};

const MAX_DISKANN_FILE_BYTES: u64 = 512 * 1024 * 1024;

pub(super) fn open(root: &Path) -> Result<Option<PositionedFile>> {
    derived_file::open(
        root,
        &relative_path(),
        MAX_DISKANN_FILE_BYTES,
        "DiskANN sector sidecar",
    )
}

pub(super) fn write(root: &Path, bytes: &[u8], sync: bool) -> Result<()> {
    derived_file::write(
        root,
        &relative_path(),
        bytes,
        MAX_DISKANN_FILE_BYTES,
        "DiskANN sector sidecar",
        sync,
    )
}

fn relative_path() -> PathBuf {
    Path::new("indexes").join("diskann-graph.bin")
}
