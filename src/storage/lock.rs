//! Cross-process collection locking.

use crate::error::{Error, Result};
use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct CollectionLock {
    file: File,
    path: PathBuf,
    exclusive: bool,
}

impl CollectionLock {
    pub fn acquire(path: &Path, read_only: bool) -> Result<Self> {
        std::fs::create_dir_all(path).map_err(|e| Error::internal(format!("create collection directory: {e}")))?;
        let lock_path = path.join(".a3s-vec.lock");
        let file = OpenOptions::new().create(true).read(true).write(true).open(&lock_path)
            .map_err(|e| Error::internal(format!("open collection lock: {e}")))?;
        let result = if read_only {
            file.try_lock_shared().map_err(|error| error.to_string())
        } else {
            file.try_lock_exclusive().map_err(|error| error.to_string())
        };
        if let Err(error) = result {
            return Err(Error::new(
                if read_only { crate::error::ErrorCode::Unavailable } else { crate::error::ErrorCode::AlreadyExists },
                format!("collection is locked by another process: {error}"),
            ));
        }
        Ok(Self { file, path: lock_path, exclusive: !read_only })
    }
    pub fn is_exclusive(&self) -> bool { self.exclusive }
    pub fn path(&self) -> &Path { &self.path }
}

impl Drop for CollectionLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}
