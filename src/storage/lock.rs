//! Cross-process collection locking.

use crate::error::{Error, Result};
use std::fs::{File, OpenOptions};
use std::path::Path;

#[derive(Debug)]
pub(crate) struct CollectionLock {
    file: File,
    exclusive: bool,
}

impl CollectionLock {
    pub fn acquire(path: &Path, read_only: bool) -> Result<Self> {
        if !read_only {
            std::fs::create_dir_all(path)
                .map_err(|e| Error::internal(format!("create collection directory: {e}")))?;
        }
        let lock_path = path.join(".a3s-vec.lock");
        let file = if read_only {
            OpenOptions::new().read(true).open(&lock_path)
        } else {
            OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(&lock_path)
        }
        .map_err(|e| Error::internal(format!("open collection lock: {e}")))?;
        let result = if read_only {
            fs2::FileExt::try_lock_shared(&file).map_err(|error| error.to_string())
        } else {
            fs2::FileExt::try_lock_exclusive(&file).map_err(|error| error.to_string())
        };
        if let Err(error) = result {
            return Err(Error::new(
                if read_only {
                    crate::error::ErrorCode::Unavailable
                } else {
                    crate::error::ErrorCode::AlreadyExists
                },
                format!("collection is locked by another process: {error}"),
            ));
        }
        Ok(Self {
            file,
            exclusive: !read_only,
        })
    }
    pub fn is_exclusive(&self) -> bool {
        self.exclusive
    }
}

impl Drop for CollectionLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}
