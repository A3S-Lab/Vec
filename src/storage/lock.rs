//! Cross-process collection locking.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const LOCK_FILE_NAME: &str = ".a3s-vec.lock";
const MAX_OWNER_BYTES: u64 = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LockOwner {
    pid: u32,
    acquired_unix_ms: u64,
}

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
        let lock_path = path.join(LOCK_FILE_NAME);
        let mut file = if read_only {
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
            let owner = describe_owner(&lock_path);
            return Err(Error::new(
                if read_only {
                    crate::error::ErrorCode::Unavailable
                } else {
                    crate::error::ErrorCode::AlreadyExists
                },
                format!("collection is locked by another process ({owner}): {error}"),
            ));
        }
        if !read_only {
            if let Err(error) = write_owner(&mut file) {
                let _ = fs2::FileExt::unlock(&file);
                return Err(error);
            }
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

fn write_owner(file: &mut File) -> Result<()> {
    let acquired_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        });
    let owner = LockOwner {
        pid: std::process::id(),
        acquired_unix_ms,
    };
    let bytes = serde_json::to_vec(&owner)
        .map_err(|error| Error::internal(format!("serialize collection lock owner: {error}")))?;
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)).map(|_| ()))
        .and_then(|()| file.write_all(&bytes))
        .and_then(|()| file.sync_data())
        .map_err(|error| Error::internal(format!("write collection lock owner: {error}")))
}

fn describe_owner(path: &Path) -> String {
    read_owner(path).map_or_else(
        || "owner metadata unavailable or stale".to_string(),
        |owner| {
            format!(
                "recorded exclusive owner pid={}, acquired_unix_ms={}",
                owner.pid, owner.acquired_unix_ms
            )
        },
    )
}

fn read_owner(path: &Path) -> Option<LockOwner> {
    let file = File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_OWNER_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .ok()?;
    if u64::try_from(bytes.len()).ok()? > MAX_OWNER_BYTES {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

impl Drop for CollectionLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::{read_owner, CollectionLock, LockOwner, LOCK_FILE_NAME};
    use crate::error::ErrorCode;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn lock_contention_reports_the_recorded_exclusive_owner() {
        let temporary = tempdir().expect("temporary directory must be available");
        let owner = CollectionLock::acquire(temporary.path(), false)
            .expect("exclusive lock must be acquired");

        let writer_error = CollectionLock::acquire(temporary.path(), false)
            .expect_err("a second writer must be rejected");
        assert_eq!(writer_error.code, ErrorCode::AlreadyExists);
        assert!(writer_error
            .message
            .contains(&format!("pid={}", std::process::id())));
        assert!(writer_error.message.contains("acquired_unix_ms="));

        let reader_error = CollectionLock::acquire(temporary.path(), true)
            .expect_err("a reader must not bypass an exclusive writer");
        assert_eq!(reader_error.code, ErrorCode::Unavailable);
        assert!(reader_error
            .message
            .contains(&format!("pid={}", std::process::id())));
        drop(owner);
    }

    #[test]
    fn stale_owner_metadata_is_replaced_after_the_kernel_lock_is_acquired() {
        let temporary = tempdir().expect("temporary directory must be available");
        let stale = LockOwner {
            pid: u32::MAX,
            acquired_unix_ms: 1,
        };
        fs::write(
            temporary.path().join(LOCK_FILE_NAME),
            serde_json::to_vec(&stale).expect("stale owner fixture must serialize"),
        )
        .expect("stale owner fixture must be writable");

        let lock = CollectionLock::acquire(temporary.path(), false)
            .expect("metadata alone must never block the kernel lock");
        let current = read_owner(&temporary.path().join(LOCK_FILE_NAME))
            .expect("current lock owner must be readable");
        assert_eq!(current.pid, std::process::id());
        assert_ne!(current, stale);
        drop(lock);

        CollectionLock::acquire(temporary.path(), false)
            .expect("a released kernel lock must be reusable");
    }
}
