//! Deterministic storage-boundary fault injection.
//!
//! Production builds keep the injector disabled. Unit tests can arm one named
//! boundary on an individual `StorageHandle`, which avoids global state and
//! keeps concurrent recovery tests isolated from each other.

use crate::error::{Error, ErrorCode, Result};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FaultPoint {
    WalPrepared,
    WalHeaderWritten,
    WalPayloadWritten,
    WalSynced,
    SnapshotWritten,
    SnapshotSynced,
    SnapshotRenamed,
    SnapshotDirectorySynced,
    ManifestWritten,
    ManifestSynced,
    ManifestRenamed,
    ManifestDirectorySynced,
    WalPruneBeforeRemove,
    WalPruneAfterRemove,
    WalPruneDirectorySynced,
    SnapshotPruneBeforeRemove,
    SnapshotPruneAfterRemove,
    SnapshotPruneDirectorySynced,
}

impl FaultPoint {
    const fn name(self) -> &'static str {
        match self {
            Self::WalPrepared => "wal.prepared",
            Self::WalHeaderWritten => "wal.header-written",
            Self::WalPayloadWritten => "wal.payload-written",
            Self::WalSynced => "wal.synced",
            Self::SnapshotWritten => "snapshot.written",
            Self::SnapshotSynced => "snapshot.synced",
            Self::SnapshotRenamed => "snapshot.renamed",
            Self::SnapshotDirectorySynced => "snapshot.directory-synced",
            Self::ManifestWritten => "manifest.written",
            Self::ManifestSynced => "manifest.synced",
            Self::ManifestRenamed => "manifest.renamed",
            Self::ManifestDirectorySynced => "manifest.directory-synced",
            Self::WalPruneBeforeRemove => "wal-prune.before-remove",
            Self::WalPruneAfterRemove => "wal-prune.after-remove",
            Self::WalPruneDirectorySynced => "wal-prune.directory-synced",
            Self::SnapshotPruneBeforeRemove => "snapshot-prune.before-remove",
            Self::SnapshotPruneAfterRemove => "snapshot-prune.after-remove",
            Self::SnapshotPruneDirectorySynced => "snapshot-prune.directory-synced",
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct FaultInjector {
    enabled: AtomicBool,
    #[cfg(test)]
    state: Mutex<FaultState>,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct FaultState {
    armed: Option<FaultPoint>,
    fired: Option<FaultPoint>,
}

impl FaultInjector {
    pub(super) fn hit(&self, point: FaultPoint) -> Result<()> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(());
        }

        #[cfg(test)]
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::internal("fault injector lock poisoned"))?;
            if state.armed == Some(point) {
                state.armed = None;
                state.fired = Some(point);
                self.enabled.store(false, Ordering::Relaxed);
                return Err(Error::new(
                    ErrorCode::Unavailable,
                    format!("injected storage fault at {}", point.name()),
                ));
            }
        }

        #[cfg(test)]
        return Ok(());

        #[cfg(not(test))]
        Err(Error::new(
            ErrorCode::Unavailable,
            format!("storage fault injection is unavailable at {}", point.name()),
        ))
    }

    #[cfg(test)]
    pub(super) fn arm(&self, point: FaultPoint) {
        let mut state = self.state.lock().expect("fault injector lock poisoned");
        state.armed = Some(point);
        state.fired = None;
        self.enabled.store(true, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn fired(&self, point: FaultPoint) -> bool {
        self.state
            .lock()
            .expect("fault injector lock poisoned")
            .fired
            == Some(point)
    }
}
