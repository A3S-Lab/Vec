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
    /// Fail-once point at the start of a `DiskANN` sidecar write.
    DiskannWritten,
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
            Self::DiskannWritten => "diskann.written",
        }
    }
}

/// Test gate for a durability sync that must be observable while parked.
///
/// The waiting thread publishes `entered` before blocking and polls `release`,
/// so the test does not need the storage mutex to let the sync continue.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct StallGate {
    entered: std::sync::Arc<AtomicBool>,
    release: std::sync::Arc<AtomicBool>,
}

#[cfg(test)]
impl StallGate {
    pub(crate) fn wait_entered(&self, timeout: std::time::Duration) -> bool {
        let start = std::time::Instant::now();
        while !self.entered.load(Ordering::Acquire) {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        true
    }

    pub(crate) fn release(&self) {
        self.release.store(true, Ordering::Release);
    }
}

#[cfg(test)]
#[derive(Debug)]
struct ArmedStall {
    point: FaultPoint,
    entered: std::sync::Arc<AtomicBool>,
    release: std::sync::Arc<AtomicBool>,
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
    stall: Option<ArmedStall>,
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

    /// Parks before a named boundary until the test releases the gate.
    ///
    /// The fault mutex is dropped before the wait. Production builds do not arm
    /// a gate, so this returns immediately.
    pub(super) fn stall(&self, point: FaultPoint) {
        #[cfg(test)]
        {
            let Some((entered, release)) = self.take_stall(point) else {
                return;
            };
            entered.store(true, Ordering::Release);
            while !release.load(Ordering::Acquire) {
                std::thread::park_timeout(std::time::Duration::from_millis(5));
            }
        }
        #[cfg(not(test))]
        {
            let _ = (self, point);
        }
    }

    #[cfg(test)]
    fn take_stall(
        &self,
        point: FaultPoint,
    ) -> Option<(std::sync::Arc<AtomicBool>, std::sync::Arc<AtomicBool>)> {
        let mut state = self.state.lock().ok()?;
        let stall = state.stall.as_ref()?;
        if stall.point != point {
            return None;
        }
        let entered = std::sync::Arc::clone(&stall.entered);
        let release = std::sync::Arc::clone(&stall.release);
        state.stall = None;
        Some((entered, release))
    }

    #[cfg(test)]
    pub(super) fn arm_stall(&self, point: FaultPoint) -> StallGate {
        let entered = std::sync::Arc::new(AtomicBool::new(false));
        let release = std::sync::Arc::new(AtomicBool::new(false));
        let mut state = self.state.lock().expect("fault injector lock poisoned");
        state.stall = Some(ArmedStall {
            point,
            entered: std::sync::Arc::clone(&entered),
            release: std::sync::Arc::clone(&release),
        });
        self.enabled.store(true, Ordering::Relaxed);
        StallGate { entered, release }
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
