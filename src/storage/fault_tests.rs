use super::fault::FaultPoint;
use super::test_support::{doc, schema};
use super::{wal, StorageHandle, WalOperation};
use crate::config::{ConfigBuilder, Durability};
use crate::error::ErrorCode;
use tempfile::tempdir;

const APPEND_FAULT_POINTS: [FaultPoint; 8] = [
    FaultPoint::WalPrepared,
    FaultPoint::WalHeaderWritten,
    FaultPoint::WalPayloadWritten,
    FaultPoint::WalSynced,
    FaultPoint::ManifestWritten,
    FaultPoint::ManifestSynced,
    FaultPoint::ManifestRenamed,
    FaultPoint::ManifestDirectorySynced,
];

const CHECKPOINT_FAULT_POINTS: [FaultPoint; 14] = [
    FaultPoint::SnapshotWritten,
    FaultPoint::SnapshotSynced,
    FaultPoint::SnapshotRenamed,
    FaultPoint::SnapshotDirectorySynced,
    FaultPoint::ManifestWritten,
    FaultPoint::ManifestSynced,
    FaultPoint::ManifestRenamed,
    FaultPoint::ManifestDirectorySynced,
    FaultPoint::WalPruneBeforeRemove,
    FaultPoint::WalPruneAfterRemove,
    FaultPoint::WalPruneDirectorySynced,
    FaultPoint::SnapshotPruneBeforeRemove,
    FaultPoint::SnapshotPruneAfterRemove,
    FaultPoint::SnapshotPruneDirectorySynced,
];

fn manifest_was_published(point: FaultPoint) -> bool {
    matches!(
        point,
        FaultPoint::ManifestRenamed
            | FaultPoint::ManifestDirectorySynced
            | FaultPoint::WalPruneBeforeRemove
            | FaultPoint::WalPruneAfterRemove
            | FaultPoint::WalPruneDirectorySynced
            | FaultPoint::SnapshotPruneBeforeRemove
            | FaultPoint::SnapshotPruneAfterRemove
            | FaultPoint::SnapshotPruneDirectorySynced
    )
}

fn is_prune_fault(point: FaultPoint) -> bool {
    matches!(
        point,
        FaultPoint::WalPruneBeforeRemove
            | FaultPoint::WalPruneAfterRemove
            | FaultPoint::WalPruneDirectorySynced
            | FaultPoint::SnapshotPruneBeforeRemove
            | FaultPoint::SnapshotPruneAfterRemove
            | FaultPoint::SnapshotPruneDirectorySynced
    )
}

#[test]
fn append_recovers_at_every_write_sync_and_manifest_publication_boundary() {
    for point in APPEND_FAULT_POINTS {
        let temporary = tempdir().expect("temporary directory must be available");
        let root = temporary.path().join("collection");
        let schema = schema();
        let stored_doc = doc("doc-1");
        let mut storage =
            StorageHandle::create(&root, &schema, false).expect("storage must be created");
        storage.arm_fault(point);

        let error = storage
            .append(
                1,
                WalOperation::Insert {
                    docs: vec![stored_doc.clone()],
                },
                &ConfigBuilder::default().durability(Durability::Always),
            )
            .expect_err("armed append boundary must interrupt the transaction");
        assert_eq!(error.code, ErrorCode::Unavailable, "point={point:?}");
        assert!(storage.fault_fired(point), "point={point:?}");
        drop(storage);

        let (mut recovered, _, docs) =
            StorageHandle::open(&root, false).expect("interrupted append must recover");
        if manifest_was_published(point) {
            assert_eq!(recovered.manifest.revision, 1, "point={point:?}");
            assert_eq!(docs, [stored_doc], "point={point:?}");
        } else {
            assert_eq!(recovered.manifest.revision, 0, "point={point:?}");
            assert!(docs.is_empty(), "point={point:?}");

            recovered
                .append(
                    1,
                    WalOperation::Insert {
                        docs: vec![stored_doc.clone()],
                    },
                    &ConfigBuilder::default().durability(Durability::Always),
                )
                .expect("the next append must replace an uncommitted WAL tail");
            drop(recovered);
            let (recovered, _, docs) = StorageHandle::open(&root, false)
                .expect("replacement append must remain recoverable");
            assert_eq!(recovered.manifest.revision, 1, "point={point:?}");
            assert_eq!(docs, [stored_doc], "point={point:?}");
        }
    }
}

#[test]
fn checkpoint_recovers_at_every_snapshot_manifest_and_prune_boundary() {
    for point in CHECKPOINT_FAULT_POINTS {
        let temporary = tempdir().expect("temporary directory must be available");
        let root = temporary.path().join("collection");
        let schema = schema();
        let stored_doc = doc("doc-1");
        let mut storage =
            StorageHandle::create(&root, &schema, false).expect("storage must be created");
        storage
            .append(
                1,
                WalOperation::Insert {
                    docs: vec![stored_doc.clone()],
                },
                &ConfigBuilder::default().durability(Durability::Always),
            )
            .expect("setup WAL append must commit");
        storage.arm_fault(point);

        let checkpoint = storage.checkpoint(&schema, std::slice::from_ref(&stored_doc), 1, true);
        if is_prune_fault(point) {
            checkpoint.expect("post-commit cleanup faults must not report a false rollback");
        } else {
            let error = checkpoint.expect_err("armed checkpoint boundary must interrupt writing");
            assert_eq!(error.code, ErrorCode::Unavailable, "point={point:?}");
        }
        assert!(storage.fault_fired(point), "point={point:?}");
        drop(storage);

        let (recovered, recovered_schema, docs) =
            StorageHandle::open(&root, false).expect("interrupted checkpoint must recover");
        assert_eq!(recovered_schema, schema, "point={point:?}");
        assert_eq!(recovered.manifest.revision, 1, "point={point:?}");
        assert_eq!(docs, [stored_doc], "point={point:?}");
        assert_eq!(
            recovered.manifest.generation,
            if manifest_was_published(point) { 2 } else { 1 },
            "point={point:?}"
        );

        let old_wal = wal::segment_path(&root, 1);
        if point == FaultPoint::WalPruneBeforeRemove {
            assert!(old_wal.exists(), "point={point:?}");
        } else if matches!(
            point,
            FaultPoint::WalPruneAfterRemove | FaultPoint::WalPruneDirectorySynced
        ) {
            assert!(!old_wal.exists(), "point={point:?}");
        }
        let old_snapshot = root.join("segments/snapshot-00000000000000000001.json");
        if point == FaultPoint::SnapshotPruneBeforeRemove {
            assert!(old_snapshot.exists(), "point={point:?}");
        } else if matches!(
            point,
            FaultPoint::SnapshotPruneAfterRemove | FaultPoint::SnapshotPruneDirectorySynced
        ) {
            assert!(!old_snapshot.exists(), "point={point:?}");
        }
    }
}
