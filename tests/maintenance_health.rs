use a3s_vec::{
    Collection, CollectionHealthStatus, CollectionMaintenanceOptions, CollectionMaintenancePhase,
    CollectionOptions, CollectionSchema, DataType, Doc, ErrorCode, FieldSchema, IndexParams,
};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const TEST_INTERVAL: Duration = Duration::from_millis(10);
const WAIT_TIMEOUT: Duration = Duration::from_secs(5);

fn indexed_schema() -> CollectionSchema {
    let mut category =
        FieldSchema::new("category", DataType::String, false, 0).expect("field must be valid");
    category
        .set_index_params(&IndexParams::invert(false, false).expect("index params must be valid"))
        .expect("index must be attachable");
    CollectionSchema::builder("maintenance-health")
        .add_field(category)
        .build()
        .expect("schema must be valid")
}

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(a3s_vec::Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn maintenance_options() -> CollectionMaintenanceOptions {
    CollectionMaintenanceOptions::new()
        .try_with_interval(TEST_INTERVAL)
        .expect("test interval must be valid")
}

fn document(index: usize) -> Doc {
    let id = format!("doc-{index:04}");
    let mut doc = Doc::with_pk(&id).expect("primary key must be valid");
    doc.add_string("category", if index % 2 == 0 { "even" } else { "odd" })
        .expect("category must be valid");
    doc
}

fn wait_until(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    while !predicate() {
        assert!(Instant::now() < deadline, "condition timed out");
        thread::sleep(TEST_INTERVAL);
    }
}

#[test]
fn collection_health_distinguishes_readiness_from_normal_checkpoint_lag() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let collection = Collection::create(
        collection_path.to_str().expect("path must be UTF-8"),
        &indexed_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");

    let initial = collection.health().expect("health must be available");
    assert_eq!(initial.status, CollectionHealthStatus::Healthy);
    assert!(initial.is_healthy());
    assert_eq!(initial.revision, 0);
    assert_eq!(initial.storage_revision, 0);
    assert_eq!(initial.index_count, 1);
    assert_eq!(initial.ready_index_count, 1);
    assert!(!initial.checkpoint_pending);
    assert!(!initial.maintenance_active);
    assert!(initial.reasons.is_empty());

    let doc = document(1);
    collection.insert(&[&doc]).expect("insert must succeed");
    let pending = collection.health().expect("health must be available");
    assert_eq!(pending.status, CollectionHealthStatus::Healthy);
    assert_eq!(pending.revision, 1);
    assert_eq!(pending.storage_revision, 1);
    assert!(pending.checkpoint_pending);
    assert_eq!(pending.wal_ops_since_checkpoint, 1);

    let observer = collection.clone();
    collection.close().expect("close must succeed");
    let closed = observer
        .health()
        .expect("closed health must remain observable");
    assert_eq!(closed.status, CollectionHealthStatus::Closed);
    assert!(!closed.is_healthy());
    assert!(closed
        .reasons
        .iter()
        .any(|reason| reason.contains("closed")));
}

#[test]
fn background_maintenance_checkpoints_the_latest_generation_and_releases_ownership() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let collection = Collection::create(
        collection_path.to_str().expect("path must be UTF-8"),
        &indexed_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let runtime = collection
        .start_maintenance(maintenance_options())
        .expect("maintenance must start");
    assert_eq!(runtime.health().phase, CollectionMaintenancePhase::Running);
    assert!(runtime.health().worker_alive);
    assert!(
        collection
            .health()
            .expect("health must succeed")
            .maintenance_active
    );

    let duplicate = collection
        .start_maintenance(maintenance_options())
        .expect_err("one collection cannot have two maintenance owners");
    assert_eq!(duplicate.code, ErrorCode::AlreadyExists);

    let writer = collection.clone();
    let write_thread = thread::spawn(move || {
        for index in 0..24 {
            let doc = document(index);
            let result = writer.insert(&[&doc]).expect("insert must succeed");
            assert_eq!(result.success_count, 1);
        }
    });
    write_thread.join().expect("writer must not panic");
    let target_revision = collection.stats().expect("stats must succeed").revision;
    runtime
        .trigger()
        .expect("an immediate run must be requestable");
    wait_until(|| {
        let maintenance = runtime.health();
        let stats = collection.stats().expect("stats must succeed");
        maintenance.last_successful_revision == Some(target_revision)
            && stats.wal_ops_since_checkpoint == 0
    });

    let maintenance = runtime.health();
    assert_eq!(maintenance.phase, CollectionMaintenancePhase::Running);
    assert!(maintenance.successful_runs > 0);
    assert_eq!(maintenance.failed_runs, 0);
    assert_eq!(maintenance.last_error, None);
    assert_eq!(collection.count().expect("count must succeed"), 24);
    let health = collection.health().expect("health must succeed");
    assert_eq!(health.status, CollectionHealthStatus::Healthy);
    assert_eq!(health.ready_index_count, health.index_count);

    runtime.close().expect("maintenance must close cleanly");
    let closed_runtime = runtime.health();
    assert_eq!(closed_runtime.phase, CollectionMaintenancePhase::Closed);
    assert!(!closed_runtime.worker_alive);
    assert_eq!(
        runtime
            .trigger()
            .expect_err("closed runtime must reject triggers")
            .code,
        ErrorCode::FailedPrecondition
    );
    assert!(
        !collection
            .health()
            .expect("health must succeed")
            .maintenance_active
    );

    let replacement = collection
        .start_maintenance(maintenance_options())
        .expect("closing must release maintenance ownership");
    replacement.close().expect("replacement must close");
    collection.close().expect("collection must close");
}

#[test]
fn scheduled_maintenance_runs_without_a_trigger_and_skips_unchanged_revisions() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let collection = Collection::create(
        collection_path.to_str().expect("path must be UTF-8"),
        &indexed_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let doc = document(1);
    collection.insert(&[&doc]).expect("insert must succeed");
    let revision = collection.stats().expect("stats must succeed").revision;

    let runtime = collection
        .start_maintenance(maintenance_options())
        .expect("maintenance must start");
    wait_until(|| runtime.health().last_successful_revision == Some(revision));
    let successful_runs = runtime.health().successful_runs;
    assert_eq!(successful_runs, 1);
    assert_eq!(
        collection
            .stats()
            .expect("stats must succeed")
            .wal_ops_since_checkpoint,
        0
    );

    wait_until(|| runtime.health().skipped_runs > 0);
    assert_eq!(runtime.health().successful_runs, successful_runs);

    runtime.close().expect("maintenance must close");
    collection.close().expect("collection must close");
}

#[test]
fn dropping_maintenance_wakes_the_worker_and_releases_the_claim() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let collection = Collection::create(
        collection_path.to_str().expect("path must be UTF-8"),
        &indexed_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let options = CollectionMaintenanceOptions::new()
        .try_with_interval(Duration::from_secs(2))
        .expect("interval must be valid");
    let runtime = collection
        .start_maintenance(options)
        .expect("maintenance must start");
    let started = Instant::now();
    drop(runtime);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "drop must wake the worker instead of waiting for the interval"
    );

    let replacement = collection
        .start_maintenance(options)
        .expect("drop must release ownership");
    replacement.close().expect("replacement must close");
    collection.close().expect("collection must close");
}

#[test]
fn worker_health_records_a_closed_collection_failure() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let collection = Collection::create(
        collection_path.to_str().expect("path must be UTF-8"),
        &indexed_schema(),
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let options = CollectionMaintenanceOptions::new()
        .try_with_interval(Duration::from_secs(2))
        .expect("interval must be valid");
    let runtime = collection
        .start_maintenance(options)
        .expect("maintenance must start");
    let observer = collection.clone();
    collection.close().expect("collection must close");
    runtime
        .trigger()
        .expect("the live worker must accept a final trigger");
    wait_until(|| !runtime.health().worker_alive);

    let degraded = runtime.health();
    assert_eq!(degraded.phase, CollectionMaintenancePhase::Degraded);
    assert_eq!(degraded.failed_runs, 1);
    assert!(degraded
        .last_error
        .as_deref()
        .is_some_and(|error| error.contains("closed")));
    assert!(!degraded.is_healthy());
    assert!(
        observer
            .health()
            .expect("health must remain available")
            .maintenance_active
    );

    runtime.close().expect("failed worker must remain joinable");
    assert_eq!(runtime.health().phase, CollectionMaintenancePhase::Closed);
    assert!(!runtime.health().is_healthy());
    assert!(
        !observer
            .health()
            .expect("health must remain available")
            .maintenance_active
    );
}

#[test]
fn read_only_collections_reject_background_maintenance() {
    let temporary = tempdir().expect("temporary directory must be available");
    let collection_path = temporary.path().join("collection");
    let path = collection_path.to_str().expect("path must be UTF-8");
    Collection::create(path, &indexed_schema(), None)
        .expect("collection must be created")
        .close()
        .expect("collection must close");

    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_read_only(true)
        .expect("read-only mode must be valid");
    let collection = Collection::open(path, Some(&options)).expect("collection must open");
    let error = collection
        .start_maintenance(maintenance_options())
        .expect_err("read-only maintenance must be rejected");
    assert_eq!(error.code, ErrorCode::PermissionDenied);
    let health = collection.health().expect("health must succeed");
    assert_eq!(health.status, CollectionHealthStatus::Healthy);
    assert!(health.read_only);
    assert!(!health.maintenance_active);
    collection.close().expect("collection must close");
}

#[test]
fn maintenance_options_reject_spin_loops_and_unbounded_intervals() {
    let too_short = CollectionMaintenanceOptions::new()
        .try_with_interval(Duration::from_millis(1))
        .expect_err("spin-loop interval must be rejected");
    assert_eq!(too_short.code, ErrorCode::InvalidArgument);

    let too_long = CollectionMaintenanceOptions::new()
        .try_with_interval(Duration::from_secs(366 * 24 * 60 * 60))
        .expect_err("unbounded interval must be rejected");
    assert_eq!(too_long.code, ErrorCode::InvalidArgument);
}
