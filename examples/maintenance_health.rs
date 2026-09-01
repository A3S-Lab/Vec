use a3s_vec::{
    initialize, shutdown, Collection, CollectionHealthStatus, CollectionMaintenanceOptions,
    CollectionMaintenancePhase, CollectionOptions, CollectionSchema, DataType, Doc, Durability,
    FieldSchema, IndexParams,
};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

struct ExampleDirectory(PathBuf);

impl ExampleDirectory {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("stale example directory must be removable");
        }
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ExampleDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn main() -> a3s_vec::Result<()> {
    initialize(None)?;
    let directory = ExampleDirectory::new("a3s-vec-maintenance-health");
    let data_path = directory.path().to_string_lossy();
    let mut category = FieldSchema::new("category", DataType::String, false, 0)?;
    category.set_index_params(&IndexParams::invert(false, false)?)?;
    let schema = CollectionSchema::builder("maintenance_health")
        .add_field(category)
        .build()?;
    let mut collection_options = CollectionOptions::new()?;
    collection_options.set_durability(Durability::Manual)?;
    let collection = Collection::create_and_open(&data_path, &schema, Some(&collection_options))?;

    let maintenance_options =
        CollectionMaintenanceOptions::new().try_with_interval(Duration::from_millis(25))?;
    let maintenance = collection.start_maintenance(maintenance_options)?;
    let mut doc = Doc::with_pk("doc-1")?;
    doc.add_string("category", "systems")?;
    assert_eq!(collection.insert(&[&doc])?.success_count, 1);
    assert!(collection.health()?.checkpoint_pending);

    maintenance.trigger()?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let worker = maintenance.health();
        let health = collection.health()?;
        if worker.last_successful_revision == Some(health.revision) && !health.checkpoint_pending {
            assert_eq!(worker.phase, CollectionMaintenancePhase::Running);
            assert_eq!(health.status, CollectionHealthStatus::Healthy);
            assert!(health.maintenance_active);
            break;
        }
        assert!(Instant::now() < deadline, "maintenance example timed out");
        thread::sleep(Duration::from_millis(10));
    }

    maintenance.close()?;
    assert_eq!(
        maintenance.health().phase,
        CollectionMaintenancePhase::Closed
    );
    assert!(!collection.health()?.maintenance_active);
    collection.close()?;
    shutdown()?;
    println!("Background maintenance and collection health workflow passed.");
    Ok(())
}
