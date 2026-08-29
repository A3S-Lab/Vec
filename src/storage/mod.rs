//! Persistence primitives used by the collection handle.

mod lock;
pub mod manifest;
pub mod snapshot;
pub mod wal;

use crate::config::{ConfigBuilder, Durability};
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use lock::CollectionLock;
use manifest::Manifest;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) use lock::CollectionLock as WriterLock;
pub use manifest::Manifest as StorageManifest;
pub use wal::WalRecord;

/// Open storage state. The lock is held for the lifetime of the collection.
#[derive(Debug)]
pub(crate) struct StorageHandle {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub lock: CollectionLock,
}

impl StorageHandle {
    pub fn create(path: &Path, schema: &CollectionSchema, read_only: bool) -> Result<Self> {
        schema.validate()?;
        if path.exists() && (path.join("manifest.json").exists() || path.join("snapshot.json").exists()) {
            return Err(Error::already_exists(format!("collection already exists at {}", path.display())));
        }
        fs::create_dir_all(path).map_err(|e| Error::internal(format!("create collection path: {e}")))?;
        let lock = CollectionLock::acquire(path, read_only)?;
        let mut manifest = Manifest::new(schema.name.clone(), schema.digest());
        snapshot::write(path, schema, &[], &mut manifest, true)?;
        manifest.wal_active_seq = 1;
        manifest.wal_checkpoint_seq = 0;
        manifest.revision = 0;
        manifest.wal_ops_since_checkpoint = 0;
        manifest.wal_bytes_since_checkpoint = 0;
        manifest::write(path, &manifest, true)?;
        Ok(Self { root: path.to_path_buf(), manifest, lock })
    }

    pub fn open(path: &Path, read_only: bool) -> Result<(Self, CollectionSchema, Vec<Doc>)> {
        if !path.exists() {
            return Err(Error::not_found(format!("collection path does not exist: {}", path.display())));
        }
        let lock = CollectionLock::acquire(path, read_only)?;
        let manifest = manifest::read(path)?;
        let (schema, mut docs) = snapshot::read(path, Some(&manifest))?;
        let records = wal::replay(&path, manifest.wal_checkpoint_seq.saturating_add(1), manifest.wal_active_seq)?;
        for record in records {
            apply_record(&mut docs, &schema, record)?;
        }
        Ok((Self { root: path.to_path_buf(), manifest, lock }, schema, docs))
    }

    pub fn append(&mut self, record: &WalRecord, config: &ConfigBuilder) -> Result<u64> {
        if !self.lock.is_exclusive() {
            return Err(Error::permission_denied("collection is read-only"));
        }
        let sync = matches!(config.durability, Durability::Always);
        let bytes = wal::append(&self.root, self.manifest.wal_active_seq, record, sync)?;
        self.manifest.wal_ops_since_checkpoint = self.manifest.wal_ops_since_checkpoint.saturating_add(1);
        self.manifest.wal_bytes_since_checkpoint = self.manifest.wal_bytes_since_checkpoint.saturating_add(bytes);
        Ok(bytes)
    }

    pub fn checkpoint(&mut self, schema: &CollectionSchema, docs: &[Doc], config: &ConfigBuilder) -> Result<()> {
        if !self.lock.is_exclusive() {
            return Err(Error::permission_denied("collection is read-only"));
        }
        let sync = !matches!(config.durability, Durability::Manual);
        snapshot::write(&self.root, schema, docs, &mut self.manifest, sync)?;
        self.manifest.wal_checkpoint_seq = self.manifest.wal_active_seq;
        self.manifest.wal_active_seq = self.manifest.wal_active_seq.saturating_add(1);
        self.manifest.wal_ops_since_checkpoint = 0;
        self.manifest.wal_bytes_since_checkpoint = 0;
        manifest::write(&self.root, &self.manifest, sync)?;
        wal::prune(&self.root, self.manifest.wal_checkpoint_seq)?;
        Ok(())
    }

    pub fn should_checkpoint(&self, config: &ConfigBuilder) -> bool {
        config.wal_max_ops.is_some_and(|v| self.manifest.wal_ops_since_checkpoint >= v)
            || config.wal_max_bytes.is_some_and(|v| self.manifest.wal_bytes_since_checkpoint >= v)
    }
}

fn apply_record(docs: &mut Vec<Doc>, schema: &CollectionSchema, record: WalRecord) -> Result<()> {
    match record {
        WalRecord::Insert { docs: incoming } => {
            for doc in incoming {
                if let Some(pk) = doc.get_pk() {
                    if let Some(existing) = docs.iter_mut().find(|d| d.get_pk() == Some(pk)) {
                        *existing = doc;
                    } else {
                        docs.push(doc);
                    }
                }
            }
        }
        WalRecord::Upsert { docs: incoming } => {
            for doc in incoming {
                let pk = doc.get_pk().ok_or_else(|| Error::invalid_argument("WAL upsert document has no primary key"))?.to_string();
                if let Some(existing) = docs.iter_mut().find(|d| d.get_pk() == Some(pk.as_str())) { *existing = doc; } else { docs.push(doc); }
            }
        }
        WalRecord::Update { docs: patches } => {
            for patch in patches {
                let pk = patch.get_pk().ok_or_else(|| Error::invalid_argument("WAL update document has no primary key"))?.to_string();
                let existing = docs.iter_mut().find(|d| d.get_pk() == Some(pk.as_str())).ok_or_else(|| Error::not_found(format!("document '{pk}' not found during WAL replay")))?;
                for (name, value) in patch.fields().iter() { existing.set_field_value(name, value.clone())?; }
                for (name, value) in patch.vectors().iter() { existing.set_vector_value(name, value.clone())?; }
                if patch.get_score() != 0.0 { existing.set_score(patch.get_score())?; }
            }
        }
        WalRecord::Delete { ids } => docs.retain(|doc| !doc.get_pk().is_some_and(|pk| ids.iter().any(|id| id == pk))),
        WalRecord::Schema { schema: next } => {
            if next.name != schema.name { return Err(Error::internal("WAL schema name mismatch")); }
        }
    }
    Ok(())
}
