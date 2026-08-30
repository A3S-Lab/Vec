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

pub use wal::WalOperation;

/// Open storage state. The lock is held for the lifetime of the collection.
#[derive(Debug)]
pub(crate) struct StorageHandle {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub lock: CollectionLock,
}

impl StorageHandle {
    pub fn create(path: &Path, schema: &CollectionSchema, read_only: bool) -> Result<Self> {
        if read_only {
            return Err(Error::permission_denied(
                "cannot create a collection through a read-only handle",
            ));
        }
        schema.validate()?;
        if path.exists() && (path.join("manifest.json").exists() || path.join("segments").exists())
        {
            return Err(Error::already_exists(format!(
                "collection already exists at {}",
                path.display()
            )));
        }
        fs::create_dir_all(path)
            .map_err(|e| Error::internal(format!("create collection path: {e}")))?;
        let lock = CollectionLock::acquire(path, false)?;
        let mut manifest = Manifest::new(schema.name.clone(), schema.digest());
        manifest.generation = 1;
        manifest.docs_checksum = snapshot::write(path, schema, &[], manifest.generation, 0, true)?;
        manifest.wal_active_seq = 1;
        manifest.wal_checkpoint_seq = 0;
        manifest.revision = 0;
        manifest.checkpoint_revision = 0;
        manifest.wal_ops_since_checkpoint = 0;
        manifest.wal_bytes_since_checkpoint = 0;
        manifest::write(path, &manifest, true)?;
        Ok(Self {
            root: path.to_path_buf(),
            manifest,
            lock,
        })
    }

    pub fn open(path: &Path, read_only: bool) -> Result<(Self, CollectionSchema, Vec<Doc>)> {
        if !path.exists() {
            return Err(Error::not_found(format!(
                "collection path does not exist: {}",
                path.display()
            )));
        }
        let lock = CollectionLock::acquire(path, read_only)?;
        let manifest = manifest::read(path)?;
        let (mut schema, mut docs) = snapshot::read(path, &manifest)?;
        let records = wal::replay(
            path,
            manifest.wal_checkpoint_seq.saturating_add(1),
            manifest.wal_active_seq,
            manifest.wal_bytes_since_checkpoint,
        )?;
        let mut recovered_revision = manifest.checkpoint_revision;
        for record in records {
            let expected_revision = recovered_revision
                .checked_add(1)
                .ok_or_else(|| Error::resource_exhausted("collection revision overflow"))?;
            if record.revision != expected_revision {
                return Err(Error::internal(format!(
                    "non-monotonic WAL revision: expected {expected_revision}, got {}",
                    record.revision
                )));
            }
            apply_operation(&mut docs, &mut schema, record.operation)?;
            recovered_revision = record.revision;
        }
        if recovered_revision != manifest.revision {
            return Err(Error::internal(format!(
                "manifest revision {} is not recoverable from checkpoint revision {} and committed WAL",
                manifest.revision, manifest.checkpoint_revision
            )));
        }
        schema.validate()?;
        Ok((
            Self {
                root: path.to_path_buf(),
                manifest,
                lock,
            },
            schema,
            docs,
        ))
    }

    pub fn append(
        &mut self,
        revision: u64,
        operation: WalOperation,
        config: &ConfigBuilder,
    ) -> Result<u64> {
        if !self.lock.is_exclusive() {
            return Err(Error::permission_denied("collection is read-only"));
        }
        let expected_revision = self
            .manifest
            .revision
            .checked_add(1)
            .ok_or_else(|| Error::resource_exhausted("collection revision overflow"))?;
        if revision != expected_revision {
            return Err(Error::failed_precondition(format!(
                "WAL revision must advance from {} to {expected_revision}, got {revision}",
                self.manifest.revision
            )));
        }
        let record = wal::WalRecord::new(revision, operation)?;
        let sync = matches!(config.durability, Durability::Always);
        let bytes = wal::append(
            &self.root,
            self.manifest.wal_active_seq,
            self.manifest.wal_bytes_since_checkpoint,
            &record,
            sync,
        )?;
        let mut next_manifest = self.manifest.clone();
        next_manifest.revision = revision;
        next_manifest.wal_ops_since_checkpoint = next_manifest
            .wal_ops_since_checkpoint
            .checked_add(1)
            .ok_or_else(|| Error::resource_exhausted("WAL operation count overflow"))?;
        next_manifest.wal_bytes_since_checkpoint = next_manifest
            .wal_bytes_since_checkpoint
            .checked_add(bytes)
            .ok_or_else(|| Error::resource_exhausted("WAL byte count overflow"))?;
        manifest::write(&self.root, &next_manifest, sync)?;
        self.manifest = next_manifest;
        Ok(bytes)
    }

    pub fn checkpoint(
        &mut self,
        schema: &CollectionSchema,
        docs: &[Doc],
        revision: u64,
        sync: bool,
    ) -> Result<()> {
        if !self.lock.is_exclusive() {
            return Err(Error::permission_denied("collection is read-only"));
        }
        if revision < self.manifest.revision {
            return Err(Error::failed_precondition(format!(
                "checkpoint revision {revision} precedes committed revision {}",
                self.manifest.revision
            )));
        }
        let generation = self
            .manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::resource_exhausted("snapshot generation overflow"))?;
        let checksum = snapshot::write(&self.root, schema, docs, generation, revision, sync)?;
        let mut next_manifest = self.manifest.clone();
        next_manifest.collection_name.clone_from(&schema.name);
        next_manifest.schema_digest = schema.digest();
        next_manifest.generation = generation;
        next_manifest.revision = revision;
        next_manifest.checkpoint_revision = revision;
        next_manifest.docs_checksum = checksum;
        next_manifest.wal_checkpoint_seq = self.manifest.wal_active_seq;
        next_manifest.wal_active_seq = self
            .manifest
            .wal_active_seq
            .checked_add(1)
            .ok_or_else(|| Error::resource_exhausted("WAL sequence overflow"))?;
        next_manifest.wal_ops_since_checkpoint = 0;
        next_manifest.wal_bytes_since_checkpoint = 0;
        manifest::write(&self.root, &next_manifest, sync)?;
        self.manifest = next_manifest;

        // The manifest is already committed. Old generations are harmless,
        // so cleanup is best-effort: reporting a prune error here would make
        // callers believe the transaction failed after it actually committed.
        let _wal_prune = wal::prune(&self.root, self.manifest.wal_checkpoint_seq);
        let _snapshot_prune = snapshot::prune(&self.root, self.manifest.generation);
        Ok(())
    }

    pub fn should_checkpoint(&self, config: &ConfigBuilder) -> bool {
        config
            .wal_max_ops
            .is_some_and(|v| self.manifest.wal_ops_since_checkpoint >= v)
            || config
                .wal_max_bytes
                .is_some_and(|v| self.manifest.wal_bytes_since_checkpoint >= v)
    }
}

fn apply_operation(
    docs: &mut Vec<Doc>,
    schema: &mut CollectionSchema,
    operation: WalOperation,
) -> Result<()> {
    match operation {
        WalOperation::Insert { docs: incoming } => {
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
        WalOperation::Upsert { docs: incoming } => {
            for doc in incoming {
                let pk = doc
                    .get_pk()
                    .ok_or_else(|| {
                        Error::invalid_argument("WAL upsert document has no primary key")
                    })?
                    .to_string();
                if let Some(existing) = docs.iter_mut().find(|d| d.get_pk() == Some(pk.as_str())) {
                    *existing = doc;
                } else {
                    docs.push(doc);
                }
            }
        }
        WalOperation::Update { docs: patches } => {
            for patch in patches {
                let pk = patch
                    .get_pk()
                    .ok_or_else(|| {
                        Error::invalid_argument("WAL update document has no primary key")
                    })?
                    .to_string();
                let existing = docs
                    .iter_mut()
                    .find(|d| d.get_pk() == Some(pk.as_str()))
                    .ok_or_else(|| {
                        Error::not_found(format!("document '{pk}' not found during WAL replay"))
                    })?;
                for (name, value) in patch.fields() {
                    existing.set_field_value(name, value.clone())?;
                }
                for (name, value) in patch.vectors() {
                    existing.set_vector_value(name, value.clone())?;
                }
                if patch.get_score() != 0.0 {
                    existing.set_score(patch.get_score())?;
                }
            }
        }
        WalOperation::Delete { ids } => {
            docs.retain(|doc| !doc.get_pk().is_some_and(|pk| ids.iter().any(|id| id == pk)));
        }
        WalOperation::Schema {
            schema: next,
            docs: next_docs,
        } => {
            next.validate()?;
            if next.name != schema.name {
                return Err(Error::internal("WAL schema name mismatch"));
            }
            *schema = next;
            *docs = next_docs;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
