//! Persistence primitives used by the collection handle.

mod derived_file;
mod diskann_file;
mod fault;
mod index_cache;
mod lock;
pub mod manifest;
pub mod snapshot;
pub mod wal;

use crate::config::{ConfigBuilder, Durability};
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use crate::storage_ceilings::StorageCeilings;
use fault::FaultInjector;
use lock::CollectionLock;
use manifest::Manifest;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) use derived_file::{PositionedFile, RandomAccessReader};
#[cfg(test)]
pub(crate) use fault::StallGate;
pub use wal::WalOperation;

/// Open storage state. The lock is held for the lifetime of the collection.
#[derive(Debug)]
pub(crate) struct StorageHandle {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub lock: CollectionLock,
    pub ceilings: StorageCeilings,
    faults: FaultInjector,
}

impl StorageHandle {
    pub fn create(
        path: &Path,
        schema: &CollectionSchema,
        read_only: bool,
        ceilings: StorageCeilings,
    ) -> Result<Self> {
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
        let faults = FaultInjector::default();
        let mut manifest = Manifest::new(schema.name.clone(), schema.digest());
        manifest.generation = 1;
        manifest.docs_checksum = snapshot::write_with_faults(
            path,
            schema,
            &[],
            manifest.generation,
            0,
            true,
            &faults,
            ceilings.max_snapshot_bytes(),
        )?;
        manifest.wal_active_seq = 1;
        manifest.wal_checkpoint_seq = 0;
        manifest.revision = 0;
        manifest.checkpoint_revision = 0;
        manifest.wal_ops_since_checkpoint = 0;
        manifest.wal_bytes_since_checkpoint = 0;
        manifest::write_with_faults(path, &manifest, true, &faults)?;
        Ok(Self {
            root: path.to_path_buf(),
            manifest,
            lock,
            ceilings,
            faults,
        })
    }

    pub fn open(
        path: &Path,
        read_only: bool,
        ceilings: StorageCeilings,
    ) -> Result<(Self, CollectionSchema, Vec<Doc>)> {
        if !path.exists() {
            return Err(Error::not_found(format!(
                "collection path does not exist: {}",
                path.display()
            )));
        }
        let lock = CollectionLock::acquire(path, read_only)?;
        let manifest = manifest::read(path)?;
        let (mut schema, mut docs) =
            snapshot::read(path, &manifest, ceilings.max_snapshot_bytes())?;
        let records = wal::replay(
            path,
            manifest.wal_checkpoint_seq.saturating_add(1),
            manifest.wal_active_seq,
            manifest.wal_bytes_since_checkpoint,
            ceilings.max_wal_replay_bytes(),
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
                ceilings,
                faults: FaultInjector::default(),
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
        let bytes = wal::append_with_faults(
            &self.root,
            self.manifest.wal_active_seq,
            self.manifest.wal_bytes_since_checkpoint,
            &record,
            sync,
            &self.faults,
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
        manifest::write_with_faults(&self.root, &next_manifest, sync, &self.faults)?;
        self.manifest = next_manifest;
        Ok(bytes)
    }

    pub fn checkpoint(
        &mut self,
        schema: &CollectionSchema,
        docs: &(impl snapshot::SnapshotDocs + ?Sized),
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
        let Some(checksum) =
            self.next_snapshot_checksum(schema, docs, generation, revision, sync)?
        else {
            return Ok(());
        };
        let mut next_manifest = self.manifest.clone();
        next_manifest.format_version = manifest::FORMAT_VERSION;
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
        manifest::write_with_faults(&self.root, &next_manifest, sync, &self.faults)?;
        self.manifest = next_manifest;

        // The manifest is already committed. Old generations are harmless,
        // so cleanup is best-effort: reporting a prune error here would make
        // callers believe the transaction failed after it actually committed.
        // If WAL cleanup is interrupted, stop this maintenance pass so the
        // remaining files match an actual crash at that boundary. Both prune
        // operations are optional after manifest publication and can be
        // retried by a later checkpoint.
        if wal::prune_with_faults(&self.root, self.manifest.wal_checkpoint_seq, &self.faults)
            .is_ok()
        {
            let _snapshot_prune = snapshot::prune_with_faults(
                &self.root,
                self.manifest.generation,
                self.manifest.docs_checksum,
                self.ceilings.max_snapshot_bytes(),
                &self.faults,
            );
        }
        Ok(())
    }

    /// `Ok(None)` means the committed snapshot already matches this revision.
    fn next_snapshot_checksum(
        &self,
        schema: &CollectionSchema,
        docs: &(impl snapshot::SnapshotDocs + ?Sized),
        generation: u64,
        revision: u64,
        sync: bool,
    ) -> Result<Option<u32>> {
        // A schema change cannot be a document delta. Reading the previous
        // snapshot only to throw it away is the whole decode of every body.
        let previous = if self.manifest.schema_digest == schema.digest()
            && self.manifest.format_version >= manifest::FORMAT_VERSION
            && self.manifest.generation > 0
        {
            snapshot::read(
                &self.root,
                &self.manifest,
                self.ceilings.max_snapshot_bytes(),
            )
            .ok()
        } else {
            None
        };
        if let Some((previous_schema, previous_docs)) = previous.as_ref() {
            // A second flush of an already checkpointed revision has no new
            // bytes to publish. Leaving the existing snapshot in place keeps
            // one authoritative generation instead of an empty delta.
            if previous_schema.digest() == schema.digest()
                && self.manifest.revision == revision
                && self.manifest.wal_ops_since_checkpoint == 0
                && snapshot::documents_unchanged(previous_docs, docs)
            {
                return Ok(None);
            }
            if !previous_docs.is_empty() && previous_schema.digest() == schema.digest() {
                return Ok(Some(snapshot::write_delta_with_faults(
                    &self.root,
                    schema,
                    docs,
                    previous_docs,
                    generation,
                    revision,
                    self.manifest.generation,
                    self.manifest.docs_checksum,
                    sync,
                    &self.faults,
                    self.ceilings.max_snapshot_bytes(),
                )?));
            }
        }
        Ok(Some(snapshot::write_with_faults(
            &self.root,
            schema,
            docs,
            generation,
            revision,
            sync,
            &self.faults,
            self.ceilings.max_snapshot_bytes(),
        )?))
    }

    pub fn should_checkpoint(&self, config: &ConfigBuilder) -> bool {
        config
            .wal_max_ops
            .is_some_and(|v| self.manifest.wal_ops_since_checkpoint >= v)
            || config
                .wal_max_bytes
                .is_some_and(|v| self.manifest.wal_bytes_since_checkpoint >= v)
    }

    pub(crate) fn read_index_cache(&self) -> Result<Option<Vec<u8>>> {
        index_cache::read(&self.root, self.ceilings.max_index_cache_file_bytes())
    }

    pub(crate) fn write_index_cache(&self, bytes: &[u8], sync: bool) -> Result<()> {
        if !self.lock.is_exclusive() {
            return Err(Error::permission_denied(
                "cannot write an index cache through a read-only handle",
            ));
        }
        index_cache::write(
            &self.root,
            bytes,
            sync,
            self.ceilings.max_index_cache_file_bytes(),
        )
    }

    pub(crate) fn open_diskann_file(&self) -> Result<Option<PositionedFile>> {
        diskann_file::open(&self.root, self.ceilings.max_diskann_file_bytes())
    }

    pub(crate) fn write_diskann_file(&self, bytes: &[u8], sync: bool) -> Result<()> {
        if !self.lock.is_exclusive() {
            return Err(Error::permission_denied(
                "cannot write a DiskANN sidecar through a read-only handle",
            ));
        }
        self.faults.hit(fault::FaultPoint::DiskannWritten)?;
        diskann_file::write(
            &self.root,
            bytes,
            sync,
            self.ceilings.max_diskann_file_bytes(),
        )
    }

    pub(crate) fn index_cache_identity(&self) -> String {
        let manifest = &self.manifest;
        format!(
            "v2:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            manifest.format_version,
            manifest.collection_name.len(),
            manifest.collection_name,
            manifest.schema_digest,
            manifest.generation,
            manifest.revision,
            manifest.checkpoint_revision,
            manifest.wal_active_seq,
            manifest.wal_checkpoint_seq,
            manifest.wal_ops_since_checkpoint,
            manifest.wal_bytes_since_checkpoint,
            manifest.docs_checksum,
        )
    }

    #[cfg(test)]
    fn arm_fault(&self, point: fault::FaultPoint) {
        self.faults.arm(point);
    }

    #[cfg(test)]
    pub(crate) fn arm_wal_sync_stall(&self) -> StallGate {
        self.faults.arm_stall(fault::FaultPoint::WalSynced)
    }

    #[cfg(test)]
    pub(crate) fn arm_diskann_write_fault(&self) {
        self.faults.arm(fault::FaultPoint::DiskannWritten);
    }

    #[cfg(test)]
    fn fault_fired(&self, point: fault::FaultPoint) -> bool {
        self.faults.fired(point)
    }

    #[cfg(test)]
    pub(crate) fn diskann_write_fault_fired(&self) -> bool {
        self.fault_fired(fault::FaultPoint::DiskannWritten)
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
        WalOperation::SchemaOnly { schema: next } => {
            next.validate()?;
            if next.name != schema.name {
                return Err(Error::internal("WAL schema name mismatch"));
            }
            *schema = next;
        }
    }
    Ok(())
}

#[cfg(test)]
mod fault_tests;
#[cfg(test)]
mod recovery_fuzz_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
