//! Thread-safe collection handle and transaction coordinator.

mod query_api;
mod query_engine;
mod validation;

use crate::config::{current_config, ConfigBuilder, Durability, IoBackend};
use crate::doc::{Doc, FieldValue};
use crate::error::{Error, ErrorCode, Result};
use crate::schema::{
    AddColumnOption, AlterColumnOption, CollectionSchema, FieldSchema, IndexParams,
};
pub use crate::stats::IndexStat;
use crate::stats::{StatsRegistry, StatsSnapshot};
use crate::storage::{StorageHandle, WalOperation};
use query_engine::{matches_filter, parse_filter_expression};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};
use validation::{merge_patch, prepare_mutation_batch, runtime_indexes};

/// Options for creating or opening a collection.
#[derive(Debug, Clone)]
pub struct CollectionOptions {
    pub read_only: bool,
    pub enable_mmap: bool,
    pub max_buffer_size: u64,
    pub segment_num: u32,
    pub durability: Durability,
    pub io_backend: IoBackend,
}

impl CollectionOptions {
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }
    pub fn set_enable_mmap(&mut self, enable: bool) -> Result<()> {
        self.enable_mmap = enable;
        Ok(())
    }
    pub fn enable_mmap(&self) -> bool {
        self.enable_mmap
    }
    pub fn set_max_buffer_size(&mut self, size: u64) -> Result<()> {
        self.max_buffer_size = size;
        Ok(())
    }
    pub fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }
    pub fn set_read_only(&mut self, read_only: bool) -> Result<()> {
        self.read_only = read_only;
        Ok(())
    }
    pub fn read_only(&self) -> bool {
        self.read_only
    }
    pub fn set_segment_num(&mut self, count: u32) -> Result<()> {
        if count > 1024 {
            return Err(Error::invalid_argument("segment_num must be at most 1024"));
        }
        self.segment_num = count;
        Ok(())
    }
    pub fn set_durability(&mut self, value: Durability) -> Result<()> {
        self.durability = value;
        Ok(())
    }
    pub fn set_io_backend(&mut self, value: IoBackend) -> Result<()> {
        self.io_backend = value;
        Ok(())
    }
}

impl Default for CollectionOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            enable_mmap: true,
            max_buffer_size: 0,
            segment_num: 0,
            durability: Durability::Always,
            io_backend: IoBackend::Portable,
        }
    }
}

/// Per-document outcome in a batch write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocWriteResult {
    pub success: bool,
    pub code: ErrorCode,
    pub message: String,
}

impl DocWriteResult {
    pub fn is_success(&self) -> bool {
        self.success
    }
}

/// Result of a batch write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    pub success_count: u64,
    pub error_count: u64,
    pub results: Vec<DocWriteResult>,
}

/// Public collection statistics (the fields used by the official SDK are kept
/// first; additional counters are additive).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionStats {
    pub doc_count: u64,
    pub indexes: Vec<IndexStat>,
    pub revision: u64,
    pub read_only: bool,
    pub wal_active_seq: u64,
    pub wal_checkpoint_seq: u64,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
}

#[derive(Debug, Clone)]
struct RuntimeIndex {
    params: IndexParams,
    source_revision: u64,
    document_count: u64,
}

#[derive(Debug, Clone)]
struct CollectionState {
    path: PathBuf,
    schema: CollectionSchema,
    docs: BTreeMap<String, Doc>,
    revision: u64,
    options: CollectionOptions,
    indexes: BTreeMap<String, RuntimeIndex>,
    stats: Arc<StatsRegistry>,
}

#[derive(Debug)]
struct CollectionInner {
    state: RwLock<CollectionState>,
    storage: Mutex<StorageHandle>,
    writer: Mutex<()>,
    closed: AtomicBool,
}

/// Cheap, cloneable, thread-safe handle to one collection.
#[derive(Clone, Debug)]
pub struct Collection {
    inner: Arc<CollectionInner>,
}

impl Collection {
    pub fn create_and_open(
        path: &str,
        schema: &CollectionSchema,
        options: Option<&CollectionOptions>,
    ) -> Result<Self> {
        let options = options.cloned().unwrap_or_default();
        let root = Path::new(path);
        let storage = StorageHandle::create(root, schema, options.read_only)?;
        let state = CollectionState {
            path: root.to_path_buf(),
            schema: schema.clone(),
            docs: BTreeMap::new(),
            revision: 0,
            options,
            indexes: runtime_indexes(schema, 0, 0),
            stats: Arc::new(StatsRegistry::default()),
        };
        Ok(Self {
            inner: Arc::new(CollectionInner {
                state: RwLock::new(state),
                storage: Mutex::new(storage),
                writer: Mutex::new(()),
                closed: AtomicBool::new(false),
            }),
        })
    }

    pub fn create(
        path: &str,
        schema: &CollectionSchema,
        options: Option<&CollectionOptions>,
    ) -> Result<Self> {
        Self::create_and_open(path, schema, options)
    }

    pub fn open(path: &str, options: Option<&CollectionOptions>) -> Result<Self> {
        let options = options.cloned().unwrap_or_default();
        let (storage, schema, docs) = StorageHandle::open(Path::new(path), options.read_only)?;
        if schema.name.trim().is_empty() {
            return Err(Error::internal("persisted collection has an empty name"));
        }
        let revision = storage.manifest.revision;
        let docs: BTreeMap<String, Doc> = docs
            .into_iter()
            .filter_map(|doc| doc.get_pk().map(str::to_string).map(|id| (id, doc)))
            .collect();
        let document_count = docs.len() as u64;
        let state = CollectionState {
            path: PathBuf::from(path),
            schema: schema.clone(),
            docs,
            revision,
            options,
            indexes: runtime_indexes(&schema, revision, document_count),
            stats: Arc::new(StatsRegistry::default()),
        };
        Ok(Self {
            inner: Arc::new(CollectionInner {
                state: RwLock::new(state),
                storage: Mutex::new(storage),
                writer: Mutex::new(()),
                closed: AtomicBool::new(false),
            }),
        })
    }

    pub fn path(&self) -> PathBuf {
        self.inner
            .state
            .read()
            .map(|state| state.path.clone())
            .unwrap_or_default()
    }

    pub fn is_open(&self) -> bool {
        !self.inner.closed.load(AtomicOrdering::Acquire)
    }

    pub fn flush(&self) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let (schema, docs) = {
            let state = self
                .inner
                .state
                .read()
                .map_err(|_| Error::internal("collection state lock poisoned"))?;
            (
                state.schema.clone(),
                state.docs.values().cloned().collect::<Vec<_>>(),
            )
        };
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        let revision = storage.manifest.revision;
        storage.checkpoint(&schema, &docs, revision, true)?;
        Ok(())
    }

    pub fn close(self) -> Result<()> {
        if self.is_open() {
            let read_only = self
                .inner
                .state
                .read()
                .map_err(|_| Error::internal("collection state lock poisoned"))?
                .options
                .read_only;
            if !read_only {
                self.flush()?;
            }
            self.inner.closed.store(true, AtomicOrdering::Release);
        }
        Ok(())
    }

    pub fn destroy(self) -> Result<()> {
        let path = self.path();
        self.close()?;
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| Error::internal(format!("destroy collection: {e}")))?;
        }
        Ok(())
    }

    pub fn schema(&self) -> Result<CollectionSchema> {
        self.ensure_open()?;
        self.inner
            .state
            .read()
            .map(|state| state.schema.clone())
            .map_err(|_| Error::internal("collection state lock poisoned"))
    }

    pub fn stats(&self) -> Result<CollectionStats> {
        self.ensure_open()?;
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        let storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        let indexes = state
            .indexes
            .iter()
            .map(|(name, index)| IndexStat {
                name: name.clone(),
                index_type: index.params.index_type,
                completeness: if index.source_revision == state.revision {
                    1.0
                } else {
                    0.0
                },
                source_revision: index.source_revision,
                document_count: index.document_count,
                state: if index.source_revision == state.revision {
                    "ready".into()
                } else {
                    "stale".into()
                },
            })
            .collect();
        Ok(CollectionStats {
            doc_count: state.docs.len() as u64,
            indexes,
            revision: state.revision,
            read_only: state.options.read_only,
            wal_active_seq: storage.manifest.wal_active_seq,
            wal_checkpoint_seq: storage.manifest.wal_checkpoint_seq,
            wal_ops_since_checkpoint: storage.manifest.wal_ops_since_checkpoint,
            wal_bytes_since_checkpoint: storage.manifest.wal_bytes_since_checkpoint,
        })
    }

    pub fn stats_snapshot(&self) -> Result<StatsSnapshot> {
        let basic = self.stats()?;
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        let registry = Arc::clone(&state.stats);
        Ok(StatsSnapshot {
            collection_name: state.schema.name.clone(),
            revision: basic.revision,
            doc_count: basic.doc_count,
            query_count: registry.query_count.load(AtomicOrdering::Relaxed),
            fts_query_count: registry.fts_query_count.load(AtomicOrdering::Relaxed),
            ann_query_count: registry.ann_query_count.load(AtomicOrdering::Relaxed),
            exact_query_count: registry.exact_query_count.load(AtomicOrdering::Relaxed),
            filtered_query_count: registry.filtered_query_count.load(AtomicOrdering::Relaxed),
            radius_query_count: registry.radius_query_count.load(AtomicOrdering::Relaxed),
            candidates_scanned: registry.candidates_scanned.load(AtomicOrdering::Relaxed),
            indexed_field_count: basic.indexes.len(),
            indexes: basic.indexes,
            read_only: basic.read_only,
            wal_active_seq: basic.wal_active_seq,
            wal_checkpoint_seq: basic.wal_checkpoint_seq,
            wal_ops_since_checkpoint: basic.wal_ops_since_checkpoint,
            wal_bytes_since_checkpoint: basic.wal_bytes_since_checkpoint,
        })
    }

    pub fn count(&self) -> Result<usize> {
        self.ensure_open()?;
        self.inner
            .state
            .read()
            .map(|state| state.docs.len())
            .map_err(|_| Error::internal("collection state lock poisoned"))
    }

    // ---------------------------------------------------------------------
    // DML
    // ---------------------------------------------------------------------

    pub fn insert(&self, docs: &[&Doc]) -> Result<WriteResult> {
        self.mutate_documents(docs, Mutation::Insert)
    }

    pub fn update(&self, docs: &[&Doc]) -> Result<WriteResult> {
        self.mutate_documents(docs, Mutation::Update)
    }

    pub fn upsert(&self, docs: &[&Doc]) -> Result<WriteResult> {
        self.mutate_documents(docs, Mutation::Upsert)
    }

    pub fn delete(&self, pks: &[&str]) -> Result<WriteResult> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut results = Vec::with_capacity(pks.len());
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for pk in pks {
            if pk.is_empty() {
                results.push(write_error(
                    ErrorCode::InvalidArgument,
                    "primary key is empty",
                ));
            } else if !seen.insert(*pk) {
                results.push(write_error(
                    ErrorCode::AlreadyExists,
                    "duplicate primary key in delete batch",
                ));
            } else if state.docs.contains_key(*pk) {
                ids.push((*pk).to_string());
                results.push(write_success());
            } else {
                results.push(write_error(
                    ErrorCode::NotFound,
                    format!("document '{pk}' not found"),
                ));
            }
        }
        if !ids.is_empty() {
            let config = options_config(&state.options);
            let revision = next_revision(state.revision)?;
            let mut next_docs = state.docs.clone();
            for id in &ids {
                next_docs.remove(id);
            }
            let mut storage = self
                .inner
                .storage
                .lock()
                .map_err(|_| Error::internal("storage lock poisoned"))?;
            storage.append(revision, WalOperation::Delete { ids: ids.clone() }, &config)?;
            state.docs = next_docs;
            state.revision = revision;
            maybe_checkpoint(&mut storage, &state, &config)?;
        }
        Ok(write_result(results))
    }

    pub fn delete_by_filter(&self, filter: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let ids: Vec<String> = state
            .docs
            .values()
            .filter(|doc| matches_filter(doc, Some(filter)))
            .filter_map(|doc| doc.get_pk().map(str::to_string))
            .collect();
        if ids.is_empty() {
            // Parse even when no documents match, so malformed filters are not
            // silently accepted.
            parse_filter_expression(filter)?;
            return Ok(());
        }
        let config = options_config(&state.options);
        let revision = next_revision(state.revision)?;
        let mut next_docs = state.docs.clone();
        for id in &ids {
            next_docs.remove(id);
        }
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(revision, WalOperation::Delete { ids: ids.clone() }, &config)?;
        state.docs = next_docs;
        state.revision = revision;
        maybe_checkpoint(&mut storage, &state, &config)?;
        Ok(())
    }

    fn mutate_documents(&self, docs: &[&Doc], mutation: Mutation) -> Result<WriteResult> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;

        let (accepted, outcomes) = prepare_mutation_batch(&state, docs, mutation);

        if accepted.is_empty() {
            return Ok(write_result(outcomes));
        }
        let operation = match mutation {
            Mutation::Insert => WalOperation::Insert {
                docs: accepted.clone(),
            },
            Mutation::Update => WalOperation::Update {
                docs: accepted.clone(),
            },
            Mutation::Upsert => WalOperation::Upsert {
                docs: accepted.clone(),
            },
        };
        let config = options_config(&state.options);
        let revision = next_revision(state.revision)?;
        let mut next_docs = state.docs.clone();
        for doc in &accepted {
            let Some(pk) = doc.get_pk().map(str::to_string) else {
                continue;
            };
            match mutation {
                Mutation::Insert | Mutation::Upsert => {
                    next_docs.insert(pk, doc.clone());
                }
                Mutation::Update => {
                    if let Some(existing) = next_docs.get_mut(&pk) {
                        merge_patch(existing, doc)?;
                    }
                }
            }
        }
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(revision, operation, &config)?;
        state.docs = next_docs;
        state.revision = revision;
        maybe_checkpoint(&mut storage, &state, &config)?;
        Ok(write_result(outcomes))
    }

    // ---------------------------------------------------------------------
    // Index and schema management
    // ---------------------------------------------------------------------

    pub fn create_index(&self, field_name: &str, params: &IndexParams) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        next.schema
            .add_index(field_name, &params.marked_built(true))?;
        let count = next.docs.len() as u64;
        let revision = next_revision(state.revision)?;
        next.indexes.insert(
            field_name.to_string(),
            RuntimeIndex {
                params: params.marked_built(true),
                source_revision: revision,
                document_count: count,
            },
        );
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn drop_index(&self, field_name: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        next.schema.drop_index(field_name)?;
        next.indexes.remove(field_name);
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn optimize(&self) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        let revision = next_revision(next.revision)?;
        let document_count = next.docs.len() as u64;
        for index in next.indexes.values_mut() {
            index.source_revision = revision;
            index.document_count = document_count;
        }
        next.revision = revision;
        let schema = next.schema.clone();
        let docs: Vec<Doc> = next.docs.values().cloned().collect();
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.checkpoint(&schema, &docs, revision, true)?;
        *state = next;
        Ok(())
    }

    pub fn add_column(&self, field_schema: &FieldSchema, default_expr: Option<&str>) -> Result<()> {
        self.add_column_with_options(field_schema, default_expr, AddColumnOption::default())
    }

    pub fn add_column_with_options(
        &self,
        field_schema: &FieldSchema,
        default_expr: Option<&str>,
        _option: AddColumnOption,
    ) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        next.schema.add_field(field_schema)?;
        let default = default_expr.map(parse_default_expression);
        if let Some(value) = default {
            for doc in next.docs.values_mut() {
                doc.set_field_value(&field_schema.name, value.clone())?;
            }
        }
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn drop_column(&self, name: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        next.schema.drop_field(name)?;
        for doc in next.docs.values_mut() {
            doc.remove_field(name)?;
        }
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
        if old_name == new_name {
            return Ok(());
        }
        if new_name.trim().is_empty() || new_name.contains('\0') {
            return Err(Error::invalid_argument("new field name is invalid"));
        }
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        if next.schema.has_field(new_name) {
            return Err(Error::already_exists(format!(
                "field '{new_name}' already exists"
            )));
        }
        if let Some(field) = next
            .schema
            .fields
            .iter_mut()
            .find(|field| field.name == old_name)
        {
            field.name = new_name.to_string();
        } else if let Some(field) = next
            .schema
            .vectors
            .iter_mut()
            .find(|field| field.name == old_name)
        {
            field.name = new_name.to_string();
        } else {
            return Err(Error::not_found(format!("field '{old_name}' not found")));
        }
        for doc in next.docs.values_mut() {
            if let Some(value) = doc.field(old_name).cloned() {
                doc.remove_field(old_name)?;
                doc.set_field_value(new_name, value)?;
            } else if let Some(value) = doc.vector(old_name).cloned() {
                doc.remove_field(old_name)?;
                doc.set_vector_value(new_name, value)?;
            }
        }
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn alter_column(
        &self,
        field_schema: &FieldSchema,
        _option: AlterColumnOption,
    ) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let mut next = state.clone();
        let target = next
            .schema
            .fields
            .iter_mut()
            .find(|field| field.name == field_schema.name)
            .ok_or_else(|| Error::not_found(format!("field '{}' not found", field_schema.name)))?;
        if target.data_type != field_schema.data_type || target.dimension != field_schema.dimension
        {
            return Err(Error::invalid_argument(
                "altering a field's data type or dimension would invalidate existing data",
            ));
        }
        *target = field_schema.clone();
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    fn snapshot_state(&self) -> Result<(CollectionSchema, Vec<Doc>, u64, Arc<StatsRegistry>)> {
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        Ok((
            state.schema.clone(),
            state.docs.values().cloned().collect(),
            state.revision,
            Arc::clone(&state.stats),
        ))
    }

    fn ensure_open(&self) -> Result<()> {
        if self.inner.closed.load(AtomicOrdering::Acquire) {
            Err(Error::failed_precondition("collection is closed"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Mutation {
    Insert,
    Update,
    Upsert,
}

fn options_config(options: &CollectionOptions) -> ConfigBuilder {
    let mut config = current_config();
    config.durability = options.durability;
    config.io_backend = options.io_backend;
    config
}

fn ensure_writable(options: &CollectionOptions) -> Result<()> {
    if options.read_only {
        Err(Error::permission_denied("collection is read-only"))
    } else {
        Ok(())
    }
}

fn write_success() -> DocWriteResult {
    DocWriteResult {
        success: true,
        code: ErrorCode::Unknown,
        message: String::new(),
    }
}

fn write_error(code: ErrorCode, message: impl Into<String>) -> DocWriteResult {
    DocWriteResult {
        success: false,
        code,
        message: message.into(),
    }
}

fn write_result(results: Vec<DocWriteResult>) -> WriteResult {
    let success_count = results.iter().filter(|result| result.success).count() as u64;
    WriteResult {
        success_count,
        error_count: results.len() as u64 - success_count,
        results,
    }
}

fn commit_schema_change(
    storage: &mut StorageHandle,
    state: &mut CollectionState,
    mut next: CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    let revision = next_revision(state.revision)?;
    next.revision = revision;
    let schema = next.schema.clone();
    let docs: Vec<Doc> = next.docs.values().cloned().collect();
    storage.append(
        revision,
        WalOperation::Schema {
            schema: schema.clone(),
            docs: docs.clone(),
        },
        config,
    )?;

    // The WAL + manifest pair is the commit point. Publish the same state in
    // memory before checkpoint maintenance so a checkpoint error cannot leave
    // this process behind the already committed revision.
    *state = next;
    let sync = !matches!(config.durability, Durability::Manual);
    storage.checkpoint(&schema, &docs, revision, sync)
}

fn maybe_checkpoint(
    storage: &mut StorageHandle,
    state: &CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    let should =
        matches!(config.durability, Durability::Interval) && storage.should_checkpoint(config);
    if should {
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        storage.checkpoint(&state.schema, &docs, state.revision, true)?;
    }
    Ok(())
}

fn next_revision(current: u64) -> Result<u64> {
    current
        .checked_add(1)
        .ok_or_else(|| Error::resource_exhausted("collection revision overflow"))
}

fn parse_default_expression(expression: &str) -> FieldValue {
    let trimmed = expression.trim();
    if trimmed.eq_ignore_ascii_case("null") {
        return FieldValue::Null;
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return FieldValue::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return FieldValue::Bool(false);
    }
    if let Ok(value) = trimmed.parse::<i64>() {
        return FieldValue::Int64(value);
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        return FieldValue::Double(value);
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return FieldValue::Json(value);
    }
    FieldValue::String(trimmed.trim_matches(['\'', '"']).to_string())
}
