//! Thread-safe collection handle and transaction coordinator.

mod checkpoint;
mod configuration;
mod index_api;
mod query_api;
mod query_contract;
mod query_engine;
mod validation;

use crate::config::{ConfigBuilder, Durability};
use crate::doc::{Doc, DocumentMap};
use crate::error::{Error, ErrorCode, Result};
use crate::index::IndexRegistry;
use crate::schema::{AddColumnOption, AlterColumnOption, CollectionSchema, FieldSchema};
pub use crate::stats::IndexStat;
use crate::stats::{StatsRegistry, StatsSnapshot};
use crate::storage::{StorageHandle, WalOperation};
use crate::types::IndexType;
use checkpoint::{commit_prepared_schema_change, maybe_checkpoint, persist_index_cache};
use configuration::options_config;
use query_engine::{matches_filter, parse_filter_expression};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};
use validation::{
    merge_patch, normalize_doc, parse_default_expression, prepare_mutation_batch, validate_doc,
};

/// Supported options for creating or opening a collection.
///
/// Storage layout and I/O backend knobs remain outside the public contract
/// until distinct implementations exist:
///
/// ```compile_fail
/// use a3s_vec::CollectionOptions;
///
/// let mut options = CollectionOptions::new().unwrap();
/// options.set_enable_mmap(false).unwrap();
/// options.set_max_buffer_size(1024).unwrap();
/// options.set_segment_num(2).unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct CollectionOptions {
    read_only: bool,
    durability: Option<Durability>,
}

impl CollectionOptions {
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }
    pub fn set_read_only(&mut self, read_only: bool) -> Result<()> {
        self.read_only = read_only;
        Ok(())
    }
    pub fn read_only(&self) -> bool {
        self.read_only
    }
    pub fn set_durability(&mut self, value: Durability) -> Result<()> {
        self.durability = Some(value);
        Ok(())
    }
    pub fn durability(&self) -> Option<Durability> {
        self.durability
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
    #[serde(default)]
    pub index_cache_hit: bool,
    pub read_only: bool,
    pub wal_active_seq: u64,
    pub wal_checkpoint_seq: u64,
    pub wal_ops_since_checkpoint: u64,
    pub wal_bytes_since_checkpoint: u64,
}

#[derive(Debug, Clone)]
struct CollectionState {
    path: PathBuf,
    schema: CollectionSchema,
    docs: Arc<DocumentMap>,
    revision: u64,
    options: CollectionOptions,
    config: ConfigBuilder,
    stats: Arc<StatsRegistry>,
    indexes: Arc<IndexRegistry>,
    index_cache_hit: bool,
}

#[derive(Debug, Clone)]
struct CollectionSnapshot {
    schema: CollectionSchema,
    docs: Arc<DocumentMap>,
    revision: u64,
    stats: Arc<StatsRegistry>,
    indexes: Arc<IndexRegistry>,
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
        let config = options_config(&options);
        let root = Path::new(path);
        let storage = StorageHandle::create(root, schema, options.read_only)?;
        let indexes = IndexRegistry::build(schema, &DocumentMap::new(), 0)?;
        let state = CollectionState {
            path: root.to_path_buf(),
            schema: schema.clone(),
            docs: Arc::new(DocumentMap::new()),
            revision: 0,
            options,
            config,
            stats: Arc::new(StatsRegistry::default()),
            indexes: Arc::new(indexes),
            index_cache_hit: false,
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
        let config = options_config(&options);
        let (storage, schema, docs) = StorageHandle::open(Path::new(path), options.read_only)?;
        if schema.name.trim().is_empty() {
            return Err(Error::internal("persisted collection has an empty name"));
        }
        let revision = storage.manifest.revision;
        let mut recovered_docs = DocumentMap::new();
        for doc in docs {
            let doc = normalize_doc(&schema, &doc).map_err(|error| {
                Error::internal(format!(
                    "persisted document cannot be normalized: {}",
                    error.message
                ))
            })?;
            validate_doc(&schema, &doc, true).map_err(|error| {
                Error::internal(format!("persisted document is invalid: {}", error.message))
            })?;
            let id = doc
                .get_pk()
                .ok_or_else(|| Error::internal("persisted document has no primary key"))?
                .to_string();
            if recovered_docs.insert(id.clone(), Arc::new(doc)).is_some() {
                return Err(Error::internal(format!(
                    "persisted collection contains duplicate primary key '{id}'"
                )));
            }
        }
        let docs = recovered_docs;
        let cached_indexes = storage.read_index_cache().ok().flatten().and_then(|bytes| {
            IndexRegistry::restore_cache(
                &bytes,
                &schema,
                &docs,
                revision,
                &storage.index_cache_identity(),
            )
        });
        let index_cache_hit = cached_indexes.is_some();
        let indexes = cached_indexes.map_or_else(
            || {
                IndexRegistry::build(&schema, &docs, revision).map_err(|error| {
                    Error::internal(format!(
                        "rebuild persisted indexes at revision {revision}: {}",
                        error.message
                    ))
                })
            },
            Ok,
        )?;
        if !index_cache_hit && !options.read_only {
            persist_index_cache(&storage, &schema, &indexes, revision, false);
        }
        let state = CollectionState {
            path: PathBuf::from(path),
            schema: schema.clone(),
            docs: Arc::new(docs),
            revision,
            options,
            config,
            stats: Arc::new(StatsRegistry::default()),
            indexes: Arc::new(indexes),
            index_cache_hit,
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
        let (schema, docs, indexes, revision) = {
            let state = self
                .inner
                .state
                .read()
                .map_err(|_| Error::internal("collection state lock poisoned"))?;
            (
                state.schema.clone(),
                state
                    .docs
                    .values()
                    .map(|doc| doc.as_ref().clone())
                    .collect::<Vec<_>>(),
                Arc::clone(&state.indexes),
                state.revision,
            )
        };
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.checkpoint(&schema, &docs, revision, true)?;
        persist_index_cache(&storage, &schema, &indexes, revision, true);
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
        let mut indexes = state.indexes.stats(&state.schema);
        indexes.extend(
            state
                .schema
                .vectors
                .iter()
                .filter(|field| {
                    field
                        .index_params
                        .as_ref()
                        .is_some_and(|params| params.index_type == IndexType::Flat)
                })
                .map(|field| IndexStat {
                    name: field.name.clone(),
                    index_type: IndexType::Flat,
                    completeness: 1.0,
                    source_revision: state.revision,
                    document_count: u64::try_from(
                        state
                            .docs
                            .values()
                            .filter(|doc| doc.vector(&field.name).is_some())
                            .count(),
                    )
                    .unwrap_or(u64::MAX),
                    estimated_payload_bytes: None,
                    state: "ready".into(),
                }),
        );
        indexes.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(CollectionStats {
            doc_count: state.docs.len() as u64,
            indexes,
            revision: state.revision,
            index_cache_hit: state.index_cache_hit,
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
            fts_index_query_count: registry.fts_index_query_count.load(AtomicOrdering::Relaxed),
            ann_query_count: registry.ann_query_count.load(AtomicOrdering::Relaxed),
            exact_query_count: registry.exact_query_count.load(AtomicOrdering::Relaxed),
            filtered_query_count: registry.filtered_query_count.load(AtomicOrdering::Relaxed),
            scalar_index_query_count: registry
                .scalar_index_query_count
                .load(AtomicOrdering::Relaxed),
            radius_query_count: registry.radius_query_count.load(AtomicOrdering::Relaxed),
            candidates_scanned: registry.candidates_scanned.load(AtomicOrdering::Relaxed),
            indexed_field_count: basic.indexes.len(),
            indexes: basic.indexes,
            index_cache_hit: basic.index_cache_hit,
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
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;
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
            } else if current.docs.contains_key(*pk) {
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
            let config = current.config.clone();
            let revision = next_revision(current.revision)?;
            let mut next_docs = current.docs.as_ref().clone();
            for id in &ids {
                next_docs.remove(id);
            }
            let changed_ids = ids.iter().cloned().collect::<BTreeSet<_>>();
            let next_indexes = current.indexes.apply_document_changes(
                &current.schema,
                current.docs.as_ref(),
                &next_docs,
                revision,
                &changed_ids,
            )?;
            let mut state = self
                .inner
                .state
                .write()
                .map_err(|_| Error::internal("collection state lock poisoned"))?;
            ensure_same_generation(&state, &current)?;
            let mut storage = self
                .inner
                .storage
                .lock()
                .map_err(|_| Error::internal("storage lock poisoned"))?;
            storage.append(revision, WalOperation::Delete { ids: ids.clone() }, &config)?;
            state.docs = Arc::new(next_docs);
            state.indexes = Arc::new(next_indexes);
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
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;
        let parsed_filter = parse_filter_expression(filter)?;
        let indexed = current
            .indexes
            .scalar_candidates(current.revision, &parsed_filter);
        let ids: Vec<String> = if let Some(indexed) = indexed {
            indexed
                .ids()
                .filter_map(|id| current.docs.get(id))
                .filter(|doc| matches_filter(doc, Some(&parsed_filter)))
                .filter_map(|doc| doc.get_pk().map(str::to_string))
                .collect()
        } else {
            current
                .docs
                .values()
                .filter(|doc| matches_filter(doc, Some(&parsed_filter)))
                .filter_map(|doc| doc.get_pk().map(str::to_string))
                .collect()
        };
        if ids.is_empty() {
            return Ok(());
        }
        let config = current.config.clone();
        let revision = next_revision(current.revision)?;
        let mut next_docs = current.docs.as_ref().clone();
        for id in &ids {
            next_docs.remove(id);
        }
        let changed_ids = ids.iter().cloned().collect::<BTreeSet<_>>();
        let next_indexes = current.indexes.apply_document_changes(
            &current.schema,
            current.docs.as_ref(),
            &next_docs,
            revision,
            &changed_ids,
        )?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, &current)?;
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(revision, WalOperation::Delete { ids: ids.clone() }, &config)?;
        state.docs = Arc::new(next_docs);
        state.indexes = Arc::new(next_indexes);
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
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;

        let (accepted, outcomes) = prepare_mutation_batch(&current, docs, mutation);

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
        let config = current.config.clone();
        let revision = next_revision(current.revision)?;
        let mut next_docs = current.docs.as_ref().clone();
        for doc in &accepted {
            let Some(pk) = doc.get_pk().map(str::to_string) else {
                continue;
            };
            match mutation {
                Mutation::Insert | Mutation::Upsert => {
                    next_docs.insert(pk, Arc::new(doc.clone()));
                }
                Mutation::Update => {
                    if let Some(existing) = next_docs.get_mut(&pk) {
                        merge_patch(Arc::make_mut(existing), doc)?;
                    }
                }
            }
        }
        let changed_ids = accepted
            .iter()
            .filter_map(|doc| doc.get_pk().map(str::to_string))
            .collect::<BTreeSet<_>>();
        let next_indexes = current.indexes.apply_document_changes(
            &current.schema,
            current.docs.as_ref(),
            &next_docs,
            revision,
            &changed_ids,
        )?;
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, &current)?;
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(revision, operation, &config)?;
        state.docs = Arc::new(next_docs);
        state.indexes = Arc::new(next_indexes);
        state.revision = revision;
        maybe_checkpoint(&mut storage, &state, &config)?;
        Ok(write_result(outcomes))
    }

    // ---------------------------------------------------------------------
    // Index and schema management
    // ---------------------------------------------------------------------

    pub fn add_column(&self, field_schema: &FieldSchema, default_expr: Option<&str>) -> Result<()> {
        self.add_column_with_options(field_schema, default_expr, AddColumnOption::default())
    }

    pub fn add_column_with_options(
        &self,
        field_schema: &FieldSchema,
        default_expr: Option<&str>,
        option: AddColumnOption,
    ) -> Result<()> {
        self.ensure_open()?;
        if option.concurrency != 0 {
            return Err(Error::not_supported(
                "add-column concurrency has no parallel schema executor",
            ));
        }
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
        let default = default_expr
            .map(|expression| parse_default_expression(expression, field_schema.data_type))
            .transpose()?;
        if let Some(value) = default {
            next.docs = Arc::new(transform_documents(&next.docs, |doc| {
                doc.set_field_value(&field_schema.name, value.clone())
            })?);
        }
        for doc in next.docs.values() {
            validate_doc(&next.schema, doc, true)?;
        }
        let config = state.config.clone();
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
        next.docs = Arc::new(transform_documents(&next.docs, |doc| {
            doc.remove_field(name)
        })?);
        let config = state.config.clone();
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
        next.docs = Arc::new(transform_documents(&next.docs, |doc| {
            if let Some(value) = doc.field(old_name).cloned() {
                doc.remove_field(old_name)?;
                doc.set_field_value(new_name, value)?;
            } else if let Some(value) = doc.vector(old_name).cloned() {
                doc.remove_field(old_name)?;
                doc.set_vector_value(new_name, value)?;
            }
            Ok(())
        })?);
        let config = state.config.clone();
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
        option: AlterColumnOption,
    ) -> Result<()> {
        self.ensure_open()?;
        if option.concurrency != 0 {
            return Err(Error::not_supported(
                "alter-column concurrency has no parallel schema executor",
            ));
        }
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
        let config = state.config.clone();
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_schema_change(&mut storage, &mut state, next, &config)
    }

    fn snapshot_state(&self) -> Result<CollectionSnapshot> {
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        Ok(CollectionSnapshot {
            schema: state.schema.clone(),
            docs: state.docs.clone(),
            revision: state.revision,
            stats: Arc::clone(&state.stats),
            indexes: state.indexes.clone(),
        })
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

fn ensure_writable(options: &CollectionOptions) -> Result<()> {
    if options.read_only {
        Err(Error::permission_denied("collection is read-only"))
    } else {
        Ok(())
    }
}

fn ensure_same_generation(current: &CollectionState, expected: &CollectionState) -> Result<()> {
    if current.revision == expected.revision && current.schema == expected.schema {
        Ok(())
    } else {
        Err(Error::failed_precondition(
            "collection generation changed during index construction",
        ))
    }
}

fn transform_documents(
    docs: &DocumentMap,
    mut transform: impl FnMut(&mut Doc) -> Result<()>,
) -> Result<DocumentMap> {
    let mut transformed = DocumentMap::new();
    for (id, doc) in docs {
        let mut next = doc.as_ref().clone();
        transform(&mut next)?;
        transformed.insert(id.clone(), Arc::new(next));
    }
    Ok(transformed)
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
    next: CollectionState,
    config: &ConfigBuilder,
) -> Result<()> {
    let next = prepare_schema_change(next)?;
    commit_prepared_schema_change(storage, state, next, config)
}

fn prepare_schema_change(mut next: CollectionState) -> Result<CollectionState> {
    let revision = next_revision(next.revision)?;
    next.revision = revision;
    next.indexes = Arc::new(IndexRegistry::build(&next.schema, &next.docs, revision)?);
    Ok(next)
}

fn next_revision(current: u64) -> Result<u64> {
    current
        .checked_add(1)
        .ok_or_else(|| Error::resource_exhausted("collection revision overflow"))
}

#[cfg(test)]
mod tests;
