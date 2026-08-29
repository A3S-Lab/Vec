//! Thread-safe collection handle and the reference query engine.

use crate::config::{current_config, ConfigBuilder, Durability, IoBackend};
use crate::doc::{Doc, FieldValue, VectorValue};
use crate::error::{Error, ErrorCode, Result};
use crate::index::dense_score;
use crate::iterator::DocIterator;
use crate::multi_query::{MultiQuery, RerankMethod};
use crate::query::{GroupBySearchQuery, SearchQuery};
use crate::schema::{AddColumnOption, AlterColumnOption, CollectionSchema, FieldSchema, IndexParams};
pub use crate::stats::IndexStat;
use crate::stats::{StatsRegistry, StatsSnapshot};
use crate::storage::{StorageHandle, WalRecord};
use crate::types::{DataType, IndexType, MetricType};
use serde_json::Value;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};

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
    pub fn enable_mmap(&self) -> bool { self.enable_mmap }
    pub fn set_max_buffer_size(&mut self, size: u64) -> Result<()> {
        self.max_buffer_size = size;
        Ok(())
    }
    pub fn max_buffer_size(&self) -> u64 { self.max_buffer_size }
    pub fn set_read_only(&mut self, read_only: bool) -> Result<()> {
        self.read_only = read_only;
        Ok(())
    }
    pub fn read_only(&self) -> bool { self.read_only }
    pub fn set_segment_num(&mut self, count: u32) -> Result<()> {
        if count > 1024 { return Err(Error::invalid_argument("segment_num must be at most 1024")); }
        self.segment_num = count;
        Ok(())
    }
    pub fn set_durability(&mut self, value: Durability) -> Result<()> { self.durability = value; Ok(()) }
    pub fn set_io_backend(&mut self, value: IoBackend) -> Result<()> { self.io_backend = value; Ok(()) }
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
    pub fn is_success(&self) -> bool { self.success }
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

#[derive(Debug)]
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
        let state = CollectionState {
            path: PathBuf::from(path),
            schema: schema.clone(),
            docs: docs
                .into_iter()
                .filter_map(|doc| doc.get_pk().map(str::to_string).map(|id| (id, doc)))
                .collect(),
            revision,
            options,
            indexes: runtime_indexes(&schema, revision, 0),
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
        let (schema, docs, config) = {
            let state = self
                .inner
                .state
                .read()
                .map_err(|_| Error::internal("collection state lock poisoned"))?;
            (state.schema.clone(), state.docs.values().cloned().collect::<Vec<_>>(), options_config(&state.options))
        };
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
    }

    pub fn close(self) -> Result<()> {
        if self.is_open() {
            self.flush()?;
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
                completeness: if index.source_revision == state.revision { 1.0 } else { 0.0 },
                source_revision: index.source_revision,
                document_count: index.document_count,
                state: if index.source_revision == state.revision { "ready".into() } else { "stale".into() },
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
        let stats = Arc::clone(&state.stats);
        Ok(StatsSnapshot {
            collection_name: state.schema.name.clone(),
            revision: basic.revision,
            doc_count: basic.doc_count,
            query_count: stats.query_count.load(AtomicOrdering::Relaxed),
            fts_query_count: stats.fts_query_count.load(AtomicOrdering::Relaxed),
            ann_query_count: stats.ann_query_count.load(AtomicOrdering::Relaxed),
            exact_query_count: stats.exact_query_count.load(AtomicOrdering::Relaxed),
            filtered_query_count: stats.filtered_query_count.load(AtomicOrdering::Relaxed),
            radius_query_count: stats.radius_query_count.load(AtomicOrdering::Relaxed),
            candidates_scanned: stats.candidates_scanned.load(AtomicOrdering::Relaxed),
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
                results.push(write_error(ErrorCode::InvalidArgument, "primary key is empty"));
            } else if !seen.insert(*pk) {
                results.push(write_error(ErrorCode::AlreadyExists, "duplicate primary key in delete batch"));
            } else if state.docs.contains_key(*pk) {
                ids.push((*pk).to_string());
                results.push(write_success());
            } else {
                results.push(write_error(ErrorCode::NotFound, format!("document '{pk}' not found")));
            }
        }
        if !ids.is_empty() {
            let config = options_config(&state.options);
            let mut storage = self
                .inner
                .storage
                .lock()
                .map_err(|_| Error::internal("storage lock poisoned"))?;
            storage.append(&WalRecord::Delete { ids: ids.clone() }, &config)?;
            for id in &ids {
                state.docs.remove(id);
            }
            state.revision = state.revision.saturating_add(1);
            storage.manifest.revision = state.revision;
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
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Delete { ids: ids.clone() }, &config)?;
        for id in &ids {
            state.docs.remove(id);
        }
        state.revision = state.revision.saturating_add(1);
        storage.manifest.revision = state.revision;
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

        let mut outcomes = Vec::with_capacity(docs.len());
        let mut accepted = Vec::new();
        let mut batch_ids = HashSet::new();
        for doc in docs {
            let validation = match mutation {
                Mutation::Insert => validate_doc(&state.schema, doc, true),
                Mutation::Update | Mutation::Upsert => validate_doc(&state.schema, doc, false),
            };
            if let Err(error) = validation {
                outcomes.push(write_error(error.code, error.message));
                continue;
            }
            let Some(pk) = doc.get_pk() else {
                outcomes.push(write_error(ErrorCode::InvalidArgument, "primary key is required"));
                continue;
            };
            if !batch_ids.insert(pk.to_string()) {
                outcomes.push(write_error(ErrorCode::AlreadyExists, format!("duplicate primary key '{pk}' in batch")));
                continue;
            }
            let exists = state.docs.contains_key(pk);
            let allowed = match mutation {
                Mutation::Insert => !exists,
                Mutation::Update => exists,
                Mutation::Upsert => true,
            };
            if !allowed {
                outcomes.push(write_error(
                    if exists { ErrorCode::AlreadyExists } else { ErrorCode::NotFound },
                    if exists { format!("document '{pk}' already exists") } else { format!("document '{pk}' not found") },
                ));
                continue;
            }
            accepted.push((*doc).clone());
            outcomes.push(write_success());
        }

        if accepted.is_empty() {
            return Ok(write_result(outcomes));
        }
        let record = match mutation {
            Mutation::Insert => WalRecord::Insert { docs: accepted.clone() },
            Mutation::Update => WalRecord::Update { docs: accepted.clone() },
            Mutation::Upsert => WalRecord::Upsert { docs: accepted.clone() },
        };
        let config = options_config(&state.options);
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&record, &config)?;
        for doc in accepted {
            let Some(pk) = doc.get_pk().map(str::to_string) else { continue };
            match mutation {
                Mutation::Insert | Mutation::Upsert => {
                    state.docs.insert(pk, doc);
                }
                Mutation::Update => {
                    if let Some(existing) = state.docs.get_mut(&pk) {
                        merge_patch(existing, &doc)?;
                    }
                }
            }
        }
        state.revision = state.revision.saturating_add(1);
        storage.manifest.revision = state.revision;
        maybe_checkpoint(&mut storage, &state, &config)?;
        Ok(write_result(outcomes))
    }

    // ---------------------------------------------------------------------
    // DQL
    // ---------------------------------------------------------------------

    pub fn query(&self, query: &SearchQuery) -> Result<Vec<Doc>> {
        self.ensure_open()?;
        let (schema, docs, revision, stats) = self.snapshot_state()?;
        let result = execute_query(&schema, &docs, query)?;
        let has_fts = query.fts.is_some();
        let has_ann = schema_index_params(&schema, &query.field_name)
            .is_some_and(|p| p.index_type.is_vector_index() && p.index_type != IndexType::Flat);
        let filtered = query.filter.as_ref().is_some_and(|v| !v.trim().is_empty());
        let radius = query.params.get("radius").is_some();
        stats.record_query(has_fts, has_ann, filtered, radius, docs.len() as u64);
        let _ = revision;
        Ok(result)
    }

    pub fn multi_query(&self, query: &MultiQuery) -> Result<Vec<Doc>> {
        self.ensure_open()?;
        if query.queries.is_empty() {
            return Err(Error::invalid_argument("multi-query must contain at least one sub-query"));
        }
        let (schema, docs, _revision, stats) = self.snapshot_state()?;
        let mut branches: Vec<Vec<Doc>> = Vec::with_capacity(query.queries.len());
        for sub in &query.queries {
            let mut branch = sub.to_search_query()?;
            if let Some(filter) = query.effective_filter() {
                branch.set_filter(filter)?;
            }
            branch.include_vector = query.include_vector_value;
            branch.output_fields = query.output_fields.clone();
            branches.push(execute_query(&schema, &docs, &branch)?);
        }
        let normalization = query.normalization.as_deref().unwrap_or("none");
        for branch in &mut branches {
            normalize_scores(branch, normalization)?;
        }
        let mut fused: BTreeMap<String, (f64, Doc)> = BTreeMap::new();
        for (branch_idx, branch) in branches.into_iter().enumerate() {
            let weight = match &query.rerank {
                RerankMethod::Weighted { weights } => weights.get(branch_idx).copied().unwrap_or(1.0),
                RerankMethod::ReciprocalRank { .. } => 1.0,
            };
            for (rank, doc) in branch.into_iter().enumerate() {
                let Some(id) = doc.get_pk().map(str::to_string) else { continue };
                let score = match query.rerank {
                    RerankMethod::ReciprocalRank { rank_constant } => weight / (rank_constant + rank as f64 + 1.0),
                    RerankMethod::Weighted { .. } => weight * f64::from(doc.get_score()),
                };
                fused
                    .entry(id)
                    .and_modify(|entry| entry.0 += score)
                    .or_insert((score, doc));
            }
        }
        let mut output: Vec<Doc> = fused
            .into_values()
            .map(|(score, mut doc)| {
                let _ = doc.set_score(score as f32);
                doc
            })
            .collect();
        sort_docs(&mut output);
        output.truncate(query.topk_value as usize);
        stats.record_query(false, query.queries.len() > 1, query.filter.is_some(), false, docs.len() as u64);
        Ok(output)
    }

    pub fn group_by(&self, query: &GroupBySearchQuery) -> Result<HashMap<String, Vec<Doc>>> {
        self.ensure_open()?;
        let mut vector_query = SearchQuery::new(&query.field_name, &query.vector, i32::MAX.min((query.group_count.saturating_mul(query.group_topk).max(1)) as i32))?;
        vector_query.include_vector = query.include_vector;
        vector_query.output_fields = query.output_fields.clone();
        vector_query.params = query.params.clone();
        if let Some(filter) = &query.filter { vector_query.set_filter(filter)?; }
        let docs = self.query(&vector_query)?;
        let mut groups: HashMap<String, Vec<Doc>> = HashMap::new();
        for doc in docs {
            let key = doc
                .scalar_json(&query.group_by_field)
                .map(|value| match value { Value::String(v) => v, other => other.to_string() })
                .unwrap_or_else(|| "__null__".to_string());
            let group = groups.entry(key).or_default();
            if group.len() < query.group_topk as usize { group.push(doc); }
        }
        if groups.len() > query.group_count as usize {
            let mut ranked: Vec<(String, f32)> = groups
                .iter()
                .map(|(key, values)| (key.clone(), values.first().map_or(f32::NEG_INFINITY, Doc::get_score)))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
            let keep: HashSet<String> = ranked.into_iter().take(query.group_count as usize).map(|(key, _)| key).collect();
            groups.retain(|key, _| keep.contains(key));
        }
        Ok(groups)
    }

    pub fn group_by_query(&self, query: &GroupBySearchQuery) -> Result<HashMap<String, Vec<Doc>>> {
        self.group_by(query)
    }

    pub fn fetch(&self, pks: &[&str]) -> Result<Vec<Doc>> {
        self.fetch_with_options(pks, None, true)
    }

    pub fn fetch_with_options(
        &self,
        pks: &[&str],
        output_fields: Option<&[&str]>,
        include_vector: bool,
    ) -> Result<Vec<Doc>> {
        self.ensure_open()?;
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        let fields = output_fields.map(|values| values.iter().map(|v| (*v).to_string()).collect::<Vec<_>>());
        Ok(pks
            .iter()
            .filter_map(|pk| state.docs.get(*pk))
            .map(|doc| doc.project(fields.as_deref(), include_vector))
            .collect())
    }

    pub fn iter(&self) -> Result<DocIterator> {
        self.iter_with_options(None, true)
    }

    pub fn iter_with_options(
        &self,
        output_fields: Option<&[&str]>,
        include_vector: bool,
    ) -> Result<DocIterator> {
        self.ensure_open()?;
        let state = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        let fields = output_fields.map(|values| values.iter().map(|v| (*v).to_string()).collect::<Vec<_>>());
        let docs = state
            .docs
            .values()
            .map(|doc| doc.project(fields.as_deref(), include_vector))
            .collect();
        Ok(DocIterator::new(docs, state.revision))
    }

    // ---------------------------------------------------------------------
    // Index and schema management
    // ---------------------------------------------------------------------

    pub fn create_index(&self, field_name: &str, params: &IndexParams) -> Result<()> {
        self.ensure_open()?;
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        state.schema.add_index(field_name, &params.marked_built(true))?;
        let count = state.docs.len() as u64;
        let source_revision = state.revision;
        state.indexes.insert(field_name.to_string(), RuntimeIndex { params: params.marked_built(true), source_revision, document_count: count });
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Schema { schema: schema.clone() }, &config)?;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
    }

    pub fn drop_index(&self, field_name: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        state.schema.drop_index(field_name)?;
        state.indexes.remove(field_name);
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Schema { schema: schema.clone() }, &config)?;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
    }

    pub fn optimize(&self) -> Result<()> {
        self.ensure_open()?;
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let document_count = state.docs.len() as u64;
        for index in state.indexes.values_mut() {
            index.source_revision = revision;
            index.document_count = document_count;
        }
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.manifest.revision = state.revision;
        storage.checkpoint(&schema, &docs, &config)?;
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
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        state.schema.add_field(field_schema)?;
        let default = default_expr.map(parse_default_expression);
        if let Some(value) = default {
            for doc in state.docs.values_mut() {
                doc.set_field_value(&field_schema.name, value.clone())?;
            }
        }
        state.revision = state.revision.saturating_add(1);
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Schema { schema: schema.clone() }, &config)?;
        storage.manifest.revision = state.revision;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
    }

    pub fn drop_column(&self, name: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        state.schema.drop_field(name)?;
        for doc in state.docs.values_mut() {
            doc.remove_field(name)?;
        }
        state.revision = state.revision.saturating_add(1);
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Schema { schema: schema.clone() }, &config)?;
        storage.manifest.revision = state.revision;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
    }

    pub fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
        if old_name == new_name {
            return Ok(());
        }
        if new_name.trim().is_empty() || new_name.contains('\0') {
            return Err(Error::invalid_argument("new field name is invalid"));
        }
        self.ensure_open()?;
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        if state.schema.has_field(new_name) {
            return Err(Error::already_exists(format!("field '{new_name}' already exists")));
        }
        if let Some(field) = state.schema.fields.iter_mut().find(|field| field.name == old_name) {
            field.name = new_name.to_string();
        } else if let Some(field) = state.schema.vectors.iter_mut().find(|field| field.name == old_name) {
            field.name = new_name.to_string();
        } else {
            return Err(Error::not_found(format!("field '{old_name}' not found")));
        }
        for doc in state.docs.values_mut() {
            if let Some(value) = doc.field(old_name).cloned() {
                doc.remove_field(old_name)?;
                doc.set_field_value(new_name, value)?;
            } else if let Some(value) = doc.vector(old_name).cloned() {
                doc.remove_field(old_name)?;
                doc.set_vector_value(new_name, value)?;
            }
        }
        state.revision = state.revision.saturating_add(1);
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Schema { schema: schema.clone() }, &config)?;
        storage.manifest.revision = state.revision;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
    }

    pub fn alter_column(
        &self,
        field_schema: &FieldSchema,
        _option: AlterColumnOption,
    ) -> Result<()> {
        self.ensure_open()?;
        let _writer = self.inner.writer.lock().map_err(|_| Error::internal("writer lock poisoned"))?;
        let mut state = self.inner.state.write().map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_writable(&state.options)?;
        let target = state.schema.fields.iter_mut().find(|field| field.name == field_schema.name)
            .ok_or_else(|| Error::not_found(format!("field '{}' not found", field_schema.name)))?;
        if target.data_type != field_schema.data_type || target.dimension != field_schema.dimension {
            return Err(Error::invalid_argument("altering a field's data type or dimension would invalidate existing data"));
        }
        *target = field_schema.clone();
        state.revision = state.revision.saturating_add(1);
        let schema = state.schema.clone();
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        let config = options_config(&state.options);
        let mut storage = self.inner.storage.lock().map_err(|_| Error::internal("storage lock poisoned"))?;
        storage.append(&WalRecord::Schema { schema: schema.clone() }, &config)?;
        storage.manifest.revision = state.revision;
        storage.checkpoint(&schema, &docs, &config)?;
        Ok(())
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

fn schema_index_params<'a>(schema: &'a CollectionSchema, field_name: &str) -> Option<&'a IndexParams> {
    schema
        .fields
        .iter()
        .find(|field| field.name == field_name)
        .and_then(|field| field.index_params.as_ref())
        .or_else(|| {
            schema
                .vectors
                .iter()
                .find(|field| field.name == field_name)
                .and_then(|field| field.index_params.as_ref())
        })
}

fn ensure_writable(options: &CollectionOptions) -> Result<()> {
    if options.read_only {
        Err(Error::permission_denied("collection is read-only"))
    } else {
        Ok(())
    }
}

fn runtime_indexes(schema: &CollectionSchema, revision: u64, document_count: u64) -> BTreeMap<String, RuntimeIndex> {
    schema
        .fields
        .iter()
        .filter_map(|field| field.index_params.clone().map(|params| (field.name.clone(), params)))
        .chain(schema.vectors.iter().filter_map(|field| field.index_params.clone().map(|params| (field.name.clone(), params))))
        .map(|(name, params)| (name, RuntimeIndex { params, source_revision: revision, document_count }))
        .collect()
}

fn validate_doc(schema: &CollectionSchema, doc: &Doc, require_all: bool) -> Result<()> {
    let pk = doc
        .get_pk()
        .ok_or_else(|| Error::invalid_argument("document primary key is required"))?;
    if pk.is_empty() || pk.contains('\0') {
        return Err(Error::invalid_argument("document primary key must be non-empty and contain no NUL byte"));
    }
    for name in doc.fields().keys() {
        if !schema.fields.iter().any(|field| field.name == *name) {
            return Err(Error::invalid_argument(format!("unknown scalar field '{name}'")));
        }
    }
    for name in doc.vectors().keys() {
        if !schema.vectors.iter().any(|field| field.name == *name) {
            return Err(Error::invalid_argument(format!("unknown vector field '{name}'")));
        }
    }
    for field in &schema.fields {
        match doc.field(&field.name) {
            None if require_all && !field.nullable => {
                return Err(Error::invalid_argument(format!("required field '{}' is missing", field.name)));
            }
            Some(FieldValue::Null) if !field.nullable => {
                return Err(Error::invalid_argument(format!("field '{}' is not nullable", field.name)));
            }
            Some(value) if !matches_field_type(value, field.data_type) => {
                return Err(Error::invalid_argument(format!("field '{}' has a value incompatible with {}", field.name, field.data_type)));
            }
            _ => {}
        }
    }
    for field in &schema.vectors {
        if let Some(value) = doc.vector(&field.name) {
            if value.data_type() != field.data_type {
                // Numeric dense vectors can be losslessly converted for query,
                // but their declared storage type remains part of the schema.
                let numeric_dense = field.data_type.is_dense_vector() && value.data_type().is_dense_vector();
                if !numeric_dense {
                    return Err(Error::invalid_argument(format!("vector '{}' has a value incompatible with {}", field.name, field.data_type)));
                }
            }
            if field.dimension > 0 && !value.is_sparse() && value.dimension() != field.dimension as usize {
                return Err(Error::invalid_argument(format!("vector '{}' dimension mismatch: expected {}, got {}", field.name, field.dimension, value.dimension())));
            }
            if value.is_sparse() {
                let Some(sparse) = value.to_sparse_f64() else { return Err(Error::invalid_argument(format!("invalid sparse vector '{}'", field.name))); };
                if sparse.keys().any(|index| *index >= field.dimension as u32 && field.dimension > 0) {
                    return Err(Error::invalid_argument(format!("sparse vector '{}' contains an out-of-range index", field.name)));
                }
            }
        } else if require_all && !field.data_type.is_sparse_vector() {
            return Err(Error::invalid_argument(format!("required vector field '{}' is missing", field.name)));
        }
    }
    Ok(())
}

fn matches_field_type(value: &FieldValue, data_type: DataType) -> bool {
    matches!(value, FieldValue::Null)
        || matches!(value, FieldValue::Json(_))
        || value.data_type() == data_type
}

fn merge_patch(target: &mut Doc, patch: &Doc) -> Result<()> {
    for (name, value) in patch.fields() {
        target.set_field_value(name, value.clone())?;
    }
    for (name, value) in patch.vectors() {
        target.set_vector_value(name, value.clone())?;
    }
    if patch.get_score() != 0.0 {
        target.set_score(patch.get_score())?;
    }
    Ok(())
}

fn write_success() -> DocWriteResult {
    DocWriteResult { success: true, code: ErrorCode::Unknown, message: String::new() }
}

fn write_error(code: ErrorCode, message: impl Into<String>) -> DocWriteResult {
    DocWriteResult { success: false, code, message: message.into() }
}

fn write_result(results: Vec<DocWriteResult>) -> WriteResult {
    let success_count = results.iter().filter(|result| result.success).count() as u64;
    WriteResult { success_count, error_count: results.len() as u64 - success_count, results }
}

fn maybe_checkpoint(storage: &mut StorageHandle, state: &CollectionState, config: &ConfigBuilder) -> Result<()> {
    let should = matches!(config.durability, Durability::Interval)
        && storage.should_checkpoint(config);
    if should {
        let docs: Vec<Doc> = state.docs.values().cloned().collect();
        storage.checkpoint(&state.schema, &docs, config)?;
    }
    Ok(())
}

fn sort_docs(docs: &mut [Doc]) {
    docs.sort_by(|left, right| {
        right
            .get_score()
            .partial_cmp(&left.get_score())
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.get_pk().unwrap_or_default().cmp(right.get_pk().unwrap_or_default()))
    });
}

fn parse_filter_expression(expression: &str) -> Result<zvec_core::filter::FilterExpr> {
    if expression.trim().is_empty() {
        return Err(Error::invalid_argument("filter expression must not be empty"));
    }
    zvec_core::filter::parse_filter(expression).map_err(|error| Error::invalid_argument(error.to_string()))
}

fn matches_filter(doc: &Doc, expression: Option<&str>) -> bool {
    let Some(expression) = expression else { return true };
    let Ok(parsed) = parse_filter_expression(expression) else { return false };
    parsed.matches(&doc.to_core())
}

fn execute_query(schema: &CollectionSchema, docs: &[Doc], query: &SearchQuery) -> Result<Vec<Doc>> {
    let field = schema
        .fields
        .iter()
        .find(|field| field.name == query.field_name)
        .map(|field| (field.data_type, field.index_params.as_ref()))
        .or_else(|| schema.vectors.iter().find(|field| field.name == query.field_name).map(|field| (field.data_type, field.index_params.as_ref())))
        .ok_or_else(|| Error::not_found(format!("query field '{}' not found", query.field_name)))?;
    if let Some(filter) = query.filter.as_deref() {
        parse_filter_expression(filter)?;
    }
    let metric = query
        .params
        .get("metric")
        .and_then(|value| value.as_str())
        .map(parse_metric)
        .transpose()?
        .or_else(|| field.1.map(|params| params.metric_type))
        .unwrap_or(MetricType::Cosine);
    let mut scored = if query.fts.is_some() {
        execute_fts(schema, docs, query, field.1)?
    } else {
        execute_vector(docs, query, metric)?
    };
    sort_docs(&mut scored);
    let topk = usize::try_from(query.topk.max(0)).unwrap_or(0);
    scored.truncate(topk);
    let output_fields = query.output_fields.as_deref();
    Ok(scored
        .into_iter()
        .map(|doc| doc.project(output_fields, query.include_vector))
        .collect())
}

fn execute_vector(docs: &[Doc], query: &SearchQuery, metric: MetricType) -> Result<Vec<Doc>> {
    let dense_query = if let Some(vector) = &query.vector {
        Some(vector.clone())
    } else if let Some(id) = &query.id {
        docs.iter().find(|doc| doc.get_pk() == Some(id.as_str())).and_then(|doc| doc.vector(&query.field_name)).and_then(VectorValue::to_dense_f32)
    } else {
        None
    };
    let sparse_query: Option<BTreeMap<u32, f64>> = query.sparse_vector.as_ref().map(|values| values.iter().map(|(i, v)| (*i, *v as f64)).collect());
    if dense_query.is_none() && sparse_query.is_none() {
        return Err(Error::invalid_argument("query requires a dense vector, sparse vector, or source id"));
    }
    let radius = query.params.get("radius").and_then(Value::as_f64);
    let mut result = Vec::new();
    for doc in docs {
        if !matches_filter(doc, query.filter.as_deref()) {
            continue;
        }
        let Some(vector) = doc.vector(&query.field_name) else { continue };
        let score = if let Some(ref dense) = dense_query {
            let Some(score) = dense_score(dense, vector, metric) else { continue };
            f64::from(score)
        } else {
            let Some(stored) = vector.to_sparse_f64() else { continue };
            sparse_query.as_ref().map(|q| q.iter().filter_map(|(i, value)| stored.get(i).map(|other| value * other)).sum()).unwrap_or(0.0)
        };
        if let Some(radius) = radius {
            let passes = if metric == MetricType::L2 { score >= -radius * radius } else { score >= radius };
            if !passes { continue; }
        }
        let mut copy = doc.clone();
        copy.set_score(score as f32)?;
        result.push(copy);
    }
    Ok(result)
}

fn execute_fts(
    schema: &CollectionSchema,
    docs: &[Doc],
    query: &SearchQuery,
    index_params: Option<&IndexParams>,
) -> Result<Vec<Doc>> {
    let fts = query.fts.as_ref().ok_or_else(|| Error::invalid_argument("FTS payload is missing"))?;
    let expression = fts.match_string.as_deref().or(fts.query_string.as_deref()).unwrap_or_default();
    if expression.trim().is_empty() { return Err(Error::invalid_argument("FTS query is empty")); }
    let tokenizer = index_params
        .and_then(|params| params.params.get("tokenizer_name"))
        .and_then(Value::as_str)
        .unwrap_or("standard");
    let terms = fts_terms(expression, fts.query_string.is_some());
    let total_docs = docs.len().max(1) as f64;
    let avg_len = docs.iter().filter_map(|doc| text_value(doc, &query.field_name).map(|text| tokenize(text, tokenizer).len() as f64)).sum::<f64>() / total_docs;
    let mut result = Vec::new();
    for doc in docs {
        if !matches_filter(doc, query.filter.as_deref()) { continue; }
        let Some(text) = text_value(doc, &query.field_name) else { continue };
        let tokens = tokenize(text, tokenizer);
        let score = bm25(&tokens, &terms, docs, &query.field_name, tokenizer, avg_len);
        if score <= 0.0 { continue; }
        let mut copy = doc.clone();
        copy.set_score(score as f32)?;
        result.push(copy);
    }
    let _ = schema;
    Ok(result)
}

fn parse_metric(value: &str) -> Result<MetricType> {
    match value.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(MetricType::L2),
        "ip" | "inner_product" | "dot" => Ok(MetricType::Ip),
        "cosine" => Ok(MetricType::Cosine),
        "mips_l2" | "mips-l2" => Ok(MetricType::MipsL2),
        _ => Err(Error::invalid_argument(format!("unknown metric '{value}'"))),
    }
}

fn text_value<'a>(doc: &'a Doc, field: &str) -> Option<&'a str> {
    match doc.field(field) {
        Some(FieldValue::String(value)) => Some(value.as_str()),
        Some(FieldValue::Json(Value::String(value))) => Some(value.as_str()),
        _ => None,
    }
}

fn tokenize(text: &str, tokenizer: &str) -> Vec<String> {
    // zvec-core owns the optional jieba dictionary and the portable standard /
    // whitespace analyzers.  If a caller requests an unknown tokenizer, the
    // core intentionally falls back to its standard analyzer.
    zvec_core::engine::fts::tokenize_with(text, tokenizer)
}

fn fts_terms(expression: &str, advanced: bool) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for character in expression.chars() {
        match character {
            '"' => {
                quoted = !quoted;
                if !quoted && !current.trim().is_empty() {
                    terms.push(current.trim().to_string());
                    current.clear();
                }
            }
            value if value.is_whitespace() && !quoted => {
                if !current.trim().is_empty() {
                    terms.push(current.trim().to_string());
                    current.clear();
                }
            }
            value => current.push(value),
        }
    }
    if !current.trim().is_empty() {
        terms.push(current.trim().to_string());
    }
    if advanced {
        terms
            .into_iter()
            .filter(|term| {
                let upper = term.to_ascii_uppercase();
                !matches!(upper.as_str(), "AND" | "OR" | "NOT")
            })
            .collect()
    } else {
        terms
    }
}

fn bm25(
    document_tokens: &[String],
    query_terms: &[String],
    docs: &[Doc],
    field_name: &str,
    tokenizer: &str,
    average_length: f64,
) -> f64 {
    if document_tokens.is_empty() || query_terms.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    let document_length = document_tokens.len() as f64;
    let k1 = 1.2;
    let b = 0.75;
    for raw_term in query_terms {
        let normalized = tokenize(raw_term, tokenizer);
        if normalized.is_empty() {
            continue;
        }
        let term = &normalized[0];
        let frequency = document_tokens.iter().filter(|token| *token == term).count() as f64;
        if frequency == 0.0 {
            // Support simple prefix/suffix wildcard syntax used by zvec FTS.
            let wildcard_hit = if let Some(prefix) = term.strip_suffix('*') {
                document_tokens.iter().any(|token| token.starts_with(prefix))
            } else if let Some(suffix) = term.strip_prefix('*') {
                document_tokens.iter().any(|token| token.ends_with(suffix))
            } else {
                false
            };
            if !wildcard_hit {
                continue;
            }
        }
        let document_frequency = docs
            .iter()
            .filter_map(|doc| text_value(doc, field_name))
            .filter(|text| tokenize(text, tokenizer).iter().any(|token| token == term))
            .count() as f64;
        let idf = ((docs.len() as f64 - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1.0)
            .ln();
        let denominator = frequency + k1 * (1.0 - b + b * document_length / average_length.max(1.0));
        score += idf * (frequency * (k1 + 1.0) / denominator.max(1e-12));
    }
    score.max(0.0)
}

fn normalize_scores(docs: &mut [Doc], method: &str) -> Result<()> {
    if docs.is_empty() || method == "none" {
        return Ok(());
    }
    let values: Vec<f64> = docs.iter().map(|doc| f64::from(doc.get_score())).collect();
    match method {
        "minmax" => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator = (max - min).max(1e-12);
            for doc in docs {
                doc.set_score(((f64::from(doc.get_score()) - min) / denominator) as f32)?;
            }
        }
        "zscore" => {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / values.len() as f64;
            let denominator = variance.sqrt().max(1e-12);
            for doc in docs {
                doc.set_score(((f64::from(doc.get_score()) - mean) / denominator) as f32)?;
            }
        }
        _ => return Err(Error::invalid_argument("unknown score normalization method")),
    }
    Ok(())
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
