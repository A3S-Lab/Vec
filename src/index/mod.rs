//! Immutable, revision-tagged in-memory ANN index generations.

mod cache;
mod fts;
mod hnsw;
mod ivf;
mod ordinal_map;
mod ordinals;
mod quantization;
mod rebuild;
mod scalar;

use crate::doc::{DocumentMap, VectorValue};
use crate::error::{Error, Result};
use crate::query::SearchQuery;
use crate::schema::{CollectionSchema, IndexParams};
use crate::stats::IndexStat;
use crate::types::{IndexType, MetricType};
use fts::FtsIndexRegistry;
use hnsw::{HnswFilter, HnswIndex};
use ivf::{scaled_candidate_limit, IvfIndex};
use ordinal_map::OrdinalMap;
pub(crate) use ordinals::OrdinalScores;
use ordinals::{OrdinalSet, OrdinalTable};
use quantization::{score, QuantizedVector};
use roaring::RoaringTreemap;
use scalar::{ScalarCandidates, ScalarIndexRegistry};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MIN_DELTA_COMPACTION: usize = 64;
const MAX_DELTA_COMPACTION: usize = 2_048;
const DELTA_COMPACTION_DIVISOR: usize = 8;
const SCALAR_EXACT_PREFILTER_MIN: usize = 4_096;
const SCALAR_EXACT_PREFILTER_PER_RESULT: usize = 64;

#[derive(Debug, Clone, Default)]
pub(crate) struct IndexRegistry {
    ordinals: OrdinalTable,
    indexes: BTreeMap<String, VectorIndex>,
    scalar_indexes: ScalarIndexRegistry,
    fts_indexes: FtsIndexRegistry,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct VectorIndex {
    #[serde(with = "cache::index_params_serde")]
    params: IndexParams,
    source_revision: u64,
    base: Arc<VectorIndexBase>,
    /// Changed vectors since the last full build. An updated base vector is
    /// also present in `tombstones`, so it shadows the graph/posting entry.
    delta: BTreeMap<u64, QuantizedVector>,
    delta_ordinals: RoaringTreemap,
    /// Base entries removed or superseded by this generation.
    tombstones: RoaringTreemap,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct VectorIndexBase {
    vectors: OrdinalMap<QuantizedVector>,
    vector_ordinals: RoaringTreemap,
    kind: VectorIndexKind,
}

struct AnnSearchContext<'a> {
    query: &'a SearchQuery,
    vector: &'a [f32],
    topk: usize,
    metric: MetricType,
    allowed: Option<&'a RoaringTreemap>,
    eligible_count: usize,
    ordinals: &'a OrdinalTable,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum VectorIndexKind {
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
}

#[derive(Debug, Clone)]
pub(crate) struct CandidateSelection {
    ids: OrdinalSet,
}

#[derive(Debug, Clone)]
pub(crate) struct CandidatePlan {
    pub selection: Option<CandidateSelection>,
    pub fts_scores: Option<OrdinalScores>,
    pub used_ann: bool,
    pub used_scalar: bool,
    pub used_fts_index: bool,
}

impl CandidatePlan {
    pub(crate) fn candidate_count(&self, document_count: usize) -> u64 {
        self.fts_scores.as_ref().map_or_else(
            || {
                self.selection.as_ref().map_or_else(
                    || u64::try_from(document_count).unwrap_or(u64::MAX),
                    CandidateSelection::count,
                )
            },
            OrdinalScores::candidate_count,
        )
    }
}

impl CandidateSelection {
    pub(crate) fn count(&self) -> u64 {
        u64::try_from(self.ids.len()).unwrap_or(u64::MAX)
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = &str> {
        self.ids.ids()
    }

    pub(crate) fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }
}

impl IndexRegistry {
    /// Builds a complete generation before it is published into collection
    /// state. A failure therefore leaves the previous generation untouched.
    pub(crate) fn build(
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
    ) -> Result<Self> {
        let ordinals = OrdinalTable::build(docs)?;
        let mut indexes = BTreeMap::new();
        for field in &schema.vectors {
            let Some(params) = field.index_params.as_ref() else {
                continue;
            };
            if !matches!(params.index_type, IndexType::Hnsw | IndexType::Ivf) {
                continue;
            }
            indexes.insert(
                field.name.clone(),
                build_vector_index(docs, &field.name, params, source_revision, &ordinals)?,
            );
        }
        let scalar_indexes = ScalarIndexRegistry::build(schema, docs, source_revision, &ordinals)?;
        let fts_indexes = FtsIndexRegistry::build(schema, docs, source_revision, &ordinals)?;
        Ok(Self {
            ordinals,
            indexes,
            scalar_indexes,
            fts_indexes,
        })
    }

    pub(crate) fn restore_cache(
        bytes: &[u8],
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
        source_identity: &str,
    ) -> Option<Self> {
        cache::restore(bytes, schema, docs, source_revision, source_identity)
    }

    pub(crate) fn cache_bytes(
        &self,
        schema: &CollectionSchema,
        source_revision: u64,
        source_identity: &str,
    ) -> Result<Vec<u8>> {
        cache::encode(self, schema, source_revision, source_identity)
    }

    pub(crate) fn has_cacheable_indexes(&self) -> bool {
        !self.indexes.is_empty() || !self.scalar_indexes.is_empty() || !self.fts_indexes.is_empty()
    }

    /// Publishes a lightweight index generation for document mutations. The
    /// immutable graph/posting base is shared with readers of the previous
    /// collection generation; only changed vectors and tombstones are copied.
    /// Once the overlay is large enough to hurt query latency, it is folded
    /// into a new complete base before publication.
    pub(crate) fn apply_document_changes(
        &self,
        schema: &CollectionSchema,
        previous_docs: &DocumentMap,
        docs: &DocumentMap,
        source_revision: u64,
        changed_ids: &BTreeSet<String>,
    ) -> Result<Self> {
        let mut ordinals = self.ordinals.clone();
        for id in changed_ids {
            match (previous_docs.contains_key(id), docs.contains_key(id)) {
                (_, true) => {
                    ordinals.ensure_live(id)?;
                }
                (true, false) => ordinals.remove_live(id),
                (false, false) => {}
            }
        }
        if ordinals.should_compact() {
            return Self::build(schema, docs, source_revision);
        }

        let mut indexes = BTreeMap::new();
        for field in &schema.vectors {
            let Some(params) = field.index_params.as_ref() else {
                continue;
            };
            if !matches!(params.index_type, IndexType::Hnsw | IndexType::Ivf) {
                continue;
            }

            let Some(current) = self
                .indexes
                .get(&field.name)
                .filter(|index| index.params == *params)
            else {
                indexes.insert(
                    field.name.clone(),
                    build_vector_index(docs, &field.name, params, source_revision, &ordinals)?,
                );
                continue;
            };

            let mut next = current.clone();
            next.source_revision = source_revision;
            for id in changed_ids {
                let previous = previous_docs
                    .get(id)
                    .and_then(|doc| doc.vector(&field.name));
                let current = docs.get(id).and_then(|doc| doc.vector(&field.name));
                if previous == current {
                    continue;
                }

                let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                    Error::internal(format!("vector ordinal is missing for document '{id}'"))
                })?;

                next.delta.remove(&ordinal);
                next.delta_ordinals.remove(ordinal);
                if next.base.vectors.contains_key(ordinal) {
                    next.tombstones.insert(ordinal);
                } else {
                    next.tombstones.remove(ordinal);
                }
                if let Some(vector) = current {
                    next.delta
                        .insert(ordinal, encode_vector(id, &field.name, params, vector)?);
                    next.delta_ordinals.insert(ordinal);
                }
            }

            if next.should_compact() {
                next = build_vector_index(docs, &field.name, params, source_revision, &ordinals)?;
            }
            indexes.insert(field.name.clone(), next);
        }
        let scalar_indexes = self.scalar_indexes.apply_document_changes(
            schema,
            previous_docs,
            docs,
            source_revision,
            changed_ids,
            &ordinals,
        )?;
        let fts_indexes = self.fts_indexes.apply_document_changes(
            schema,
            previous_docs,
            docs,
            source_revision,
            changed_ids,
            &ordinals,
        )?;
        Ok(Self {
            ordinals,
            indexes,
            scalar_indexes,
            fts_indexes,
        })
    }

    /// Selects an ANN candidate generation only when it exactly matches the
    /// collection revision and query metric. Any stale/missing generation
    /// returns `None`, which is the caller's signal to use the exact oracle.
    fn candidates(
        &self,
        docs: &DocumentMap,
        revision: u64,
        query: &SearchQuery,
        allowed: Option<&OrdinalSet>,
    ) -> Result<Option<CandidateSelection>> {
        let Some(index) = self.indexes.get(&query.field_name) else {
            return Ok(None);
        };
        if index.source_revision != revision
            || query
                .params
                .get("is_linear")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        {
            return Ok(None);
        }
        if requested_metric(query).is_some_and(|metric| metric != index.params.metric_type) {
            return Ok(None);
        }
        let Some(query_vector) = query_vector(docs, query)? else {
            return Ok(None);
        };
        let topk = usize::try_from(query.topk)
            .map_err(|_| Error::invalid_argument("query topk must be positive"))?;
        let metric = index.params.metric_type;
        let eligible_count = allowed.map_or_else(
            || index.live_vector_count(),
            |allowed| index.eligible_vector_count(allowed.bitmap()),
        );
        if eligible_count == 0 {
            return Ok(Some(CandidateSelection {
                ids: OrdinalSet::new(&self.ordinals, RoaringTreemap::new()),
            }));
        }
        let search = AnnSearchContext {
            query,
            vector: &query_vector,
            topk,
            metric,
            allowed: allowed.map(OrdinalSet::bitmap),
            eligible_count,
            ordinals: &self.ordinals,
        };
        let ids = match &index.base.kind {
            VectorIndexKind::Hnsw(hnsw) => index.hnsw_candidates(hnsw, &search),
            VectorIndexKind::Ivf(ivf) => index.ivf_candidates(ivf, &search),
        };
        Ok(ids.map(|ids| CandidateSelection {
            ids: OrdinalSet::new(&self.ordinals, ids),
        }))
    }

    pub(crate) fn scalar_candidates(
        &self,
        source_revision: u64,
        filter: &zvec_core::filter::FilterExpr,
    ) -> Option<ScalarCandidates> {
        self.scalar_indexes
            .candidates(source_revision, filter, &self.ordinals)
    }

    fn fts_scores(
        &self,
        docs: &DocumentMap,
        source_revision: u64,
        query: &SearchQuery,
        candidates: Option<&OrdinalSet>,
        topk: Option<usize>,
    ) -> Result<Option<OrdinalScores>> {
        self.fts_indexes
            .search(source_revision, query, candidates, docs, &self.ordinals)?
            .map(|scores| match topk {
                Some(topk) => OrdinalScores::new_topk(&self.ordinals, scores, topk),
                None => OrdinalScores::new(&self.ordinals, scores),
            })
            .transpose()
    }

    fn plan_fts_candidates(
        &self,
        docs: &DocumentMap,
        source_revision: u64,
        query: &SearchQuery,
        scalar: Option<ScalarCandidates>,
        filter_exact: bool,
    ) -> Result<CandidatePlan> {
        let used_scalar = scalar.is_some();
        let topk = filter_exact
            .then(|| {
                usize::try_from(query.topk)
                    .ok()
                    .filter(|topk| *topk > 0)
                    .ok_or_else(|| Error::invalid_argument("query topk must be positive"))
            })
            .transpose()?;
        let fts_scores = self.fts_scores(
            docs,
            source_revision,
            query,
            scalar.as_ref().map(|value| &value.ids),
            topk,
        )?;
        if let Some(fts_scores) = fts_scores {
            return Ok(CandidatePlan {
                selection: None,
                fts_scores: Some(fts_scores),
                used_ann: false,
                used_scalar,
                used_fts_index: true,
            });
        }
        Ok(CandidatePlan {
            selection: scalar.map(|selection| CandidateSelection {
                ids: selection.into_ids(),
            }),
            fts_scores: None,
            used_ann: false,
            used_scalar,
            used_fts_index: false,
        })
    }

    pub(crate) fn plan_candidates(
        &self,
        docs: &DocumentMap,
        source_revision: u64,
        query: &SearchQuery,
        filter: Option<&zvec_core::filter::FilterExpr>,
    ) -> Result<CandidatePlan> {
        let scalar = filter.and_then(|filter| self.scalar_candidates(source_revision, filter));
        if query.fts.is_some() {
            let filter_exact =
                filter.is_none() || scalar.as_ref().is_some_and(|candidates| candidates.exact);
            return self.plan_fts_candidates(docs, source_revision, query, scalar, filter_exact);
        }
        let used_scalar = scalar.is_some();
        let Some(mut scalar) = scalar else {
            let ann = self.candidates(docs, source_revision, query, None)?;
            let used_ann = ann.is_some();
            return Ok(CandidatePlan {
                selection: ann,
                fts_scores: None,
                used_ann,
                used_scalar,
                used_fts_index: false,
            });
        };
        let topk = usize::try_from(query.topk)
            .map_err(|_| Error::invalid_argument("query topk must be positive"))?;
        let exact_limit = topk
            .saturating_mul(SCALAR_EXACT_PREFILTER_PER_RESULT)
            .max(SCALAR_EXACT_PREFILTER_MIN);
        if scalar.len() <= exact_limit {
            return Ok(CandidatePlan {
                selection: Some(CandidateSelection {
                    ids: scalar.into_ids(),
                }),
                fts_scores: None,
                used_ann: false,
                used_scalar,
                used_fts_index: false,
            });
        }
        if !scalar.exact {
            let Some(filter) = filter else {
                return Err(Error::internal(
                    "scalar candidate refinement requires a parsed filter",
                ));
            };
            scalar.retain_ids(|id| {
                docs.get(id)
                    .is_some_and(|doc| filter.matches(&doc.to_core()))
            });
            scalar.exact = true;
        }
        if scalar.len() <= exact_limit {
            return Ok(CandidatePlan {
                selection: Some(CandidateSelection {
                    ids: scalar.into_ids(),
                }),
                fts_scores: None,
                used_ann: false,
                used_scalar,
                used_fts_index: false,
            });
        }
        let Some(ann) = self.candidates(docs, source_revision, query, Some(&scalar.ids))? else {
            return Ok(CandidatePlan {
                selection: Some(CandidateSelection {
                    ids: scalar.into_ids(),
                }),
                fts_scores: None,
                used_ann: false,
                used_scalar,
                used_fts_index: false,
            });
        };
        Ok(CandidatePlan {
            selection: Some(ann),
            fts_scores: None,
            used_ann: true,
            used_scalar,
            used_fts_index: false,
        })
    }

    pub(crate) fn stats(&self, schema: &CollectionSchema) -> Vec<IndexStat> {
        let mut stats = Vec::new();
        for field in &schema.vectors {
            let Some(params) = field.index_params.as_ref() else {
                continue;
            };
            if !matches!(params.index_type, IndexType::Hnsw | IndexType::Ivf) {
                continue;
            }
            if let Some(index) = self.indexes.get(&field.name) {
                stats.push(IndexStat {
                    name: field.name.clone(),
                    index_type: index.params.index_type,
                    completeness: 1.0,
                    source_revision: index.source_revision,
                    document_count: u64::try_from(index.live_vector_count()).unwrap_or(u64::MAX),
                    estimated_payload_bytes: Some(index.estimated_payload_bytes()),
                    state: "ready".into(),
                });
            } else {
                stats.push(IndexStat {
                    name: field.name.clone(),
                    index_type: params.index_type,
                    completeness: 0.0,
                    source_revision: 0,
                    document_count: 0,
                    estimated_payload_bytes: None,
                    state: "missing".into(),
                });
            }
        }
        stats.extend(self.scalar_indexes.stats());
        stats.extend(self.fts_indexes.stats());
        stats
    }
}

impl VectorIndex {
    fn estimated_payload_bytes(&self) -> u64 {
        let base_vectors = self
            .base
            .vectors
            .values()
            .fold(self.base.vectors.slot_count(), |total, vector| {
                total.saturating_add(vector.encoded_bytes())
            });
        let vectors = self.delta.values().fold(base_vectors, |total, vector| {
            total
                .saturating_add(std::mem::size_of::<u64>())
                .saturating_add(vector.encoded_bytes())
        });
        let membership = self
            .base
            .vector_ordinals
            .serialized_size()
            .saturating_add(self.delta_ordinals.serialized_size())
            .saturating_add(self.tombstones.serialized_size());
        let kind = match &self.base.kind {
            VectorIndexKind::Hnsw(index) => index.estimated_payload_bytes(),
            VectorIndexKind::Ivf(index) => index.estimated_payload_bytes(),
        };
        u64::try_from(vectors.saturating_add(membership).saturating_add(kind)).unwrap_or(u64::MAX)
    }

    fn live_vector_count(&self) -> usize {
        let hidden_base =
            bitmap_count_to_usize(self.tombstones.intersection_len(&self.base.vector_ordinals));
        self.base
            .vectors
            .len()
            .saturating_sub(hidden_base)
            .saturating_add(self.delta.len())
    }

    fn eligible_vector_count(&self, allowed: &RoaringTreemap) -> usize {
        let base = self
            .base
            .vector_ordinals
            .intersection_len(allowed)
            .saturating_sub(self.tombstones.intersection_len(allowed));
        let delta = self.delta_ordinals.intersection_len(allowed);
        bitmap_count_to_usize(base.saturating_add(delta))
    }

    fn base_eligible_vector_count(&self, allowed: &RoaringTreemap) -> usize {
        bitmap_count_to_usize(
            self.base
                .vector_ordinals
                .intersection_len(allowed)
                .saturating_sub(self.tombstones.intersection_len(allowed)),
        )
    }

    fn hnsw_candidates(
        &self,
        hnsw: &HnswIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<RoaringTreemap> {
        let requested_ef = optional_positive_query_parameter(search.query, "ef");
        let limit = hnsw.candidate_limit(requested_ef, search.topk, search.eligible_count);
        let base = if let Some(allowed) = search.allowed {
            let base_eligible_count = self.base_eligible_vector_count(allowed);
            let traversal_limit =
                proportional_candidate_limit(limit, self.base.vectors.len(), base_eligible_count);
            if traversal_limit >= search.eligible_count {
                return None;
            }
            hnsw.filtered_candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                limit,
                traversal_limit,
                search.metric,
                HnswFilter {
                    allowed,
                    excluded: &self.tombstones,
                    eligible_count: base_eligible_count,
                },
            )
        } else {
            let base_ef = limit
                .saturating_add(bitmap_count_to_usize(self.tombstones.len()))
                .min(self.base.vectors.len());
            hnsw.candidates(
                &self.base.vectors,
                search.ordinals,
                search.vector,
                Some(base_ef),
                search.topk,
                search.metric,
            )
        };
        let merged = self.merge_candidates(
            base,
            search.vector,
            Some(limit),
            search.metric,
            search.allowed,
            search.ordinals,
        );
        candidate_set_is_sufficient(&merged, search).then_some(merged)
    }

    fn ivf_candidates(
        &self,
        ivf: &IvfIndex,
        search: &AnnSearchContext<'_>,
    ) -> Option<RoaringTreemap> {
        let requested_nprobe = optional_positive_query_parameter(search.query, "nprobe");
        let base = search.allowed.map_or_else(
            || ivf.candidates(search.vector, requested_nprobe),
            |allowed| {
                ivf.filtered_candidates(
                    search.vector,
                    requested_nprobe,
                    search.topk,
                    allowed,
                    &self.tombstones,
                )
            },
        );
        let merged = self.merge_candidates(
            base,
            search.vector,
            None,
            search.metric,
            search.allowed,
            search.ordinals,
        );
        let Some(scale_factor) = optional_f32_query_parameter(search.query, "scale_factor") else {
            return candidate_set_is_sufficient(&merged, search).then_some(merged);
        };
        let limit = scaled_candidate_limit(search.topk, scale_factor, search.eligible_count);
        let limited = self.limit_candidates(
            &merged,
            search.vector,
            limit,
            search.metric,
            search.ordinals,
        );
        candidate_set_is_sufficient(&limited, search).then_some(limited)
    }

    fn overlay_len(&self) -> usize {
        let new_vectors = self
            .delta
            .keys()
            .filter(|&&ordinal| !self.base.vectors.contains_key(ordinal))
            .count();
        bitmap_count_to_usize(self.tombstones.len()).saturating_add(new_vectors)
    }

    fn should_compact(&self) -> bool {
        self.overlay_len() >= delta_compaction_limit(self.base.vectors.len())
    }

    fn merge_candidates(
        &self,
        mut base: RoaringTreemap,
        query: &[f32],
        limit: Option<usize>,
        metric: MetricType,
        allowed: Option<&RoaringTreemap>,
        ordinals: &OrdinalTable,
    ) -> RoaringTreemap {
        base -= &self.tombstones;
        if let Some(allowed) = allowed {
            base &= allowed;
        }
        let mut ids = base;
        let mut delta = self.delta_ordinals.clone();
        if let Some(allowed) = allowed {
            delta &= allowed;
        }
        ids |= delta;
        match limit {
            Some(limit) => self.limit_candidates(&ids, query, limit, metric, ordinals),
            None => ids,
        }
    }

    fn limit_candidates(
        &self,
        ids: &RoaringTreemap,
        query: &[f32],
        limit: usize,
        metric: MetricType,
        ordinals: &OrdinalTable,
    ) -> RoaringTreemap {
        let mut scored: Vec<(u64, f64)> = ids
            .iter()
            .filter_map(|ordinal| {
                let vector = self
                    .delta
                    .get(&ordinal)
                    .or_else(|| self.base.vectors.get(ordinal))?;
                Some((ordinal, score(query, vector, metric)))
            })
            .collect();
        scored.sort_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| {
                ordinals
                    .id(left.0)
                    .unwrap_or_default()
                    .cmp(ordinals.id(right.0).unwrap_or_default())
                    .then_with(|| left.0.cmp(&right.0))
            })
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(ordinal, _)| ordinal)
            .collect()
    }
}

fn candidate_set_is_sufficient(ids: &RoaringTreemap, search: &AnnSearchContext<'_>) -> bool {
    search.allowed.is_none()
        || bitmap_count_to_usize(ids.len()) >= search.topk.min(search.eligible_count)
}

fn bitmap_count_to_usize(count: u64) -> usize {
    usize::try_from(count).unwrap_or(usize::MAX)
}

fn proportional_candidate_limit(target: usize, population: usize, eligible: usize) -> usize {
    if target == 0 || population == 0 || eligible == 0 {
        return 0;
    }
    let target = u128::try_from(target).unwrap_or(u128::MAX);
    let population_u128 = u128::try_from(population).unwrap_or(u128::MAX);
    let eligible_u128 = u128::try_from(eligible).unwrap_or(u128::MAX);
    let scaled = target
        .saturating_mul(population_u128)
        .saturating_add(eligible_u128.saturating_sub(1))
        / eligible_u128;
    usize::try_from(scaled)
        .unwrap_or(population)
        .min(population)
}

fn build_vector_index(
    docs: &DocumentMap,
    field_name: &str,
    params: &IndexParams,
    source_revision: u64,
    ordinals: &OrdinalTable,
) -> Result<VectorIndex> {
    let vectors = collect_vectors(docs, field_name, params, ordinals)?;
    let vector_ordinals: RoaringTreemap = vectors.keys().collect();
    let kind = match params.index_type {
        IndexType::Hnsw => VectorIndexKind::Hnsw(HnswIndex::build(
            &vectors,
            ordinals,
            positive_parameter(params, "m")?,
            positive_parameter(params, "ef_construction")?,
            params.metric_type,
        )),
        IndexType::Ivf => VectorIndexKind::Ivf(IvfIndex::build(
            &vectors,
            positive_parameter(params, "n_list")?,
            nonnegative_parameter(params, "n_iters")?,
        )),
        _ => {
            return Err(Error::not_supported(format!(
                "{:?} does not have an in-memory ANN implementation",
                params.index_type
            )))
        }
    };
    Ok(VectorIndex {
        params: params.clone(),
        source_revision,
        base: Arc::new(VectorIndexBase {
            vectors,
            vector_ordinals,
            kind,
        }),
        delta: BTreeMap::new(),
        delta_ordinals: RoaringTreemap::new(),
        tombstones: RoaringTreemap::new(),
    })
}

fn delta_compaction_limit(base_len: usize) -> usize {
    let fractional =
        base_len.saturating_add(DELTA_COMPACTION_DIVISOR - 1) / DELTA_COMPACTION_DIVISOR;
    fractional.clamp(MIN_DELTA_COMPACTION, MAX_DELTA_COMPACTION)
}

fn collect_vectors(
    docs: &DocumentMap,
    field_name: &str,
    params: &IndexParams,
    ordinals: &OrdinalTable,
) -> Result<OrdinalMap<QuantizedVector>> {
    docs.iter()
        .filter_map(|(id, doc)| doc.vector(field_name).map(|vector| (id, vector)))
        .map(|(id, vector)| {
            let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                Error::internal(format!("vector ordinal is missing for document '{id}'"))
            })?;
            Ok((ordinal, encode_vector(id, field_name, params, vector)?))
        })
        .collect()
}

fn encode_vector(
    id: &str,
    field_name: &str,
    params: &IndexParams,
    vector: &VectorValue,
) -> Result<QuantizedVector> {
    let dense = vector.to_dense_f32().ok_or_else(|| {
        Error::resource_exhausted(format!(
            "document '{id}' field '{field_name}' cannot be represented by the f32 ANN kernel"
        ))
    })?;
    QuantizedVector::encode(dense, params.quantize_type).map_err(|error| {
        Error::new(
            error.code,
            format!(
                "build {:?} index for document '{id}' field '{field_name}': {}",
                params.index_type, error.message
            ),
        )
    })
}

fn positive_parameter(params: &IndexParams, name: &str) -> Result<usize> {
    let value = params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be positive"))
        })?;
    if value == 0 {
        return Err(Error::invalid_argument(format!(
            "index parameter '{name}' must be positive"
        )));
    }
    usize::try_from(value)
        .map_err(|_| Error::resource_exhausted(format!("index parameter '{name}' is too large")))
}

fn nonnegative_parameter(params: &IndexParams, name: &str) -> Result<usize> {
    let value = params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be non-negative"))
        })?;
    usize::try_from(value)
        .map_err(|_| Error::resource_exhausted(format!("index parameter '{name}' is too large")))
}

fn query_vector(docs: &DocumentMap, query: &SearchQuery) -> Result<Option<Vec<f32>>> {
    if let Some(vector) = &query.vector {
        return Ok(Some(vector.clone()));
    }
    let Some(id) = query.id.as_deref() else {
        return Ok(None);
    };
    let Some(vector) = docs.get(id).and_then(|doc| doc.vector(&query.field_name)) else {
        return Ok(None);
    };
    vector.to_dense_f32().map(Some).ok_or_else(|| {
        Error::resource_exhausted(format!(
            "source document '{id}' cannot be represented by the f32 ANN kernel"
        ))
    })
}

fn optional_positive_query_parameter(query: &SearchQuery, name: &str) -> Option<usize> {
    query
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| (value > 0).then_some(value))
        .and_then(|value| usize::try_from(value).ok())
}

#[allow(clippy::cast_possible_truncation)]
fn optional_f32_query_parameter(query: &SearchQuery, name: &str) -> Option<f32> {
    query
        .params
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0 && *value <= f64::from(f32::MAX))
        .map(|value| value as f32)
}

fn requested_metric(query: &SearchQuery) -> Option<MetricType> {
    match query
        .params
        .get("metric")?
        .as_str()?
        .to_ascii_lowercase()
        .as_str()
    {
        "l2" | "euclidean" => Some(MetricType::L2),
        "ip" | "inner_product" | "dot" => Some(MetricType::Ip),
        "cosine" => Some(MetricType::Cosine),
        "mips_l2" | "mips-l2" => Some(MetricType::MipsL2),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
