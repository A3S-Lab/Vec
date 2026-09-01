//! Collection query, fetch, and iterator APIs.

use super::query_engine::{
    count_to_f64, execute_query_with_candidates, normalize_scores, parse_optional_filter,
    score_to_f32, sort_docs,
};
use super::Collection;
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::iterator::DocIterator;
use crate::multi_query::{MultiQuery, RerankMethod};
use crate::query::{GroupBySearchQuery, SearchQuery};
use crate::stats::{IndexUsage, QueryKind, QueryObservation};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

impl Collection {
    pub fn query(&self, query: &SearchQuery) -> Result<Vec<Doc>> {
        self.ensure_open()?;
        let snapshot = self.snapshot_state()?;
        let filter = parse_optional_filter(query.filter.as_deref())?;
        let plan = snapshot.indexes.plan_candidates(
            &snapshot.docs,
            snapshot.revision,
            query,
            filter.as_ref(),
        )?;
        let result = execute_query_with_candidates(
            &snapshot.schema,
            &snapshot.docs,
            query,
            plan.selection.as_ref(),
            plan.fts_scores.as_ref(),
            filter.as_ref(),
        )?;
        let has_fts = query.fts.is_some();
        snapshot.stats.record_query(QueryObservation {
            kind: if has_fts {
                QueryKind::Fts
            } else if plan.used_ann {
                QueryKind::Ann
            } else {
                QueryKind::Exact
            },
            diskann_io_backend: plan.diskann_io_backend,
            diskann_sector_reads: plan.diskann_sector_reads,
            filtered: query
                .filter
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty()),
            index_usage: IndexUsage::new(plan.used_scalar, plan.used_fts_index),
            radius: query.params.get("radius").is_some(),
            candidates: plan.candidate_count(snapshot.docs.len()),
        });
        Ok(result)
    }

    pub fn multi_query(&self, query: &MultiQuery) -> Result<Vec<Doc>> {
        self.ensure_open()?;
        if query.queries.is_empty() {
            return Err(Error::invalid_argument(
                "multi-query must contain at least one sub-query",
            ));
        }
        let snapshot = self.snapshot_state()?;
        let mut branches: Vec<Vec<Doc>> = Vec::with_capacity(query.queries.len());
        let mut used_ann = false;
        let mut diskann_io_backend = None;
        let mut diskann_sector_reads = 0_u64;
        let mut used_scalar = false;
        let mut used_fts_index = false;
        let mut candidates = 0_u64;
        for sub in &query.queries {
            let mut branch = sub.to_search_query()?;
            if let Some(filter) = query.effective_filter() {
                branch.set_filter(filter)?;
            }
            branch.include_vector = query.include_vector_value;
            branch.output_fields.clone_from(&query.output_fields);
            let filter = parse_optional_filter(branch.filter.as_deref())?;
            let plan = snapshot.indexes.plan_candidates(
                &snapshot.docs,
                snapshot.revision,
                &branch,
                filter.as_ref(),
            )?;
            used_ann |= plan.used_ann;
            diskann_io_backend = diskann_io_backend.or(plan.diskann_io_backend);
            diskann_sector_reads = diskann_sector_reads.saturating_add(plan.diskann_sector_reads);
            used_scalar |= plan.used_scalar;
            used_fts_index |= plan.used_fts_index;
            candidates = candidates.saturating_add(plan.candidate_count(snapshot.docs.len()));
            branches.push(execute_query_with_candidates(
                &snapshot.schema,
                &snapshot.docs,
                &branch,
                plan.selection.as_ref(),
                plan.fts_scores.as_ref(),
                filter.as_ref(),
            )?);
        }
        let normalization = query.normalization.as_deref().unwrap_or("none");
        for branch in &mut branches {
            normalize_scores(branch, normalization)?;
        }
        let mut fused: BTreeMap<String, (f64, Doc)> = BTreeMap::new();
        for (branch_index, branch) in branches.into_iter().enumerate() {
            let weight = match &query.rerank {
                RerankMethod::Weighted { weights } => {
                    weights.get(branch_index).copied().unwrap_or(1.0)
                }
                RerankMethod::ReciprocalRank { .. } => 1.0,
            };
            for (rank, doc) in branch.into_iter().enumerate() {
                let Some(id) = doc.get_pk().map(str::to_string) else {
                    continue;
                };
                let score = match query.rerank {
                    RerankMethod::ReciprocalRank { rank_constant } => {
                        weight / (rank_constant + count_to_f64(rank) + 1.0)
                    }
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
                doc.set_score(score_to_f32(score)?)?;
                Ok(doc)
            })
            .collect::<Result<_>>()?;
        sort_docs(&mut output);
        let topk = usize::try_from(query.topk_value)
            .map_err(|_| Error::invalid_argument("multi-query topk must be non-negative"))?;
        output.truncate(topk);
        let has_fts = query.queries.iter().any(|branch| branch.fts.is_some());
        snapshot.stats.record_query(QueryObservation {
            kind: match (used_ann, has_fts) {
                (true, true) => QueryKind::AnnFts,
                (true, false) => QueryKind::Ann,
                (false, true) => QueryKind::Fts,
                (false, false) => QueryKind::Exact,
            },
            diskann_io_backend,
            diskann_sector_reads,
            filtered: query.filter.is_some(),
            index_usage: IndexUsage::new(used_scalar, used_fts_index),
            radius: false,
            candidates,
        });
        Ok(output)
    }

    pub fn group_by(&self, query: &GroupBySearchQuery) -> Result<HashMap<String, Vec<Doc>>> {
        self.ensure_open()?;
        let candidate_limit = query.group_count.saturating_mul(query.group_topk).max(1);
        let candidate_limit = i32::try_from(candidate_limit)
            .map_err(|_| Error::resource_exhausted("group-by candidate limit exceeds i32"))?;
        let mut vector_query = SearchQuery::new(&query.field_name, &query.vector, candidate_limit)?;
        vector_query.include_vector = query.include_vector;
        vector_query.output_fields.clone_from(&query.output_fields);
        vector_query.params.clone_from(&query.params);
        if let Some(filter) = &query.filter {
            vector_query.set_filter(filter)?;
        }
        let docs = self.query(&vector_query)?;
        let mut groups: HashMap<String, Vec<Doc>> = HashMap::new();
        for doc in docs {
            let key = doc.scalar_json(&query.group_by_field).map_or_else(
                || "__null__".to_string(),
                |value| match value {
                    Value::String(value) => value,
                    other => other.to_string(),
                },
            );
            let group = groups.entry(key).or_default();
            if group.len() < query.group_topk as usize {
                group.push(doc);
            }
        }
        if groups.len() > query.group_count as usize {
            let mut ranked: Vec<(String, f32)> = groups
                .iter()
                .map(|(key, values)| {
                    (
                        key.clone(),
                        values.first().map_or(f32::NEG_INFINITY, Doc::get_score),
                    )
                })
                .collect();
            ranked.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let keep: HashSet<String> = ranked
                .into_iter()
                .take(query.group_count as usize)
                .map(|(key, _)| key)
                .collect();
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
        let fields =
            output_fields.map(|values| values.iter().map(|v| (*v).to_string()).collect::<Vec<_>>());
        Ok(pks
            .iter()
            .filter_map(|pk| state.docs.get(*pk))
            .map(|doc| doc.project(fields.as_deref(), include_vector))
            .collect())
    }

    // This name is retained for zvec API compatibility; `DocIterator` exposes
    // fallible batch iteration rather than implementing `Iterator` directly.
    #[allow(clippy::iter_not_returning_iterator)]
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
        let fields =
            output_fields.map(|values| values.iter().map(|v| (*v).to_string()).collect::<Vec<_>>());
        let docs = state
            .docs
            .values()
            .map(|doc| doc.project(fields.as_deref(), include_vector))
            .collect();
        Ok(DocIterator::new(docs, state.revision))
    }
}
