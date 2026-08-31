//! Multi-route retrieval and deterministic reranking.

use crate::error::{Error, Result};
use crate::query::{
    apply_diskann_query_controls, apply_fts_query_controls, apply_hnsw_query_controls,
    apply_ivf_query_controls, unsupported_query_controls, DiskannQueryParams, FlatQueryParams, Fts,
    FtsQueryParams, HnswQueryParams, IvfQueryParams, IvfRabitqQueryParams, SearchQuery,
};
use serde::{Deserialize, Serialize};

/// Reranking strategy used to fuse branch results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RerankMethod {
    ReciprocalRank { rank_constant: f64 },
    Weighted { weights: Vec<f64> },
}

impl Default for RerankMethod {
    fn default() -> Self {
        Self::Weighted {
            weights: Vec::new(),
        }
    }
}

/// A branch in a [`MultiQuery`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubQuery {
    pub field_name: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub sparse_vector: Option<Vec<(u32, f32)>>,
    pub fts: Option<Fts>,
    pub num_candidates: i32,
    pub params: serde_json::Map<String, serde_json::Value>,
}

impl SubQuery {
    pub fn new() -> Result<Self> {
        Ok(Self {
            field_name: None,
            vector: None,
            sparse_vector: None,
            fts: None,
            num_candidates: 10,
            params: serde_json::Map::new(),
        })
    }
    pub fn set_num_candidates(&mut self, n: i32) -> Result<()> {
        if n <= 0 {
            return Err(Error::invalid_argument("num_candidates must be positive"));
        }
        self.num_candidates = n;
        Ok(())
    }
    pub fn num_candidates(&self) -> i32 {
        self.num_candidates
    }
    pub fn set_field_name(&mut self, name: &str) -> Result<()> {
        if name.trim().is_empty() || name.contains('\0') {
            return Err(Error::invalid_argument("field name must be non-empty"));
        }
        self.field_name = Some(name.to_string());
        Ok(())
    }
    pub fn set_query_vector(&mut self, data: &[f32]) -> Result<()> {
        if data.is_empty() || !data.iter().all(|v| v.is_finite()) {
            return Err(Error::invalid_argument(
                "query vector must be non-empty and finite",
            ));
        }
        self.vector = Some(data.to_vec());
        self.sparse_vector = None;
        Ok(())
    }
    pub fn set_sparse_vector(&mut self, indices: &[u32], values: &[f32]) -> Result<()> {
        if indices.is_empty()
            || indices.len() != values.len()
            || !values.iter().all(|v| v.is_finite())
        {
            return Err(Error::invalid_argument("invalid sparse vector"));
        }
        self.sparse_vector = Some(
            indices
                .iter()
                .copied()
                .zip(values.iter().copied())
                .collect(),
        );
        self.vector = None;
        Ok(())
    }
    pub fn set_sparse_indices(&mut self, indices: &[u32]) -> Result<()> {
        if indices.is_empty() {
            return Err(Error::invalid_argument("sparse indices cannot be empty"));
        }
        let values = self.sparse_vector.as_ref().map_or_else(
            || vec![0.0; indices.len()],
            |v| v.iter().map(|(_, x)| *x).collect(),
        );
        if values.len() != indices.len() {
            return Err(Error::invalid_argument(
                "sparse indices and values have different lengths",
            ));
        }
        self.sparse_vector = Some(indices.iter().copied().zip(values).collect());
        Ok(())
    }
    pub fn set_sparse_values(&mut self, values: &[f32]) -> Result<()> {
        if values.is_empty() || !values.iter().all(|v| v.is_finite()) {
            return Err(Error::invalid_argument(
                "sparse values cannot be empty and must be finite",
            ));
        }
        let indices: Vec<u32> = if let Some(existing) = self.sparse_vector.as_ref() {
            existing.iter().map(|(index, _)| *index).collect()
        } else {
            let length = u32::try_from(values.len())
                .map_err(|_| Error::resource_exhausted("sparse vector exceeds u32 dimensions"))?;
            (0..length).collect()
        };
        if indices.len() != values.len() {
            return Err(Error::invalid_argument(
                "sparse indices and values have different lengths",
            ));
        }
        self.sparse_vector = Some(indices.into_iter().zip(values.iter().copied()).collect());
        Ok(())
    }
    pub fn set_hnsw_params(&mut self, params: HnswQueryParams) -> Result<()> {
        apply_hnsw_query_controls(&mut self.params, params)
    }
    pub fn set_ivf_params(&mut self, params: IvfQueryParams) -> Result<()> {
        apply_ivf_query_controls(&mut self.params, params)
    }
    pub fn set_ivf_rabitq_params(&mut self, _params: IvfRabitqQueryParams) -> Result<()> {
        unsupported_query_controls("IVF RaBitQ")
    }
    pub fn set_flat_params(&mut self, _params: FlatQueryParams) -> Result<()> {
        unsupported_query_controls("Flat refinement")
    }
    pub fn set_diskann_params(&mut self, params: DiskannQueryParams) -> Result<()> {
        apply_diskann_query_controls(&mut self.params, params)
    }
    pub fn set_fts_params(&mut self, params: FtsQueryParams) -> Result<()> {
        apply_fts_query_controls(&mut self.params, params)
    }
    pub fn set_fts(&mut self, fts: &Fts) -> Result<()> {
        if fts.query_string.is_none() && fts.match_string.is_none() {
            return Err(Error::invalid_argument("FTS query has no expression"));
        }
        self.fts = Some(fts.clone());
        Ok(())
    }
    pub(crate) fn to_search_query(&self) -> Result<SearchQuery> {
        let field = self
            .field_name
            .as_deref()
            .ok_or_else(|| Error::invalid_argument("sub-query field_name is required"))?;
        let mut query = if let Some(vector) = &self.vector {
            SearchQuery::new(field, vector, self.num_candidates)?
        } else if let Some(sparse) = &self.sparse_vector {
            let (i, v): (Vec<_>, Vec<_>) = sparse.iter().copied().unzip();
            SearchQuery::sparse(field, &i, &v, self.num_candidates)?
        } else if let Some(fts) = &self.fts {
            SearchQuery::fts(field, fts, self.num_candidates)?
        } else {
            return Err(Error::invalid_argument(
                "sub-query has no vector or FTS payload",
            ));
        };
        query.params.extend(self.params.clone());
        Ok(query)
    }
}

/// A collection of sub-queries fused into one ranked result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiQuery {
    pub queries: Vec<SubQuery>,
    pub topk_value: i32,
    pub filter: Option<String>,
    pub include_vector_value: bool,
    pub output_fields: Option<Vec<String>>,
    pub rerank: RerankMethod,
    pub normalization: Option<String>,
}

impl MultiQuery {
    pub fn new() -> Result<Self> {
        Ok(Self {
            queries: Vec::new(),
            topk_value: 10,
            filter: None,
            include_vector_value: false,
            output_fields: None,
            rerank: RerankMethod::Weighted {
                weights: Vec::new(),
            },
            normalization: None,
        })
    }
    pub fn add_sub_query(&mut self, sub: &SubQuery) -> Result<()> {
        if self.queries.len() >= 1024 {
            return Err(Error::resource_exhausted(
                "multi-query branch limit exceeded",
            ));
        }
        self.queries.push(sub.clone());
        Ok(())
    }
    pub fn sub_query_count(&self) -> usize {
        self.queries.len()
    }
    pub fn set_topk(&mut self, topk: i32) -> Result<()> {
        if topk <= 0 {
            return Err(Error::invalid_argument("topk must be positive"));
        }
        self.topk_value = topk;
        Ok(())
    }
    pub fn topk(&self) -> i32 {
        self.topk_value
    }
    pub fn set_filter(&mut self, filter: &str) -> Result<()> {
        self.filter = (!filter.trim().is_empty()).then_some(filter.to_string());
        Ok(())
    }
    pub fn set_include_vector(&mut self, include: bool) -> Result<()> {
        self.include_vector_value = include;
        Ok(())
    }
    pub fn include_vector(&self) -> bool {
        self.include_vector_value
    }
    pub fn set_output_fields(&mut self, fields: &[&str]) -> Result<()> {
        if fields
            .iter()
            .any(|f| f.trim().is_empty() || f.contains('\0'))
        {
            return Err(Error::invalid_argument("output field name is invalid"));
        }
        self.output_fields = Some(fields.iter().map(|v| (*v).to_string()).collect());
        Ok(())
    }
    pub fn set_rerank_rrf(&mut self, rank_constant: i32) -> Result<()> {
        if rank_constant <= 0 {
            return Err(Error::invalid_argument(
                "RRF rank constant must be positive",
            ));
        }
        self.rerank = RerankMethod::ReciprocalRank {
            rank_constant: f64::from(rank_constant),
        };
        Ok(())
    }
    pub fn set_rerank_weighted(&mut self, weights: &[f64]) -> Result<()> {
        if weights.is_empty() || !weights.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(Error::invalid_argument(
                "weights must be non-empty, finite, and non-negative",
            ));
        }
        self.rerank = RerankMethod::Weighted {
            weights: weights.to_vec(),
        };
        Ok(())
    }
    pub fn set_normalization(&mut self, method: &str) -> Result<()> {
        let normalized = method.to_ascii_lowercase();
        if !matches!(normalized.as_str(), "none" | "minmax" | "zscore") {
            return Err(Error::invalid_argument(
                "normalization must be none, minmax, or zscore",
            ));
        }
        self.normalization = Some(normalized);
        Ok(())
    }
    pub(crate) fn effective_filter(&self) -> Option<&str> {
        self.filter.as_deref()
    }
}
