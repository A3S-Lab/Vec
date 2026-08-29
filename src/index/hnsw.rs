//! HNSW-compatible index facade.
//!
//! The portable implementation uses the exact flat oracle until a graph is
//! requested explicitly.  This preserves correctness on every platform while
//! keeping the index contract stable for a future graph builder.

use super::flat::FlatIndex;
use super::{Candidate, IndexInput, VectorIndex};
use crate::doc::Doc;
use crate::error::Result;
use crate::query::SearchQuery;
use crate::types::{IndexType, MetricType};

#[derive(Debug, Clone)]
pub struct HnswIndex {
    pub inner: FlatIndex,
    pub m: usize,
    pub ef_construction: usize,
}
impl HnswIndex {
    pub fn new(field_name: impl Into<String>, dimension: usize, metric: MetricType, m: usize, ef_construction: usize) -> Self {
        Self { inner: FlatIndex::new(field_name, dimension, metric), m, ef_construction }
    }
}
impl VectorIndex for HnswIndex {
    fn kind(&self) -> IndexType { IndexType::Hnsw }
    fn dimension(&self) -> usize { self.inner.dimension() }
    fn metric(&self) -> MetricType { self.inner.metric() }
    fn source_revision(&self) -> u64 { self.inner.source_revision() }
    fn build(&mut self, input: IndexInput<'_>) -> Result<()> { self.inner.build(input) }
    fn search(&self, query: &SearchQuery, docs: &[Doc]) -> Result<Vec<Candidate>> { self.inner.search(query, docs) }
}

