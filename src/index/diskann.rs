//! DiskANN/Vamana portable facade.

use super::flat::FlatIndex;
use super::{Candidate, IndexInput, VectorIndex};
use crate::doc::Doc;
use crate::error::Result;
use crate::query::SearchQuery;
use crate::types::{IndexType, MetricType};

#[derive(Debug, Clone)]
pub struct DiskAnnIndex {
    pub inner: FlatIndex,
    pub max_degree: usize,
    pub list_size: usize,
    pub pq_chunk_num: usize,
}
impl DiskAnnIndex {
    pub fn new(field_name: impl Into<String>, dimension: usize, metric: MetricType, max_degree: usize, list_size: usize, pq_chunk_num: usize) -> Self {
        Self { inner: FlatIndex::new(field_name, dimension, metric), max_degree, list_size, pq_chunk_num }
    }
}
impl VectorIndex for DiskAnnIndex {
    fn kind(&self) -> IndexType { IndexType::Diskann }
    fn dimension(&self) -> usize { self.inner.dimension() }
    fn metric(&self) -> MetricType { self.inner.metric() }
    fn source_revision(&self) -> u64 { self.inner.source_revision() }
    fn build(&mut self, input: IndexInput<'_>) -> Result<()> { self.inner.build(input) }
    fn search(&self, query: &SearchQuery, docs: &[Doc]) -> Result<Vec<Candidate>> { self.inner.search(query, docs) }
}

