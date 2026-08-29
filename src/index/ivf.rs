//! IVF-compatible index facade with an exact fallback.

use super::flat::FlatIndex;
use super::{Candidate, IndexInput, VectorIndex};
use crate::doc::Doc;
use crate::error::Result;
use crate::query::SearchQuery;
use crate::types::{IndexType, MetricType};

#[derive(Debug, Clone)]
pub struct IvfIndex {
    pub inner: FlatIndex,
    pub n_list: usize,
    pub n_iters: usize,
}
impl IvfIndex {
    pub fn new(field_name: impl Into<String>, dimension: usize, metric: MetricType, n_list: usize, n_iters: usize) -> Self {
        Self { inner: FlatIndex::new(field_name, dimension, metric), n_list, n_iters }
    }
}
impl VectorIndex for IvfIndex {
    fn kind(&self) -> IndexType { IndexType::Ivf }
    fn dimension(&self) -> usize { self.inner.dimension() }
    fn metric(&self) -> MetricType { self.inner.metric() }
    fn source_revision(&self) -> u64 { self.inner.source_revision() }
    fn build(&mut self, input: IndexInput<'_>) -> Result<()> { self.inner.build(input) }
    fn search(&self, query: &SearchQuery, docs: &[Doc]) -> Result<Vec<Candidate>> { self.inner.search(query, docs) }
}

