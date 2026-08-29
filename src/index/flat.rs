//! Exact flat scan index.

use super::{dense_score, Candidate, IndexInput, VectorIndex};
use crate::doc::Doc;
use crate::error::Result;
use crate::query::SearchQuery;
use crate::types::{IndexType, MetricType};

#[derive(Debug, Clone)]
pub struct FlatIndex {
    pub field_name: String,
    pub dimension: usize,
    pub metric: MetricType,
    pub source_revision: u64,
}

impl FlatIndex {
    pub fn new(field_name: impl Into<String>, dimension: usize, metric: MetricType) -> Self {
        Self { field_name: field_name.into(), dimension, metric, source_revision: 0 }
    }
}

impl VectorIndex for FlatIndex {
    fn kind(&self) -> IndexType { IndexType::Flat }
    fn dimension(&self) -> usize { self.dimension }
    fn metric(&self) -> MetricType { self.metric }
    fn source_revision(&self) -> u64 { self.source_revision }
    fn build(&mut self, input: IndexInput<'_>) -> Result<()> { self.source_revision = input.revision; Ok(()) }
    fn search(&self, query: &SearchQuery, docs: &[Doc]) -> Result<Vec<Candidate>> {
        let Some(vector) = query.vector.as_deref() else { return Ok(Vec::new()) };
        let mut out: Vec<Candidate> = docs.iter().filter_map(|doc| {
            let id = doc.get_pk()?.to_string();
            let value = doc.vector(&self.field_name)?;
            let score = dense_score(vector, value, self.metric)?;
            Some(Candidate { id, score })
        }).collect();
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
        out.truncate(query.topk.max(0) as usize);
        Ok(out)
    }
}

