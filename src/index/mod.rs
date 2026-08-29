//! Pluggable derived-index contracts.
//!
//! The collection currently keeps an exact reference path as the correctness
//! oracle.  These small contracts allow HNSW/IVF/DiskANN implementations to be
//! introduced or rebuilt without changing storage or query semantics.

use crate::doc::{Doc, VectorValue};
use crate::error::Result;
use crate::query::SearchQuery;
use crate::schema::IndexParams;
use crate::types::{IndexType, MetricType};
use serde::{Deserialize, Serialize};

pub mod diskann;
pub mod flat;
pub mod fts;
pub mod hnsw;
pub mod ivf;
pub mod pq;
pub mod quantize;
pub mod rabitq;
pub mod scalar;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexInput<'a> {
    pub docs: &'a [Doc],
    pub revision: u64,
}

#[derive(Debug, Default)]
pub struct IndexWriter {
    pub bytes: Vec<u8>,
}

pub trait VectorIndex: Send + Sync {
    fn kind(&self) -> IndexType;
    fn dimension(&self) -> usize;
    fn metric(&self) -> MetricType;
    fn source_revision(&self) -> u64;
    fn build(&mut self, input: IndexInput<'_>) -> Result<()>;
    fn search(&self, query: &SearchQuery, docs: &[Doc]) -> Result<Vec<Candidate>>;
    fn insert(&mut self, _doc: &Doc) -> Result<()> { Ok(()) }
    fn remove(&mut self, _id: &str) -> Result<()> { Ok(()) }
    fn save(&self, _writer: &mut IndexWriter) -> Result<()> { Ok(()) }
}

pub trait ScalarIndex: Send + Sync {
    fn field_name(&self) -> &str;
    fn build(&mut self, docs: &[Doc], revision: u64) -> Result<()>;
    fn candidates(&self, query: &SearchQuery, docs: &[Doc]) -> Result<Vec<String>>;
}

pub(crate) fn dense_score(query: &[f32], vector: &VectorValue, metric: MetricType) -> Option<f32> {
    let values = vector.to_dense_f32()?;
    if values.len() != query.len() { return None; }
    Some(match metric {
        MetricType::L2 => -query.iter().zip(values.iter()).map(|(a, b)| { let d = *a - *b; d * d }).sum::<f32>(),
        MetricType::Cosine => {
            let dot = query.iter().zip(values.iter()).map(|(a, b)| *a * *b).sum::<f32>();
            let qn = query.iter().map(|v| v * v).sum::<f32>().sqrt();
            let vn = values.iter().map(|v| v * v).sum::<f32>().sqrt();
            if qn == 0.0 || vn == 0.0 { 0.0 } else { dot / (qn * vn) }
        }
        MetricType::MipsL2 | MetricType::Ip | MetricType::Undefined => query.iter().zip(values.iter()).map(|(a, b)| *a * *b).sum(),
    })
}
