//! Exact scalar index facade.  Filters still have a scan fallback.

use super::{Candidate, ScalarIndex};
use crate::doc::Doc;
use crate::error::Result;
use crate::query::SearchQuery;

#[derive(Debug, Clone)]
pub struct ExactScalarIndex {
    pub field_name: String,
    pub source_revision: u64,
    pub ids: Vec<String>,
}
impl ExactScalarIndex {
    pub fn new(field_name: impl Into<String>) -> Self { Self { field_name: field_name.into(), source_revision: 0, ids: Vec::new() } }
}
impl ScalarIndex for ExactScalarIndex {
    fn field_name(&self) -> &str { &self.field_name }
    fn build(&mut self, docs: &[Doc], revision: u64) -> Result<()> { self.ids = docs.iter().filter_map(|d| d.get_pk().map(str::to_string)).collect(); self.source_revision = revision; Ok(()) }
    fn candidates(&self, _query: &SearchQuery, _docs: &[Doc]) -> Result<Vec<String>> { Ok(self.ids.clone()) }
}

