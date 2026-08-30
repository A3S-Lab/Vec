//! Caller-owned embedding interfaces.
//!
//! The database never downloads a model or performs network I/O implicitly.
//! Applications can opt into these traits when they already own an embedding
//! provider, which keeps the core deterministic and easy to run on Intel
//! macOS 12.

use crate::error::Result;

/// Input accepted by an embedding provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingInput {
    Text(String),
    Document(String),
}

impl From<&str> for EmbeddingInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for EmbeddingInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Dense embedding provider.  Implementations should be pure with respect to
/// the collection; retries and network policy belong in the adapter.
pub trait DenseEmbedding: Send + Sync {
    fn embed(&self, input: &EmbeddingInput) -> Result<Vec<f32>>;
}

/// Sparse embedding provider.
pub trait SparseEmbedding: Send + Sync {
    fn embed_sparse(&self, input: &EmbeddingInput) -> Result<Vec<(u32, f32)>>;
}

/// Optional convenience for executing a query after embedding text.
pub trait QueryExecutor: Send + Sync {
    fn execute_text(
        &self,
        input: &EmbeddingInput,
        field_name: &str,
        topk: usize,
    ) -> Result<Vec<crate::Doc>>;
}
