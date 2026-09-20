//! Caller-owned embedding interfaces.
//!
//! The database never downloads a model or performs network I/O implicitly.
//! Applications can opt into these traits when they already own an embedding
//! provider, which keeps the core deterministic and easy to run on the
//! supported Linux, Windows, and macOS platforms.

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

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::{DenseEmbedding, EmbeddingInput, QueryExecutor, SparseEmbedding};
    use crate::error::{Error, Result};
    use crate::Doc;

    struct FixedDense;

    impl DenseEmbedding for FixedDense {
        fn embed(&self, input: &EmbeddingInput) -> Result<Vec<f32>> {
            match input {
                EmbeddingInput::Text(text) | EmbeddingInput::Document(text) => {
                    Ok(vec![text.len() as f32, 1.0])
                }
            }
        }
    }

    struct FixedSparse;

    impl SparseEmbedding for FixedSparse {
        fn embed_sparse(&self, input: &EmbeddingInput) -> Result<Vec<(u32, f32)>> {
            match input {
                EmbeddingInput::Text(text) | EmbeddingInput::Document(text) => {
                    Ok(vec![(0, text.len() as f32)])
                }
            }
        }
    }

    struct EchoExecutor;

    impl QueryExecutor for EchoExecutor {
        fn execute_text(
            &self,
            input: &EmbeddingInput,
            field_name: &str,
            topk: usize,
        ) -> Result<Vec<Doc>> {
            if field_name.is_empty() || topk == 0 {
                return Err(Error::invalid_argument("field/topk"));
            }
            let _ = input;
            Ok(Vec::new())
        }
    }

    #[test]
    fn embedding_input_and_provider_traits_are_callable() {
        let from_str: EmbeddingInput = "hello".into();
        let from_string: EmbeddingInput = String::from("world").into();
        assert_eq!(from_str, EmbeddingInput::Text("hello".into()));
        assert_eq!(from_string, EmbeddingInput::Text("world".into()));
        assert_eq!(
            FixedDense
                .embed(&EmbeddingInput::Document("ab".into()))
                .unwrap(),
            vec![2.0, 1.0]
        );
        assert_eq!(
            FixedSparse
                .embed_sparse(&EmbeddingInput::Text("xyz".into()))
                .unwrap(),
            vec![(0, 3.0)]
        );
        assert!(EchoExecutor
            .execute_text(&EmbeddingInput::Text("q".into()), "embedding", 3)
            .unwrap()
            .is_empty());
        assert!(EchoExecutor
            .execute_text(&EmbeddingInput::Text("q".into()), "", 3)
            .is_err());
    }
}
