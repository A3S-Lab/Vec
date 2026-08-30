//! `a3s-vec` is A3S's native Rust in-process vector database.
//!
//! The crate provides the collection, document, schema, query, index, and
//! durability primitives needed by an embedded vector store.  It follows the
//! zvec Rust API vocabulary while keeping the implementation free of a C/C++
//! runtime dependency, which makes the same source usable on Intel macOS 12,
//! Apple Silicon, Linux, and Windows.
//!
//! A collection's document snapshot and WAL are authoritative.  ANN, scalar,
//! and full-text indexes are derived data and are rebuilt or safely bypassed
//! when their generation does not match the snapshot.

mod collection;
mod config;
mod doc;
mod embedding;
mod error;
mod iterator;
mod multi_query;
mod query;
mod schema;
mod stats;
mod storage;
mod types;

pub mod core {
    //! The pure-Rust algorithm kernel used by the high-level A3S API.
    //!
    //! This escape hatch is useful for advanced callers that need low-level
    //! centroid, graph, SIMD, or `DiskANN` format helpers.  High-level users
    //! should prefer the types re-exported at the crate root.
    pub use zvec_core::*;
}

pub use collection::{
    Collection, CollectionOptions, CollectionStats, DocWriteResult, IndexStat, WriteResult,
};
pub use config::{
    check_version, default_config, initialize, is_initialized, shutdown, version, version_major,
    version_minor, version_patch, ConfigBuilder, Durability, IoBackend,
};
pub use doc::{Doc, FieldValue, VectorValue};
pub use embedding::{DenseEmbedding, EmbeddingInput, QueryExecutor, SparseEmbedding};
pub use error::{Error, ErrorCode, Result};
pub use iterator::DocIterator;
pub use multi_query::{MultiQuery, RerankMethod, SubQuery};
pub use query::{
    DiskannQueryParams, FlatQueryParams, Fts, FtsQueryParams, GroupBySearchQuery, HnswQueryParams,
    IvfQueryParams, IvfRabitqQueryParams, SearchQuery, SearchQueryBuilder, VectorQuery,
};
pub use schema::{
    AddColumnOption, AlterColumnOption, CollectionSchema, CollectionSchemaBuilder,
    DiskANNIndexParam, DiskAnnIndexParam, FieldSchema, FlatIndexParam, FtsIndexParam,
    HnswIndexParam, IVFIndexParam, IndexParams, IndexParamsBuilder, InvertIndexParam,
    IvfIndexParam, IvfRabitqIndexParam, VamanaIndexParam, VectorSchema,
};
pub use stats::StatsSnapshot;
pub use types::{DataType, DocOperator, IndexType, LogLevel, LogType, MetricType, QuantizeType};

/// Convenient import for the common collection workflow.
pub mod prelude {
    pub use crate::{
        initialize, is_initialized, version, Collection, CollectionOptions, CollectionSchema,
        ConfigBuilder, DataType, Doc, DocIterator, Error, ErrorCode, FieldSchema, IndexParams,
        MetricType, MultiQuery, QuantizeType, Result, SearchQuery, VectorSchema, WriteResult,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
