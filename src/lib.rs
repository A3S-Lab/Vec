//! `a3s-vec` is A3S's native Rust in-process vector database.
//!
//! The crate provides the collection, document, schema, query, index, and
//! durability primitives needed by an embedded vector store.  It follows the
//! zvec Rust API vocabulary while keeping the implementation free of a C/C++
//! runtime dependency, which makes the same source usable on Intel macOS 12,
//! Apple Silicon, Linux, and Windows.
//!
//! A collection's document snapshot and WAL are authoritative. Flat queries
//! execute against that snapshot; revision-tagged HNSW, IVF, L2 Vamana, and
//! product-quantized L2 `DiskANN` generations select bounded candidates before
//! exact re-ranking. Each Vamana/DiskANN base also has an A3S-native,
//! sector-aligned recovery sidecar. Native
//! numeric vector encodings are preserved at rest and remain distinct from index-only
//! FP16/INT8/INT4 scalar quantization. Revisioned scalar and full-text indexes
//! accelerate structured and lexical retrieval, including Unicode n-grams,
//! ordered token filters, boolean groups, required/prohibited clauses, and
//! wildcard/fuzzy/range terms, boosts, and ordered phrase proximity. HNSW and
//! IVF also provide multi-bit `RaBitQ` families with exact re-ranking. `DiskANN`
//! uses deterministic PQ training, ADC graph traversal, and on-demand
//! positioned reads; mmap/async acceleration remains explicit future work.
//!
//! The external algorithm kernel is an implementation detail and is not part
//! of the stable A3S API:
//!
//! ```compile_fail
//! use a3s_vec::core;
//! ```

mod collection;
mod config;
mod doc;
mod embedding;
mod error;
mod index;
mod iterator;
mod multi_query;
mod query;
mod schema;
mod stats;
mod storage;
mod text;
mod types;

pub use collection::{
    Collection, CollectionOptions, CollectionStats, DocWriteResult, IndexStat, WriteResult,
};
pub use config::{
    check_version, default_config, initialize, is_initialized, shutdown, version, version_major,
    version_minor, version_patch, ConfigBuilder, Durability,
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
pub use types::{DataType, DocOperator, IndexType, MetricType, QuantizeType};

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
