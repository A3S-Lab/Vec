//! Runtime in-memory index lifecycle and generation publication.

use super::{
    commit_prepared_schema_change, ensure_same_generation, ensure_writable, persist_index_cache,
    prepare_schema_change, Collection,
};
use crate::error::{Error, Result};
use crate::index::IndexRegistry;
use crate::schema::{CollectionSchema, IndexParams};
use crate::types::IndexType;

impl Collection {
    pub fn create_index(&self, field_name: &str, params: &IndexParams) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;
        current
            .schema
            .check_index_configuration(field_name, params)?;
        if !matches!(
            params.index_type,
            IndexType::Flat
                | IndexType::Hnsw
                | IndexType::Ivf
                | IndexType::Diskann
                | IndexType::Vamana
                | IndexType::Invert
                | IndexType::Fts
        ) {
            return Err(Error::not_supported(format!(
                "{:?} physical index creation is not implemented",
                params.index_type
            )));
        }
        let mut next = current.clone();
        next.schema.add_index(field_name, params)?;
        let next = prepare_schema_change(next)?;
        let config = current.config.clone();
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, &current)?;
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_prepared_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn drop_index(&self, field_name: &str) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let current = self
            .inner
            .state
            .read()
            .map_err(|_| Error::internal("collection state lock poisoned"))?
            .clone();
        ensure_writable(&current.options)?;
        let mut next = current.clone();
        next.schema.drop_index(field_name)?;
        let next = prepare_schema_change(next)?;
        let config = current.config.clone();
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        ensure_same_generation(&state, &current)?;
        let mut storage = self
            .inner
            .storage
            .lock()
            .map_err(|_| Error::internal("storage lock poisoned"))?;
        commit_prepared_schema_change(&mut storage, &mut state, next, &config)
    }

    pub fn optimize(&self) -> Result<()> {
        self.rebuild_index_generation(None)
    }

    /// Rebuilds one configured in-memory index generation.
    pub fn rebuild_index(&self, field_name: &str) -> Result<()> {
        self.rebuild_index_generation(Some(field_name))
    }

    fn rebuild_index_generation(&self, field_name: Option<&str>) -> Result<()> {
        self.ensure_open()?;
        let _writer = self
            .inner
            .writer
            .lock()
            .map_err(|_| Error::internal("writer lock poisoned"))?;
        let snapshot = {
            let state = self
                .inner
                .state
                .read()
                .map_err(|_| Error::internal("collection state lock poisoned"))?;
            ensure_writable(&state.options)?;
            (
                state.schema.clone(),
                state.docs.clone(),
                state.revision,
                state.indexes.clone(),
            )
        };

        // The old immutable generation remains visible to readers throughout
        // construction. Publication is one state-lock assignment below.
        let indexes = if let Some(field_name) = field_name {
            snapshot
                .3
                .rebuild_field(&snapshot.0, &snapshot.1, snapshot.2, field_name)?
        } else {
            IndexRegistry::build(&snapshot.0, &snapshot.1, snapshot.2)?
        };
        let refresh_cache = should_rewrite_index_cache(&snapshot.0, field_name);
        let mut state = self
            .inner
            .state
            .write()
            .map_err(|_| Error::internal("collection state lock poisoned"))?;
        if state.revision != snapshot.2 || state.schema != snapshot.0 {
            return Err(Error::failed_precondition(
                "collection changed while indexes were rebuilding",
            ));
        }
        let indexes = std::sync::Arc::new(indexes);
        state.indexes = std::sync::Arc::clone(&indexes);
        let schema = state.schema.clone();
        let revision = state.revision;
        drop(state);
        if refresh_cache {
            let storage = self
                .inner
                .storage
                .lock()
                .map_err(|_| Error::internal("storage lock poisoned"))?;
            persist_index_cache(&storage, &schema, &indexes, revision, true);
        }
        Ok(())
    }
}

// Exact scalar/FTS rebuilds preserve the cache's logical generation, whereas
// an ANN rebuild can replace graph/centroid structure and bounded overlays.
fn should_rewrite_index_cache(schema: &CollectionSchema, field_name: Option<&str>) -> bool {
    field_name.is_none()
        || field_name.is_some_and(|field_name| {
            schema.vectors.iter().any(|field| {
                field.name == field_name
                    && field.index_params.as_ref().is_some_and(|params| {
                        matches!(
                            params.index_type,
                            IndexType::Hnsw
                                | IndexType::Ivf
                                | IndexType::Diskann
                                | IndexType::Vamana
                        )
                    })
            })
        })
}

#[cfg(test)]
mod tests {
    use super::should_rewrite_index_cache;
    use crate::{CollectionSchema, DataType, FieldSchema, IndexParams, MetricType};

    #[test]
    fn only_ann_and_full_rebuilds_rewrite_the_derived_cache() {
        let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
            .expect("field must be valid");
        embedding
            .set_index_params(
                &IndexParams::hnsw(MetricType::L2, 4, 16).expect("HNSW params must be valid"),
            )
            .expect("HNSW index must be valid");
        let mut language =
            FieldSchema::new("language", DataType::String, false, 0).expect("field must be valid");
        language
            .set_index_params(
                &IndexParams::invert(false, false).expect("scalar params must be valid"),
            )
            .expect("scalar index must be valid");
        let mut body =
            FieldSchema::new("body", DataType::String, false, 0).expect("field must be valid");
        body.set_index_params(
            &IndexParams::fts(Some("standard"), None, None).expect("FTS params must be valid"),
        )
        .expect("FTS index must be valid");
        let schema = CollectionSchema::builder("cache-refresh")
            .add_field(embedding)
            .add_field(language)
            .add_field(body)
            .build()
            .expect("schema must be valid");

        assert!(should_rewrite_index_cache(&schema, None));
        assert!(should_rewrite_index_cache(&schema, Some("embedding")));
        assert!(!should_rewrite_index_cache(&schema, Some("language")));
        assert!(!should_rewrite_index_cache(&schema, Some("body")));
    }
}
