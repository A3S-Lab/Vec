//! Targeted derived-index generation rebuilds.

use super::{build_vector_index, is_in_memory_ann, IndexRegistry};
use crate::doc::DocumentMap;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use crate::types::IndexType;

impl IndexRegistry {
    pub(crate) fn rebuild_field(
        &self,
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
        field_name: &str,
    ) -> Result<Self> {
        if !self.ordinals.validates(docs) {
            return Err(Error::internal(
                "cannot rebuild one index with a stale ordinal generation",
            ));
        }
        let mut next = self.clone();
        if let Some(field) = schema.vectors.iter().find(|field| field.name == field_name) {
            let params = field
                .index_params
                .as_ref()
                .ok_or_else(|| Error::not_found(format!("index '{field_name}' not found")))?;
            if is_in_memory_ann(params.index_type) {
                next.indexes.insert(
                    field_name.to_string(),
                    build_vector_index(
                        docs,
                        field_name,
                        field.dimension,
                        params,
                        source_revision,
                        &self.ordinals,
                    )?,
                );
                return Ok(next);
            }
            return Err(Error::not_supported(format!(
                "{:?} index does not have a rebuildable in-memory generation",
                params.index_type
            )));
        }
        let field = schema
            .fields
            .iter()
            .find(|field| field.name == field_name)
            .ok_or_else(|| Error::not_found(format!("index '{field_name}' not found")))?;
        let params = field
            .index_params
            .as_ref()
            .ok_or_else(|| Error::not_found(format!("index '{field_name}' not found")))?;
        match params.index_type {
            IndexType::Invert => {
                next.scalar_indexes = self.scalar_indexes.rebuild_field(
                    field,
                    docs,
                    source_revision,
                    &self.ordinals,
                )?;
            }
            IndexType::Fts => {
                next.fts_indexes =
                    self.fts_indexes
                        .rebuild_field(field, docs, source_revision, &self.ordinals)?;
            }
            index_type => {
                return Err(Error::not_supported(format!(
                    "{index_type:?} index does not have a rebuildable in-memory generation"
                )));
            }
        }
        Ok(next)
    }
}
