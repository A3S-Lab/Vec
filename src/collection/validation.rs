//! Write-boundary document validation and derived index metadata.

use super::{write_error, write_success, CollectionState, DocWriteResult, Mutation, RuntimeIndex};
use crate::doc::{Doc, FieldValue};
use crate::error::{Error, ErrorCode, Result};
use crate::schema::CollectionSchema;
use crate::types::DataType;
use std::collections::{BTreeMap, HashSet};

pub(super) fn prepare_mutation_batch(
    state: &CollectionState,
    docs: &[&Doc],
    mutation: Mutation,
) -> (Vec<Doc>, Vec<DocWriteResult>) {
    let mut outcomes = Vec::with_capacity(docs.len());
    let mut accepted = Vec::new();
    let mut batch_ids = HashSet::new();
    for doc in docs {
        let validation = match mutation {
            Mutation::Insert => validate_doc(&state.schema, doc, true),
            Mutation::Update | Mutation::Upsert => validate_doc(&state.schema, doc, false),
        };
        if let Err(error) = validation {
            outcomes.push(write_error(error.code, error.message));
            continue;
        }
        let Some(pk) = doc.get_pk() else {
            outcomes.push(write_error(
                ErrorCode::InvalidArgument,
                "primary key is required",
            ));
            continue;
        };
        if !batch_ids.insert(pk.to_string()) {
            outcomes.push(write_error(
                ErrorCode::AlreadyExists,
                format!("duplicate primary key '{pk}' in batch"),
            ));
            continue;
        }
        let exists = state.docs.contains_key(pk);
        let allowed = match mutation {
            Mutation::Insert => !exists,
            Mutation::Update => exists,
            Mutation::Upsert => true,
        };
        if !allowed {
            outcomes.push(write_error(
                if exists {
                    ErrorCode::AlreadyExists
                } else {
                    ErrorCode::NotFound
                },
                if exists {
                    format!("document '{pk}' already exists")
                } else {
                    format!("document '{pk}' not found")
                },
            ));
            continue;
        }
        accepted.push((*doc).clone());
        outcomes.push(write_success());
    }
    (accepted, outcomes)
}

pub(super) fn runtime_indexes(
    schema: &CollectionSchema,
    revision: u64,
    document_count: u64,
) -> BTreeMap<String, RuntimeIndex> {
    schema
        .fields
        .iter()
        .filter_map(|field| {
            field
                .index_params
                .clone()
                .map(|params| (field.name.clone(), params))
        })
        .chain(schema.vectors.iter().filter_map(|field| {
            field
                .index_params
                .clone()
                .map(|params| (field.name.clone(), params))
        }))
        .map(|(name, params)| {
            (
                name,
                RuntimeIndex {
                    params,
                    source_revision: revision,
                    document_count,
                },
            )
        })
        .collect()
}

pub(super) fn validate_doc(schema: &CollectionSchema, doc: &Doc, require_all: bool) -> Result<()> {
    let pk = doc
        .get_pk()
        .ok_or_else(|| Error::invalid_argument("document primary key is required"))?;
    if pk.is_empty() || pk.contains('\0') {
        return Err(Error::invalid_argument(
            "document primary key must be non-empty and contain no NUL byte",
        ));
    }
    for name in doc.fields().keys() {
        if !schema.fields.iter().any(|field| field.name == *name) {
            return Err(Error::invalid_argument(format!(
                "unknown scalar field '{name}'"
            )));
        }
    }
    for name in doc.vectors().keys() {
        if !schema.vectors.iter().any(|field| field.name == *name) {
            return Err(Error::invalid_argument(format!(
                "unknown vector field '{name}'"
            )));
        }
    }
    for field in &schema.fields {
        match doc.field(&field.name) {
            None if require_all && !field.nullable => {
                return Err(Error::invalid_argument(format!(
                    "required field '{}' is missing",
                    field.name
                )));
            }
            Some(FieldValue::Null) if !field.nullable => {
                return Err(Error::invalid_argument(format!(
                    "field '{}' is not nullable",
                    field.name
                )));
            }
            Some(value) if !matches_field_type(value, field.data_type) => {
                return Err(Error::invalid_argument(format!(
                    "field '{}' has a value incompatible with {}",
                    field.name, field.data_type
                )));
            }
            _ => {}
        }
    }
    for field in &schema.vectors {
        if let Some(value) = doc.vector(&field.name) {
            if value.data_type() != field.data_type {
                let numeric_dense =
                    field.data_type.is_dense_vector() && value.data_type().is_dense_vector();
                if !numeric_dense {
                    return Err(Error::invalid_argument(format!(
                        "vector '{}' has a value incompatible with {}",
                        field.name, field.data_type
                    )));
                }
            }
            if field.dimension > 0
                && !value.is_sparse()
                && value.dimension() != field.dimension as usize
            {
                return Err(Error::invalid_argument(format!(
                    "vector '{}' dimension mismatch: expected {}, got {}",
                    field.name,
                    field.dimension,
                    value.dimension()
                )));
            }
            if value.is_sparse() {
                let Some(sparse) = value.to_sparse_f64() else {
                    return Err(Error::invalid_argument(format!(
                        "invalid sparse vector '{}'",
                        field.name
                    )));
                };
                if sparse
                    .keys()
                    .any(|index| *index >= field.dimension && field.dimension > 0)
                {
                    return Err(Error::invalid_argument(format!(
                        "sparse vector '{}' contains an out-of-range index",
                        field.name
                    )));
                }
            }
        } else if require_all && !field.data_type.is_sparse_vector() {
            return Err(Error::invalid_argument(format!(
                "required vector field '{}' is missing",
                field.name
            )));
        }
    }
    Ok(())
}

fn matches_field_type(value: &FieldValue, data_type: DataType) -> bool {
    matches!(value, FieldValue::Null | FieldValue::Json(_)) || value.data_type() == data_type
}

pub(super) fn merge_patch(target: &mut Doc, patch: &Doc) -> Result<()> {
    for (name, value) in patch.fields() {
        target.set_field_value(name, value.clone())?;
    }
    for (name, value) in patch.vectors() {
        target.set_vector_value(name, value.clone())?;
    }
    if patch.get_score() != 0.0 {
        target.set_score(patch.get_score())?;
    }
    Ok(())
}
