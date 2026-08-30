//! Write-boundary document normalization and validation.

use super::{write_error, write_success, CollectionState, DocWriteResult, Mutation};
use crate::doc::{Doc, FieldValue};
use crate::error::{Error, ErrorCode, Result};
use crate::schema::CollectionSchema;
use crate::types::DataType;
use serde_json::Value;
use std::collections::HashSet;

pub(super) fn prepare_mutation_batch(
    state: &CollectionState,
    docs: &[&Doc],
    mutation: Mutation,
) -> (Vec<Doc>, Vec<DocWriteResult>) {
    let mut outcomes = Vec::with_capacity(docs.len());
    let mut accepted = Vec::new();
    let mut batch_ids = HashSet::new();
    for doc in docs {
        let normalized = match normalize_doc(&state.schema, doc) {
            Ok(normalized) => normalized,
            Err(error) => {
                outcomes.push(write_error(error.code, error.message));
                continue;
            }
        };
        let validation = match mutation {
            Mutation::Insert | Mutation::Upsert => validate_doc(&state.schema, &normalized, true),
            Mutation::Update => validate_doc(&state.schema, &normalized, false),
        };
        if let Err(error) = validation {
            outcomes.push(write_error(error.code, error.message));
            continue;
        }
        let Some(pk) = normalized.get_pk() else {
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
        accepted.push(normalized);
        outcomes.push(write_success());
    }
    (accepted, outcomes)
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
            Some(value) if is_null_value(value) && !field.nullable => {
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
    matches!(value, FieldValue::Null) || value.data_type() == data_type
}

fn is_null_value(value: &FieldValue) -> bool {
    matches!(value, FieldValue::Null | FieldValue::Json(Value::Null))
}

pub(super) fn normalize_doc(schema: &CollectionSchema, doc: &Doc) -> Result<Doc> {
    let mut normalized = doc.clone();
    for field in &schema.fields {
        let Some(FieldValue::Json(value)) = normalized.field(&field.name).cloned() else {
            continue;
        };
        let value = coerce_json_field(&value, field.data_type).map_err(|error| {
            Error::new(
                error.code,
                format!("field '{}': {}", field.name, error.message),
            )
        })?;
        normalized.set_field_value(&field.name, value)?;
    }
    Ok(normalized)
}

pub(super) fn coerce_json_field(value: &Value, data_type: DataType) -> Result<FieldValue> {
    let incompatible = || {
        Error::invalid_argument(format!(
            "JSON adapter value is incompatible with {data_type}"
        ))
    };
    if value.is_null() {
        return Ok(FieldValue::Null);
    }
    match data_type {
        DataType::String => value
            .as_str()
            .map(|value| FieldValue::String(value.to_string()))
            .ok_or_else(incompatible),
        DataType::Bool => value
            .as_bool()
            .map(FieldValue::Bool)
            .ok_or_else(incompatible),
        DataType::Int32 => value
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .map(FieldValue::Int32)
            .ok_or_else(incompatible),
        DataType::Int64 => value
            .as_i64()
            .map(FieldValue::Int64)
            .ok_or_else(incompatible),
        DataType::Uint32 => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(FieldValue::Uint32)
            .ok_or_else(incompatible),
        DataType::Uint64 => value
            .as_u64()
            .map(FieldValue::Uint64)
            .ok_or_else(incompatible),
        DataType::Float => value
            .as_f64()
            .and_then(json_f64_to_f32)
            .map(FieldValue::Float)
            .ok_or_else(incompatible),
        DataType::Double => value
            .as_f64()
            .filter(|value| value.is_finite())
            .map(FieldValue::Double)
            .ok_or_else(incompatible),
        DataType::ArrayString => json_array(value, |value| value.as_str().map(str::to_string))
            .map(FieldValue::ArrayString)
            .ok_or_else(incompatible),
        DataType::ArrayBool => json_array(value, Value::as_bool)
            .map(FieldValue::ArrayBool)
            .ok_or_else(incompatible),
        DataType::ArrayInt32 => json_array(value, |value| {
            value.as_i64().and_then(|value| i32::try_from(value).ok())
        })
        .map(FieldValue::ArrayInt32)
        .ok_or_else(incompatible),
        DataType::ArrayInt64 => json_array(value, Value::as_i64)
            .map(FieldValue::ArrayInt64)
            .ok_or_else(incompatible),
        DataType::ArrayUint32 => json_array(value, |value| {
            value.as_u64().and_then(|value| u32::try_from(value).ok())
        })
        .map(FieldValue::ArrayUint32)
        .ok_or_else(incompatible),
        DataType::ArrayUint64 => json_array(value, Value::as_u64)
            .map(FieldValue::ArrayUint64)
            .ok_or_else(incompatible),
        DataType::ArrayFloat => json_array(value, |value| value.as_f64().and_then(json_f64_to_f32))
            .map(FieldValue::ArrayFloat)
            .ok_or_else(incompatible),
        DataType::ArrayDouble => json_array(value, |value| {
            value.as_f64().filter(|value| value.is_finite())
        })
        .map(FieldValue::ArrayDouble)
        .ok_or_else(incompatible),
        DataType::Binary | DataType::ArrayBinary => Err(Error::invalid_argument(
            "JSON adapter values cannot represent typed binary fields",
        )),
        _ => Err(incompatible()),
    }
}

pub(super) fn parse_default_expression(
    expression: &str,
    data_type: DataType,
) -> Result<FieldValue> {
    let trimmed = expression.trim();
    let value = if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        Value::String(trimmed[1..trimmed.len() - 1].to_string())
    } else {
        serde_json::from_str::<Value>(trimmed)
            .unwrap_or_else(|_| Value::String(trimmed.trim_matches('"').to_string()))
    };
    coerce_json_field(&value, data_type)
}

fn json_array<T>(value: &Value, convert: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
    value
        .as_array()?
        .iter()
        .map(convert)
        .collect::<Option<Vec<_>>>()
}

#[allow(clippy::cast_possible_truncation)]
fn json_f64_to_f32(value: f64) -> Option<f32> {
    (value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX))
        .then_some(value as f32)
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
