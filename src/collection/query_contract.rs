//! Schema-derived query validation shared by every execution path.

use crate::error::{Error, Result};
use crate::query::SearchQuery;
use crate::schema::{CollectionSchema, IndexParams};
use crate::types::{DataType, MetricType};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy)]
pub(super) struct QueryField<'a> {
    data_type: DataType,
    dimension: u32,
    pub(super) index_params: Option<&'a IndexParams>,
}

pub(super) fn schema_index_params<'a>(
    schema: &'a CollectionSchema,
    field_name: &str,
) -> Option<&'a IndexParams> {
    schema
        .fields
        .iter()
        .find(|field| field.name == field_name)
        .and_then(|field| field.index_params.as_ref())
        .or_else(|| {
            schema
                .vectors
                .iter()
                .find(|field| field.name == field_name)
                .and_then(|field| field.index_params.as_ref())
        })
}

pub(super) fn validate_query_contract<'a>(
    schema: &'a CollectionSchema,
    query: &SearchQuery,
) -> Result<QueryField<'a>> {
    let field = resolve_query_field(schema, &query.field_name)?;
    if query.topk <= 0 {
        return Err(Error::invalid_argument("query topk must be positive"));
    }
    let route_count = usize::from(query.fts.is_some())
        + usize::from(query.vector.is_some())
        + usize::from(query.sparse_vector.is_some())
        + usize::from(query.id.is_some());
    if route_count != 1 {
        return Err(Error::invalid_argument(
            "query must select exactly one FTS, dense, sparse, or source-id route",
        ));
    }

    if query.fts.is_some() {
        if field.data_type != DataType::String {
            return Err(Error::invalid_argument(format!(
                "FTS query requires a String field, got {}",
                field.data_type
            )));
        }
        return Ok(field);
    }
    if !field.data_type.is_vector() {
        return Err(Error::invalid_argument(format!(
            "vector query requires a vector field, got {}",
            field.data_type
        )));
    }
    if matches!(
        field.data_type,
        DataType::VectorBinary32 | DataType::VectorBinary64
    ) {
        return Err(Error::not_supported(
            "binary vector queries are not implemented by the exact oracle",
        ));
    }

    validate_dense_payload(&field, query)?;
    validate_sparse_payload(&field, query)?;
    if query.id.is_some() && field.data_type.is_sparse_vector() {
        return Err(Error::not_supported(
            "source-id queries for sparse vector fields are not implemented",
        ));
    }
    if let Some(radius) = query.params.get("radius") {
        let radius = radius
            .as_f64()
            .ok_or_else(|| Error::invalid_argument("query radius must be numeric"))?;
        if !radius.is_finite() {
            return Err(Error::invalid_argument("query radius must be finite"));
        }
    }
    Ok(field)
}

pub(super) fn query_metric(field: &QueryField<'_>, query: &SearchQuery) -> Result<MetricType> {
    match query.params.get("metric") {
        Some(value) => {
            parse_metric(value.as_str().ok_or_else(|| {
                Error::invalid_argument("query metric parameter must be a string")
            })?)
        }
        None => Ok(field
            .index_params
            .map(|params| params.metric_type)
            .filter(|metric| *metric != MetricType::Undefined)
            .unwrap_or(MetricType::Cosine)),
    }
}

pub(super) fn validate_tokenizer(tokenizer: &str) -> Result<()> {
    match tokenizer {
        "standard" | "whitespace" => Ok(()),
        "jieba" | "jieba_accurate" => {
            #[cfg(feature = "jieba")]
            {
                Ok(())
            }
            #[cfg(not(feature = "jieba"))]
            {
                Err(Error::not_supported(
                    "jieba tokenizer requires the optional 'jieba' feature",
                ))
            }
        }
        _ => Err(Error::invalid_argument(format!(
            "unknown FTS tokenizer '{tokenizer}'"
        ))),
    }
}

fn resolve_query_field<'a>(
    schema: &'a CollectionSchema,
    field_name: &str,
) -> Result<QueryField<'a>> {
    schema
        .fields
        .iter()
        .find(|field| field.name == field_name)
        .map(|field| QueryField {
            data_type: field.data_type,
            dimension: field.dimension,
            index_params: field.index_params.as_ref(),
        })
        .or_else(|| {
            schema
                .vectors
                .iter()
                .find(|field| field.name == field_name)
                .map(|field| QueryField {
                    data_type: field.data_type,
                    dimension: field.dimension,
                    index_params: field.index_params.as_ref(),
                })
        })
        .ok_or_else(|| Error::not_found(format!("query field '{field_name}' not found")))
}

fn validate_dense_payload(field: &QueryField<'_>, query: &SearchQuery) -> Result<()> {
    let Some(vector) = query.vector.as_ref() else {
        return Ok(());
    };
    if !field.data_type.is_dense_vector() {
        return Err(Error::invalid_argument(
            "dense query payload requires a dense vector field",
        ));
    }
    if vector.is_empty() || !vector.iter().all(|value| value.is_finite()) {
        return Err(Error::invalid_argument(
            "dense query vector must be non-empty and finite",
        ));
    }
    let expected = usize::try_from(field.dimension)
        .map_err(|_| Error::resource_exhausted("vector dimension exceeds this platform"))?;
    if vector.len() != expected {
        return Err(Error::invalid_argument(format!(
            "query vector dimension mismatch: expected {expected}, got {}",
            vector.len()
        )));
    }
    Ok(())
}

fn validate_sparse_payload(field: &QueryField<'_>, query: &SearchQuery) -> Result<()> {
    let Some(values) = query.sparse_vector.as_ref() else {
        return Ok(());
    };
    if !field.data_type.is_sparse_vector() {
        return Err(Error::invalid_argument(
            "sparse query payload requires a sparse vector field",
        ));
    }
    if values.is_empty() || !values.iter().all(|(_, value)| value.is_finite()) {
        return Err(Error::invalid_argument(
            "sparse query vector must be non-empty and finite",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, _) in values {
        if !seen.insert(*index) {
            return Err(Error::invalid_argument(
                "sparse query vector contains a duplicate index",
            ));
        }
        if field.dimension > 0 && *index >= field.dimension {
            return Err(Error::invalid_argument(format!(
                "sparse query index {index} exceeds dimension {}",
                field.dimension
            )));
        }
    }
    Ok(())
}

fn parse_metric(value: &str) -> Result<MetricType> {
    match value.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(MetricType::L2),
        "ip" | "inner_product" | "dot" => Ok(MetricType::Ip),
        "cosine" => Ok(MetricType::Cosine),
        "mips_l2" | "mips-l2" => Ok(MetricType::MipsL2),
        _ => Err(Error::invalid_argument(format!("unknown metric '{value}'"))),
    }
}
