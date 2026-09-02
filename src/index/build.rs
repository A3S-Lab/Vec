//! Complete immutable ANN base construction.

use super::diskann_index::DiskannIndex;
use super::hnsw::HnswIndex;
use super::ivf::IvfIndex;
use super::ordinal_map::OrdinalMap;
use super::ordinals::OrdinalTable;
use super::quantization::QuantizedVector;
use super::rabitq_index::{HnswRabitqIndex, IvfRabitqIndex};
use super::{VectorIndex, VectorIndexBase, VectorIndexKind};
use crate::doc::{DocumentMap, VectorValue};
use crate::error::{Error, Result};
use crate::schema::IndexParams;
use crate::types::{IndexType, QuantizeType};
use roaring::RoaringTreemap;
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) fn build_vector_index(
    docs: &DocumentMap,
    field_name: &str,
    dimension: u32,
    params: &IndexParams,
    source_revision: u64,
    ordinals: &OrdinalTable,
) -> Result<VectorIndex> {
    let dimension = usize::try_from(dimension)
        .map_err(|_| Error::resource_exhausted("vector dimension exceeds this platform"))?;
    let vectors = collect_vectors(docs, field_name, params, ordinals)?;
    let vector_ordinals: RoaringTreemap = vectors.keys().collect();
    let kind = build_kind(&vectors, ordinals, dimension, params)?;
    Ok(VectorIndex {
        params: params.clone(),
        source_revision,
        base: Arc::new(VectorIndexBase {
            vectors,
            vector_ordinals,
            kind,
            diskann: None,
        }),
        delta: BTreeMap::new(),
        delta_ordinals: RoaringTreemap::new(),
        tombstones: RoaringTreemap::new(),
    })
}

fn build_kind(
    vectors: &OrdinalMap<QuantizedVector>,
    ordinals: &OrdinalTable,
    dimension: usize,
    params: &IndexParams,
) -> Result<VectorIndexKind> {
    match params.index_type {
        IndexType::Hnsw => Ok(VectorIndexKind::Hnsw(HnswIndex::build(
            vectors,
            ordinals,
            positive_parameter(params, "m")?,
            positive_parameter(params, "ef_construction")?,
            params.metric_type,
        ))),
        IndexType::HnswRabitq => Ok(VectorIndexKind::HnswRabitq(HnswRabitqIndex::build(
            vectors,
            ordinals,
            dimension,
            positive_parameter(params, "m")?,
            positive_parameter(params, "ef_construction")?,
            positive_parameter(params, "total_bits")?,
            positive_parameter(params, "num_clusters")?,
            nonnegative_parameter(params, "sample_count")?,
            params.metric_type,
        )?)),
        IndexType::Ivf => Ok(VectorIndexKind::Ivf(IvfIndex::build(
            vectors,
            positive_parameter(params, "n_list")?,
            nonnegative_parameter(params, "n_iters")?,
            boolean_parameter(params, "use_soar")?,
        ))),
        IndexType::IvfRabitq => Ok(VectorIndexKind::IvfRabitq(IvfRabitqIndex::build(
            vectors,
            dimension,
            positive_parameter(params, "n_list")?,
            positive_parameter(params, "total_bits")?,
            nonnegative_parameter(params, "sample_count")?,
            params.metric_type,
        )?)),
        IndexType::Diskann => Ok(VectorIndexKind::Diskann(DiskannIndex::build(
            vectors,
            ordinals,
            dimension,
            positive_parameter(params, "max_degree")?,
            positive_parameter(params, "list_size")?,
            nonnegative_parameter(params, "pq_chunk_num")?,
            finite_parameter(params, "alpha")?,
            params.metric_type,
        )?)),
        IndexType::Vamana => Ok(VectorIndexKind::Vamana(super::vamana::VamanaIndex::build(
            vectors,
            ordinals,
            positive_parameter(params, "max_degree")?,
            positive_parameter(params, "search_list_size")?,
            finite_parameter(params, "alpha")?,
            params.metric_type,
        ))),
        _ => Err(Error::not_supported(format!(
            "{:?} does not have an in-memory ANN implementation",
            params.index_type
        ))),
    }
}

fn collect_vectors(
    docs: &DocumentMap,
    field_name: &str,
    params: &IndexParams,
    ordinals: &OrdinalTable,
) -> Result<OrdinalMap<QuantizedVector>> {
    docs.iter()
        .filter_map(|(id, doc)| doc.vector(field_name).map(|vector| (id, vector)))
        .map(|(id, vector)| {
            let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                Error::internal(format!("vector ordinal is missing for document '{id}'"))
            })?;
            Ok((ordinal, encode_vector(id, field_name, params, vector)?))
        })
        .collect()
}

pub(super) fn encode_vector(
    id: &str,
    field_name: &str,
    params: &IndexParams,
    vector: &VectorValue,
) -> Result<QuantizedVector> {
    let dense = vector.to_dense_f32().ok_or_else(|| {
        Error::resource_exhausted(format!(
            "document '{id}' field '{field_name}' cannot be represented by the f32 ANN kernel"
        ))
    })?;
    let base_quantize = if params.quantize_type == QuantizeType::Rabitq {
        QuantizeType::Undefined
    } else {
        params.quantize_type
    };
    QuantizedVector::encode(dense, base_quantize).map_err(|error| {
        Error::new(
            error.code,
            format!(
                "build {:?} index for document '{id}' field '{field_name}': {}",
                params.index_type, error.message
            ),
        )
    })
}

fn positive_parameter(params: &IndexParams, name: &str) -> Result<usize> {
    let value = params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be positive"))
        })?;
    if value == 0 {
        return Err(Error::invalid_argument(format!(
            "index parameter '{name}' must be positive"
        )));
    }
    usize::try_from(value)
        .map_err(|_| Error::resource_exhausted(format!("index parameter '{name}' is too large")))
}

fn nonnegative_parameter(params: &IndexParams, name: &str) -> Result<usize> {
    let value = params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be non-negative"))
        })?;
    usize::try_from(value)
        .map_err(|_| Error::resource_exhausted(format!("index parameter '{name}' is too large")))
}

fn finite_parameter(params: &IndexParams, name: &str) -> Result<f64> {
    params
        .params
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be a finite number"))
        })
}

fn boolean_parameter(params: &IndexParams, name: &str) -> Result<bool> {
    params
        .params
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| Error::invalid_argument(format!("index parameter '{name}' must be boolean")))
}
