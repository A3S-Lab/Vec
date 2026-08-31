use super::IndexParams;
use crate::error::{Error, Result};
use crate::text::validate_tokenizer_params;
use crate::types::{DataType, IndexType, MetricType, QuantizeType};

pub(super) fn validate_index_configuration(
    field_name: &str,
    data_type: DataType,
    params: &IndexParams,
) -> Result<()> {
    if params.index_type.is_vector_index() != data_type.is_vector() {
        return Err(Error::invalid_argument(format!(
            "index type {:?} is incompatible with field '{field_name}'",
            params.index_type
        )));
    }
    if params.index_type == IndexType::Fts && data_type != DataType::String {
        return Err(Error::invalid_argument(
            "FTS configuration requires a string field",
        ));
    }

    match params.index_type {
        IndexType::Hnsw => validate_hnsw_configuration(data_type, params),
        IndexType::Ivf => validate_ivf_configuration(data_type, params),
        IndexType::Vamana => validate_vamana_configuration(data_type, params),
        IndexType::Flat => validate_flat_configuration(data_type, params),
        IndexType::Invert => validate_invert_configuration(data_type, params),
        IndexType::Fts => validate_fts_configuration(params),
        IndexType::Undefined => Err(Error::invalid_argument("index type must be defined")),
        unsupported => Err(Error::not_supported(format!(
            "{unsupported:?} index execution is not implemented"
        ))),
    }
}

fn validate_invert_configuration(data_type: DataType, params: &IndexParams) -> Result<()> {
    if !data_type.is_scalar() {
        return Err(Error::invalid_argument(
            "inverted indexes require a scalar field",
        ));
    }
    if params.metric_type != MetricType::Undefined {
        return Err(Error::invalid_argument(
            "inverted indexes cannot select a vector metric",
        ));
    }
    if params.quantize_type != QuantizeType::Undefined {
        return Err(Error::not_supported(
            "inverted-index quantization has no execution consumer",
        ));
    }
    validate_parameter_names(params, &["enable_range_optimization", "enable_wildcard"])?;
    let range = boolean_parameter(params, "enable_range_optimization")?;
    let wildcard = boolean_parameter(params, "enable_wildcard")?;
    if range && data_type == DataType::Bool {
        return Err(Error::not_supported(
            "boolean fields do not have range-index semantics",
        ));
    }
    if wildcard && data_type != DataType::String {
        return Err(Error::not_supported(
            "wildcard indexing requires a string field",
        ));
    }
    Ok(())
}

fn validate_hnsw_configuration(data_type: DataType, params: &IndexParams) -> Result<()> {
    validate_ann_base(data_type, params)?;
    validate_parameter_names(params, &["m", "ef_construction", "quantize_type"])?;
    positive_integer(params, "m")?;
    positive_integer(params, "ef_construction")?;
    validate_redundant_quantize_parameter(params)
}

fn validate_ivf_configuration(data_type: DataType, params: &IndexParams) -> Result<()> {
    validate_ann_base(data_type, params)?;
    validate_parameter_names(params, &["n_list", "n_iters", "use_soar", "quantize_type"])?;
    positive_integer(params, "n_list")?;
    nonnegative_integer(params, "n_iters")?;
    if params
        .params
        .get("use_soar")
        .and_then(serde_json::Value::as_bool)
        .is_none()
    {
        return Err(Error::invalid_argument(
            "IVF use_soar parameter must be boolean",
        ));
    }
    if params.params["use_soar"] == serde_json::Value::Bool(true) {
        return Err(Error::not_supported(
            "IVF SOAR assignment is not implemented",
        ));
    }
    validate_redundant_quantize_parameter(params)
}

fn validate_vamana_configuration(data_type: DataType, params: &IndexParams) -> Result<()> {
    validate_ann_base(data_type, params)?;
    if params.metric_type != MetricType::L2 {
        return Err(Error::not_supported(
            "Vamana currently requires the L2 metric",
        ));
    }
    if params.quantize_type != QuantizeType::Undefined {
        return Err(Error::not_supported(
            "Vamana quantization is not implemented",
        ));
    }
    validate_parameter_names(
        params,
        &[
            "max_degree",
            "search_list_size",
            "alpha",
            "max_occlusion",
            "saturate",
        ],
    )?;
    positive_integer(params, "max_degree")?;
    positive_integer(params, "search_list_size")?;
    let alpha = finite_number(params, "alpha")?;
    if alpha < 1.0 {
        return Err(Error::invalid_argument(
            "Vamana alpha must be finite and at least 1.0",
        ));
    }
    if nonnegative_integer(params, "max_occlusion")? != 0 {
        return Err(Error::not_supported(
            "Vamana max_occlusion is not implemented",
        ));
    }
    if boolean_parameter(params, "saturate")? {
        return Err(Error::not_supported(
            "Vamana graph saturation is not implemented",
        ));
    }
    Ok(())
}

fn validate_ann_base(data_type: DataType, params: &IndexParams) -> Result<()> {
    if !matches!(
        data_type,
        DataType::VectorFp16
            | DataType::VectorFp32
            | DataType::VectorFp64
            | DataType::VectorInt4
            | DataType::VectorInt8
            | DataType::VectorInt16
    ) {
        return Err(Error::not_supported(
            "ANN indexes require a numeric dense vector field",
        ));
    }
    if params.metric_type == MetricType::Undefined {
        return Err(Error::invalid_argument("ANN index requires a metric"));
    }
    if !matches!(
        params.quantize_type,
        QuantizeType::Undefined | QuantizeType::Fp16 | QuantizeType::Int8 | QuantizeType::Int4
    ) {
        return Err(Error::not_supported(format!(
            "{:?} ANN quantization is not implemented",
            params.quantize_type
        )));
    }
    Ok(())
}

fn validate_parameter_names(params: &IndexParams, allowed: &[&str]) -> Result<()> {
    if let Some(name) = params
        .params
        .keys()
        .find(|name| !allowed.contains(&name.as_str()))
    {
        return Err(Error::invalid_argument(format!(
            "unknown {:?} index parameter '{name}'",
            params.index_type
        )));
    }
    Ok(())
}

fn positive_integer(params: &IndexParams, name: &str) -> Result<u64> {
    let value = params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be positive"))
        })?;
    if value == 0 {
        Err(Error::invalid_argument(format!(
            "index parameter '{name}' must be positive"
        )))
    } else {
        Ok(value)
    }
}

fn nonnegative_integer(params: &IndexParams, name: &str) -> Result<u64> {
    params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be non-negative"))
        })
}

fn boolean_parameter(params: &IndexParams, name: &str) -> Result<bool> {
    params
        .params
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| Error::invalid_argument(format!("index parameter '{name}' must be boolean")))
}

fn finite_number(params: &IndexParams, name: &str) -> Result<f64> {
    params
        .params
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            Error::invalid_argument(format!("index parameter '{name}' must be a finite number"))
        })
}

fn validate_redundant_quantize_parameter(params: &IndexParams) -> Result<()> {
    let Some(value) = params.params.get("quantize_type") else {
        return Ok(());
    };
    let encoded = serde_json::to_value(params.quantize_type)
        .map_err(|error| Error::internal(format!("serialize quantization type: {error}")))?;
    if *value != encoded {
        return Err(Error::invalid_argument(
            "index quantize_type fields disagree",
        ));
    }
    Ok(())
}

fn validate_flat_configuration(data_type: DataType, params: &IndexParams) -> Result<()> {
    if matches!(
        data_type,
        DataType::VectorBinary32 | DataType::VectorBinary64
    ) {
        return Err(Error::not_supported(
            "Flat binary-vector execution is not implemented",
        ));
    }
    if params.metric_type == MetricType::Undefined {
        return Err(Error::invalid_argument(
            "Flat vector configuration requires a metric",
        ));
    }
    if params.quantize_type != QuantizeType::Undefined {
        return Err(Error::not_supported(
            "Flat quantization has no execution consumer",
        ));
    }
    if let Some(name) = params.params.keys().next() {
        return Err(Error::not_supported(format!(
            "Flat index parameter '{name}' has no execution consumer"
        )));
    }
    Ok(())
}

fn validate_fts_configuration(params: &IndexParams) -> Result<()> {
    if params.metric_type != MetricType::Undefined {
        return Err(Error::invalid_argument(
            "FTS configuration cannot select a vector metric",
        ));
    }
    if params.quantize_type != QuantizeType::Undefined {
        return Err(Error::not_supported(
            "FTS quantization has no execution consumer",
        ));
    }
    for name in params.params.keys() {
        match name.as_str() {
            "tokenizer_name" | "filters" | "extra_params" => {}
            unknown => {
                return Err(Error::invalid_argument(format!(
                    "unknown FTS configuration parameter '{unknown}'"
                )))
            }
        }
    }
    if let Some(tokenizer) = params.params.get("tokenizer_name") {
        let tokenizer = tokenizer.as_str().ok_or_else(|| {
            Error::invalid_argument("FTS tokenizer_name parameter must be a string")
        })?;
        if tokenizer.trim().is_empty() {
            return Err(Error::invalid_argument(
                "FTS tokenizer_name parameter must not be empty",
            ));
        }
    }
    validate_tokenizer_params(Some(params))
}
