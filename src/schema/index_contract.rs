use super::IndexParams;
use crate::error::{Error, Result};
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
        IndexType::Flat => validate_flat_configuration(data_type, params),
        IndexType::Fts => validate_fts_configuration(params),
        IndexType::Undefined => Err(Error::invalid_argument("index type must be defined")),
        unsupported => Err(Error::not_supported(format!(
            "{unsupported:?} index execution is not implemented"
        ))),
    }
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
    if let Some(filters) = params.params.get("filters") {
        let filters = filters
            .as_array()
            .ok_or_else(|| Error::invalid_argument("FTS filters parameter must be an array"))?;
        if !filters.is_empty() {
            return Err(Error::not_supported(
                "FTS token filters have no execution consumer",
            ));
        }
    }
    if params.params.contains_key("extra_params") {
        return Err(Error::not_supported(
            "FTS extra_params has no execution consumer",
        ));
    }
    Ok(())
}
