//! Collection and index schemas.

mod index_contract;

use crate::error::{Error, Result};
use crate::types::{DataType, IndexType, MetricType, QuantizeType};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

use index_contract::validate_index_configuration;

/// HNSW build parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswIndexParam {
    pub metric: MetricType,
    pub m: u32,
    pub ef_construction: u32,
    pub quantize: QuantizeType,
}

/// IVF build parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IVFIndexParam {
    pub metric: MetricType,
    pub n_list: u32,
    pub n_iters: u32,
    pub use_soar: bool,
}

pub type IvfIndexParam = IVFIndexParam;

/// IVF `RaBitQ` build parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IvfRabitqIndexParam {
    pub metric: MetricType,
    pub n_list: u32,
    pub total_bits: u32,
    pub sample_count: u32,
}

/// Exact flat index parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatIndexParam {
    pub metric: MetricType,
}

/// DiskANN/Vamana build parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskAnnIndexParam {
    pub metric: MetricType,
    pub max_degree: u32,
    pub list_size: u32,
    pub pq_chunk_num: u32,
    pub alpha: f64,
}

pub type DiskANNIndexParam = DiskAnnIndexParam;

/// Vamana graph parameters.  `DiskANN` uses the same graph with a disk layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VamanaIndexParam {
    pub metric: MetricType,
    pub max_degree: u32,
    pub search_list_size: u32,
    pub alpha: f64,
    pub max_occlusion: u32,
    pub saturate: bool,
}

/// Scalar inverted-index parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvertIndexParam {
    pub enable_range_optimization: bool,
    pub enable_wildcard: bool,
}

/// Full-text index parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FtsIndexParam {
    pub tokenizer_name: String,
    pub filters: Vec<String>,
    pub extra_params: Option<String>,
}

/// Serializable index configuration.
///
/// The open `params` map is retained for adapter compatibility, but collection
/// schemas reject entries that do not have an execution consumer. Live Flat,
/// HNSW, IVF, HNSW/IVF `RaBitQ`, Vamana, DiskANN/PQ, scalar-inverted, and
/// scan-FTS configurations are validated at the field boundary; attaching a
/// future descriptor returns [`crate::ErrorCode::NotSupported`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexParams {
    pub index_type: IndexType,
    pub metric_type: MetricType,
    pub quantize_type: QuantizeType,
    #[serde(default)]
    pub params: Map<String, Value>,
}

impl IndexParams {
    pub fn hnsw(metric: MetricType, m: i32, ef_construction: i32) -> Result<Self> {
        Self::hnsw_with_quantize(metric, m, ef_construction, QuantizeType::Undefined)
    }

    pub fn hnsw_with_quantize(
        metric: MetricType,
        m: i32,
        ef_construction: i32,
        quantize: QuantizeType,
    ) -> Result<Self> {
        if m <= 0 || ef_construction <= 0 {
            return Err(Error::invalid_argument(
                "HNSW m and ef_construction must be positive",
            ));
        }
        let mut out = Self::new(IndexType::Hnsw, metric);
        out.quantize_type = quantize;
        out.params.insert("m".into(), json!(m));
        out.params
            .insert("ef_construction".into(), json!(ef_construction));
        out.params.insert("quantize_type".into(), json!(quantize));
        Ok(out)
    }

    pub fn ivf(metric: MetricType, n_list: i32, n_iters: i32, use_soar: bool) -> Result<Self> {
        if n_list <= 0 || n_iters < 0 {
            return Err(Error::invalid_argument(
                "IVF n_list must be positive and n_iters non-negative",
            ));
        }
        let mut out = Self::new(IndexType::Ivf, metric);
        out.params.insert("n_list".into(), json!(n_list));
        out.params.insert("n_iters".into(), json!(n_iters));
        out.params.insert("use_soar".into(), json!(use_soar));
        Ok(out)
    }

    pub fn ivf_rabitq(
        metric: MetricType,
        nlist: i32,
        total_bits: i32,
        sample_count: i32,
    ) -> Result<Self> {
        if nlist <= 0 || !(1..=9).contains(&total_bits) || sample_count < 0 {
            return Err(Error::invalid_argument(
                "IVF RaBitQ requires positive nlist, total_bits in 1..=9, and non-negative sample_count",
            ));
        }
        let mut out = Self::new(IndexType::IvfRabitq, metric);
        out.quantize_type = QuantizeType::Rabitq;
        out.params.insert("n_list".into(), json!(nlist));
        out.params.insert("total_bits".into(), json!(total_bits));
        out.params
            .insert("sample_count".into(), json!(sample_count));
        out.params
            .insert("quantize_type".into(), json!(QuantizeType::Rabitq));
        Ok(out)
    }

    pub fn flat(metric: MetricType) -> Result<Self> {
        Ok(Self::new(IndexType::Flat, metric))
    }

    pub fn diskann(
        metric: MetricType,
        max_degree: i32,
        list_size: i32,
        pq_chunk_num: i32,
    ) -> Result<Self> {
        if max_degree <= 0 || list_size <= 0 || pq_chunk_num < 0 {
            return Err(Error::invalid_argument(
                "DiskANN parameters are out of range",
            ));
        }
        let mut out = Self::new(IndexType::Diskann, metric);
        out.params.insert("max_degree".into(), json!(max_degree));
        out.params.insert("list_size".into(), json!(list_size));
        out.params
            .insert("pq_chunk_num".into(), json!(pq_chunk_num));
        out.params.insert("alpha".into(), json!(1.2));
        Ok(out)
    }

    pub fn vamana(
        metric: MetricType,
        max_degree: i32,
        search_list_size: i32,
        alpha: f64,
    ) -> Result<Self> {
        Self::vamana_with_options(metric, max_degree, search_list_size, alpha, 0, false)
    }

    /// Creates a Vamana graph descriptor with explicit `RobustPrune` controls.
    ///
    /// `max_occlusion` bounds the candidate set considered by `RobustPrune`;
    /// zero keeps the historical unbounded behavior. `saturate` fills any
    /// remaining out-edge slots after diversity pruning.
    pub fn vamana_with_options(
        metric: MetricType,
        max_degree: i32,
        search_list_size: i32,
        alpha: f64,
        max_occlusion: i32,
        saturate: bool,
    ) -> Result<Self> {
        if max_degree <= 0
            || search_list_size <= 0
            || max_occlusion < 0
            || !alpha.is_finite()
            || alpha < 1.0
        {
            return Err(Error::invalid_argument(
                "Vamana parameters are out of range",
            ));
        }
        let mut out = Self::new(IndexType::Vamana, metric);
        out.params.insert("max_degree".into(), json!(max_degree));
        out.params
            .insert("search_list_size".into(), json!(search_list_size));
        out.params.insert("alpha".into(), json!(alpha));
        out.params
            .insert("max_occlusion".into(), json!(max_occlusion));
        out.params.insert("saturate".into(), json!(saturate));
        Ok(out)
    }

    pub fn hnsw_rabitq(metric: MetricType, m: i32, ef_construction: i32) -> Result<Self> {
        Self::hnsw_rabitq_with_options(metric, m, ef_construction, 7, 16, 0)
    }

    pub fn hnsw_rabitq_with_options(
        metric: MetricType,
        m: i32,
        ef_construction: i32,
        total_bits: i32,
        num_clusters: i32,
        sample_count: i32,
    ) -> Result<Self> {
        if !(1..=9).contains(&total_bits) || num_clusters <= 0 || sample_count < 0 {
            return Err(Error::invalid_argument(
                "HNSW RaBitQ requires total_bits in 1..=9, positive num_clusters, and non-negative sample_count",
            ));
        }
        let mut out = Self::hnsw_with_quantize(metric, m, ef_construction, QuantizeType::Rabitq)?;
        out.index_type = IndexType::HnswRabitq;
        out.params.insert("total_bits".into(), json!(total_bits));
        out.params
            .insert("num_clusters".into(), json!(num_clusters));
        out.params
            .insert("sample_count".into(), json!(sample_count));
        Ok(out)
    }

    pub fn invert(enable_range_opt: bool, enable_wildcard: bool) -> Result<Self> {
        let mut out = Self::new(IndexType::Invert, MetricType::Undefined);
        out.params
            .insert("enable_range_optimization".into(), json!(enable_range_opt));
        out.params
            .insert("enable_wildcard".into(), json!(enable_wildcard));
        Ok(out)
    }

    /// Creates full-text index parameters.
    ///
    /// Omitting `filters` selects the zvec-compatible `lowercase` default;
    /// passing an explicit empty slice preserves tokenizer case. Supported
    /// filters are `lowercase`, `ascii_folding`, and `stemmer`.
    ///
    /// The `ngram` tokenizer accepts `ngram_min`, `ngram_max`, and
    /// `token_chars` in `extra_params`. The standard tokenizer accepts
    /// `max_token_length`, while the stemmer filter accepts `stemmer_lang`.
    pub fn fts(
        tokenizer_name: Option<&str>,
        filters: Option<&[&str]>,
        extra_params: Option<&str>,
    ) -> Result<Self> {
        let tokenizer = tokenizer_name.unwrap_or("standard").trim();
        if tokenizer.is_empty() {
            return Err(Error::invalid_argument(
                "FTS tokenizer name must not be empty",
            ));
        }
        let mut out = Self::new(IndexType::Fts, MetricType::Undefined);
        out.params.insert("tokenizer_name".into(), json!(tokenizer));
        out.params.insert(
            "filters".into(),
            json!(filters.map_or_else(|| vec!["lowercase"], <[_]>::to_vec)),
        );
        if let Some(extra) = extra_params {
            out.params.insert("extra_params".into(), json!(extra));
        }
        Ok(out)
    }

    pub fn new(index_type: IndexType, metric_type: MetricType) -> Self {
        Self {
            index_type,
            metric_type,
            quantize_type: QuantizeType::Undefined,
            params: Map::new(),
        }
    }

    pub fn index_type(&self) -> IndexType {
        self.index_type
    }
    pub fn metric_type(&self) -> MetricType {
        self.metric_type
    }
    pub fn quantize_type(&self) -> QuantizeType {
        self.quantize_type
    }
    pub fn set_metric_type(&mut self, metric: MetricType) -> Result<()> {
        if metric == MetricType::Undefined && self.index_type.is_vector_index() {
            return Err(Error::invalid_argument("vector index requires a metric"));
        }
        self.metric_type = metric;
        Ok(())
    }
    pub fn set_quantize_type(&mut self, quantize: QuantizeType) -> Result<()> {
        self.quantize_type = quantize;
        self.params.insert("quantize_type".into(), json!(quantize));
        Ok(())
    }
    pub fn parameter(&self, name: &str) -> Option<&Value> {
        self.params.get(name)
    }
    pub fn with_parameter(mut self, name: impl Into<String>, value: Value) -> Self {
        self.params.insert(name.into(), value);
        self
    }
}

impl IndexType {
    pub fn is_vector_index(self) -> bool {
        matches!(
            self,
            Self::Hnsw
                | Self::HnswRabitq
                | Self::Ivf
                | Self::IvfRabitq
                | Self::Flat
                | Self::Diskann
                | Self::Vamana
        )
    }
}

/// Fluent index-parameter builder for callers that prefer typed construction.
#[derive(Debug, Clone)]
pub struct IndexParamsBuilder {
    params: IndexParams,
}

impl IndexParamsBuilder {
    pub fn new(index_type: IndexType) -> Self {
        Self {
            params: IndexParams::new(index_type, MetricType::Undefined),
        }
    }
    pub fn metric_type(mut self, metric: MetricType) -> Self {
        self.params.metric_type = metric;
        self
    }
    pub fn metric(self, metric: MetricType) -> Self {
        self.metric_type(metric)
    }
    pub fn quantize_type(mut self, quantize: QuantizeType) -> Self {
        self.params.quantize_type = quantize;
        self
    }
    pub fn m(mut self, value: u32) -> Self {
        self.params.params.insert("m".into(), json!(value));
        self
    }
    pub fn ef_construction(mut self, value: u32) -> Self {
        self.params
            .params
            .insert("ef_construction".into(), json!(value));
        self
    }
    pub fn n_list(mut self, value: u32) -> Self {
        self.params.params.insert("n_list".into(), json!(value));
        self
    }
    pub fn n_iters(mut self, value: u32) -> Self {
        self.params.params.insert("n_iters".into(), json!(value));
        self
    }
    pub fn use_soar(mut self, value: bool) -> Self {
        self.params.params.insert("use_soar".into(), json!(value));
        self
    }
    pub fn max_degree(mut self, value: u32) -> Self {
        self.params.params.insert("max_degree".into(), json!(value));
        self
    }
    pub fn list_size(mut self, value: u32) -> Self {
        self.params.params.insert("list_size".into(), json!(value));
        self
    }
    pub fn search_list_size(mut self, value: u32) -> Self {
        self.params
            .params
            .insert("search_list_size".into(), json!(value));
        self
    }
    pub fn alpha(mut self, value: f64) -> Self {
        self.params.params.insert("alpha".into(), json!(value));
        self
    }
    pub fn max_occlusion(mut self, value: u32) -> Self {
        self.params
            .params
            .insert("max_occlusion".into(), json!(value));
        self
    }
    pub fn saturate(mut self, value: bool) -> Self {
        self.params.params.insert("saturate".into(), json!(value));
        self
    }
    pub fn pq_chunk_num(mut self, value: u32) -> Self {
        self.params
            .params
            .insert("pq_chunk_num".into(), json!(value));
        self
    }
    pub fn tokenizer(mut self, value: impl Into<String>) -> Self {
        self.params
            .params
            .insert("tokenizer_name".into(), json!(value.into()));
        self
    }
    pub fn parameter(mut self, name: impl Into<String>, value: Value) -> Self {
        self.params.params.insert(name.into(), value);
        self
    }
    pub fn build(self) -> Result<IndexParams> {
        if self.params.index_type.is_vector_index()
            && self.params.metric_type == MetricType::Undefined
        {
            return Err(Error::invalid_argument("vector index requires a metric"));
        }
        Ok(self.params)
    }
}

/// Schema for one field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub dimension: u32,
    pub index_params: Option<IndexParams>,
}

impl FieldSchema {
    pub fn new(name: &str, data_type: DataType, nullable: bool, dimension: u32) -> Result<Self> {
        validate_field_shape(name, data_type, dimension)?;
        Ok(Self {
            name: name.to_string(),
            data_type,
            nullable,
            dimension,
            index_params: None,
        })
    }
    pub fn set_index_params(&mut self, params: &IndexParams) -> Result<()> {
        validate_field_shape(&self.name, self.data_type, self.dimension)?;
        validate_index_configuration(&self.name, self.data_type, self.dimension, params)?;
        self.index_params = Some(params.clone());
        Ok(())
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
    pub fn dimension(&self) -> u32 {
        self.dimension
    }
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }
    pub fn is_vector_field(&self) -> bool {
        self.data_type.is_vector()
    }
    pub fn is_dense_vector(&self) -> bool {
        self.data_type.is_dense_vector()
    }
    pub fn is_sparse_vector(&self) -> bool {
        self.data_type.is_sparse_vector()
    }
    pub fn has_index(&self) -> bool {
        self.index_params.is_some()
    }
    pub fn index_type(&self) -> IndexType {
        self.index_params
            .as_ref()
            .map_or(IndexType::Undefined, IndexParams::index_type)
    }
    pub fn is_array_type(&self) -> bool {
        self.data_type.is_array()
    }
    pub fn index_params(&self) -> Option<&IndexParams> {
        self.index_params.as_ref()
    }
}

/// Explicit vector schema (also useful to adapters that keep vectors separate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSchema {
    pub name: String,
    pub data_type: DataType,
    pub dimension: u32,
    pub index_params: Option<IndexParams>,
}

impl VectorSchema {
    pub fn new(name: &str, data_type: DataType, dimension: u32) -> Result<Self> {
        let field = FieldSchema::new(name, data_type, false, dimension)?;
        if !field.is_vector_field() {
            return Err(Error::invalid_argument(
                "vector schema requires a vector data type",
            ));
        }
        Ok(Self {
            name: field.name,
            data_type,
            dimension,
            index_params: None,
        })
    }
    pub fn set_index_params(&mut self, params: &IndexParams) -> Result<()> {
        let mut field = FieldSchema::new(&self.name, self.data_type, false, self.dimension)?;
        field.set_index_params(params)?;
        self.index_params = field.index_params;
        Ok(())
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dimension(&self) -> u32 {
        self.dimension
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
    pub fn has_index(&self) -> bool {
        self.index_params.is_some()
    }
    pub fn index_type(&self) -> IndexType {
        self.index_params
            .as_ref()
            .map_or(IndexType::Undefined, IndexParams::index_type)
    }
}

/// Collection schema containing scalar and vector fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: String,
    pub fields: Vec<FieldSchema>,
    pub vectors: Vec<VectorSchema>,
    pub max_doc_count_per_segment: u64,
}

impl CollectionSchema {
    pub fn new(name: &str) -> Result<Self> {
        validate_name(name)?;
        Ok(Self {
            name: name.to_string(),
            fields: Vec::new(),
            vectors: Vec::new(),
            max_doc_count_per_segment: 0,
        })
    }
    pub fn builder(name: &str) -> CollectionSchemaBuilder {
        CollectionSchemaBuilder::new(name)
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn add_field(&mut self, field: &FieldSchema) -> Result<()> {
        validate_field_shape(&field.name, field.data_type, field.dimension)?;
        self.ensure_unique(&field.name)?;
        if let Some(params) = &field.index_params {
            validate_index_configuration(&field.name, field.data_type, field.dimension, params)?;
        }
        if field.data_type.is_vector() {
            self.vectors.push(VectorSchema {
                name: field.name.clone(),
                data_type: field.data_type,
                dimension: field.dimension,
                index_params: field.index_params.clone(),
            });
        } else {
            self.fields.push(field.clone());
        }
        Ok(())
    }
    pub fn add_vector_field(&mut self, field: &VectorSchema) -> Result<()> {
        validate_field_shape(&field.name, field.data_type, field.dimension)?;
        self.ensure_unique(&field.name)?;
        if !field.data_type.is_vector() {
            return Err(Error::invalid_argument(
                "vector field requires vector data type",
            ));
        }
        if let Some(params) = &field.index_params {
            validate_index_configuration(&field.name, field.data_type, field.dimension, params)?;
        }
        self.vectors.push(field.clone());
        Ok(())
    }
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name) || self.vectors.iter().any(|f| f.name == name)
    }
    pub fn field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.iter().find(|f| f.name == name)
    }
    pub fn vector(&self, name: &str) -> Option<&VectorSchema> {
        self.vectors.iter().find(|f| f.name == name)
    }
    pub fn has_index(&self, name: &str) -> bool {
        self.fields
            .iter()
            .find(|f| f.name == name)
            .and_then(|f| f.index_params.as_ref())
            .is_some()
            || self
                .vectors
                .iter()
                .find(|f| f.name == name)
                .and_then(|f| f.index_params.as_ref())
                .is_some()
    }
    pub fn drop_field(&mut self, name: &str) -> Result<()> {
        let before = self.fields.len() + self.vectors.len();
        self.fields.retain(|f| f.name != name);
        self.vectors.retain(|f| f.name != name);
        if before == self.fields.len() + self.vectors.len() {
            return Err(Error::not_found(format!("field '{name}' not found")));
        }
        Ok(())
    }
    pub fn add_index(&mut self, field_name: &str, params: &IndexParams) -> Result<()> {
        if let Some(field) = self.fields.iter_mut().find(|f| f.name == field_name) {
            return field.set_index_params(params);
        }
        if let Some(field) = self.vectors.iter_mut().find(|f| f.name == field_name) {
            return field.set_index_params(params);
        }
        Err(Error::not_found(format!("field '{field_name}' not found")))
    }
    pub(crate) fn check_index_configuration(
        &self,
        field_name: &str,
        params: &IndexParams,
    ) -> Result<()> {
        if let Some(field) = self.fields.iter().find(|field| field.name == field_name) {
            return validate_index_configuration(
                &field.name,
                field.data_type,
                field.dimension,
                params,
            );
        }
        if let Some(field) = self.vectors.iter().find(|field| field.name == field_name) {
            return validate_index_configuration(
                &field.name,
                field.data_type,
                field.dimension,
                params,
            );
        }
        Err(Error::not_found(format!("field '{field_name}' not found")))
    }
    pub fn drop_index(&mut self, field_name: &str) -> Result<()> {
        if let Some(field) = self.fields.iter_mut().find(|f| f.name == field_name) {
            field.index_params = None;
            return Ok(());
        }
        if let Some(field) = self.vectors.iter_mut().find(|f| f.name == field_name) {
            field.index_params = None;
            return Ok(());
        }
        Err(Error::not_found(format!("field '{field_name}' not found")))
    }
    pub fn set_max_doc_count_per_segment(&mut self, count: u64) -> Result<()> {
        if count == 0 {
            self.max_doc_count_per_segment = 0;
            Ok(())
        } else {
            Err(Error::not_supported(
                "max_doc_count_per_segment requires a segmented storage executor",
            ))
        }
    }
    pub fn max_doc_count_per_segment(&self) -> u64 {
        self.max_doc_count_per_segment
    }
    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name)?;
        if self.max_doc_count_per_segment != 0 {
            return Err(Error::not_supported(
                "max_doc_count_per_segment requires a segmented storage executor",
            ));
        }
        if self.fields.len() + self.vectors.len() == 0 {
            return Err(Error::invalid_argument(
                "collection schema must contain at least one field",
            ));
        }
        let mut names = BTreeSet::new();
        for field in &self.fields {
            validate_field_shape(&field.name, field.data_type, field.dimension)?;
            if field.data_type.is_vector() {
                return Err(Error::invalid_argument(format!(
                    "vector field '{}' must be stored in the vector schema list",
                    field.name
                )));
            }
            if !names.insert(&field.name) {
                return Err(Error::invalid_argument(format!(
                    "duplicate field name '{}'",
                    field.name
                )));
            }
            if let Some(params) = &field.index_params {
                validate_index_configuration(
                    &field.name,
                    field.data_type,
                    field.dimension,
                    params,
                )?;
            }
        }
        for field in &self.vectors {
            validate_field_shape(&field.name, field.data_type, field.dimension)?;
            if !field.data_type.is_vector() {
                return Err(Error::invalid_argument(format!(
                    "non-vector field '{}' must be stored in the scalar schema list",
                    field.name
                )));
            }
            if !names.insert(&field.name) {
                return Err(Error::invalid_argument(format!(
                    "duplicate field name '{}'",
                    field.name
                )));
            }
            if let Some(params) = &field.index_params {
                validate_index_configuration(
                    &field.name,
                    field.data_type,
                    field.dimension,
                    params,
                )?;
            }
        }
        Ok(())
    }
    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        format!("{:08x}", crc32fast::hash(&bytes))
    }
    pub fn fields(&self) -> &[FieldSchema] {
        &self.fields
    }
    pub fn vectors(&self) -> &[VectorSchema] {
        &self.vectors
    }
    fn ensure_unique(&self, name: &str) -> Result<()> {
        if self.has_field(name) {
            Err(Error::already_exists(format!(
                "field '{name}' already exists"
            )))
        } else {
            Ok(())
        }
    }
}

/// Fluent collection schema builder.
#[derive(Debug, Clone)]
pub struct CollectionSchemaBuilder {
    name: String,
    fields: Vec<(FieldSchema, Option<IndexParams>)>,
    max_doc_count_per_segment: Option<u64>,
    deferred_error: Option<Error>,
}

impl CollectionSchemaBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fields: Vec::new(),
            max_doc_count_per_segment: None,
            deferred_error: None,
        }
    }
    pub fn add_field(mut self, field: FieldSchema) -> Self {
        self.fields.push((field, None));
        self
    }
    pub fn add_vector_field(
        mut self,
        name: &str,
        data_type: DataType,
        dimension: u32,
        index_params: IndexParams,
    ) -> Self {
        match FieldSchema::new(name, data_type, false, dimension) {
            Ok(field) => self.fields.push((field, Some(index_params))),
            Err(error) => self.deferred_error = Some(error),
        }
        self
    }
    pub fn add_indexed_field(
        mut self,
        name: &str,
        data_type: DataType,
        index_params: IndexParams,
    ) -> Self {
        match FieldSchema::new(name, data_type, false, 0) {
            Ok(field) => self.fields.push((field, Some(index_params))),
            Err(error) => self.deferred_error = Some(error),
        }
        self
    }
    pub fn max_doc_count_per_segment(mut self, count: u64) -> Self {
        self.max_doc_count_per_segment = Some(count);
        self
    }
    pub fn build(self) -> Result<CollectionSchema> {
        if let Some(error) = self.deferred_error {
            return Err(error);
        }
        let mut schema = CollectionSchema::new(&self.name)?;
        for (mut field, params) in self.fields {
            if let Some(params) = params {
                field.set_index_params(&params)?;
            }
            schema.add_field(&field)?;
        }
        if let Some(count) = self.max_doc_count_per_segment {
            schema.set_max_doc_count_per_segment(count)?;
        }
        schema.validate()?;
        Ok(schema)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AddColumnOption {
    /// Requested schema-backfill workers. Zero keeps the deterministic serial
    /// path; positive values are bounded by work size, host parallelism, and
    /// the engine's worker ceiling.
    pub concurrency: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AlterColumnOption {
    /// Requested candidate-schema validation workers. Zero keeps the
    /// deterministic serial path; positive values use the same bounded pool
    /// as [`AddColumnOption::concurrency`].
    pub concurrency: u32,
}

fn validate_name(name: &str) -> Result<()> {
    if name.trim().is_empty() || name.contains('\0') {
        return Err(Error::invalid_argument(
            "name must be non-empty and contain no NUL byte",
        ));
    }
    Ok(())
}

/// Validates the portion of a public field descriptor that must remain sound
/// even when callers construct or deserialize the descriptor and then mutate
/// its public fields directly.
fn validate_field_shape(name: &str, data_type: DataType, dimension: u32) -> Result<()> {
    validate_name(name)?;
    if data_type == DataType::Undefined {
        return Err(Error::invalid_argument("field data type must be defined"));
    }
    if data_type.is_vector() && !data_type.is_sparse_vector() && dimension == 0 {
        return Err(Error::invalid_argument(
            "dense vector dimension must be positive",
        ));
    }
    let binary_alignment = match data_type {
        DataType::VectorBinary32 => Some(32),
        DataType::VectorBinary64 => Some(64),
        _ => None,
    };
    if let Some(alignment) = binary_alignment {
        if dimension % alignment != 0 {
            return Err(Error::invalid_argument(format!(
                "{data_type} dimension must be a multiple of {alignment}"
            )));
        }
    }
    if !data_type.is_vector() && dimension != 0 {
        return Err(Error::invalid_argument(
            "non-vector field dimension must be zero",
        ));
    }
    Ok(())
}
