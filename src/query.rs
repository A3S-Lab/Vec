//! Query payloads and per-index search controls.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswQueryParams {
    pub ef: i32,
    pub radius: f32,
    pub is_linear: bool,
    pub is_using_refiner: bool,
}
impl HnswQueryParams {
    pub fn new(ef: i32, radius: f32, is_linear: bool, is_using_refiner: bool) -> Self { Self { ef, radius, is_linear, is_using_refiner } }
    pub fn set_ef(&mut self, ef: i32) -> Result<()> { if ef <= 0 { return Err(Error::invalid_argument("HNSW ef must be positive")); } self.ef = ef; Ok(()) }
    pub fn ef(&self) -> i32 { self.ef }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IvfQueryParams {
    pub nprobe: i32,
    pub is_using_refiner: bool,
    pub scale_factor: f32,
}
impl IvfQueryParams {
    pub fn new(nprobe: i32, is_using_refiner: bool, scale_factor: f32) -> Self { Self { nprobe, is_using_refiner, scale_factor } }
    pub fn set_nprobe(&mut self, nprobe: i32) -> Result<()> { if nprobe <= 0 { return Err(Error::invalid_argument("IVF nprobe must be positive")); } self.nprobe = nprobe; Ok(()) }
    pub fn nprobe(&self) -> i32 { self.nprobe }
    pub fn set_scale_factor(&mut self, value: f32) -> Result<()> { if !value.is_finite() || value <= 0.0 { return Err(Error::invalid_argument("scale factor must be finite and positive")); } self.scale_factor = value; Ok(()) }
    pub fn scale_factor(&self) -> f32 { self.scale_factor }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IvfRabitqQueryParams {
    pub nprobe: i32,
    pub radius: f32,
    pub is_linear: bool,
    pub is_using_refiner: bool,
    pub scale_factor: f32,
}
impl IvfRabitqQueryParams {
    pub fn new(nprobe: i32, radius: f32, is_linear: bool, is_using_refiner: bool) -> Self { Self { nprobe, radius, is_linear, is_using_refiner, scale_factor: 1.0 } }
    pub fn set_nprobe(&mut self, nprobe: i32) -> Result<()> { if nprobe <= 0 { return Err(Error::invalid_argument("IVF RaBitQ nprobe must be positive")); } self.nprobe = nprobe; Ok(()) }
    pub fn nprobe(&self) -> i32 { self.nprobe }
    pub fn set_scale_factor(&mut self, value: f32) -> Result<()> { if !value.is_finite() || value <= 0.0 { return Err(Error::invalid_argument("scale factor must be finite and positive")); } self.scale_factor = value; Ok(()) }
    pub fn scale_factor(&self) -> f32 { self.scale_factor }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatQueryParams {
    pub is_using_refiner: bool,
    pub scale_factor: f32,
}
impl FlatQueryParams {
    pub fn new(is_using_refiner: bool, scale_factor: f32) -> Self { Self { is_using_refiner, scale_factor } }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskannQueryParams {
    pub list_size: i32,
}
impl DiskannQueryParams {
    pub fn new(list_size: i32) -> Self { Self { list_size } }
    pub fn set_list_size(&mut self, value: i32) -> Result<()> { if value <= 0 { return Err(Error::invalid_argument("DiskANN list_size must be positive")); } self.list_size = value; Ok(()) }
    pub fn list_size(&self) -> i32 { self.list_size }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FtsQueryParams {
    pub default_operator: Option<String>,
}
impl FtsQueryParams {
    pub fn new(default_operator: Option<&str>) -> Result<Self> {
        let mut out = Self { default_operator: None };
        if let Some(value) = default_operator { out.set_default_operator(value)?; }
        Ok(out)
    }
    pub fn set_default_operator(&mut self, op: &str) -> Result<()> {
        let normalized = op.to_ascii_lowercase();
        if normalized != "and" && normalized != "or" { return Err(Error::invalid_argument("FTS default operator must be AND or OR")); }
        self.default_operator = Some(normalized);
        Ok(())
    }
    pub fn default_operator(&self) -> Option<String> { self.default_operator.clone() }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fts {
    pub query_string: Option<String>,
    pub match_string: Option<String>,
}
impl Fts {
    pub fn new() -> Result<Self> { Ok(Self { query_string: None, match_string: None }) }
    pub fn set_query_string(&mut self, query: &str) -> Result<()> { if query.trim().is_empty() { return Err(Error::invalid_argument("FTS query string must not be empty")); } self.query_string = Some(query.to_string()); Ok(()) }
    pub fn set_match_string(&mut self, query: &str) -> Result<()> { if query.trim().is_empty() { return Err(Error::invalid_argument("FTS match string must not be empty")); } self.match_string = Some(query.to_string()); Ok(()) }
    pub fn query_string(&self) -> Option<String> { self.query_string.clone() }
    pub fn match_string(&self) -> Option<String> { self.match_string.clone() }
}

/// A single dense, sparse, id-based, or FTS query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchQuery {
    pub field_name: String,
    pub vector: Option<Vec<f32>>,
    pub sparse_vector: Option<Vec<(u32, f32)>>,
    pub id: Option<String>,
    pub topk: i32,
    pub filter: Option<String>,
    pub include_vector: bool,
    pub include_doc_id: bool,
    pub output_fields: Option<Vec<String>>,
    pub fts: Option<Fts>,
    #[serde(default)]
    pub params: Map<String, Value>,
}

/// Compatibility alias used by the zvec vocabulary.
pub type VectorQuery = SearchQuery;

impl SearchQuery {
    pub fn new(field_name: &str, vector: &[f32], topk: i32) -> Result<Self> {
        validate_query_header(field_name, topk)?;
        if vector.is_empty() || !vector.iter().all(|v| v.is_finite()) { return Err(Error::invalid_argument("query vector must be non-empty and finite")); }
        Ok(Self { field_name: field_name.to_string(), vector: Some(vector.to_vec()), sparse_vector: None, id: None, topk, filter: None, include_vector: false, include_doc_id: false, output_fields: None, fts: None, params: Map::new() })
    }
    pub fn fts(field_name: &str, fts: &Fts, topk: i32) -> Result<Self> {
        validate_query_header(field_name, topk)?;
        if fts.query_string.is_none() && fts.match_string.is_none() { return Err(Error::invalid_argument("FTS query has no expression")); }
        Ok(Self { field_name: field_name.to_string(), vector: None, sparse_vector: None, id: None, topk, filter: None, include_vector: false, include_doc_id: false, output_fields: None, fts: Some(fts.clone()), params: Map::new() })
    }
    pub fn by_id(field_name: &str, id: &str, topk: i32) -> Result<Self> {
        validate_query_header(field_name, topk)?;
        if id.is_empty() { return Err(Error::invalid_argument("query id must not be empty")); }
        Ok(Self { field_name: field_name.to_string(), vector: None, sparse_vector: None, id: Some(id.to_string()), topk, filter: None, include_vector: false, include_doc_id: false, output_fields: None, fts: None, params: Map::new() })
    }
    pub fn sparse(field_name: &str, indices: &[u32], values: &[f32], topk: i32) -> Result<Self> {
        validate_query_header(field_name, topk)?;
        if indices.is_empty() || indices.len() != values.len() || !values.iter().all(|v| v.is_finite()) { return Err(Error::invalid_argument("sparse query indices and values must have equal non-zero length")); }
        Ok(Self { field_name: field_name.to_string(), vector: None, sparse_vector: Some(indices.iter().copied().zip(values.iter().copied()).collect()), id: None, topk, filter: None, include_vector: false, include_doc_id: false, output_fields: None, fts: None, params: Map::new() })
    }
    pub fn builder() -> SearchQueryBuilder { SearchQueryBuilder::new() }
    pub fn set_field_name(&mut self, name: &str) -> Result<()> { validate_name(name)?; self.field_name = name.to_string(); Ok(()) }
    pub fn set_query_vector(&mut self, vector: &[f32]) -> Result<()> { if vector.is_empty() || !vector.iter().all(|v| v.is_finite()) { return Err(Error::invalid_argument("query vector must be non-empty and finite")); } self.vector = Some(vector.to_vec()); self.sparse_vector = None; self.id = None; Ok(()) }
    pub fn set_sparse_vector(&mut self, indices: &[u32], values: &[f32]) -> Result<()> { if indices.is_empty() || indices.len() != values.len() || !values.iter().all(|v| v.is_finite()) { return Err(Error::invalid_argument("sparse query indices and values are invalid")); } self.sparse_vector = Some(indices.iter().copied().zip(values.iter().copied()).collect()); self.vector = None; self.id = None; Ok(()) }
    pub fn set_filter(&mut self, filter: &str) -> Result<()> { if filter.trim().is_empty() { self.filter = None; return Ok(()); } self.filter = Some(filter.to_string()); Ok(()) }
    pub fn set_topk(&mut self, topk: i32) -> Result<()> { if topk <= 0 { return Err(Error::invalid_argument("topk must be positive")); } self.topk = topk; Ok(()) }
    pub fn set_include_vector(&mut self, include: bool) -> Result<()> { self.include_vector = include; Ok(()) }
    pub fn set_include_doc_id(&mut self, include: bool) -> Result<()> { self.include_doc_id = include; Ok(()) }
    pub fn set_output_fields(&mut self, fields: &[&str]) -> Result<()> { for field in fields { validate_name(field)?; } self.output_fields = Some(fields.iter().map(|s| (*s).to_string()).collect()); Ok(()) }
    pub fn set_hnsw_params(&mut self, params: HnswQueryParams) -> Result<()> { self.params.insert("type".into(), json!("hnsw")); self.params.insert("ef".into(), json!(params.ef)); self.params.insert("radius".into(), json!(params.radius)); self.params.insert("is_linear".into(), json!(params.is_linear)); self.params.insert("is_using_refiner".into(), json!(params.is_using_refiner)); Ok(()) }
    pub fn set_ivf_params(&mut self, params: IvfQueryParams) -> Result<()> { self.params.insert("type".into(), json!("ivf")); self.params.insert("nprobe".into(), json!(params.nprobe)); self.params.insert("is_using_refiner".into(), json!(params.is_using_refiner)); self.params.insert("scale_factor".into(), json!(params.scale_factor)); Ok(()) }
    pub fn set_ivf_rabitq_params(&mut self, params: IvfRabitqQueryParams) -> Result<()> { self.params.insert("type".into(), json!("ivf_rabitq")); self.params.insert("nprobe".into(), json!(params.nprobe)); self.params.insert("radius".into(), json!(params.radius)); self.params.insert("is_linear".into(), json!(params.is_linear)); self.params.insert("is_using_refiner".into(), json!(params.is_using_refiner)); self.params.insert("scale_factor".into(), json!(params.scale_factor)); Ok(()) }
    pub fn set_flat_params(&mut self, params: FlatQueryParams) -> Result<()> { self.params.insert("type".into(), json!("flat")); self.params.insert("is_using_refiner".into(), json!(params.is_using_refiner)); self.params.insert("scale_factor".into(), json!(params.scale_factor)); Ok(()) }
    pub fn set_diskann_params(&mut self, params: DiskannQueryParams) -> Result<()> { if params.list_size <= 0 { return Err(Error::invalid_argument("DiskANN list_size must be positive")); } self.params.insert("type".into(), json!("diskann")); self.params.insert("list_size".into(), json!(params.list_size)); Ok(()) }
    pub fn set_fts_params(&mut self, params: FtsQueryParams) -> Result<()> { if let Some(op) = params.default_operator { self.params.insert("operator".into(), json!(op)); } Ok(()) }
    pub fn set_fts(&mut self, fts: &Fts) -> Result<()> { if fts.query_string.is_none() && fts.match_string.is_none() { return Err(Error::invalid_argument("FTS query has no expression")); } self.fts = Some(fts.clone()); Ok(()) }
    pub fn set_radius(&mut self, radius: f32) -> Result<()> { if !radius.is_finite() { return Err(Error::invalid_argument("radius must be finite")); } self.params.insert("radius".into(), json!(radius)); Ok(()) }
    pub fn get_filter(&self) -> Option<&str> { self.filter.as_deref() }
    pub fn has_vector(&self) -> bool { self.vector.is_some() || self.sparse_vector.is_some() }
}

/// Fluent builder matching the official SDK's naming.
#[derive(Debug, Clone, Default)]
pub struct SearchQueryBuilder {
    field_name: Option<String>,
    vector: Option<Vec<f32>>,
    topk: i32,
    filter: Option<String>,
    include_vector: Option<bool>,
    include_doc_id: Option<bool>,
    output_fields: Option<Vec<String>>,
    fts_query_string: Option<String>,
    fts_match_string: Option<String>,
}
impl SearchQueryBuilder {
    pub fn new() -> Self { Self { topk: 10, ..Self::default() } }
    pub fn field_name(mut self, name: &str) -> Self { self.field_name = Some(name.to_string()); self }
    pub fn vector(mut self, vector: &[f32]) -> Self { self.vector = Some(vector.to_vec()); self }
    pub fn topk(mut self, topk: i32) -> Self { self.topk = topk; self }
    pub fn filter(mut self, filter: &str) -> Self { self.filter = Some(filter.to_string()); self }
    pub fn include_vector(mut self, include: bool) -> Self { self.include_vector = Some(include); self }
    pub fn include_doc_id(mut self, include: bool) -> Self { self.include_doc_id = Some(include); self }
    pub fn output_fields(mut self, fields: &[&str]) -> Self { self.output_fields = Some(fields.iter().map(|v| (*v).to_string()).collect()); self }
    pub fn fts_query_string(mut self, query: &str) -> Self { self.fts_query_string = Some(query.to_string()); self }
    pub fn fts_match_string(mut self, query: &str) -> Self { self.fts_match_string = Some(query.to_string()); self }
    pub fn build(self) -> Result<SearchQuery> {
        let field = self.field_name.ok_or_else(|| Error::invalid_argument("field_name is required"))?;
        let vector = self.vector.ok_or_else(|| Error::invalid_argument("vector is required"))?;
        let mut query = SearchQuery::new(&field, &vector, self.topk)?;
        if let Some(filter) = self.filter { query.set_filter(&filter)?; }
        if let Some(value) = self.include_vector { query.set_include_vector(value)?; }
        if let Some(value) = self.include_doc_id { query.set_include_doc_id(value)?; }
        if let Some(fields) = self.output_fields { let refs: Vec<&str> = fields.iter().map(String::as_str).collect(); query.set_output_fields(&refs)?; }
        if self.fts_query_string.is_some() || self.fts_match_string.is_some() {
            let mut fts = Fts::new()?;
            if let Some(value) = self.fts_query_string { fts.set_query_string(&value)?; }
            if let Some(value) = self.fts_match_string { fts.set_match_string(&value)?; }
            query.set_fts(&fts)?;
        }
        Ok(query)
    }
}

/// Group-by vector query payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupBySearchQuery {
    pub field_name: String,
    pub group_by_field: String,
    pub vector: Vec<f32>,
    pub group_count: u32,
    pub group_topk: u32,
    pub filter: Option<String>,
    pub include_vector: bool,
    pub output_fields: Option<Vec<String>>,
    pub params: Map<String, Value>,
}
impl GroupBySearchQuery {
    pub fn new(field_name: &str, group_by_field: &str, vector: &[f32], group_count: u32, group_topk: u32) -> Result<Self> {
        validate_name(field_name)?; validate_name(group_by_field)?;
        if vector.is_empty() || !vector.iter().all(|v| v.is_finite()) || group_count == 0 || group_topk == 0 { return Err(Error::invalid_argument("invalid group-by query parameters")); }
        Ok(Self { field_name: field_name.to_string(), group_by_field: group_by_field.to_string(), vector: vector.to_vec(), group_count, group_topk, filter: None, include_vector: false, output_fields: None, params: Map::new() })
    }
    pub fn set_filter(&mut self, filter: &str) -> Result<()> { self.filter = (!filter.trim().is_empty()).then_some(filter.to_string()); Ok(()) }
    pub fn set_include_vector(&mut self, include: bool) -> Result<()> { self.include_vector = include; Ok(()) }
    pub fn set_output_fields(&mut self, fields: &[&str]) -> Result<()> { for field in fields { validate_name(field)?; } self.output_fields = Some(fields.iter().map(|s| (*s).to_string()).collect()); Ok(()) }
    pub fn set_hnsw_params(&mut self, params: HnswQueryParams) -> Result<()> { self.params.insert("type".into(), json!("hnsw")); self.params.insert("ef".into(), json!(params.ef)); Ok(()) }
    pub fn set_ivf_params(&mut self, params: IvfQueryParams) -> Result<()> { self.params.insert("type".into(), json!("ivf")); self.params.insert("nprobe".into(), json!(params.nprobe)); Ok(()) }
    pub fn set_ivf_rabitq_params(&mut self, params: IvfRabitqQueryParams) -> Result<()> { self.params.insert("type".into(), json!("ivf_rabitq")); self.params.insert("nprobe".into(), json!(params.nprobe)); Ok(()) }
    pub fn set_flat_params(&mut self, params: FlatQueryParams) -> Result<()> { self.params.insert("type".into(), json!("flat")); self.params.insert("is_using_refiner".into(), json!(params.is_using_refiner)); Ok(()) }
    pub fn set_diskann_params(&mut self, params: DiskannQueryParams) -> Result<()> { if params.list_size <= 0 { return Err(Error::invalid_argument("DiskANN list_size must be positive")); } self.params.insert("type".into(), json!("diskann")); self.params.insert("list_size".into(), json!(params.list_size)); Ok(()) }
}

fn validate_name(name: &str) -> Result<()> { if name.trim().is_empty() || name.contains('\0') { Err(Error::invalid_argument("field name must be non-empty and contain no NUL byte")) } else { Ok(()) } }
fn validate_query_header(name: &str, topk: i32) -> Result<()> { validate_name(name)?; if topk <= 0 { return Err(Error::invalid_argument("topk must be positive")); } Ok(()) }
