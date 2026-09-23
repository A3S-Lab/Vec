//! Revisioned scalar postings and bitmap filter evaluation.

use super::ordinals::{OrdinalSet, OrdinalTable};
use crate::doc::{DocumentMap, FieldValue};
use crate::error::{Error, Result};
use crate::filter::{like_match, CmpOp, FilterExpr, Literal};
use crate::schema::{CollectionSchema, FieldSchema, IndexParams};
use crate::stats::IndexStat;
use crate::types::{DataType, IndexType};
use im::OrdMap;
use roaring::RoaringTreemap;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::Arc;

const CONJUNCTION_EARLY_STOP: u64 = 4_096;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct ScalarIndexRegistry {
    source_revision: u64,
    indexes: BTreeMap<String, ScalarIndex>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScalarCandidates {
    pub(super) ids: OrdinalSet,
    pub(super) exact: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ScalarIndex {
    data_type: DataType,
    #[serde(with = "super::cache::index_params_serde")]
    params: IndexParams,
    values: OrdMap<ScalarKey, Arc<RoaringTreemap>>,
    present: Arc<RoaringTreemap>,
    non_null: Arc<RoaringTreemap>,
}

#[derive(Debug)]
struct BitmapEvaluation {
    bitmap: RoaringTreemap,
    exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
enum ScalarKey {
    Null,
    Bool(bool),
    Number(ScalarNumber),
    String(String),
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct ScalarNumber(f64);

impl ScalarCandidates {
    pub(crate) fn ids(&self) -> impl Iterator<Item = &str> {
        self.ids.ids()
    }

    pub(super) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(super) fn retain_ids(&mut self, keep: impl FnMut(&str) -> bool) {
        self.ids.retain_ids(keep);
    }

    pub(super) fn into_ids(self) -> OrdinalSet {
        self.ids
    }
}

impl ScalarNumber {
    fn new(value: f64) -> Option<Self> {
        value
            .is_finite()
            .then_some(Self(if value == 0.0 { 0.0 } else { value }))
    }
}

impl PartialEq for ScalarNumber {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ScalarNumber {}

impl PartialOrd for ScalarNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScalarNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl ScalarIndexRegistry {
    pub(super) fn build(
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        let configured: Vec<_> = schema
            .fields
            .iter()
            .filter_map(|field| {
                field
                    .index_params
                    .as_ref()
                    .filter(|params| params.index_type == IndexType::Invert)
                    .map(|params| (field, params))
            })
            .collect();
        if configured.is_empty() {
            return Ok(Self {
                source_revision,
                ..Self::default()
            });
        }

        let mut indexes = BTreeMap::new();
        for (field, params) in configured {
            indexes.insert(
                field.name.clone(),
                ScalarIndex::build(&field.name, field.data_type, params, docs, ordinals)?,
            );
        }
        Ok(Self {
            source_revision,
            indexes,
        })
    }

    pub(super) fn rebuild_field(
        &self,
        field: &FieldSchema,
        docs: &DocumentMap,
        source_revision: u64,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        let params = field
            .index_params
            .as_ref()
            .filter(|params| params.index_type == IndexType::Invert)
            .ok_or_else(|| Error::internal("scalar rebuild requires an inverted index field"))?;
        let mut next = self.clone();
        next.source_revision = source_revision;
        next.indexes.insert(
            field.name.clone(),
            ScalarIndex::build(&field.name, field.data_type, params, docs, ordinals)?,
        );
        Ok(next)
    }

    pub(super) fn apply_document_changes(
        &self,
        schema: &CollectionSchema,
        previous_docs: &DocumentMap,
        docs: &DocumentMap,
        source_revision: u64,
        changed_ids: &BTreeSet<String>,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        if !self.matches_schema(schema) {
            return Self::build(schema, docs, source_revision, ordinals);
        }
        if self.indexes.is_empty() {
            return Ok(Self {
                source_revision,
                ..Self::default()
            });
        }

        let mut indexes = BTreeMap::new();
        for field in &schema.fields {
            let Some(params) = field
                .index_params
                .as_ref()
                .filter(|params| params.index_type == IndexType::Invert)
            else {
                continue;
            };
            let Some(current) = self
                .indexes
                .get(&field.name)
                .filter(|index| index.params == *params && index.data_type == field.data_type)
            else {
                return Self::build(schema, docs, source_revision, ordinals);
            };
            let mut next = current.clone();
            for id in changed_ids {
                let previous = previous_docs.get(id).and_then(|doc| doc.field(&field.name));
                let current = docs.get(id).and_then(|doc| doc.field(&field.name));
                if previous == current {
                    continue;
                }
                let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                    Error::internal(format!("scalar ordinal is missing for document '{id}'"))
                })?;
                if let Some(value) = previous {
                    next.remove_value(value, ordinal)?;
                }
                if let Some(value) = current {
                    next.insert_value(value, ordinal)?;
                }
            }
            indexes.insert(field.name.clone(), next);
        }
        Ok(Self {
            source_revision,
            indexes,
        })
    }

    pub(super) fn candidates(
        &self,
        source_revision: u64,
        filter: &FilterExpr,
        ordinals: &OrdinalTable,
    ) -> Option<ScalarCandidates> {
        if self.source_revision != source_revision || self.indexes.is_empty() {
            return None;
        }
        let evaluation = self.evaluate(filter, ordinals)?;
        Some(ScalarCandidates {
            ids: OrdinalSet::new(ordinals, evaluation.bitmap),
            exact: evaluation.exact,
        })
    }

    pub(super) fn stats(&self) -> Vec<IndexStat> {
        self.indexes
            .iter()
            .map(|(name, index)| IndexStat {
                name: name.clone(),
                index_type: IndexType::Invert,
                completeness: 1.0,
                source_revision: self.source_revision,
                document_count: index.present.len(),
                estimated_payload_bytes: None,
                state: "ready".into(),
            })
            .collect()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    pub(super) fn validates(
        &self,
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_revision: u64,
        ordinals: &OrdinalTable,
    ) -> bool {
        self.source_revision == source_revision
            && self.matches_schema(schema)
            && self
                .indexes
                .iter()
                .all(|(field_name, index)| index.validates(field_name, docs, ordinals))
    }

    fn matches_schema(&self, schema: &CollectionSchema) -> bool {
        let configured: Vec<_> = schema
            .fields
            .iter()
            .filter_map(|field| {
                field
                    .index_params
                    .as_ref()
                    .filter(|params| params.index_type == IndexType::Invert)
                    .map(|params| (&field.name, field.data_type, params))
            })
            .collect();
        configured.len() == self.indexes.len()
            && configured.into_iter().all(|(name, data_type, params)| {
                self.indexes
                    .get(name)
                    .is_some_and(|index| index.data_type == data_type && index.params == *params)
            })
    }

    fn evaluate(&self, filter: &FilterExpr, ordinals: &OrdinalTable) -> Option<BitmapEvaluation> {
        match filter {
            FilterExpr::And(left, right) => self.evaluate_conjunction(left, right, ordinals),
            FilterExpr::Or(left, right) => {
                let left = self.evaluate(left, ordinals)?;
                let right = self.evaluate(right, ordinals)?;
                Some(BitmapEvaluation {
                    bitmap: &left.bitmap | &right.bitmap,
                    exact: left.exact && right.exact,
                })
            }
            FilterExpr::Not(inner) => {
                let inner = self.evaluate(inner, ordinals)?;
                inner.exact.then(|| BitmapEvaluation {
                    bitmap: ordinals.live() - &inner.bitmap,
                    exact: true,
                })
            }
            FilterExpr::Compare { field, op, value } => self
                .indexes
                .get(field)?
                .compare(*op, value)
                .map(BitmapEvaluation::exact),
            FilterExpr::In {
                field,
                values,
                negated,
            } => Some(BitmapEvaluation::exact(self.indexes.get(field)?.in_values(
                values,
                *negated,
                ordinals.live(),
            ))),
            FilterExpr::Like {
                field,
                pattern,
                negated,
            } => self
                .indexes
                .get(field)?
                .wildcard(
                    |value| like_match(value, pattern),
                    like_literal_prefix(pattern).as_deref(),
                    *negated,
                    ordinals.live(),
                )
                .map(BitmapEvaluation::exact),
            FilterExpr::IsNull { field, negated } => {
                let index = self.indexes.get(field)?;
                Some(BitmapEvaluation::exact(if *negated {
                    (*index.non_null).clone()
                } else {
                    ordinals.live() - &*index.non_null
                }))
            }
            FilterExpr::ContainAll { .. } => None,
            FilterExpr::HasPrefix {
                field,
                prefix,
                negated,
            } => self
                .indexes
                .get(field)?
                .wildcard(
                    |value| value.starts_with(prefix),
                    Some(prefix),
                    *negated,
                    ordinals.live(),
                )
                .map(BitmapEvaluation::exact),
            FilterExpr::HasSuffix {
                field,
                suffix,
                negated,
            } => self
                .indexes
                .get(field)?
                .wildcard(
                    |value| value.ends_with(suffix),
                    None,
                    *negated,
                    ordinals.live(),
                )
                .map(BitmapEvaluation::exact),
        }
    }

    fn evaluate_conjunction(
        &self,
        left: &FilterExpr,
        right: &FilterExpr,
        ordinals: &OrdinalTable,
    ) -> Option<BitmapEvaluation> {
        let left = self.evaluate(left, ordinals);
        if left.as_ref().is_some_and(BitmapEvaluation::is_empty) {
            return Some(BitmapEvaluation::exact(RoaringTreemap::new()));
        }
        if left.as_ref().is_some_and(BitmapEvaluation::is_selective) {
            return left.map(BitmapEvaluation::into_conservative);
        }

        let right = self.evaluate(right, ordinals);
        if right.as_ref().is_some_and(BitmapEvaluation::is_empty) {
            return Some(BitmapEvaluation::exact(RoaringTreemap::new()));
        }
        if right.as_ref().is_some_and(BitmapEvaluation::is_selective) {
            return right.map(BitmapEvaluation::into_conservative);
        }

        match (left, right) {
            (Some(left), Some(right)) => Some(BitmapEvaluation {
                bitmap: &left.bitmap & &right.bitmap,
                exact: left.exact && right.exact,
            }),
            (Some(indexed), None) | (None, Some(indexed)) => Some(indexed.into_conservative()),
            (None, None) => None,
        }
    }
}

impl BitmapEvaluation {
    fn exact(bitmap: RoaringTreemap) -> Self {
        Self {
            bitmap,
            exact: true,
        }
    }

    fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    fn is_selective(&self) -> bool {
        self.bitmap.len() <= CONJUNCTION_EARLY_STOP
    }

    fn into_conservative(self) -> Self {
        Self {
            bitmap: self.bitmap,
            exact: false,
        }
    }
}

impl ScalarIndex {
    fn build(
        field_name: &str,
        data_type: DataType,
        params: &IndexParams,
        docs: &DocumentMap,
        ordinals: &OrdinalTable,
    ) -> Result<Self> {
        let mut values = BTreeMap::<ScalarKey, RoaringTreemap>::new();
        let mut present = RoaringTreemap::new();
        let mut non_null = RoaringTreemap::new();
        for (id, doc) in docs {
            if let Some(value) = doc.field(field_name) {
                let ordinal = ordinals.ordinal(id).ok_or_else(|| {
                    Error::internal(format!("scalar ordinal is missing for document '{id}'"))
                })?;
                let key = scalar_key(value).ok_or_else(|| {
                    Error::internal(format!(
                        "{:?} value cannot enter an inverted scalar index",
                        value.data_type()
                    ))
                })?;
                present.insert(ordinal);
                if key != ScalarKey::Null {
                    non_null.insert(ordinal);
                }
                values.entry(key).or_default().insert(ordinal);
            }
        }
        Ok(Self {
            data_type,
            params: params.clone(),
            values: values
                .into_iter()
                .map(|(key, bitmap)| (key, Arc::new(bitmap)))
                .collect(),
            present: Arc::new(present),
            non_null: Arc::new(non_null),
        })
    }

    fn insert_value(&mut self, value: &FieldValue, ordinal: u64) -> Result<()> {
        let key = scalar_key(value).ok_or_else(|| {
            Error::internal(format!(
                "{:?} value cannot enter an inverted scalar index",
                value.data_type()
            ))
        })?;
        Arc::make_mut(&mut self.present).insert(ordinal);
        if key != ScalarKey::Null {
            Arc::make_mut(&mut self.non_null).insert(ordinal);
        }
        if let Some(bitmap) = self.values.get_mut(&key) {
            Arc::make_mut(bitmap).insert(ordinal);
        } else {
            let mut bitmap = RoaringTreemap::new();
            bitmap.insert(ordinal);
            self.values.insert(key, Arc::new(bitmap));
        }
        Ok(())
    }

    fn remove_value(&mut self, value: &FieldValue, ordinal: u64) -> Result<()> {
        let key = scalar_key(value).ok_or_else(|| {
            Error::internal(format!(
                "{:?} value cannot leave an inverted scalar index",
                value.data_type()
            ))
        })?;
        Arc::make_mut(&mut self.present).remove(ordinal);
        Arc::make_mut(&mut self.non_null).remove(ordinal);
        let empty = self.values.get_mut(&key).is_some_and(|bitmap| {
            let bitmap = Arc::make_mut(bitmap);
            bitmap.remove(ordinal);
            bitmap.is_empty()
        });
        if empty {
            self.values.remove(&key);
        }
        Ok(())
    }

    fn validates(&self, field_name: &str, docs: &DocumentMap, ordinals: &OrdinalTable) -> bool {
        if !self.present.is_subset(ordinals.live())
            || !self.non_null.is_subset(&self.present)
            || self.values.values().any(|bitmap| bitmap.is_empty())
        {
            return false;
        }

        let mut covered = RoaringTreemap::new();
        for (key, bitmap) in &self.values {
            if !key.validates(self.data_type)
                || !bitmap.is_subset(&self.present)
                || !(&covered & &**bitmap).is_empty()
            {
                return false;
            }
            covered |= &**bitmap;
        }
        if covered != *self.present {
            return false;
        }
        let expected_non_null = self.values.get(&ScalarKey::Null).map_or_else(
            || (*self.present).clone(),
            |nulls| &*self.present - &**nulls,
        );
        if expected_non_null != *self.non_null {
            return false;
        }

        docs.iter().all(|(id, doc)| {
            let Some(ordinal) = ordinals.ordinal(id) else {
                return false;
            };
            let Some(value) = doc.field(field_name) else {
                return !self.present.contains(ordinal);
            };
            scalar_key(value).is_some_and(|key| {
                self.values
                    .get(&key)
                    .is_some_and(|bitmap| bitmap.contains(ordinal))
            })
        })
    }

    fn compare(&self, op: CmpOp, literal: &Literal) -> Option<RoaringTreemap> {
        let equal = self.equal(literal);
        match op {
            CmpOp::Eq => Some(equal),
            CmpOp::Ne => Some(&*self.present - &equal),
            CmpOp::Gt | CmpOp::Ge | CmpOp::Lt | CmpOp::Le => {
                if !self.range_enabled() {
                    return None;
                }
                let literal = literal_key(literal)?;
                let mut result = RoaringTreemap::new();
                let bounds = match op {
                    CmpOp::Gt => (Excluded(literal.clone()), Unbounded),
                    CmpOp::Ge => (Included(literal.clone()), Unbounded),
                    CmpOp::Lt => (Unbounded, Excluded(literal.clone())),
                    CmpOp::Le => (Unbounded, Included(literal.clone())),
                    CmpOp::Eq | CmpOp::Ne => return None,
                };
                for (key, bitmap) in self.values.range(bounds) {
                    if comparable_order(key, &literal).is_some() {
                        result |= &**bitmap;
                    }
                }
                Some(result)
            }
        }
    }

    fn in_values(
        &self,
        literals: &[Literal],
        negated: bool,
        live: &RoaringTreemap,
    ) -> RoaringTreemap {
        let mut result = RoaringTreemap::new();
        for literal in literals {
            result |= self.equal(literal);
        }
        if negated {
            live - &result
        } else {
            result
        }
    }

    fn equal(&self, literal: &Literal) -> RoaringTreemap {
        literal_key(literal)
            .and_then(|key| self.values.get(&key))
            .map_or_else(RoaringTreemap::new, |bitmap| (**bitmap).clone())
    }

    fn wildcard(
        &self,
        matches: impl Fn(&str) -> bool,
        literal_prefix: Option<&str>,
        negated: bool,
        live: &RoaringTreemap,
    ) -> Option<RoaringTreemap> {
        if !self.wildcard_enabled() || self.data_type != DataType::String {
            return None;
        }
        let mut result = RoaringTreemap::new();
        if let Some(prefix) = literal_prefix.filter(|prefix| !prefix.is_empty()) {
            let start = ScalarKey::String(prefix.to_string());
            for (key, bitmap) in self.values.range(start..) {
                let ScalarKey::String(value) = key else {
                    continue;
                };
                if !value.starts_with(prefix) {
                    break;
                }
                if matches(value) {
                    result |= &**bitmap;
                }
            }
        } else {
            for (key, bitmap) in &self.values {
                if let ScalarKey::String(value) = key {
                    if matches(value) {
                        result |= &**bitmap;
                    }
                }
            }
        }
        Some(if negated { live - &result } else { result })
    }

    fn range_enabled(&self) -> bool {
        self.params
            .params
            .get("enable_range_optimization")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }

    fn wildcard_enabled(&self) -> bool {
        self.params
            .params
            .get("enable_wildcard")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }
}

impl ScalarKey {
    fn validates(&self, data_type: DataType) -> bool {
        match self {
            Self::Null => true,
            Self::Bool(_) => data_type == DataType::Bool,
            Self::Number(value) => {
                value.0.is_finite()
                    && matches!(
                        data_type,
                        DataType::Int32
                            | DataType::Int64
                            | DataType::Uint32
                            | DataType::Uint64
                            | DataType::Float
                            | DataType::Double
                    )
            }
            Self::String(_) => matches!(data_type, DataType::Binary | DataType::String),
        }
    }
}

fn scalar_key(value: &FieldValue) -> Option<ScalarKey> {
    match value.to_json() {
        serde_json::Value::Null => Some(ScalarKey::Null),
        serde_json::Value::Bool(value) => Some(ScalarKey::Bool(value)),
        serde_json::Value::Number(value) => value
            .as_f64()
            .and_then(ScalarNumber::new)
            .map(ScalarKey::Number),
        serde_json::Value::String(value) => Some(ScalarKey::String(value)),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => None,
    }
}

fn literal_key(literal: &Literal) -> Option<ScalarKey> {
    match literal {
        Literal::Str(value) => Some(ScalarKey::String(value.clone())),
        Literal::Num(value) => ScalarNumber::new(*value).map(ScalarKey::Number),
        Literal::Bool(value) => Some(ScalarKey::Bool(*value)),
        Literal::Null => Some(ScalarKey::Null),
    }
}

fn comparable_order(left: &ScalarKey, right: &ScalarKey) -> Option<Ordering> {
    match (left, right) {
        (ScalarKey::Number(left), ScalarKey::Number(right)) => Some(left.cmp(right)),
        (ScalarKey::String(left), ScalarKey::String(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

fn like_literal_prefix(pattern: &str) -> Option<String> {
    let prefix: String = pattern
        .chars()
        .take_while(|character| !matches!(character, '%' | '*' | '_'))
        .collect();
    (!prefix.is_empty()).then_some(prefix)
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use super::{like_literal_prefix, literal_key, scalar_key, ScalarKey, ScalarNumber};
    use crate::doc::FieldValue;
    use crate::filter::{like_match, Literal};
    use crate::types::DataType;
    use std::cmp::Ordering;

    #[test]
    fn wildcard_matching_agrees_with_filter_syntax() {
        assert!(like_match("src/lib.rs", "src/%"));
        assert!(like_match("main.rs", "*.rs"));
        assert!(like_match("abc", "a_c"));
        assert!(!like_match("src/lib.rs", "tests/%"));
        assert!(like_match("abcdef", "a%c%f"));
        assert!(like_match("abc", "%%%"));
        assert!(!like_match("abc", "ab%d"));
        assert!(!like_match("ab", "a_c"));
        assert!(like_match("", "%"));
        assert!(!like_match("", "_"));
    }

    #[test]
    fn like_prefix_stops_before_the_first_wildcard() {
        assert_eq!(like_literal_prefix("src/%/lib.rs").as_deref(), Some("src/"));
        assert_eq!(like_literal_prefix("main.rs").as_deref(), Some("main.rs"));
        assert_eq!(like_literal_prefix("_ain.rs"), None);
        assert_eq!(like_literal_prefix("*.rs"), None);
    }

    #[test]
    fn scalar_number_and_literal_keys_reject_non_finite_values() {
        assert!(ScalarNumber::new(0.0).is_some());
        assert!(ScalarNumber::new(-1.5).is_some());
        assert!(ScalarNumber::new(f64::NAN).is_none());
        assert!(ScalarNumber::new(f64::INFINITY).is_none());
        assert_eq!(
            literal_key(&Literal::Num(1.0))
                .map(|key| format!("{key:?}"))
                .is_some(),
            true
        );
        assert!(literal_key(&Literal::Num(f64::NAN)).is_none());
        assert!(literal_key(&Literal::Null).is_some());
        assert!(literal_key(&Literal::Bool(true)).is_some());
        assert!(literal_key(&Literal::Str("x".into())).is_some());
    }

    #[test]
    fn scalar_number_ord_collapses_signed_zero_and_orders_finite_values() {
        let zero = ScalarNumber::new(0.0).expect("zero");
        let neg_zero = ScalarNumber::new(-0.0).expect("neg zero");
        let low = ScalarNumber::new(-2.5).expect("low");
        let high = ScalarNumber::new(3.0).expect("high");
        assert_eq!(zero, neg_zero);
        assert_eq!(zero.partial_cmp(&neg_zero), Some(Ordering::Equal));
        assert_eq!(low.cmp(&high), Ordering::Less);
        assert_eq!(high.cmp(&low), Ordering::Greater);
        assert_eq!(low.partial_cmp(&high), Some(Ordering::Less));
        let mut values = vec![high, low, zero];
        values.sort();
        assert_eq!(values, vec![low, zero, high]);
    }

    #[test]
    fn scalar_key_validation_and_json_mapping_cover_supported_shapes() {
        assert!(ScalarKey::Null.validates(DataType::String));
        assert!(ScalarKey::Bool(true).validates(DataType::Bool));
        assert!(!ScalarKey::Bool(true).validates(DataType::String));
        let number = ScalarKey::Number(ScalarNumber::new(1.0).expect("number"));
        assert!(number.validates(DataType::Int32));
        assert!(number.validates(DataType::Double));
        assert!(!number.validates(DataType::String));
        assert!(ScalarKey::String("x".into()).validates(DataType::String));
        assert!(ScalarKey::String("x".into()).validates(DataType::Binary));
        assert!(!ScalarKey::String("x".into()).validates(DataType::Bool));

        assert_eq!(scalar_key(&FieldValue::Null), Some(ScalarKey::Null));
        assert_eq!(
            scalar_key(&FieldValue::Bool(false)),
            Some(ScalarKey::Bool(false))
        );
        assert_eq!(
            scalar_key(&FieldValue::Int32(7)),
            Some(ScalarKey::Number(ScalarNumber::new(7.0).expect("7")))
        );
        assert_eq!(
            scalar_key(&FieldValue::String("ok".into())),
            Some(ScalarKey::String("ok".into()))
        );
        assert!(scalar_key(&FieldValue::ArrayString(vec!["a".into()])).is_none());
    }
}
