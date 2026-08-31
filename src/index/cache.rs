//! Versioned, checksummed serialization for non-authoritative derived indexes.

use super::fts::FtsIndexRegistry;
use super::ordinals::OrdinalTable;
use super::scalar::ScalarIndexRegistry;
use super::{encode_vector, IndexRegistry, VectorIndex, VectorIndexKind};
use crate::doc::DocumentMap;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use crate::types::IndexType;
use bincode::Options;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const CACHE_MAGIC: &[u8; 8] = b"A3SIDX01";
const CACHE_FORMAT_VERSION: u32 = 7;
const HEADER_BYTES: usize = CACHE_MAGIC.len() + 8 + 4;
const MAX_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;

pub(super) mod index_params_serde {
    use crate::schema::IndexParams;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S>(value: &IndexParams, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let json = serde_json::to_string(value).map_err(serde::ser::Error::custom)?;
        json.serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<IndexParams, D::Error>
    where
        D: Deserializer<'de>,
    {
        let json = String::deserialize(deserializer)?;
        serde_json::from_str(&json).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize)]
struct CachePayload {
    format_version: u32,
    source_revision: u64,
    source_identity: String,
    schema_digest: String,
    ordinals: OrdinalTable,
    indexes: BTreeMap<String, VectorIndex>,
    scalar_indexes: ScalarIndexRegistry,
    fts_indexes: FtsIndexRegistry,
}

pub(super) fn encode(
    registry: &IndexRegistry,
    schema: &CollectionSchema,
    source_revision: u64,
    source_identity: &str,
) -> Result<Vec<u8>> {
    let payload = CachePayload {
        format_version: CACHE_FORMAT_VERSION,
        source_revision,
        source_identity: source_identity.to_string(),
        schema_digest: schema.digest(),
        ordinals: registry.ordinals.clone(),
        indexes: registry.indexes.clone(),
        scalar_indexes: registry.scalar_indexes.clone(),
        fts_indexes: registry.fts_indexes.clone(),
    };
    encode_payload(&payload)
}

fn encode_payload(payload: &CachePayload) -> Result<Vec<u8>> {
    let encoded = codec()
        .serialize(payload)
        .map_err(|error| Error::internal(format!("serialize derived index cache: {error}")))?;
    let payload_len = u64::try_from(encoded.len())
        .map_err(|_| Error::resource_exhausted("derived index cache exceeds u64 bytes"))?;
    if payload_len > MAX_PAYLOAD_BYTES {
        return Err(Error::resource_exhausted(format!(
            "derived index cache exceeds the {MAX_PAYLOAD_BYTES}-byte storage limit"
        )));
    }
    let mut output = Vec::with_capacity(HEADER_BYTES.saturating_add(encoded.len()));
    output.extend_from_slice(CACHE_MAGIC);
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend_from_slice(&crc32fast::hash(&encoded).to_le_bytes());
    output.extend_from_slice(&encoded);
    Ok(output)
}

pub(super) fn restore(
    bytes: &[u8],
    schema: &CollectionSchema,
    docs: &DocumentMap,
    source_revision: u64,
    source_identity: &str,
) -> Option<IndexRegistry> {
    let payload = decode_payload(bytes)?;
    if payload.format_version != CACHE_FORMAT_VERSION
        || payload.source_revision != source_revision
        || payload.source_identity != source_identity
        || payload.schema_digest != schema.digest()
        || !payload.ordinals.validates(docs)
        || !validate_indexes(
            schema,
            docs,
            source_revision,
            &payload.ordinals,
            &payload.indexes,
        )
        || !payload
            .scalar_indexes
            .validates(schema, docs, source_revision, &payload.ordinals)
        || !payload
            .fts_indexes
            .validates(schema, docs, source_revision, &payload.ordinals)
    {
        return None;
    }
    Some(IndexRegistry {
        ordinals: payload.ordinals,
        indexes: payload.indexes,
        scalar_indexes: payload.scalar_indexes,
        fts_indexes: payload.fts_indexes,
    })
}

fn decode_payload(bytes: &[u8]) -> Option<CachePayload> {
    if bytes.len() < HEADER_BYTES || &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
        return None;
    }
    let payload_len = u64::from_le_bytes(
        bytes[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 8]
            .try_into()
            .ok()?,
    );
    if payload_len > MAX_PAYLOAD_BYTES {
        return None;
    }
    let payload_len = usize::try_from(payload_len).ok()?;
    if bytes.len() != HEADER_BYTES.checked_add(payload_len)? {
        return None;
    }
    let expected_checksum =
        u32::from_le_bytes(bytes[CACHE_MAGIC.len() + 8..HEADER_BYTES].try_into().ok()?);
    let encoded = &bytes[HEADER_BYTES..];
    if crc32fast::hash(encoded) != expected_checksum {
        return None;
    }
    codec().deserialize(encoded).ok()
}

fn validate_indexes(
    schema: &CollectionSchema,
    docs: &DocumentMap,
    source_revision: u64,
    ordinals: &OrdinalTable,
    indexes: &BTreeMap<String, VectorIndex>,
) -> bool {
    let configured: Vec<_> = schema
        .vectors
        .iter()
        .filter_map(|field| {
            field
                .index_params
                .as_ref()
                .filter(|params| {
                    matches!(
                        params.index_type,
                        IndexType::Hnsw | IndexType::Ivf | IndexType::Vamana
                    )
                })
                .map(|params| (field, params))
        })
        .collect();
    if configured.len() != indexes.len() {
        return false;
    }
    configured.into_iter().all(|(field, params)| {
        indexes.get(&field.name).is_some_and(|index| {
            index.params == *params
                && index.source_revision == source_revision
                && validate_vector_index(index, docs, &field.name, field.dimension, ordinals)
        })
    })
}

fn validate_vector_index(
    index: &VectorIndex,
    docs: &DocumentMap,
    field_name: &str,
    dimension: u32,
    ordinals: &OrdinalTable,
) -> bool {
    let Ok(dimension) = usize::try_from(dimension) else {
        return false;
    };
    let base_ordinals: RoaringTreemap = index.base.vectors.keys().collect();
    let delta_ordinals: RoaringTreemap = index.delta.keys().copied().collect();
    if !index.base.vectors.validates(ordinals.allocated_len())
        || base_ordinals != index.base.vector_ordinals
        || delta_ordinals != index.delta_ordinals
        || !index.tombstones.is_subset(&index.base.vector_ordinals)
        || index.should_compact()
    {
        return false;
    }
    if index.delta.keys().any(|ordinal| {
        index.base.vectors.contains_key(*ordinal) != index.tombstones.contains(*ordinal)
    }) {
        return false;
    }
    let Some(expected_live) = expected_vector_ordinals(docs, field_name, ordinals) else {
        return false;
    };
    let mut actual_live = &index.base.vector_ordinals - &index.tombstones;
    actual_live |= &index.delta_ordinals;
    if actual_live != expected_live
        || index
            .base
            .vectors
            .values()
            .chain(index.delta.values())
            .any(|vector| !vector.validates(dimension))
        || docs.iter().any(|(id, doc)| {
            let Some(vector) = doc.vector(field_name) else {
                return false;
            };
            let Some(ordinal) = ordinals.ordinal(id) else {
                return true;
            };
            let cached = index.delta.get(&ordinal).or_else(|| {
                (!index.tombstones.contains(ordinal))
                    .then(|| index.base.vectors.get(ordinal))
                    .flatten()
            });
            encode_vector(id, field_name, &index.params, vector)
                .ok()
                .as_ref()
                != cached
        })
    {
        return false;
    }
    match &index.base.kind {
        VectorIndexKind::Hnsw(hnsw) => {
            if index.params.index_type != IndexType::Hnsw {
                return false;
            }
            let Some(m) = positive_param(index, "m") else {
                return false;
            };
            let Some(ef_construction) = positive_param(index, "ef_construction") else {
                return false;
            };
            hnsw.validates(&index.base.vectors, m, ef_construction)
        }
        VectorIndexKind::Ivf(ivf) => {
            if index.params.index_type != IndexType::Ivf {
                return false;
            }
            let Some(n_list) = positive_param(index, "n_list") else {
                return false;
            };
            ivf.validates(&index.base.vectors, dimension, n_list)
        }
        VectorIndexKind::Vamana(vamana) => {
            if index.params.index_type != IndexType::Vamana {
                return false;
            }
            let Some(max_degree) = positive_param(index, "max_degree") else {
                return false;
            };
            let Some(search_list_size) = positive_param(index, "search_list_size") else {
                return false;
            };
            let Some(alpha) = finite_param(index, "alpha") else {
                return false;
            };
            vamana.validates(&index.base.vectors, max_degree, search_list_size, alpha)
        }
    }
}

fn expected_vector_ordinals(
    docs: &DocumentMap,
    field_name: &str,
    ordinals: &OrdinalTable,
) -> Option<RoaringTreemap> {
    docs.iter()
        .filter(|(_, doc)| doc.vector(field_name).is_some())
        .map(|(id, _)| ordinals.ordinal(id))
        .collect()
}

fn positive_param(index: &VectorIndex, name: &str) -> Option<usize> {
    index
        .params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .and_then(|value| usize::try_from(value).ok())
}

fn finite_param(index: &VectorIndex, name: &str) -> Option<f64> {
    index
        .params
        .params
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_PAYLOAD_BYTES)
}

#[cfg(test)]
mod tests {
    use super::{
        codec, decode_payload, encode, encode_payload, restore, validate_indexes, CachePayload,
        CACHE_FORMAT_VERSION, CACHE_MAGIC, HEADER_BYTES,
    };
    use crate::doc::{Doc, DocumentMap};
    use crate::index::{quantization::QuantizedVector, IndexRegistry, VectorIndex};
    use crate::{CollectionSchema, DataType, FieldSchema, IndexParams, MetricType};
    use bincode::Options;
    use im::OrdMap;
    use roaring::RoaringTreemap;
    use serde::Serialize;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    #[derive(Serialize)]
    struct LegacyOrdinalTableV2 {
        by_id: OrdMap<String, u64>,
        by_ordinal: OrdMap<u64, String>,
        live: Arc<RoaringTreemap>,
        next: u64,
    }

    #[derive(Serialize)]
    struct LegacyCachePayloadV2 {
        format_version: u32,
        source_revision: u64,
        source_identity: String,
        schema_digest: String,
        ordinals: LegacyOrdinalTableV2,
        indexes: BTreeMap<String, VectorIndex>,
    }

    #[derive(Serialize)]
    struct LegacyCachePayloadV3 {
        format_version: u32,
        source_revision: u64,
        source_identity: String,
        schema_digest: String,
        ordinals: super::OrdinalTable,
        indexes: BTreeMap<String, VectorIndex>,
    }

    fn wrap_payload(encoded: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(HEADER_BYTES + encoded.len());
        output.extend_from_slice(CACHE_MAGIC);
        output.extend_from_slice(
            &u64::try_from(encoded.len())
                .expect("legacy payload length fits u64")
                .to_le_bytes(),
        );
        output.extend_from_slice(&crc32fast::hash(encoded).to_le_bytes());
        output.extend_from_slice(encoded);
        output
    }

    fn legacy_v2_bytes(
        registry: &IndexRegistry,
        schema: &CollectionSchema,
        docs: &DocumentMap,
    ) -> Vec<u8> {
        let by_ordinal: OrdMap<u64, String> = docs
            .keys()
            .enumerate()
            .map(|(ordinal, id)| {
                (
                    u64::try_from(ordinal).expect("fixture ordinal fits u64"),
                    id.clone(),
                )
            })
            .collect();
        let by_id = by_ordinal
            .iter()
            .map(|(ordinal, id)| (id.clone(), *ordinal))
            .collect();
        let live = Arc::new(by_ordinal.keys().copied().collect());
        let payload = LegacyCachePayloadV2 {
            format_version: 2,
            source_revision: 1,
            source_identity: "fixture-source".to_string(),
            schema_digest: schema.digest(),
            ordinals: LegacyOrdinalTableV2 {
                by_id,
                next: u64::try_from(by_ordinal.len()).expect("fixture length fits u64"),
                by_ordinal,
                live,
            },
            indexes: registry.indexes.clone(),
        };
        let encoded = codec()
            .serialize(&payload)
            .expect("legacy cache payload must encode");
        wrap_payload(&encoded)
    }

    fn legacy_v3_bytes(registry: &IndexRegistry, schema: &CollectionSchema) -> Vec<u8> {
        let payload = LegacyCachePayloadV3 {
            format_version: 3,
            source_revision: 1,
            source_identity: "fixture-source".to_string(),
            schema_digest: schema.digest(),
            ordinals: registry.ordinals.clone(),
            indexes: registry.indexes.clone(),
        };
        let encoded = codec()
            .serialize(&payload)
            .expect("legacy cache payload must encode");
        wrap_payload(&encoded)
    }

    fn fixture(params: &IndexParams) -> (CollectionSchema, DocumentMap, IndexRegistry) {
        let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
            .expect("field must be valid");
        field.set_index_params(params).expect("index must be valid");
        let schema = CollectionSchema::builder("cache-fixture")
            .add_field(field)
            .build()
            .expect("schema must be valid");
        let empty = DocumentMap::new();
        let initial = IndexRegistry::build(&schema, &empty, 0).expect("empty index must build");
        let docs: DocumentMap = (0_u16..128)
            .map(|value| {
                let id = format!("doc-{value:03}");
                let mut doc = Doc::with_pk(&id).expect("document must be valid");
                doc.add_vector_f32("embedding", &[f32::from(value), 0.0])
                    .expect("vector must be valid");
                (id, Arc::new(doc))
            })
            .collect();
        let changed = docs.keys().cloned().collect::<BTreeSet<_>>();
        let registry = initial
            .apply_document_changes(&schema, &empty, &docs, 1, &changed)
            .expect("incremental index must build");
        (schema, docs, registry)
    }

    #[test]
    fn malformed_headers_and_checksums_are_cache_misses() {
        assert!(decode_payload(&[]).is_none());
        let mut bytes = vec![0_u8; HEADER_BYTES];
        bytes[..CACHE_MAGIC.len()].copy_from_slice(CACHE_MAGIC);
        bytes[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 8].copy_from_slice(&1_u64.to_le_bytes());
        bytes.push(1);
        assert!(decode_payload(&bytes).is_none());
    }

    #[test]
    fn encoded_ann_generation_round_trips_after_compaction() {
        let params = IndexParams::hnsw(MetricType::L2, 8, 32).expect("params must be valid");
        let (schema, docs, registry) = fixture(&params);
        let bytes = encode(&registry, &schema, 1, "fixture-source").expect("cache must encode");
        assert_eq!(
            bytes,
            encode(&registry, &schema, 1, "fixture-source").expect("cache must re-encode")
        );
        let _: CachePayload = codec()
            .deserialize(&bytes[HEADER_BYTES..])
            .expect("bincode payload must decode");
        let payload = decode_payload(&bytes).expect("cache payload must decode");
        assert!(payload.ordinals.validates(&docs));
        assert!(validate_indexes(
            &schema,
            &docs,
            1,
            &payload.ordinals,
            &payload.indexes
        ));
        assert!(restore(&bytes, &schema, &docs, 1, "fixture-source").is_some());
        assert!(restore(&bytes, &schema, &docs, 1, "different-source").is_none());

        let mut obsolete = decode_payload(&bytes).expect("cache payload must decode");
        obsolete.format_version = CACHE_FORMAT_VERSION - 1;
        let obsolete = encode_payload(&obsolete).expect("obsolete fixture must encode");
        assert!(restore(&obsolete, &schema, &docs, 1, "fixture-source").is_none());
        let legacy = legacy_v2_bytes(&registry, &schema, &docs);
        assert!(bytes.len() < legacy.len());
        assert!(restore(&legacy, &schema, &docs, 1, "fixture-source").is_none());
        let legacy = legacy_v3_bytes(&registry, &schema);
        assert!(restore(&legacy, &schema, &docs, 1, "fixture-source").is_none());

        let mut invalid = payload;
        let index = invalid
            .indexes
            .values_mut()
            .next()
            .expect("fixture index must exist");
        std::sync::Arc::make_mut(&mut index.base).vectors.remove(0);
        let invalid = encode_payload(&invalid).expect("invalid fixture must encode");
        assert!(restore(&invalid, &schema, &docs, 1, "fixture-source").is_none());
    }

    #[test]
    fn ivf_generation_round_trips_through_the_same_cache_contract() {
        let params =
            IndexParams::ivf(MetricType::L2, 16, 5, false).expect("IVF params must be valid");
        let (schema, docs, registry) = fixture(&params);
        let bytes = encode(&registry, &schema, 1, "ivf-source").expect("cache must encode");
        assert!(restore(&bytes, &schema, &docs, 1, "ivf-source").is_some());
    }

    #[test]
    fn cache_vectors_must_match_authoritative_documents() {
        let params = IndexParams::hnsw(MetricType::L2, 8, 32).expect("params must be valid");
        let (schema, docs, registry) = fixture(&params);
        let bytes = encode(&registry, &schema, 1, "fixture-source").expect("cache must encode");
        let mut payload = decode_payload(&bytes).expect("cache payload must decode");
        let vector = std::sync::Arc::make_mut(
            &mut payload
                .indexes
                .values_mut()
                .next()
                .expect("fixture index must exist")
                .base,
        )
        .vectors
        .get_mut(0)
        .expect("fixture vector must exist");
        let QuantizedVector::F32(values) = vector else {
            panic!("fixture must use unquantized vectors");
        };
        values[0] = 1.0;
        let drifted = encode_payload(&payload).expect("drifted cache must encode");
        assert!(restore(&drifted, &schema, &docs, 1, "fixture-source").is_none());
    }
}
