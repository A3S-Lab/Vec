//! Versioned, checksummed serialization for non-authoritative derived indexes.

use super::fts::FtsIndexRegistry;
use super::ordinals::OrdinalTable;
use super::scalar::ScalarIndexRegistry;
use super::{encode_vector, IndexRegistry, VectorIndex, VectorIndexKind};
use crate::config::IoBackend;
use crate::doc::DocumentMap;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use crate::storage::PositionedFile;
use crate::types::IndexType;
use bincode::Options;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const CACHE_MAGIC: &[u8; 8] = b"A3SIDX01";
const CACHE_FORMAT_VERSION: u32 = 10;
const HEADER_BYTES: usize = CACHE_MAGIC.len() + 8 + 4;
pub(crate) use crate::storage_ceilings::DEFAULT_INDEX_CACHE_BYTES as MAX_PAYLOAD_BYTES;

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

#[allow(dead_code)]
pub(super) fn encode(
    registry: &IndexRegistry,
    schema: &CollectionSchema,
    source_revision: u64,
    source_identity: &str,
) -> Result<Vec<u8>> {
    encode_with_limit(
        registry,
        schema,
        source_revision,
        source_identity,
        MAX_PAYLOAD_BYTES,
    )
}

pub(super) fn encode_with_limit(
    registry: &IndexRegistry,
    schema: &CollectionSchema,
    source_revision: u64,
    source_identity: &str,
    max_payload_bytes: u64,
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
    encode_payload(&payload, max_payload_bytes)
}

fn encode_payload(payload: &CachePayload, max_payload_bytes: u64) -> Result<Vec<u8>> {
    let encoded = codec()
        .serialize(payload)
        .map_err(|error| Error::internal(format!("serialize derived index cache: {error}")))?;
    let payload_len = u64::try_from(encoded.len())
        .map_err(|_| Error::resource_exhausted("derived index cache exceeds u64 bytes"))?;
    if payload_len > max_payload_bytes {
        return Err(Error::resource_exhausted(format!(
            "derived index cache exceeds the {max_payload_bytes}-byte storage limit"
        )));
    }
    let mut output = Vec::with_capacity(HEADER_BYTES.saturating_add(encoded.len()));
    output.extend_from_slice(CACHE_MAGIC);
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend_from_slice(&crc32fast::hash(&encoded).to_le_bytes());
    output.extend_from_slice(&encoded);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn restore(
    bytes: &[u8],
    diskann_file: Option<PositionedFile>,
    io_backend: IoBackend,
    schema: &CollectionSchema,
    docs: &DocumentMap,
    source_revision: u64,
    source_identity: &str,
    max_payload_bytes: u64,
    max_diskann_file_bytes: u64,
) -> Option<IndexRegistry> {
    let payload = decode_payload(bytes, max_payload_bytes)?;
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
    let registry = IndexRegistry {
        ordinals: payload.ordinals,
        indexes: payload.indexes,
        scalar_indexes: payload.scalar_indexes,
        fts_indexes: payload.fts_indexes,
    };
    let mut registry = registry;
    super::diskann::attach(
        diskann_file,
        io_backend,
        &mut registry,
        schema,
        source_revision,
        source_identity,
        max_diskann_file_bytes,
    )
    .then_some(registry)
}

fn decode_payload(bytes: &[u8], max_payload_bytes: u64) -> Option<CachePayload> {
    if bytes.len() < HEADER_BYTES || &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
        return None;
    }
    let payload_len = u64::from_le_bytes(
        bytes[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 8]
            .try_into()
            .ok()?,
    );
    if payload_len > max_payload_bytes {
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
                    super::builds_packed_vector_index(params.index_type, field.data_type)
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
    validate_vector_kind(index, dimension)
}

fn validate_vector_kind(index: &VectorIndex, dimension: usize) -> bool {
    match &index.base.kind {
        VectorIndexKind::Flat(_) => index.params.index_type == IndexType::Flat,
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
        VectorIndexKind::HnswRabitq(hnsw) => validate_hnsw_rabitq(index, hnsw, dimension),
        VectorIndexKind::Ivf(ivf) => {
            if index.params.index_type != IndexType::Ivf {
                return false;
            }
            let Some(n_list) = positive_param(index, "n_list") else {
                return false;
            };
            let Some(use_soar) = boolean_param(index, "use_soar") else {
                return false;
            };
            ivf.validates(&index.base.vectors, dimension, n_list, use_soar)
        }
        VectorIndexKind::IvfRabitq(ivf) => validate_ivf_rabitq(index, ivf, dimension),
        VectorIndexKind::Diskann(diskann) => {
            if index.params.index_type != IndexType::Diskann {
                return false;
            }
            let Some(max_degree) = positive_param(index, "max_degree") else {
                return false;
            };
            let Some(list_size) = positive_param(index, "list_size") else {
                return false;
            };
            let Some(pq_chunk_num) = nonnegative_param(index, "pq_chunk_num") else {
                return false;
            };
            let Some(alpha) = finite_param(index, "alpha") else {
                return false;
            };
            diskann.validates(
                &index.base.vectors,
                dimension,
                max_degree,
                list_size,
                pq_chunk_num,
                alpha,
            )
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
            let Some(max_occlusion) = nonnegative_param(index, "max_occlusion") else {
                return false;
            };
            let Some(saturate) = boolean_param(index, "saturate") else {
                return false;
            };
            vamana.validates(
                &index.base.vectors,
                max_degree,
                search_list_size,
                alpha,
                max_occlusion,
                saturate,
            )
        }
    }
}

fn validate_hnsw_rabitq(
    index: &VectorIndex,
    hnsw: &super::rabitq_index::HnswRabitqIndex,
    dimension: usize,
) -> bool {
    if index.params.index_type != IndexType::HnswRabitq {
        return false;
    }
    let Some(m) = positive_param(index, "m") else {
        return false;
    };
    let Some(ef_construction) = positive_param(index, "ef_construction") else {
        return false;
    };
    let Some(total_bits) = positive_param(index, "total_bits") else {
        return false;
    };
    let Some(num_clusters) = positive_param(index, "num_clusters") else {
        return false;
    };
    let Some(sample_count) = nonnegative_param(index, "sample_count") else {
        return false;
    };
    hnsw.validates(
        &index.base.vectors,
        dimension,
        m,
        ef_construction,
        total_bits,
        num_clusters,
        sample_count,
        index.params.metric_type,
    )
}

fn validate_ivf_rabitq(
    index: &VectorIndex,
    ivf: &super::rabitq_index::IvfRabitqIndex,
    dimension: usize,
) -> bool {
    if index.params.index_type != IndexType::IvfRabitq {
        return false;
    }
    let Some(n_list) = positive_param(index, "n_list") else {
        return false;
    };
    let Some(total_bits) = positive_param(index, "total_bits") else {
        return false;
    };
    let Some(sample_count) = nonnegative_param(index, "sample_count") else {
        return false;
    };
    ivf.validates(
        &index.base.vectors,
        dimension,
        n_list,
        total_bits,
        sample_count,
        index.params.metric_type,
    )
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

fn nonnegative_param(index: &VectorIndex, name: &str) -> Option<usize> {
    index
        .params
        .params
        .get(name)
        .and_then(serde_json::Value::as_u64)
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

fn boolean_param(index: &VectorIndex, name: &str) -> Option<bool> {
    index
        .params
        .params
        .get(name)
        .and_then(serde_json::Value::as_bool)
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_PAYLOAD_BYTES)
}

#[cfg(test)]
#[allow(clippy::too_many_lines, clippy::type_complexity)]
mod tests {
    use super::{
        codec, decode_payload, encode, encode_payload, restore, validate_indexes, CachePayload,
        CACHE_FORMAT_VERSION, CACHE_MAGIC, HEADER_BYTES, MAX_PAYLOAD_BYTES,
    };
    use crate::doc::{Doc, DocumentMap};
    use crate::index::{quantization::QuantizedVector, IndexRegistry, VectorIndex};
    use crate::{CollectionSchema, DataType, FieldSchema, IndexParams, IoBackend, MetricType};
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

    fn restore_fixture(
        bytes: &[u8],
        schema: &CollectionSchema,
        docs: &DocumentMap,
        source_identity: &str,
    ) -> Option<IndexRegistry> {
        restore(
            bytes,
            None,
            IoBackend::Positioned,
            schema,
            docs,
            1,
            source_identity,
            MAX_PAYLOAD_BYTES,
            crate::storage_ceilings::DEFAULT_DISKANN_FILE_BYTES,
        )
    }

    #[test]
    fn malformed_headers_and_checksums_are_cache_misses() {
        assert!(decode_payload(&[], MAX_PAYLOAD_BYTES).is_none());
        let mut bytes = vec![0_u8; HEADER_BYTES];
        bytes[..CACHE_MAGIC.len()].copy_from_slice(CACHE_MAGIC);
        bytes[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 8].copy_from_slice(&1_u64.to_le_bytes());
        bytes.push(1);
        assert!(decode_payload(&bytes, MAX_PAYLOAD_BYTES).is_none());
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
        let payload = decode_payload(&bytes, MAX_PAYLOAD_BYTES).expect("cache payload must decode");
        assert!(payload.ordinals.validates(&docs));
        assert!(validate_indexes(
            &schema,
            &docs,
            1,
            &payload.ordinals,
            &payload.indexes
        ));
        assert!(restore_fixture(&bytes, &schema, &docs, "fixture-source").is_some());
        assert!(restore_fixture(&bytes, &schema, &docs, "different-source").is_none());

        let mut obsolete =
            decode_payload(&bytes, MAX_PAYLOAD_BYTES).expect("cache payload must decode");
        obsolete.format_version = CACHE_FORMAT_VERSION - 1;
        let obsolete =
            encode_payload(&obsolete, MAX_PAYLOAD_BYTES).expect("obsolete fixture must encode");
        assert!(restore_fixture(&obsolete, &schema, &docs, "fixture-source").is_none());
        let legacy = legacy_v2_bytes(&registry, &schema, &docs);
        assert!(bytes.len() < legacy.len());
        assert!(restore_fixture(&legacy, &schema, &docs, "fixture-source").is_none());
        let legacy = legacy_v3_bytes(&registry, &schema);
        assert!(restore_fixture(&legacy, &schema, &docs, "fixture-source").is_none());

        let mut invalid = payload;
        let index = invalid
            .indexes
            .values_mut()
            .next()
            .expect("fixture index must exist");
        std::sync::Arc::make_mut(&mut index.base).vectors.remove(0);
        let invalid =
            encode_payload(&invalid, MAX_PAYLOAD_BYTES).expect("invalid fixture must encode");
        assert!(restore_fixture(&invalid, &schema, &docs, "fixture-source").is_none());
    }

    #[test]
    fn ivf_and_soar_generations_round_trip_through_the_same_cache_contract() {
        for use_soar in [false, true] {
            let params = IndexParams::ivf(MetricType::L2, 16, 5, use_soar)
                .expect("IVF params must be valid");
            let (schema, docs, registry) = fixture(&params);
            let bytes = encode(&registry, &schema, 1, "ivf-source").expect("cache must encode");
            assert!(restore_fixture(&bytes, &schema, &docs, "ivf-source").is_some());
        }
    }

    #[test]
    fn cache_vectors_must_match_authoritative_documents() {
        let params = IndexParams::hnsw(MetricType::L2, 8, 32).expect("params must be valid");
        let (schema, docs, registry) = fixture(&params);
        let bytes = encode(&registry, &schema, 1, "fixture-source").expect("cache must encode");
        let mut payload =
            decode_payload(&bytes, MAX_PAYLOAD_BYTES).expect("cache payload must decode");
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
        let drifted =
            encode_payload(&payload, MAX_PAYLOAD_BYTES).expect("drifted cache must encode");
        assert!(restore_fixture(&drifted, &schema, &docs, "fixture-source").is_none());
    }

    fn assert_restore_rejects_after(
        params: &IndexParams,
        source: &str,
        mutate: impl FnOnce(&mut CachePayload),
    ) {
        let (mut schema, docs, registry) = if params.index_type == crate::types::IndexType::Flat {
            // Flat is rebuild-only; incremental apply leaves the packed index absent.
            let mut field = FieldSchema::new("embedding", DataType::VectorFp32, false, 2)
                .expect("field must be valid");
            field.set_index_params(params).expect("index must be valid");
            let schema = CollectionSchema::builder("cache-fixture-flat")
                .add_field(field)
                .build()
                .expect("schema must be valid");
            let docs: DocumentMap = (0_u16..128)
                .map(|value| {
                    let id = format!("doc-{value:03}");
                    let mut doc = Doc::with_pk(&id).expect("document must be valid");
                    doc.add_vector_f32("embedding", &[f32::from(value), 0.0])
                        .expect("vector must be valid");
                    (id, Arc::new(doc))
                })
                .collect();
            let registry = IndexRegistry::build(&schema, &docs, 1).expect("flat index must build");
            (schema, docs, registry)
        } else {
            fixture(params)
        };
        let bytes = encode(&registry, &schema, 1, source).expect("cache must encode");
        let mut payload =
            decode_payload(&bytes, MAX_PAYLOAD_BYTES).expect("cache payload must decode");
        mutate(&mut payload);
        // Keep schema params aligned with the corrupted index so restore reaches
        // validate_vector_kind instead of failing the shallow params equality check.
        if let Some(index) = payload.indexes.get("embedding") {
            if let Some(field) = schema
                .vectors
                .iter_mut()
                .find(|field| field.name == "embedding")
            {
                field.index_params = Some(index.params.clone());
            }
        }
        let corrupted =
            encode_payload(&payload, MAX_PAYLOAD_BYTES).expect("corrupted fixture must encode");
        assert!(
            restore_fixture(&corrupted, &schema, &docs, source).is_none(),
            "corrupted {source} cache must miss"
        );
    }

    #[test]
    fn validate_rejects_mismatched_kind_params_and_index_inventory() {
        use crate::types::IndexType;
        use serde_json::json;

        // Wrong index_type vs packed kind.
        assert_restore_rejects_after(
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
            "hnsw-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Flat;
            },
        );
        assert_restore_rejects_after(
            &IndexParams::ivf(MetricType::L2, 16, 5, false).expect("ivf"),
            "ivf-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Hnsw;
            },
        );
        assert_restore_rejects_after(
            &IndexParams::flat(MetricType::L2).expect("flat"),
            "flat-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Hnsw;
            },
        );
        assert_restore_rejects_after(
            &IndexParams::diskann(MetricType::L2, 16, 64, 0).expect("diskann"),
            "diskann-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Vamana;
            },
        );
        assert_restore_rejects_after(
            &IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("vamana"),
            "vamana-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Diskann;
            },
        );
        assert_restore_rejects_after(
            &IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
            "hnsw-rabitq-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Hnsw;
            },
        );
        assert_restore_rejects_after(
            &IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
            "ivf-rabitq-type",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .index_type = IndexType::Ivf;
            },
        );

        // Missing / non-positive construction parameters.
        assert_restore_rejects_after(
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
            "hnsw-m-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("m");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
            "hnsw-ef-zero",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .insert("ef_construction".into(), json!(0));
            },
        );
        assert_restore_rejects_after(
            &IndexParams::ivf(MetricType::L2, 16, 5, true).expect("ivf"),
            "ivf-nlist-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("n_list");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::ivf(MetricType::L2, 16, 5, true).expect("ivf"),
            "ivf-soar-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("use_soar");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
            "diskann-degree-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("max_degree");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
            "diskann-list-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("list_size");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
            "diskann-pq-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("pq_chunk_num");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
            "diskann-alpha-nan",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .insert("alpha".into(), json!(f64::NAN));
            },
        );
        assert_restore_rejects_after(
            &IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("vamana"),
            "vamana-search-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("search_list_size");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::vamana_with_options(MetricType::L2, 16, 64, 1.2, 8, true)
                .expect("vamana-opts"),
            "vamana-occlusion-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("max_occlusion");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::vamana_with_options(MetricType::L2, 16, 64, 1.2, 8, true)
                .expect("vamana-opts"),
            "vamana-saturate-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("saturate");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
            "hnsw-rabitq-bits-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("total_bits");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
            "hnsw-rabitq-clusters-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("num_clusters");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
            "hnsw-rabitq-sample-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("sample_count");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
            "ivf-rabitq-bits-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("total_bits");
            },
        );
        assert_restore_rejects_after(
            &IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
            "ivf-rabitq-sample-missing",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .params
                    .params
                    .remove("sample_count");
            },
        );

        // Inventory mismatch and revision drift.
        assert_restore_rejects_after(
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
            "inventory",
            |payload| {
                payload.indexes.clear();
            },
        );
        assert_restore_rejects_after(
            &IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
            "revision",
            |payload| {
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index")
                    .source_revision = 99;
            },
        );
    }

    #[test]
    fn validate_indexes_hits_kind_false_arms_directly() {
        use crate::types::IndexType;
        use serde_json::json;

        let cases: Vec<(&str, IndexParams, Box<dyn Fn(&mut VectorIndex)>)> = vec![
            (
                "hnsw-type",
                IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
                Box::new(|index| index.params.index_type = IndexType::Flat),
            ),
            (
                "hnsw-m-missing",
                IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
                Box::new(|index| {
                    index.params.params.remove("m");
                }),
            ),
            (
                "hnsw-ef-zero",
                IndexParams::hnsw(MetricType::L2, 8, 32).expect("hnsw"),
                Box::new(|index| {
                    index
                        .params
                        .params
                        .insert("ef_construction".into(), json!(0));
                }),
            ),
            (
                "ivf-type",
                IndexParams::ivf(MetricType::L2, 16, 5, true).expect("ivf"),
                Box::new(|index| index.params.index_type = IndexType::Hnsw),
            ),
            (
                "ivf-nlist-missing",
                IndexParams::ivf(MetricType::L2, 16, 5, true).expect("ivf"),
                Box::new(|index| {
                    index.params.params.remove("n_list");
                }),
            ),
            (
                "ivf-soar-missing",
                IndexParams::ivf(MetricType::L2, 16, 5, true).expect("ivf"),
                Box::new(|index| {
                    index.params.params.remove("use_soar");
                }),
            ),
            (
                "flat-type",
                IndexParams::flat(MetricType::L2).expect("flat"),
                Box::new(|index| index.params.index_type = IndexType::Hnsw),
            ),
            (
                "diskann-degree-missing",
                IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
                Box::new(|index| {
                    index.params.params.remove("max_degree");
                }),
            ),
            (
                "diskann-list-missing",
                IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
                Box::new(|index| {
                    index.params.params.remove("list_size");
                }),
            ),
            (
                "diskann-pq-missing",
                IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
                Box::new(|index| {
                    index.params.params.remove("pq_chunk_num");
                }),
            ),
            (
                "diskann-alpha-missing",
                IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
                Box::new(|index| {
                    index.params.params.remove("alpha");
                }),
            ),
            (
                "vamana-degree-missing",
                IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("vamana"),
                Box::new(|index| {
                    index.params.params.remove("max_degree");
                }),
            ),
            (
                "vamana-search-missing",
                IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("vamana"),
                Box::new(|index| {
                    index.params.params.remove("search_list_size");
                }),
            ),
            (
                "vamana-alpha-missing",
                IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("vamana"),
                Box::new(|index| {
                    index.params.params.remove("alpha");
                }),
            ),
            (
                "hnsw-rabitq-m-missing",
                IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("m");
                }),
            ),
            (
                "hnsw-rabitq-ef-missing",
                IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("ef_construction");
                }),
            ),
            (
                "ivf-rabitq-nlist-missing",
                IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("n_list");
                }),
            ),
            (
                "vamana-occlusion-missing",
                IndexParams::vamana_with_options(MetricType::L2, 16, 64, 1.2, 8, true)
                    .expect("vamana"),
                Box::new(|index| {
                    index.params.params.remove("max_occlusion");
                }),
            ),
            (
                "vamana-saturate-missing",
                IndexParams::vamana_with_options(MetricType::L2, 16, 64, 1.2, 8, true)
                    .expect("vamana"),
                Box::new(|index| {
                    index.params.params.remove("saturate");
                }),
            ),
            (
                "hnsw-rabitq-bits-missing",
                IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("total_bits");
                }),
            ),
            (
                "hnsw-rabitq-clusters-missing",
                IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("num_clusters");
                }),
            ),
            (
                "hnsw-rabitq-sample-missing",
                IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("sample_count");
                }),
            ),
            (
                "ivf-rabitq-bits-missing",
                IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("total_bits");
                }),
            ),
            (
                "ivf-rabitq-sample-missing",
                IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
                Box::new(|index| {
                    index.params.params.remove("sample_count");
                }),
            ),
            (
                "hnsw-rabitq-type",
                IndexParams::hnsw_rabitq(MetricType::L2, 8, 32).expect("hnsw-rabitq"),
                Box::new(|index| index.params.index_type = IndexType::Hnsw),
            ),
            (
                "ivf-rabitq-type",
                IndexParams::ivf_rabitq(MetricType::L2, 16, 4, 64).expect("ivf-rabitq"),
                Box::new(|index| index.params.index_type = IndexType::Ivf),
            ),
            (
                "diskann-type",
                IndexParams::diskann(MetricType::L2, 16, 64, 1).expect("diskann"),
                Box::new(|index| index.params.index_type = IndexType::Vamana),
            ),
            (
                "vamana-type",
                IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("vamana"),
                Box::new(|index| index.params.index_type = IndexType::Diskann),
            ),
        ];

        for (label, params, mutate) in cases {
            let (mut schema, docs, registry) = if params.index_type == IndexType::Flat {
                let mut field =
                    FieldSchema::new("embedding", DataType::VectorFp32, false, 2).expect("field");
                field.set_index_params(&params).expect("index");
                let schema = CollectionSchema::builder(&format!("direct-{label}"))
                    .add_field(field)
                    .build()
                    .expect("schema");
                let docs: DocumentMap = (0_u16..32)
                    .map(|value| {
                        let id = format!("doc-{value:03}");
                        let mut doc = Doc::with_pk(&id).expect("doc");
                        doc.add_vector_f32("embedding", &[f32::from(value), 0.0])
                            .expect("vector");
                        (id, Arc::new(doc))
                    })
                    .collect();
                let registry = IndexRegistry::build(&schema, &docs, 1).expect("build");
                (schema, docs, registry)
            } else {
                fixture(&params)
            };
            let bytes = encode(&registry, &schema, 1, label).expect("encode");
            let mut payload = decode_payload(&bytes, MAX_PAYLOAD_BYTES).expect("decode");
            mutate(
                payload
                    .indexes
                    .values_mut()
                    .next()
                    .expect("index must exist"),
            );
            if let Some(index) = payload.indexes.get("embedding") {
                if let Some(field) = schema
                    .vectors
                    .iter_mut()
                    .find(|field| field.name == "embedding")
                {
                    field.index_params = Some(index.params.clone());
                }
            }
            assert!(
                !validate_indexes(&schema, &docs, 1, &payload.ordinals, &payload.indexes),
                "{label} must fail validate_indexes"
            );
        }
    }

    #[test]
    fn decode_payload_rejects_oversized_and_length_mismatched_headers() {
        let mut bytes = vec![0_u8; HEADER_BYTES];
        bytes[..CACHE_MAGIC.len()].copy_from_slice(CACHE_MAGIC);
        let huge = (super::MAX_PAYLOAD_BYTES + 1).to_le_bytes();
        bytes[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 8].copy_from_slice(&huge);
        assert!(decode_payload(&bytes, MAX_PAYLOAD_BYTES).is_none());

        let mut short = vec![0_u8; HEADER_BYTES];
        short[..CACHE_MAGIC.len()].copy_from_slice(CACHE_MAGIC);
        short[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 8].copy_from_slice(&8_u64.to_le_bytes());
        assert!(decode_payload(&short, MAX_PAYLOAD_BYTES).is_none());
    }
}
