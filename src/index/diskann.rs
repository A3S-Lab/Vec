//! Native, sector-aligned persistence for Vamana and `DiskANN` generations.
//!
//! This is an A3S-owned format, not the Microsoft `DiskANN` C++ wire format.
//! It mirrors the immutable in-memory Vamana base so recovery can validate the
//! graph before a later disk reader makes it authoritative for search.

mod codec;
mod reader;

use super::product_quantization::{ProductCodebook, ProductQuantizer};
use super::vamana::VamanaIndex;
use super::{IndexRegistry, VectorIndex, VectorIndexKind};
use crate::config::IoBackend;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use crate::storage::PositionedFile;
use crate::types::IndexType;
use codec::{
    align_up, encoded_bytes_len, push_bytes, push_u32, push_u64, put_u32, put_u64, read_u32,
    read_u64, usize_to_u32, usize_to_u64, SliceReader,
};
pub(super) use reader::FieldReader;
use std::sync::Arc;

pub(super) const SECTOR_BYTES: usize = 4_096;
pub(super) const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;

const MAGIC: &[u8; 8] = b"A3SDAN01";
const FORMAT_VERSION: u32 = 2;
const FIXED_HEADER_BYTES: usize = 64;
const CHECKSUM_OFFSET: usize = 48;
const FIELD_METADATA_BYTES: usize = 80;
const RECORD_PREFIX_BYTES: usize = 16;
const NONE_ORDINAL: u64 = u64::MAX;

struct FieldLayout<'a> {
    name: &'a str,
    index: &'a VectorIndex,
    index_type: IndexType,
    graph: &'a VamanaIndex,
    pq: Option<&'a ProductQuantizer>,
    dimension: usize,
    max_degree: usize,
    list_size: usize,
    alpha: f64,
    entry_ordinal: Option<u64>,
    record_bytes: usize,
    nodes_per_sector: usize,
    sectors_per_node: usize,
    data_offset: usize,
    data_bytes: usize,
}

struct PreparedLayout<'a> {
    fields: Vec<FieldLayout<'a>>,
    metadata_bytes: usize,
    data_offset: usize,
    total_bytes: usize,
}

struct ReaderSpec {
    name: String,
    dimension: usize,
    max_degree: usize,
    entry_ordinal: Option<u64>,
    record_bytes: usize,
    nodes_per_sector: usize,
    sectors_per_node: usize,
    data_offset: usize,
    ordinals: Vec<u64>,
    codebook: Option<ProductCodebook>,
}

pub(super) fn encode(
    registry: &IndexRegistry,
    schema: &CollectionSchema,
    source_revision: u64,
    source_identity: &str,
) -> Result<Option<Vec<u8>>> {
    let prepared = prepare(registry, schema, source_identity)?;
    if prepared.fields.is_empty() {
        return Ok(None);
    }

    let mut output = vec![0_u8; prepared.total_bytes];
    output[..MAGIC.len()].copy_from_slice(MAGIC);
    put_u32(&mut output, 8, FORMAT_VERSION);
    put_u32(&mut output, 12, usize_to_u32(SECTOR_BYTES, "sector size")?);
    put_u32(
        &mut output,
        16,
        usize_to_u32(prepared.fields.len(), "field count")?,
    );
    put_u32(
        &mut output,
        20,
        usize_to_u32(prepared.metadata_bytes, "metadata length")?,
    );
    put_u64(
        &mut output,
        24,
        usize_to_u64(prepared.data_offset, "data offset")?,
    );
    put_u64(
        &mut output,
        32,
        usize_to_u64(prepared.total_bytes, "file length")?,
    );
    put_u64(&mut output, 40, source_revision);

    let metadata_end = FIXED_HEADER_BYTES
        .checked_add(prepared.metadata_bytes)
        .ok_or_else(|| Error::resource_exhausted("DiskANN metadata length overflow"))?;
    let metadata = encode_metadata(&prepared, schema, source_identity)?;
    output[FIXED_HEADER_BYTES..metadata_end].copy_from_slice(&metadata);

    for field in &prepared.fields {
        encode_field(&mut output, field)?;
    }
    let checksum = crc32fast::hash(&output[FIXED_HEADER_BYTES..]);
    put_u32(&mut output, CHECKSUM_OFFSET, checksum);
    Ok(Some(output))
}

fn encode_metadata(
    prepared: &PreparedLayout<'_>,
    schema: &CollectionSchema,
    source_identity: &str,
) -> Result<Vec<u8>> {
    let mut metadata = Vec::with_capacity(prepared.metadata_bytes);
    push_bytes(&mut metadata, source_identity.as_bytes())?;
    push_bytes(&mut metadata, schema.digest().as_bytes())?;
    for field in &prepared.fields {
        push_bytes(&mut metadata, field.name.as_bytes())?;
        encode_field_metadata(&mut metadata, field)?;
    }
    if metadata.len() != prepared.metadata_bytes {
        return Err(Error::internal("DiskANN metadata layout drifted"));
    }
    Ok(metadata)
}

fn encode_field_metadata(metadata: &mut Vec<u8>, field: &FieldLayout<'_>) -> Result<()> {
    push_u64(
        metadata,
        usize_to_u64(field.index.base.vectors.len(), "node count")?,
    );
    for (value, label) in [
        (field.dimension, "vector dimension"),
        (field.max_degree, "maximum degree"),
        (field.list_size, "search list size"),
        (field.record_bytes, "record length"),
        (field.nodes_per_sector, "nodes per sector"),
        (field.sectors_per_node, "sectors per node"),
    ] {
        push_u32(metadata, usize_to_u32(value, label)?);
    }
    push_u64(metadata, field.alpha.to_bits());
    push_u64(metadata, field.entry_ordinal.unwrap_or(NONE_ORDINAL));
    push_u64(
        metadata,
        usize_to_u64(field.data_offset, "field data offset")?,
    );
    push_u64(
        metadata,
        usize_to_u64(field.data_bytes, "field data length")?,
    );
    push_u32(metadata, u32::from(field.index_type));
    push_u32(metadata, u32::from(field.pq.is_some()));
    push_u32(
        metadata,
        usize_to_u32(
            field.pq.map_or(0, |pq| pq.codebook().chunk_count()),
            "PQ chunk count",
        )?,
    );
    push_u32(
        metadata,
        usize_to_u32(
            field.pq.map_or(0, |pq| pq.codebook().centroid_count()),
            "PQ centroid count",
        )?,
    );
    if let Some(pq) = field.pq {
        for value in pq.codebook().flattened_centroids() {
            push_u32(metadata, value.to_bits());
        }
    }
    Ok(())
}

pub(super) fn validates(
    bytes: Option<&[u8]>,
    registry: &IndexRegistry,
    schema: &CollectionSchema,
    source_revision: u64,
    source_identity: &str,
) -> bool {
    let Ok(prepared) = prepare(registry, schema, source_identity) else {
        return false;
    };
    if prepared.fields.is_empty() {
        return true;
    }
    let Some(bytes) = bytes else {
        return false;
    };
    if bytes.len() != prepared.total_bytes
        || bytes.len() % SECTOR_BYTES != 0
        || bytes.len() < SECTOR_BYTES
        || bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice())
        || read_u32(bytes, 8) != Some(FORMAT_VERSION)
        || read_u32(bytes, 12) != u32::try_from(SECTOR_BYTES).ok()
        || read_u32(bytes, 16) != u32::try_from(prepared.fields.len()).ok()
        || read_u32(bytes, 20) != u32::try_from(prepared.metadata_bytes).ok()
        || read_u64(bytes, 24) != u64::try_from(prepared.data_offset).ok()
        || read_u64(bytes, 32) != u64::try_from(prepared.total_bytes).ok()
        || read_u64(bytes, 40) != Some(source_revision)
        || bytes
            .get(52..FIXED_HEADER_BYTES)
            .map_or(true, |padding| padding.iter().any(|byte| *byte != 0))
        || read_u32(bytes, CHECKSUM_OFFSET) != Some(crc32fast::hash(&bytes[FIXED_HEADER_BYTES..]))
    {
        return false;
    }

    let metadata_end = FIXED_HEADER_BYTES + prepared.metadata_bytes;
    if bytes[metadata_end..prepared.data_offset]
        .iter()
        .any(|byte| *byte != 0)
    {
        return false;
    }
    let mut reader = SliceReader::new(&bytes[FIXED_HEADER_BYTES..metadata_end]);
    if reader.read_bytes() != Some(source_identity.as_bytes())
        || reader.read_bytes() != Some(schema.digest().as_bytes())
    {
        return false;
    }
    for field in &prepared.fields {
        if reader.read_bytes() != Some(field.name.as_bytes())
            || reader.read_u64() != u64::try_from(field.index.base.vectors.len()).ok()
            || reader.read_u32() != u32::try_from(field.dimension).ok()
            || reader.read_u32() != u32::try_from(field.max_degree).ok()
            || reader.read_u32() != u32::try_from(field.list_size).ok()
            || reader.read_u32() != u32::try_from(field.record_bytes).ok()
            || reader.read_u32() != u32::try_from(field.nodes_per_sector).ok()
            || reader.read_u32() != u32::try_from(field.sectors_per_node).ok()
            || reader.read_u64() != Some(field.alpha.to_bits())
            || reader.read_u64() != Some(field.entry_ordinal.unwrap_or(NONE_ORDINAL))
            || reader.read_u64() != u64::try_from(field.data_offset).ok()
            || reader.read_u64() != u64::try_from(field.data_bytes).ok()
            || reader.read_u32() != Some(u32::from(field.index_type))
            || reader.read_u32() != Some(u32::from(field.pq.is_some()))
            || reader.read_u32()
                != u32::try_from(field.pq.map_or(0, |pq| pq.codebook().chunk_count())).ok()
            || reader.read_u32()
                != u32::try_from(field.pq.map_or(0, |pq| pq.codebook().centroid_count())).ok()
            || !validate_codebook_metadata(&mut reader, field)
            || !validate_field(bytes, field)
        {
            return false;
        }
    }
    reader.is_empty()
}

pub(super) fn attach(
    file: Option<PositionedFile>,
    io_backend: IoBackend,
    registry: &mut IndexRegistry,
    schema: &CollectionSchema,
    source_revision: u64,
    source_identity: &str,
) -> bool {
    let Ok(prepared) = prepare(registry, schema, source_identity) else {
        return false;
    };
    if prepared.fields.is_empty() {
        return true;
    }
    let Some(file) = file else {
        return false;
    };
    let Ok(bytes) = file.read_all() else {
        return false;
    };
    if !validates(
        Some(&bytes),
        registry,
        schema,
        source_revision,
        source_identity,
    ) {
        return false;
    }
    let Ok(reader_source) = file.into_random_access(io_backend, &bytes) else {
        return false;
    };
    let readers: Vec<ReaderSpec> = prepared
        .fields
        .iter()
        .map(|field| ReaderSpec {
            name: field.name.to_string(),
            dimension: field.dimension,
            max_degree: field.max_degree,
            entry_ordinal: field.entry_ordinal,
            record_bytes: field.record_bytes,
            nodes_per_sector: field.nodes_per_sector,
            sectors_per_node: field.sectors_per_node,
            data_offset: field.data_offset,
            ordinals: field.index.base.vectors.keys().collect(),
            codebook: field.pq.map(|pq| pq.codebook().clone()),
        })
        .collect();
    drop(prepared);
    for spec in readers {
        let Ok(reader) = FieldReader::new(
            reader_source.clone(),
            spec.dimension,
            spec.max_degree,
            spec.entry_ordinal,
            spec.record_bytes,
            spec.nodes_per_sector,
            spec.sectors_per_node,
            spec.data_offset,
            spec.ordinals.into_iter(),
            spec.codebook,
        ) else {
            return false;
        };
        let Some(index) = registry.indexes.get_mut(&spec.name) else {
            return false;
        };
        Arc::make_mut(&mut index.base).diskann = Some(Arc::new(reader));
    }
    true
}

fn prepare<'a>(
    registry: &'a IndexRegistry,
    schema: &CollectionSchema,
    source_identity: &str,
) -> Result<PreparedLayout<'a>> {
    let mut fields = Vec::new();
    for (name, index) in &registry.indexes {
        let Some((graph, pq)) = graph_storage(&index.base.kind) else {
            continue;
        };
        let dimension = schema
            .vectors
            .iter()
            .find(|field| field.name == *name)
            .ok_or_else(|| Error::internal(format!("graph field '{name}' is absent from schema")))?
            .dimension;
        let dimension = usize::try_from(dimension)
            .map_err(|_| Error::resource_exhausted("graph dimension is too large"))?;
        let max_degree = graph.max_degree();
        let vector_bytes = pq.map_or_else(
            || {
                dimension
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        Error::resource_exhausted("DiskANN vector record length overflow")
                    })
            },
            |pq| Ok(pq.codebook().chunk_count()),
        )?;
        let neighbor_bytes = max_degree
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| Error::resource_exhausted("DiskANN neighbor record length overflow"))?;
        let raw_record_bytes = RECORD_PREFIX_BYTES
            .checked_add(vector_bytes)
            .and_then(|value| value.checked_add(neighbor_bytes))
            .ok_or_else(|| Error::resource_exhausted("DiskANN node record length overflow"))?;
        let record_bytes = align_up(raw_record_bytes, std::mem::size_of::<u64>())
            .ok_or_else(|| Error::resource_exhausted("DiskANN node record alignment overflow"))?;
        let (nodes_per_sector, sectors_per_node, data_bytes) =
            field_storage(record_bytes, index.base.vectors.len())?;
        fields.push(FieldLayout {
            name,
            index,
            index_type: index.params.index_type,
            graph,
            pq,
            dimension,
            max_degree,
            list_size: graph.default_list_size(),
            alpha: graph.alpha(),
            entry_ordinal: graph.entry_ordinal(),
            record_bytes,
            nodes_per_sector,
            sectors_per_node,
            data_offset: 0,
            data_bytes,
        });
    }

    let mut metadata_bytes = encoded_bytes_len(source_identity.as_bytes())?
        .checked_add(encoded_bytes_len(schema.digest().as_bytes())?)
        .ok_or_else(|| Error::resource_exhausted("DiskANN metadata length overflow"))?;
    for field in &fields {
        let codebook_bytes = field
            .pq
            .map_or(0, |pq| pq.codebook().flattened_centroids().count())
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::resource_exhausted("DiskANN PQ codebook length overflow"))?;
        metadata_bytes = metadata_bytes
            .checked_add(encoded_bytes_len(field.name.as_bytes())?)
            .and_then(|value| value.checked_add(FIELD_METADATA_BYTES))
            .and_then(|value| value.checked_add(codebook_bytes))
            .ok_or_else(|| Error::resource_exhausted("DiskANN field metadata length overflow"))?;
    }
    let data_offset = align_up(
        FIXED_HEADER_BYTES
            .checked_add(metadata_bytes)
            .ok_or_else(|| Error::resource_exhausted("DiskANN header length overflow"))?,
        SECTOR_BYTES,
    )
    .ok_or_else(|| Error::resource_exhausted("DiskANN header alignment overflow"))?;
    let mut total_bytes = data_offset;
    for field in &mut fields {
        field.data_offset = total_bytes;
        total_bytes = total_bytes
            .checked_add(field.data_bytes)
            .ok_or_else(|| Error::resource_exhausted("DiskANN file length overflow"))?;
    }
    if u64::try_from(total_bytes).unwrap_or(u64::MAX) > MAX_FILE_BYTES {
        return Err(Error::resource_exhausted(format!(
            "DiskANN sidecar exceeds the {MAX_FILE_BYTES}-byte storage limit"
        )));
    }
    Ok(PreparedLayout {
        fields,
        metadata_bytes,
        data_offset,
        total_bytes,
    })
}

fn field_storage(record_bytes: usize, node_count: usize) -> Result<(usize, usize, usize)> {
    if record_bytes <= SECTOR_BYTES {
        let nodes_per_sector = SECTOR_BYTES / record_bytes;
        let sectors = node_count
            .checked_add(nodes_per_sector.saturating_sub(1))
            .map(|value| value / nodes_per_sector)
            .ok_or_else(|| Error::resource_exhausted("DiskANN sector count overflow"))?;
        let bytes = sectors
            .checked_mul(SECTOR_BYTES)
            .ok_or_else(|| Error::resource_exhausted("DiskANN field length overflow"))?;
        Ok((nodes_per_sector, 0, bytes))
    } else {
        let stride = align_up(record_bytes, SECTOR_BYTES)
            .ok_or_else(|| Error::resource_exhausted("DiskANN record stride overflow"))?;
        let sectors_per_node = stride / SECTOR_BYTES;
        let bytes = node_count
            .checked_mul(stride)
            .ok_or_else(|| Error::resource_exhausted("DiskANN field length overflow"))?;
        Ok((0, sectors_per_node, bytes))
    }
}

fn encode_field(output: &mut [u8], field: &FieldLayout<'_>) -> Result<()> {
    if field
        .graph
        .nodes()
        .map(|(ordinal, _)| ordinal)
        .ne(field.index.base.vectors.keys())
    {
        return Err(Error::internal("graph and vector ordinals differ"));
    }
    for (sequence, (ordinal, vector)) in field.index.base.vectors.iter().enumerate() {
        let position = record_position(field, sequence)
            .ok_or_else(|| Error::resource_exhausted("DiskANN record offset overflow"))?;
        let end = position
            .checked_add(field.record_bytes)
            .ok_or_else(|| Error::resource_exhausted("DiskANN record end overflow"))?;
        let record = output
            .get_mut(position..end)
            .ok_or_else(|| Error::internal("DiskANN record exceeds its field"))?;
        let neighbors = field
            .graph
            .neighbors(ordinal)
            .ok_or_else(|| Error::internal("node is missing from its graph"))?;
        if neighbors.len() > field.max_degree {
            return Err(Error::internal("graph node exceeds its maximum degree"));
        }
        put_u64(record, 0, ordinal);
        put_u32(record, 8, usize_to_u32(neighbors.len(), "node degree")?);
        let mut cursor = RECORD_PREFIX_BYTES;
        if let Some(pq) = field.pq {
            let code = pq
                .code(ordinal)
                .ok_or_else(|| Error::internal("DiskANN PQ code is missing"))?;
            let end = cursor
                .checked_add(code.len())
                .ok_or_else(|| Error::resource_exhausted("DiskANN PQ record overflow"))?;
            record
                .get_mut(cursor..end)
                .ok_or_else(|| Error::internal("DiskANN PQ code exceeds its record"))?
                .copy_from_slice(code);
            cursor = end;
        } else {
            let decoded = vector.decode();
            if decoded.len() != field.dimension {
                return Err(Error::internal("graph vector dimension drifted"));
            }
            for value in decoded {
                put_u32(record, cursor, value.to_bits());
                cursor += std::mem::size_of::<f32>();
            }
        }
        for neighbor in neighbors {
            put_u64(record, cursor, *neighbor);
            cursor += std::mem::size_of::<u64>();
        }
    }
    Ok(())
}

fn validate_field(bytes: &[u8], field: &FieldLayout<'_>) -> bool {
    for (sequence, (ordinal, vector)) in field.index.base.vectors.iter().enumerate() {
        let Some(position) = record_position(field, sequence) else {
            return false;
        };
        let Some(record) = bytes.get(position..position.saturating_add(field.record_bytes)) else {
            return false;
        };
        let Some(neighbors) = field.graph.neighbors(ordinal) else {
            return false;
        };
        if read_u64(record, 0) != Some(ordinal)
            || read_u32(record, 8) != u32::try_from(neighbors.len()).ok()
            || read_u32(record, 12) != Some(0)
        {
            return false;
        }
        let mut cursor = RECORD_PREFIX_BYTES;
        if let Some(pq) = field.pq {
            let Some(expected) = pq.code(ordinal) else {
                return false;
            };
            let Some(end) = cursor.checked_add(expected.len()) else {
                return false;
            };
            if record.get(cursor..end) != Some(expected) {
                return false;
            }
            cursor = end;
        } else {
            let decoded = vector.decode();
            for expected in decoded {
                if read_u32(record, cursor) != Some(expected.to_bits()) {
                    return false;
                }
                cursor += std::mem::size_of::<f32>();
            }
        }
        for slot in 0..field.max_degree {
            let expected = neighbors.get(slot).copied().unwrap_or(0);
            if read_u64(record, cursor) != Some(expected) {
                return false;
            }
            cursor += std::mem::size_of::<u64>();
        }
        if record[cursor..].iter().any(|byte| *byte != 0) {
            return false;
        }
    }
    validate_field_padding(bytes, field)
}

fn graph_storage(kind: &VectorIndexKind) -> Option<(&VamanaIndex, Option<&ProductQuantizer>)> {
    match kind {
        VectorIndexKind::Diskann(index) => Some((index.graph(), index.quantizer())),
        VectorIndexKind::Vamana(index) => Some((index, None)),
        VectorIndexKind::Hnsw(_)
        | VectorIndexKind::HnswRabitq(_)
        | VectorIndexKind::Ivf(_)
        | VectorIndexKind::IvfRabitq(_) => None,
    }
}

fn validate_codebook_metadata(reader: &mut SliceReader<'_>, field: &FieldLayout<'_>) -> bool {
    field.pq.map_or(true, |pq| {
        pq.codebook()
            .flattened_centroids()
            .all(|expected| reader.read_u32() == Some(expected.to_bits()))
    })
}

fn validate_field_padding(bytes: &[u8], field: &FieldLayout<'_>) -> bool {
    let Some(data) = bytes.get(field.data_offset..field.data_offset + field.data_bytes) else {
        return false;
    };
    let node_count = field.index.base.vectors.len();
    if field.nodes_per_sector > 0 {
        for (sector_index, sector) in data.chunks_exact(SECTOR_BYTES).enumerate() {
            let first = sector_index.saturating_mul(field.nodes_per_sector);
            let used_nodes = node_count.saturating_sub(first).min(field.nodes_per_sector);
            let used_bytes = used_nodes.saturating_mul(field.record_bytes);
            if sector[used_bytes..].iter().any(|byte| *byte != 0) {
                return false;
            }
        }
    } else {
        let stride = field.sectors_per_node.saturating_mul(SECTOR_BYTES);
        for node in data.chunks_exact(stride) {
            if node[field.record_bytes..].iter().any(|byte| *byte != 0) {
                return false;
            }
        }
    }
    true
}

fn record_position(field: &FieldLayout<'_>, sequence: usize) -> Option<usize> {
    let relative = if field.nodes_per_sector > 0 {
        let sector = sequence.checked_div(field.nodes_per_sector)?;
        let slot = sequence.checked_rem(field.nodes_per_sector)?;
        sector
            .checked_mul(SECTOR_BYTES)?
            .checked_add(slot.checked_mul(field.record_bytes)?)?
    } else {
        sequence.checked_mul(field.sectors_per_node.checked_mul(SECTOR_BYTES)?)?
    };
    field.data_offset.checked_add(relative)
}

#[cfg(test)]
mod tests;
