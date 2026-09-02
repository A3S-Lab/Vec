//! On-demand Vamana traversal over sector-aligned node records.

use super::SECTOR_BYTES;
use crate::config::IoBackend;
use crate::error::{Error, Result};
use crate::index::ordinals::OrdinalTable;
use crate::index::product_quantization::{AdcTable, ProductCodebook};
use crate::index::quantization::{dense_query_norm, score_dense_with_query_norm};
use crate::storage::RandomAccessReader;
use crate::types::MetricType;
use roaring::RoaringTreemap;
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug)]
pub(in crate::index) struct FieldReader {
    file: RandomAccessReader,
    dimension: usize,
    max_degree: usize,
    entry_ordinal: Option<u64>,
    record_bytes: usize,
    nodes_per_sector: usize,
    sectors_per_node: usize,
    data_offset: u64,
    record_sequences: Vec<Option<usize>>,
    node_count: usize,
    codebook: Option<ProductCodebook>,
}

#[derive(Debug)]
struct DiskNode {
    vector: Vec<f32>,
    code: Vec<u8>,
    neighbors: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
struct ScoredOrdinal {
    ordinal: u64,
    score: f64,
}

#[derive(Debug)]
struct GraphSearch {
    candidates: Vec<u64>,
    visited: Vec<u64>,
}

#[derive(Debug)]
pub(in crate::index) struct ReadResult {
    pub(in crate::index) candidates: RoaringTreemap,
    pub(in crate::index) sector_reads: u64,
    pub(in crate::index) io_backend: IoBackend,
}

struct ReadSession<'a> {
    reader: &'a FieldReader,
    extents: HashMap<u64, Vec<u8>>,
    nodes: HashMap<u64, DiskNode>,
    sector_reads: u64,
}

impl FieldReader {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::index) fn new(
        file: RandomAccessReader,
        dimension: usize,
        max_degree: usize,
        entry_ordinal: Option<u64>,
        record_bytes: usize,
        nodes_per_sector: usize,
        sectors_per_node: usize,
        data_offset: usize,
        ordinals: impl Iterator<Item = u64>,
        codebook: Option<ProductCodebook>,
    ) -> Result<Self> {
        let mut record_sequences = Vec::new();
        let mut node_count = 0_usize;
        for (sequence, ordinal) in ordinals.enumerate() {
            let slot = usize::try_from(ordinal)
                .map_err(|_| Error::resource_exhausted("DiskANN ordinal exceeds usize"))?;
            if record_sequences.len() <= slot {
                record_sequences.resize(slot.saturating_add(1), None);
            }
            record_sequences[slot] = Some(sequence);
            node_count = node_count.saturating_add(1);
        }
        if entry_ordinal.is_some_and(|entry| {
            usize::try_from(entry)
                .ok()
                .and_then(|slot| record_sequences.get(slot))
                .and_then(|sequence| *sequence)
                .is_none()
        }) {
            return Err(Error::internal(
                "DiskANN entry ordinal is absent from the field",
            ));
        }
        if codebook
            .as_ref()
            .is_some_and(|codebook| codebook.dimension() != dimension)
        {
            return Err(Error::internal("DiskANN PQ codebook dimension is invalid"));
        }
        Ok(Self {
            file,
            dimension,
            max_degree,
            entry_ordinal,
            record_bytes,
            nodes_per_sector,
            sectors_per_node,
            data_offset: u64::try_from(data_offset)
                .map_err(|_| Error::resource_exhausted("DiskANN data offset exceeds u64"))?,
            record_sequences,
            node_count,
            codebook,
        })
    }

    pub(in crate::index) fn candidates(
        &self,
        query: &[f32],
        list_size: usize,
        metric: MetricType,
        ordinals: &OrdinalTable,
    ) -> Result<ReadResult> {
        let Some(entry) = self.entry_ordinal else {
            return Ok(ReadResult {
                candidates: RoaringTreemap::new(),
                sector_reads: 0,
                io_backend: self.file.io_backend(),
            });
        };
        let table = self
            .codebook
            .as_ref()
            .map(|codebook| codebook.table(query, metric))
            .transpose()?;
        let query_norm = if metric == MetricType::Cosine {
            dense_query_norm(query)
        } else {
            0.0
        };
        let mut session = ReadSession::new(self);
        let search = greedy_search(
            &mut session,
            ordinals,
            query,
            entry,
            list_size,
            metric,
            table.as_ref(),
            query_norm,
        )?;
        Ok(ReadResult {
            candidates: search.candidates.into_iter().collect(),
            sector_reads: session.sector_reads,
            io_backend: self.file.io_backend(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::index) fn filtered_candidates(
        &self,
        query: &[f32],
        result_limit: usize,
        traversal_limit: usize,
        metric: MetricType,
        allowed: &RoaringTreemap,
        excluded: &RoaringTreemap,
        ordinals: &OrdinalTable,
    ) -> Result<ReadResult> {
        let Some(entry) = self.entry_ordinal else {
            return Ok(ReadResult {
                candidates: RoaringTreemap::new(),
                sector_reads: 0,
                io_backend: self.file.io_backend(),
            });
        };
        let table = self
            .codebook
            .as_ref()
            .map(|codebook| codebook.table(query, metric))
            .transpose()?;
        let query_norm = if metric == MetricType::Cosine {
            dense_query_norm(query)
        } else {
            0.0
        };
        let mut session = ReadSession::new(self);
        let search = greedy_search(
            &mut session,
            ordinals,
            query,
            entry,
            traversal_limit.min(self.node_count).max(1),
            metric,
            table.as_ref(),
            query_norm,
        )?;
        let mut scored = Vec::new();
        for ordinal in search
            .visited
            .into_iter()
            .chain(search.candidates)
            .filter(|ordinal| allowed.contains(*ordinal) && !excluded.contains(*ordinal))
            .collect::<BTreeSet<_>>()
        {
            let score = score_node(
                session.node(ordinal)?,
                query,
                metric,
                table.as_ref(),
                query_norm,
            )?;
            scored.push(ScoredOrdinal { ordinal, score });
        }
        sort_scored(&mut scored, ordinals);
        Ok(ReadResult {
            candidates: scored
                .into_iter()
                .take(result_limit)
                .map(|candidate| candidate.ordinal)
                .collect(),
            sector_reads: session.sector_reads,
            io_backend: self.file.io_backend(),
        })
    }

    fn contains_ordinal(&self, ordinal: u64) -> bool {
        usize::try_from(ordinal)
            .ok()
            .and_then(|slot| self.record_sequences.get(slot))
            .and_then(|sequence| *sequence)
            .is_some()
    }

    fn record_position(&self, ordinal: u64) -> Option<u64> {
        let sequence = usize::try_from(ordinal)
            .ok()
            .and_then(|slot| self.record_sequences.get(slot))
            .and_then(|sequence| *sequence)?;
        let relative = if self.nodes_per_sector > 0 {
            let sector = sequence.checked_div(self.nodes_per_sector)?;
            let slot = sequence.checked_rem(self.nodes_per_sector)?;
            sector
                .checked_mul(SECTOR_BYTES)?
                .checked_add(slot.checked_mul(self.record_bytes)?)?
        } else {
            sequence.checked_mul(self.sectors_per_node.checked_mul(SECTOR_BYTES)?)?
        };
        self.data_offset.checked_add(u64::try_from(relative).ok()?)
    }
}

impl<'a> ReadSession<'a> {
    fn new(reader: &'a FieldReader) -> Self {
        Self {
            reader,
            extents: HashMap::new(),
            nodes: HashMap::new(),
            sector_reads: 0,
        }
    }

    fn node(&mut self, ordinal: u64) -> Result<&DiskNode> {
        if !self.nodes.contains_key(&ordinal) {
            let node = self.read_node(ordinal)?;
            self.nodes.insert(ordinal, node);
        }
        self.nodes
            .get(&ordinal)
            .ok_or_else(|| Error::internal("DiskANN node cache lost a decoded record"))
    }

    fn read_node(&mut self, ordinal: u64) -> Result<DiskNode> {
        let position = self
            .reader
            .record_position(ordinal)
            .ok_or_else(|| Error::internal("DiskANN graph referenced an unknown ordinal"))?;
        let (extent_offset, extent_bytes, record_offset) = if self.reader.nodes_per_sector > 0 {
            let sector_bytes = u64::try_from(SECTOR_BYTES).unwrap_or(u64::MAX);
            let extent_offset = position - position % sector_bytes;
            let record_offset = usize::try_from(position - extent_offset)
                .map_err(|_| Error::resource_exhausted("DiskANN record offset exceeds usize"))?;
            (extent_offset, SECTOR_BYTES, record_offset)
        } else {
            let extent_bytes = self
                .reader
                .sectors_per_node
                .checked_mul(SECTOR_BYTES)
                .ok_or_else(|| Error::resource_exhausted("DiskANN node extent overflow"))?;
            (position, extent_bytes, 0)
        };
        if !self.extents.contains_key(&extent_offset) {
            let end = extent_offset
                .checked_add(u64::try_from(extent_bytes).unwrap_or(u64::MAX))
                .ok_or_else(|| Error::resource_exhausted("DiskANN read extent overflow"))?;
            if end > self.reader.file.len() {
                return Err(Error::internal("DiskANN node extent exceeds the sidecar"));
            }
            let mut bytes = vec![0_u8; extent_bytes];
            self.reader.file.read_exact_at(extent_offset, &mut bytes)?;
            self.sector_reads = self
                .sector_reads
                .saturating_add(u64::try_from(extent_bytes / SECTOR_BYTES).unwrap_or(u64::MAX));
            self.extents.insert(extent_offset, bytes);
        }
        let extent = self
            .extents
            .get(&extent_offset)
            .ok_or_else(|| Error::internal("DiskANN sector cache lost a loaded extent"))?;
        let end = record_offset
            .checked_add(self.reader.record_bytes)
            .ok_or_else(|| Error::resource_exhausted("DiskANN record boundary overflow"))?;
        let record = extent
            .get(record_offset..end)
            .ok_or_else(|| Error::internal("DiskANN record crosses its stored extent"))?
            .to_vec();
        self.parse_node(ordinal, &record)
    }

    fn parse_node(&self, ordinal: u64, record: &[u8]) -> Result<DiskNode> {
        if read_u64(record, 0) != Some(ordinal) || read_u32(record, 12) != Some(0) {
            return Err(Error::internal("DiskANN node prefix is invalid"));
        }
        let degree = read_u32(record, 8)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|degree| *degree <= self.reader.max_degree)
            .ok_or_else(|| Error::internal("DiskANN node degree is invalid"))?;
        let mut cursor = 16_usize;
        let (vector, code) = if let Some(codebook) = &self.reader.codebook {
            let end = cursor
                .checked_add(codebook.chunk_count())
                .ok_or_else(|| Error::resource_exhausted("DiskANN PQ code boundary overflow"))?;
            let code = record
                .get(cursor..end)
                .ok_or_else(|| Error::internal("DiskANN PQ code is truncated"))?
                .to_vec();
            if code
                .iter()
                .any(|centroid| usize::from(*centroid) >= codebook.centroid_count())
            {
                return Err(Error::internal("DiskANN PQ code is invalid"));
            }
            cursor = end;
            (Vec::new(), code)
        } else {
            let mut vector = Vec::with_capacity(self.reader.dimension);
            for _ in 0..self.reader.dimension {
                let value = read_u32(record, cursor)
                    .map(f32::from_bits)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| Error::internal("DiskANN node vector is invalid"))?;
                vector.push(value);
                cursor = cursor.saturating_add(std::mem::size_of::<f32>());
            }
            (vector, Vec::new())
        };
        let mut neighbors = Vec::with_capacity(degree);
        for slot in 0..self.reader.max_degree {
            let neighbor = read_u64(record, cursor)
                .ok_or_else(|| Error::internal("DiskANN neighbor slot is truncated"))?;
            if slot < degree {
                if neighbor == ordinal || !self.reader.contains_ordinal(neighbor) {
                    return Err(Error::internal("DiskANN neighbor ordinal is invalid"));
                }
                neighbors.push(neighbor);
            } else if neighbor != 0 {
                return Err(Error::internal("DiskANN unused neighbor slot is not zero"));
            }
            cursor = cursor.saturating_add(std::mem::size_of::<u64>());
        }
        if neighbors.iter().copied().collect::<HashSet<_>>().len() != neighbors.len()
            || record
                .get(cursor..)
                .map_or(true, |padding| padding.iter().any(|byte| *byte != 0))
        {
            return Err(Error::internal("DiskANN node record is not canonical"));
        }
        Ok(DiskNode {
            vector,
            code,
            neighbors,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn greedy_search(
    session: &mut ReadSession<'_>,
    ordinals: &OrdinalTable,
    query: &[f32],
    entry: u64,
    list_size: usize,
    metric: MetricType,
    table: Option<&AdcTable>,
    query_norm: f64,
) -> Result<GraphSearch> {
    let limit = list_size.max(1);
    let entry_score = score_node(session.node(entry)?, query, metric, table, query_norm)?;
    let mut pool = vec![ScoredOrdinal {
        ordinal: entry,
        score: entry_score,
    }];
    let mut visited =
        HashSet::with_capacity(limit.saturating_mul(2).min(session.reader.node_count));
    let mut visited_order = Vec::with_capacity(visited.capacity());

    loop {
        sort_scored(&mut pool, ordinals);
        let Some(current) = pool
            .iter()
            .find(|candidate| !visited.contains(&candidate.ordinal))
            .copied()
        else {
            break;
        };
        visited.insert(current.ordinal);
        visited_order.push(current.ordinal);
        let neighbors = session.node(current.ordinal)?.neighbors.clone();
        for neighbor in neighbors {
            if visited.contains(&neighbor)
                || pool.iter().any(|candidate| candidate.ordinal == neighbor)
            {
                continue;
            }
            let score = score_node(session.node(neighbor)?, query, metric, table, query_norm)?;
            pool.push(ScoredOrdinal {
                ordinal: neighbor,
                score,
            });
        }
        sort_scored(&mut pool, ordinals);
        pool.truncate(limit);
    }
    sort_scored(&mut pool, ordinals);
    Ok(GraphSearch {
        candidates: pool
            .into_iter()
            .map(|candidate| candidate.ordinal)
            .collect(),
        visited: visited_order,
    })
}

fn score_node(
    node: &DiskNode,
    query: &[f32],
    metric: MetricType,
    table: Option<&AdcTable>,
    query_norm: f64,
) -> Result<f64> {
    table.map_or_else(
        || {
            Ok(score_dense_with_query_norm(
                query,
                &node.vector,
                metric,
                query_norm,
            ))
        },
        |table| {
            table
                .score(&node.code)
                .ok_or_else(|| Error::internal("DiskANN PQ code does not match its ADC table"))
        },
    )
}

fn sort_scored(values: &mut [ScoredOrdinal], ordinals: &OrdinalTable) {
    values.sort_unstable_by(|left, right| {
        right.score.total_cmp(&left.score).then_with(|| {
            ordinals
                .id(left.ordinal)
                .unwrap_or_default()
                .cmp(ordinals.id(right.ordinal).unwrap_or_default())
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        })
    });
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
    ))
}
