//! Stable scalar, vector, metric, and operation types.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Data type of a collection field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum DataType {
    Undefined = 0,
    Binary = 1,
    String = 2,
    Bool = 3,
    Int32 = 4,
    Int64 = 5,
    Uint32 = 6,
    Uint64 = 7,
    Float = 8,
    Double = 9,
    VectorBinary32 = 20,
    VectorBinary64 = 21,
    VectorFp16 = 22,
    VectorFp32 = 23,
    VectorFp64 = 24,
    VectorInt4 = 25,
    VectorInt8 = 26,
    VectorInt16 = 27,
    SparseVectorFp16 = 30,
    SparseVectorFp32 = 31,
    ArrayBinary = 40,
    ArrayString = 41,
    ArrayBool = 42,
    ArrayInt32 = 43,
    ArrayInt64 = 44,
    ArrayUint32 = 45,
    ArrayUint64 = 46,
    ArrayFloat = 47,
    ArrayDouble = 48,
}

impl From<u32> for DataType {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::Binary,
            2 => Self::String,
            3 => Self::Bool,
            4 => Self::Int32,
            5 => Self::Int64,
            6 => Self::Uint32,
            7 => Self::Uint64,
            8 => Self::Float,
            9 => Self::Double,
            20 => Self::VectorBinary32,
            21 => Self::VectorBinary64,
            22 => Self::VectorFp16,
            23 => Self::VectorFp32,
            24 => Self::VectorFp64,
            25 => Self::VectorInt4,
            26 => Self::VectorInt8,
            27 => Self::VectorInt16,
            30 => Self::SparseVectorFp16,
            31 => Self::SparseVectorFp32,
            40 => Self::ArrayBinary,
            41 => Self::ArrayString,
            42 => Self::ArrayBool,
            43 => Self::ArrayInt32,
            44 => Self::ArrayInt64,
            45 => Self::ArrayUint32,
            46 => Self::ArrayUint64,
            47 => Self::ArrayFloat,
            48 => Self::ArrayDouble,
            _ => Self::Undefined,
        }
    }
}

impl From<DataType> for u32 {
    fn from(value: DataType) -> Self {
        value as u32
    }
}

impl DataType {
    pub fn is_vector(self) -> bool {
        matches!(
            self,
            Self::VectorBinary32
                | Self::VectorBinary64
                | Self::VectorFp16
                | Self::VectorFp32
                | Self::VectorFp64
                | Self::VectorInt4
                | Self::VectorInt8
                | Self::VectorInt16
                | Self::SparseVectorFp16
                | Self::SparseVectorFp32
        )
    }

    pub fn is_dense_vector(self) -> bool {
        self.is_vector() && !self.is_sparse_vector()
    }

    pub fn is_sparse_vector(self) -> bool {
        matches!(self, Self::SparseVectorFp16 | Self::SparseVectorFp32)
    }

    pub fn is_array(self) -> bool {
        matches!(
            self,
            Self::ArrayBinary
                | Self::ArrayString
                | Self::ArrayBool
                | Self::ArrayInt32
                | Self::ArrayInt64
                | Self::ArrayUint32
                | Self::ArrayUint64
                | Self::ArrayFloat
                | Self::ArrayDouble
        )
    }

    pub fn is_scalar(self) -> bool {
        !self.is_vector() && !self.is_array() && self != Self::Undefined
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Index implementation selected for a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum IndexType {
    Undefined = 0,
    Hnsw = 1,
    Ivf = 2,
    Flat = 3,
    Diskann = 5,
    /// Vamana is the graph-construction name used by `DiskANN`.
    Vamana = 6,
    IvfRabitq = 7,
    HnswRabitq = 8,
    Invert = 10,
    Fts = 11,
}

impl From<u32> for IndexType {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::Hnsw,
            2 => Self::Ivf,
            3 => Self::Flat,
            5 => Self::Diskann,
            6 => Self::Vamana,
            7 => Self::IvfRabitq,
            8 => Self::HnswRabitq,
            10 => Self::Invert,
            11 => Self::Fts,
            _ => Self::Undefined,
        }
    }
}

impl From<IndexType> for u32 {
    fn from(value: IndexType) -> Self {
        value as u32
    }
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Similarity metric.  Search scores are always ordered from high to low;
/// L2 is represented as negative squared distance at the API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum MetricType {
    Undefined = 0,
    L2 = 1,
    Ip = 2,
    Cosine = 3,
    MipsL2 = 4,
}

impl From<u32> for MetricType {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::L2,
            2 => Self::Ip,
            3 => Self::Cosine,
            4 => Self::MipsL2,
            _ => Self::Undefined,
        }
    }
}

impl From<MetricType> for u32 {
    fn from(value: MetricType) -> Self {
        value as u32
    }
}

impl MetricType {
    pub fn higher_is_better(self) -> bool {
        true
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Scalar/vector quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum QuantizeType {
    Undefined = 0,
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
    Rabitq = 4,
    Pq = 5,
}

impl From<u32> for QuantizeType {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::Fp16,
            2 => Self::Int8,
            3 => Self::Int4,
            4 => Self::Rabitq,
            5 => Self::Pq,
            _ => Self::Undefined,
        }
    }
}

impl From<QuantizeType> for u32 {
    fn from(value: QuantizeType) -> Self {
        value as u32
    }
}

impl fmt::Display for QuantizeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// DML operation kind used by write-ahead records and telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocOperator {
    Insert,
    Update,
    Upsert,
    Delete,
}

impl fmt::Display for DocOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
