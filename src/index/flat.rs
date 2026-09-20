//! Exact Flat index: contiguous unquantized coordinates, no graph.

/// Marker for an exact Flat base. Coordinates live on `VectorIndexBase`;
/// this kind only distinguishes Flat from approximate ANN layouts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct FlatIndex;

impl FlatIndex {
    pub(super) fn build() -> Self {
        Self
    }

    pub(super) const fn estimated_payload_bytes() -> usize {
        0
    }
}
