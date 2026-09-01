//! Primitive little-endian helpers for the native sector format.

use crate::error::{Error, Result};

pub(super) fn encoded_bytes_len(bytes: &[u8]) -> Result<usize> {
    let _ = usize_to_u32(bytes.len(), "metadata string length")?;
    bytes
        .len()
        .checked_add(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::resource_exhausted("DiskANN metadata string length overflow"))
}

pub(super) fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    push_u32(output, usize_to_u32(bytes.len(), "metadata string length")?);
    output.extend_from_slice(bytes);
    Ok(())
}

pub(super) fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

pub(super) fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

pub(super) fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::resource_exhausted(format!("DiskANN {label} exceeds u32")))
}

pub(super) fn usize_to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| Error::resource_exhausted(format!("DiskANN {label} exceeds u64")))
}

pub(super) fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|rounded| rounded / alignment * alignment)
}

pub(super) struct SliceReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> SliceReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(super) fn read_u32(&mut self) -> Option<u32> {
        let value = read_u32(self.bytes, self.position)?;
        self.position = self.position.checked_add(4)?;
        Some(value)
    }

    pub(super) fn read_u64(&mut self) -> Option<u64> {
        let value = read_u64(self.bytes, self.position)?;
        self.position = self.position.checked_add(8)?;
        Some(value)
    }

    pub(super) fn read_bytes(&mut self) -> Option<&'a [u8]> {
        let length = usize::try_from(self.read_u32()?).ok()?;
        let end = self.position.checked_add(length)?;
        let value = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(value)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}
