//! Atomic document/schema snapshots.

use super::manifest::{atomic_write, checksum, Manifest};
use crate::doc::Doc;
use crate::error::{Error, Result};
use crate::schema::CollectionSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snapshot {
    schema: CollectionSchema,
    docs: Vec<Doc>,
}

pub fn write(
    root: &Path,
    schema: &CollectionSchema,
    docs: &[Doc],
    manifest: &mut Manifest,
    sync: bool,
) -> Result<()> {
    let snapshot = Snapshot {
        schema: schema.clone(),
        docs: docs.to_vec(),
    };
    let bytes = serde_json::to_vec(&snapshot)
        .map_err(|e| Error::internal(format!("serialize document snapshot: {e}")))?;
    let digest = checksum(&bytes);
    atomic_write(root, Path::new("snapshot.json"), &bytes, sync)?;
    manifest.docs_checksum = digest;
    manifest.schema_digest = schema.digest();
    manifest.collection_name = schema.name.clone();
    manifest.generation = manifest.generation.saturating_add(1);
    Ok(())
}

pub fn read(root: &Path, expected: Option<&Manifest>) -> Result<(CollectionSchema, Vec<Doc>)> {
    let bytes = fs::read(root.join("snapshot.json"))
        .map_err(|e| Error::internal(format!("read document snapshot: {e}")))?;
    if let Some(manifest) = expected {
        let actual = checksum(&bytes);
        if manifest.docs_checksum != 0 && manifest.docs_checksum != actual {
            return Err(Error::internal(format!(
                "document snapshot checksum mismatch: expected {}, got {}",
                manifest.docs_checksum, actual
            )));
        }
    }
    let snapshot: Snapshot = serde_json::from_slice(&bytes)
        .map_err(|e| Error::internal(format!("parse document snapshot: {e}")))?;
    if let Some(manifest) = expected {
        if snapshot.schema.digest() != manifest.schema_digest {
            return Err(Error::internal("schema digest does not match manifest"));
        }
    }
    Ok((snapshot.schema, snapshot.docs))
}

