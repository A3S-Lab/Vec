use super::{
    encode, prepare, put_u32, record_position, validates, CHECKSUM_OFFSET, FIXED_HEADER_BYTES,
    SECTOR_BYTES,
};
use crate::doc::{Doc, DocumentMap};
use crate::index::IndexRegistry;
use crate::{CollectionSchema, DataType, FieldSchema, IndexParams, MetricType};
use std::sync::Arc;

fn fixture(dimension: u32, count: u16) -> (CollectionSchema, DocumentMap, IndexRegistry) {
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, dimension)
        .expect("field must be valid");
    embedding
        .set_index_params(
            &IndexParams::vamana(MetricType::L2, 16, 64, 1.2).expect("Vamana params must be valid"),
        )
        .expect("Vamana field must be valid");
    let schema = CollectionSchema::builder("diskann-sector-fixture")
        .add_field(embedding)
        .build()
        .expect("schema must be valid");
    let docs: DocumentMap = (0..count)
        .map(|index| {
            let id = format!("doc-{index:03}");
            let mut doc = Doc::with_pk(&id).expect("document must be valid");
            let vector: Vec<f32> = (0..dimension)
                .map(|coordinate| {
                    f32::from(index)
                        + f32::from(u16::try_from(coordinate).expect("fixture coordinate fits u16"))
                            / 1_024.0
                })
                .collect();
            doc.add_vector_f32("embedding", &vector)
                .expect("vector must be valid");
            (id, Arc::new(doc))
        })
        .collect();
    let registry = IndexRegistry::build(&schema, &docs, 7).expect("Vamana index must build");
    (schema, docs, registry)
}

#[test]
fn small_records_pack_without_crossing_sectors_and_validate_strictly() {
    let (schema, _docs, registry) = fixture(2, 64);
    let prepared = prepare(&registry, &schema, "source").expect("layout must prepare");
    let field = &prepared.fields[0];
    assert!(field.nodes_per_sector > 1);
    assert_eq!(field.sectors_per_node, 0);
    let first = record_position(field, 0).expect("first record must have an offset");
    let second = record_position(field, 1).expect("second record must have an offset");
    assert_eq!(first / SECTOR_BYTES, second / SECTOR_BYTES);
    assert_eq!(second - first, field.record_bytes);

    let bytes = encode(&registry, &schema, 7, "source")
        .expect("sidecar must encode")
        .expect("Vamana requires a sidecar");
    assert_eq!(bytes.len() % SECTOR_BYTES, 0);
    assert!(validates(Some(&bytes), &registry, &schema, 7, "source"));
    assert!(!validates(Some(&bytes), &registry, &schema, 8, "source"));
    assert!(!validates(Some(&bytes), &registry, &schema, 7, "tamper"));

    let mut corrupted = bytes.clone();
    *corrupted.last_mut().expect("sidecar must not be empty") ^= 0x5a;
    assert!(!validates(
        Some(&corrupted),
        &registry,
        &schema,
        7,
        "source"
    ));
    assert!(!validates(
        Some(&bytes[..bytes.len() - 1]),
        &registry,
        &schema,
        7,
        "source"
    ));
    let mut trailing = bytes.clone();
    trailing.extend_from_slice(&[0_u8; SECTOR_BYTES]);
    assert!(!validates(Some(&trailing), &registry, &schema, 7, "source"));

    let mut noncanonical = bytes;
    let sector_padding = field.data_offset + field.nodes_per_sector * field.record_bytes;
    noncanonical[sector_padding] = 1;
    let checksum = crc32fast::hash(&noncanonical[FIXED_HEADER_BYTES..]);
    put_u32(&mut noncanonical, CHECKSUM_OFFSET, checksum);
    assert!(!validates(
        Some(&noncanonical),
        &registry,
        &schema,
        7,
        "source"
    ));
}

#[test]
fn oversized_records_start_on_sector_boundaries() {
    let (schema, _docs, registry) = fixture(1_024, 3);
    let prepared = prepare(&registry, &schema, "source").expect("layout must prepare");
    let field = &prepared.fields[0];
    assert!(field.record_bytes > SECTOR_BYTES);
    assert_eq!(field.nodes_per_sector, 0);
    assert!(field.sectors_per_node >= 2);
    for sequence in 0..3 {
        assert_eq!(
            record_position(field, sequence).expect("record must have an offset") % SECTOR_BYTES,
            0
        );
    }
    let bytes = encode(&registry, &schema, 7, "source")
        .expect("sidecar must encode")
        .expect("Vamana requires a sidecar");
    assert!(validates(Some(&bytes), &registry, &schema, 7, "source"));
}
