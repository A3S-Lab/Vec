use super::{
    attach, encode, prepare, put_u32, record_position, validates, CHECKSUM_OFFSET,
    FIXED_HEADER_BYTES, SECTOR_BYTES,
};
use crate::doc::{Doc, DocumentMap};
use crate::index::{IndexRegistry, VectorIndexKind};
use crate::storage::StorageHandle;
use crate::{CollectionSchema, DataType, FieldSchema, IndexParams, MetricType};
use roaring::RoaringTreemap;
use std::sync::Arc;
use tempfile::tempdir;

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

fn pq_fixture(
    dimension: u32,
    count: u16,
    chunks: i32,
) -> (CollectionSchema, DocumentMap, IndexRegistry) {
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, dimension)
        .expect("field must be valid");
    embedding
        .set_index_params(
            &IndexParams::diskann(MetricType::L2, 16, 64, chunks)
                .expect("DiskANN params must be valid"),
        )
        .expect("DiskANN field must be valid");
    let schema = CollectionSchema::builder("diskann-pq-sector-fixture")
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
    let registry = IndexRegistry::build(&schema, &docs, 7).expect("DiskANN index must build");
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

#[test]
fn positioned_reader_matches_the_memory_graph_for_packed_and_oversized_records() {
    for (dimension, count, list_size) in [(2_u32, 64_u16, 16_usize), (1_024, 6, 3)] {
        let (schema, _docs, registry) = fixture(dimension, count);
        let bytes = encode(&registry, &schema, 7, "source")
            .expect("sidecar must encode")
            .expect("Vamana requires a sidecar");
        let temporary = tempdir().expect("temporary directory must be available");
        let storage = StorageHandle::create(temporary.path(), &schema, false)
            .expect("storage must be created");
        storage
            .write_diskann_file(&bytes, false)
            .expect("sidecar must be written");
        let file = storage
            .open_diskann_file()
            .expect("sidecar must open")
            .expect("sidecar must exist");
        let mut attached = registry.clone();
        assert!(attach(Some(file), &mut attached, &schema, 7, "source"));

        let index = attached
            .indexes
            .get("embedding")
            .expect("Vamana index must exist");
        let VectorIndexKind::Vamana(vamana) = &index.base.kind else {
            panic!("fixture must build Vamana");
        };
        let reader = index
            .base
            .diskann
            .as_ref()
            .expect("positioned reader must attach");
        let query: Vec<f32> = (0..dimension)
            .map(|coordinate| {
                2.5 + f32::from(u16::try_from(coordinate).expect("fixture coordinate fits u16"))
                    / 1_024.0
            })
            .collect();
        let expected = vamana.candidates(
            &index.base.vectors,
            &attached.ordinals,
            &query,
            Some(list_size),
            1,
            MetricType::L2,
        );
        let actual = reader
            .candidates(&query, list_size, MetricType::L2, &attached.ordinals)
            .expect("positioned traversal must succeed");
        assert_eq!(actual.candidates, expected, "dimension={dimension}");
        assert!(actual.sector_reads > 0, "dimension={dimension}");

        let allowed: RoaringTreemap = index
            .base
            .vectors
            .keys()
            .filter(|ordinal| ordinal % 2 == 0)
            .collect();
        let excluded = RoaringTreemap::new();
        let result_limit = 3;
        let expected = vamana.filtered_candidates(
            &index.base.vectors,
            &attached.ordinals,
            &query,
            result_limit,
            list_size,
            MetricType::L2,
            &allowed,
            &excluded,
        );
        let actual = reader
            .filtered_candidates(
                &query,
                result_limit,
                list_size,
                MetricType::L2,
                &allowed,
                &excluded,
                &attached.ordinals,
            )
            .expect("filtered positioned traversal must succeed");
        assert_eq!(actual.candidates, expected, "dimension={dimension}");
        assert!(actual.sector_reads > 0, "dimension={dimension}");
    }
}

#[test]
fn pq_records_are_compact_validated_and_match_in_memory_adc() {
    let (schema, _docs, registry) = pq_fixture(128, 64, 8);
    let prepared = prepare(&registry, &schema, "source").expect("layout must prepare");
    let field = &prepared.fields[0];
    let dense_record_bytes = super::align_up(
        16 + 128 * std::mem::size_of::<f32>() + 16 * std::mem::size_of::<u64>(),
        std::mem::size_of::<u64>(),
    )
    .expect("dense fixture record must align");
    assert!(field.record_bytes < dense_record_bytes);
    assert_eq!(field.record_bytes, 152);

    let bytes = encode(&registry, &schema, 7, "source")
        .expect("sidecar must encode")
        .expect("DiskANN requires a sidecar");
    assert!(validates(Some(&bytes), &registry, &schema, 7, "source"));
    let temporary = tempdir().expect("temporary directory must be available");
    let storage =
        StorageHandle::create(temporary.path(), &schema, false).expect("storage must be created");
    storage
        .write_diskann_file(&bytes, false)
        .expect("sidecar must be written");
    let file = storage
        .open_diskann_file()
        .expect("sidecar must open")
        .expect("sidecar must exist");
    let mut attached = registry.clone();
    assert!(attach(Some(file), &mut attached, &schema, 7, "source"));
    let index = attached
        .indexes
        .get("embedding")
        .expect("DiskANN index must exist");
    let VectorIndexKind::Diskann(diskann) = &index.base.kind else {
        panic!("fixture must build DiskANN");
    };
    let query = vec![17.25_f32; 128];
    let expected = diskann
        .candidates(
            &index.base.vectors,
            &attached.ordinals,
            &query,
            Some(24),
            10,
            MetricType::L2,
        )
        .expect("memory ADC must succeed");
    let actual = index
        .base
        .diskann
        .as_ref()
        .expect("positioned reader must attach")
        .candidates(&query, 24, MetricType::L2, &attached.ordinals)
        .expect("positioned ADC must succeed");
    assert_eq!(actual.candidates, expected);
    assert!(actual.sector_reads > 0);

    let allowed: RoaringTreemap = index
        .base
        .vectors
        .keys()
        .filter(|ordinal| ordinal % 2 == 0)
        .collect();
    let excluded: RoaringTreemap = [18_u64].into_iter().collect();
    let expected = diskann
        .filtered_candidates(
            &index.base.vectors,
            &attached.ordinals,
            &query,
            5,
            24,
            MetricType::L2,
            &allowed,
            &excluded,
        )
        .expect("filtered memory ADC must succeed");
    let actual = index
        .base
        .diskann
        .as_ref()
        .expect("positioned reader must attach")
        .filtered_candidates(
            &query,
            5,
            24,
            MetricType::L2,
            &allowed,
            &excluded,
            &attached.ordinals,
        )
        .expect("filtered positioned ADC must succeed");
    assert_eq!(actual.candidates, expected);
    assert!(actual.sector_reads > 0);

    let mut invalid_code = bytes;
    invalid_code[field.data_offset + 16] = u8::MAX;
    let checksum = crc32fast::hash(&invalid_code[FIXED_HEADER_BYTES..]);
    put_u32(&mut invalid_code, CHECKSUM_OFFSET, checksum);
    assert!(!validates(
        Some(&invalid_code),
        &registry,
        &schema,
        7,
        "source"
    ));
}

#[test]
fn zero_pq_chunks_keep_diskann_on_the_full_vector_record_path() {
    let (schema, _docs, registry) = pq_fixture(4, 16, 0);
    let prepared = prepare(&registry, &schema, "source").expect("layout must prepare");
    let field = &prepared.fields[0];
    assert!(field.pq.is_none());
    assert_eq!(field.index_type, crate::IndexType::Diskann);
    assert_eq!(field.record_bytes, 160);
    let bytes = encode(&registry, &schema, 7, "source")
        .expect("sidecar must encode")
        .expect("DiskANN requires a sidecar");
    assert!(validates(Some(&bytes), &registry, &schema, 7, "source"));
}
