//! Reproducible single-document ANN update fixture.

use a3s_vec::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, Durability, FieldSchema,
    HnswQueryParams, IndexParams, MetricType, SearchQuery,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DOCUMENTS: usize = 2_000;
const DIMENSIONS: usize = 32;
const WRITES: usize = 48;
const GENERATION_SIZES: [usize; 3] = [2_000, 20_000, 100_000];
const GENERATION_WRITES: usize = 64;

fn vector_for(index: usize) -> Vec<f32> {
    (0..DIMENSIONS)
        .map(|dimension| {
            let raw = (index * 31 + dimension * 17 + index * dimension * 3) % 1_009;
            let raw = u16::try_from(raw).expect("fixture value fits u16");
            (f32::from(raw) - 504.0) / 504.0
        })
        .collect()
}

fn replacement(index: usize) -> Doc {
    let mut doc =
        Doc::with_pk(format!("doc-{index:05}")).expect("document must have a valid primary key");
    doc.add_vector_f32("embedding", &vector_for(DOCUMENTS + index + 1))
        .expect("replacement vector must be valid");
    doc
}

#[allow(clippy::cast_precision_loss)]
fn micros_per_operation(duration: Duration, operations: usize) -> f64 {
    duration.as_micros() as f64 / operations as f64
}

fn manual_options() -> CollectionOptions {
    let mut options = CollectionOptions::new().expect("options must be valid");
    options
        .set_durability(Durability::Manual)
        .expect("manual durability must be valid");
    options
}

fn benchmark_document_generation(document_count: usize) -> f64 {
    let temporary = tempdir().expect("temporary directory must be available");
    let schema_name = format!("generation-{document_count}");
    let schema = CollectionSchema::builder(&schema_name)
        .add_field(
            FieldSchema::new("epoch", DataType::Int32, false, 0).expect("field must be valid"),
        )
        .build()
        .expect("schema must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema,
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let docs: Vec<Doc> = (0..document_count)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:07}")).expect("document must be valid");
            doc.add_i32("epoch", 0).expect("epoch must be valid");
            doc
        })
        .collect();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");

    let mut warmup = Doc::with_pk("doc-0000000").expect("document must be valid");
    warmup
        .add_i32("epoch", -1)
        .expect("warmup epoch must be valid");
    black_box(
        collection
            .update(&[&warmup])
            .expect("warmup update must succeed"),
    );

    let started = Instant::now();
    for epoch in 0..GENERATION_WRITES {
        let mut patch = Doc::with_pk("doc-0000000").expect("document must be valid");
        patch
            .add_i32(
                "epoch",
                i32::try_from(epoch).expect("benchmark epoch fits i32"),
            )
            .expect("epoch must be valid");
        black_box(
            collection
                .update(&[black_box(&patch)])
                .expect("scalar update must succeed"),
        );
    }
    micros_per_operation(started.elapsed(), GENERATION_WRITES)
}

fn main() {
    let temporary = tempdir().expect("temporary directory must be available");
    let schema = CollectionSchema::builder("incremental-write-benchmark")
        .add_field(
            FieldSchema::new(
                "embedding",
                DataType::VectorFp32,
                false,
                u32::try_from(DIMENSIONS).expect("dimension fits u32"),
            )
            .expect("field must be valid"),
        )
        .build()
        .expect("schema must be valid");
    let collection = Collection::create(
        temporary
            .path()
            .join("collection")
            .to_str()
            .expect("benchmark path must be UTF-8"),
        &schema,
        Some(&manual_options()),
    )
    .expect("collection must be created");
    let docs: Vec<Doc> = (0..DOCUMENTS)
        .map(|index| {
            let mut doc = Doc::with_pk(format!("doc-{index:05}")).expect("document must be valid");
            doc.add_vector_f32("embedding", &vector_for(index))
                .expect("vector must be valid");
            doc
        })
        .collect();
    collection
        .insert(&docs.iter().collect::<Vec<_>>())
        .expect("documents must be inserted");
    collection
        .create_index(
            "embedding",
            &IndexParams::hnsw(MetricType::Cosine, 16, 96).expect("HNSW descriptor must be valid"),
        )
        .expect("HNSW index must build");

    let warmup = replacement(0);
    black_box(
        collection
            .upsert(&[&warmup])
            .expect("warmup upsert must succeed"),
    );
    let started = Instant::now();
    for index in 1..=WRITES {
        let doc = replacement(index);
        black_box(
            collection
                .upsert(&[black_box(&doc)])
                .expect("incremental upsert must succeed"),
        );
    }
    let incremental_time = started.elapsed();

    let last = replacement(WRITES);
    let mut query = SearchQuery::new("embedding", &vector_for(DOCUMENTS + WRITES + 1), 1)
        .expect("verification query must be valid");
    query
        .set_hnsw_params(HnswQueryParams::new(64, 0.0, false, true))
        .expect("HNSW controls must be valid");
    assert_eq!(
        collection
            .query(&query)
            .expect("incremental generation must be searchable")[0]
            .get_pk(),
        last.get_pk()
    );

    let started = Instant::now();
    collection
        .rebuild_index("embedding")
        .expect("full HNSW rebuild must succeed");
    let rebuild_time = started.elapsed();

    println!("operation,documents,dimensions,samples,micros_per_operation");
    println!(
        "incremental_upsert,{DOCUMENTS},{DIMENSIONS},{WRITES},{:.2}",
        micros_per_operation(incremental_time, WRITES)
    );
    println!(
        "full_hnsw_rebuild,{DOCUMENTS},{DIMENSIONS},1,{:.2}",
        micros_per_operation(rebuild_time, 1)
    );
    for document_count in GENERATION_SIZES {
        println!(
            "scalar_generation_update,{document_count},0,{GENERATION_WRITES},{:.2}",
            benchmark_document_generation(document_count)
        );
    }
}
