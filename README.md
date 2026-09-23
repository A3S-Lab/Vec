<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec: in-process vector and full-text search">
</p>

<p align="center">
  <a href="https://crates.io/crates/a3s-vec"><img alt="crates.io" src="https://img.shields.io/crates/v/a3s-vec.svg"></a>
  <a href="https://docs.rs/a3s-vec"><img alt="docs.rs" src="https://img.shields.io/docsrs/a3s-vec"></a>
  <a href="https://github.com/A3S-Lab/Vec/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Vec/ci.yml?branch=main"></a>
  <img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.75-informational">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

# a3s-vec

In-process vector and full-text search for a Coding Agent workspace. A collection is a directory. The document snapshot and the WAL are the record. HNSW, IVF, RaBitQ, Vamana, DiskANN, scalar postings, and BM25 are built from that record and can be built again.

[Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [Testing](TESTING.md) · [Benchmarks](BENCHMARKS.md) · [docs.rs](https://docs.rs/a3s-vec)

Current release: [`0.1.8`](https://crates.io/crates/a3s-vec), tag `0.1.8` at `a26d59e`, crate SHA-256 `755d3bee…`. Details and older tags are in [RELEASE.md](RELEASE.md). `0.1.7` has the same library; its tests call `f32::next_up` and do not build on Rust 1.75.

## Install

```toml
[dependencies]
a3s-vec = "0.1.8"
```

Queries on a Tokio runtime use the same planner on `spawn_blocking`:

```toml
a3s-vec = { version = "0.1.8", features = ["async"] }
```

From the A3S monorepo: `a3s-vec = { path = "crates/vec" }`.

## Example

```rust
use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, IndexParams, MetricType, Result,
    SearchQuery,
};

fn main() -> Result<()> {
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 4)?;
    embedding.set_index_params(&IndexParams::flat(MetricType::Cosine)?)?;
    let schema = CollectionSchema::builder("notes")
        .add_field(embedding)
        .build()?;
    let collection = Collection::create("./notes-index", &schema, None)?;

    let mut doc = Doc::with_pk("src/index.rs")?;
    doc.add_vector_f32("embedding", &[1.0, 0.0, 0.0, 0.0])?;
    collection.insert(&[&doc])?;

    let hits = collection.query(&SearchQuery::new("embedding", &[1.0, 0.0, 0.0, 0.0], 1)?)?;
    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

Full text uses the same collection. `standard`, `whitespace`, and `ngram` are built in. `jieba` is the `jieba` feature.

```rust
use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams, Result,
    SearchQuery,
};

fn main() -> Result<()> {
    let mut body = FieldSchema::new("body", DataType::String, false, 0)?;
    body.set_index_params(&IndexParams::fts(Some("standard"), None, None)?)?;
    let schema = CollectionSchema::builder("workspace")
        .add_field(body)
        .build()?;
    let collection = Collection::create("./workspace-index", &schema, None)?;

    let mut doc = Doc::with_pk("src/index.rs")?;
    doc.add_string("body", "Rust vector database for workspace retrieval")?;
    collection.insert(&[&doc])?;

    let mut expression = Fts::new()?;
    expression.set_query_string("rust AND \"vector database\"")?;
    let hits = collection.query(&SearchQuery::fts("body", &expression, 10)?)?;
    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

```rust
async fn search(
    collection: &a3s_vec::Collection,
    query: &a3s_vec::SearchQuery,
) -> a3s_vec::Result<Vec<a3s_vec::Doc>> {
    collection.query_async(query).await
}
```

More programs: [`examples/README.md`](examples/README.md).

## Score

A query freezes one schema, document, and index revision, then checks the route, types, dimensions, and limits. An index may return candidates. The score written on the hit is the exact `f64` score of the stored vector. If the index is missing or stale, the scan reads the documents. Flat recall is 1. Equal scores keep the smaller primary key, and that key is resolved for the retained hits.

The process default for durability is `Always`. The default HNSW `ef` is 64. Exact re-rank stays on. IVF has no default `scale_factor`.

Filter parsing, tokenization, and FP16/INT8/INT4 quantization live in this crate. They are not re-exported.

## Indexes and fields

| Index | Notes |
| --- | --- |
| Flat | Exact scan of stored vectors. |
| HNSW | `m` on upper layers, `2m` on layer 0. |
| IVF | Optional SOAR. |
| HNSW RaBitQ, IVF RaBitQ | 1-to-9-bit codes for traversal. The public score is still the full vector. |
| Vamana | L2, inner product, cosine, MIPS-L2. |
| DiskANN | PQ/ADC, positioned reads or a validated anonymous mmap snapshot. |

BM25 supports boolean, phrase, wildcard, fuzzy, and range queries. A character trigram prunes wildcard and fuzzy expansion before the matcher runs.

Scalar filters are equality, range, `IN`, null, wildcard, prefix, suffix, and boolean composition. They use the same planner as vector and full-text search.

Encodings: FP16, FP32, FP64, INT4, INT8, INT16, Binary32, Binary64, sparse FP16, sparse FP32. Metrics: L2, inner product, cosine, MIPS-L2. Binary search is exact Flat L2 or Hamming.

`StorageCeilings` defaults to 8 GiB for the snapshot, the index cache, and WAL replay, and 512 MiB for a DiskANN sidecar. Zero is rejected. The library does not size these from host RAM.

Read-only open, flush, rebuild, optimize, health, and one owned maintenance scheduler are on `Collection`. DiskANN I/O, RaBitQ, analyzers, resource limits, and recovery are specified in [ARCHITECTURE.md](ARCHITECTURE.md).

## Measurement

One same-host run, Apple M5 Max, 2026-09-23, a3s-vec `0.1.7` ranking code (`0.1.8` changes a Rust 1.75 test helper) against zvec 0.7.0. Cosine, top-10, 32 queries × 3 rounds, batch 512, HNSW `m=16`, `ef_construction=96`, `ef=64`, one worker. a3s-vec re-ranks with `f64`. zvec runs with `is_using_refiner=False`. Insert time includes the final flush. 2,000×32 and 100,000×128 are three-process medians. 1,000,000×128 is one process.

Protocol: [docs/scale-compare-protocol.md](docs/scale-compare-protocol.md). Full tables: [BENCHMARKS.md](BENCHMARKS.md).

| Fixture | a3s insert | zvec insert | a3s Flat p50 | zvec Flat p50 | a3s HNSW build | zvec HNSW build | a3s HNSW p50 | zvec HNSW p50 | a3s Recall@10 | zvec Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000×32 | 28.2 ms | 28.8 ms | 11.5 µs | 53.4 µs | 66.2 ms | 67.8 ms | 29.3 µs | 57.6 µs | 1.0000 | 1.0000 |
| 100,000×128 | 346 ms | 964 ms | 770 µs | 1,694 µs | 12.4 s | 45.3 s | 98.3 µs | 136 µs | 0.6000 | 0.5813 |
| 1,000,000×128 | 3.33 s | 10.0 s | 7.54 ms | 23.6 ms | 202 s | 559 s | 136 µs | 180 µs | 0.3063 | 0.2594 |

Recall@10 at `ef=64` is the value this protocol produced. The 2026-09-20 million-document insert of `77,339.081` ms is an older unsplit measurement, kept in [BENCHMARKS.md](BENCHMARKS.md).

## Limits

- The on-disk format is not Alibaba zvec's C++ storage, and this crate does not speak that ABI.
- There is no C++ wire import or export, and no binary ANN.
- Async file reads and file-backed mmap wait on a failing test ([VEC-R2](ROADMAP.md)).
- macOS 12 Monterey on Intel is unsupported.

## Develop

```sh
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --all-features
cargo clippy --locked --all-targets -- -D warnings
cargo +1.75.0 test --locked
```

Hosted CI runs those gates, recovery fuzz, and smoke benches on Linux x86_64/aarch64, Windows x86_64, and macOS arm64/x86_64. The macOS deployment target is 15.0. The default build does not require `io_uring` or an architecture-specific SIMD path.

Repository: [A3S-Lab/Vec](https://github.com/A3S-Lab/Vec). The monorepo mounts it at `crates/vec`. [MIT](LICENSE).
