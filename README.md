<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec: fast process-local vector and full-text retrieval for Coding Agent workspaces">
</p>

`a3s-vec` is a native Rust, process-local retrieval engine for Coding Agent
workspaces. It combines dense and sparse vectors, scalar filtering, and BM25
inside one durable collection—without a server process or a C/C++ runtime.

The project is an active prototype. HNSW, IVF, HNSW/IVF RaBitQ, L2 Vamana,
product-quantized L2 DiskANN with native sector-aligned positioned traversal,
scalar inverted indexes, and FTS are live;
exact execution remains the correctness oracle whenever an index is missing,
stale, or not selective enough.

[Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) ·
[Reproducible benchmarks](BENCHMARKS.md)

## What it delivers

| Need | Current implementation |
| --- | --- |
| Local semantic retrieval | Dense and sparse exact search, HNSW, IVF, HNSW/IVF RaBitQ, L2 Vamana, PQ/ADC DiskANN, and exact re-ranking |
| Workspace text search | BM25, Unicode n-grams, boolean groups, exact phrases, and token filters |
| Structured narrowing | Typed scalar indexes, range/null/IN/wildcard predicates, and bitmap prefiltering |
| Durable embedding | WAL, checksummed snapshots, manifest commits, file locking, a validated derived-index cache, and a Vamana/DiskANN sector sidecar |
| Predictable failure | Typed validation errors and exact fallbacks instead of silent approximation |

All vector, scalar, and FTS indexes share one revisioned `u64` ordinal domain.
That lets the planner compose bitmaps and candidates without building
query-sized primary-key maps, then resolve only the exact top-k documents.

## Measured proof

`cargo bench --bench structured_fts` builds 25,000 workspace-shaped documents
and checks every indexed result and public score bit against scan execution.
The second of two consecutive post-change runs on the current development
machine produced:

| Query | Planner path | Candidates/query | Latency/query |
| --- | --- | ---: | ---: |
| Selective phrase | Indexed | 1 | 3.50 µs |
| Selective phrase baseline | Scan | 25,000 | 13.22 ms |
| Selective required + optional | Indexed | 1 | 1.88 µs |
| Selective required + optional baseline | Scan | 25,000 | 13.47 ms |
| Common phrase | Automatic scan fallback | 25,000 | 13.52 ms |
| Broad boolean + NOT | Automatic scan fallback | 25,000 | 14.70 ms |

The selective cases reduce scored candidates by 25,000×. Broad structured
queries deliberately switch to the exact scan path when candidate-set work is
unlikely to pay for itself. These are local regression measurements—not a
cross-project zvec benchmark. Full methodology and repeated observations live
in [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

The A3S monorepo consumes this repository as `crates/vec`. Until a crate
release is published, use a path dependency from the monorepo root:

```toml
[dependencies]
a3s-vec = { path = "crates/vec" }
```

This complete example creates a durable FTS collection, inserts one workspace
document, and executes a structured query:

```rust
use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams,
    Result, SearchQuery,
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
    let query = SearchQuery::fts("body", &expression, 10)?;
    let hits = collection.query(&query)?;

    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

## Retrieval capabilities

### Vectors and indexes

- Dense FP16, FP32, FP64, INT4, INT8, INT16, Binary32, and Binary64 payloads.
- Sparse FP16 and FP32 payloads.
- Exact L2, inner product, cosine, and MIPS-L2 scoring with `f64`
  intermediates.
- Native HNSW and IVF candidate generation with exact full-vector re-ranking.
- Portable HNSW/IVF RaBitQ with deterministic random rotation, compact
  1-to-9-bit codes, bounded refinement, and exact full-vector re-ranking.
- Deterministic two-pass L2 Vamana construction, bounded `list_size` search,
  incremental overlays, and exact full-vector re-ranking.
- Deterministic product-quantizer training with up to 256 centroids per chunk,
  one-byte codes, query-local ADC tables, and exact full-vector re-ranking.
- Native 4 KiB-sector Vamana/DiskANN files with fixed full-vector or PQ-code
  records, CRC validation, bounded positioned reads, and failure-closed
  in-memory fallback.
- Index-only FP16, symmetric INT8, and symmetric INT4 quantization.
- Scalar inverted indexes for equality, range, `IN`, null, wildcard, prefix,
  suffix, and boolean filter composition.

Vamana accepts unquantized L2 vectors. `IndexParams::diskann` uses the same
deterministic graph and enables corpus-trained PQ when `pq_chunk_num > 0`;
zero selects full-vector graph scoring. A freshly built or rebuilt generation
traverses in memory. After a validated cache reopen, bounded queries load only
the required fixed-record sectors through positioned reads and retain a
request-local sector/node cache. PQ queries build one centroid-distance table
and sum code distances during graph traversal. Incremental overlays share the
reader; a full rebuild retrains the codebook and invalidates the reader until
the next validated reopen. A short read or malformed record falls back to the
equivalent in-memory full-vector or ADC graph, and authoritative vectors still
perform final re-ranking. The file is an A3S-native format, not the Microsoft
DiskANN C++ format; mmap and asynchronous I/O remain optional accelerators.
The same query control selects the bounded list size for both index types:

```rust
use a3s_vec::{DiskannQueryParams, IndexParams, MetricType, Result, SearchQuery};

fn configure_diskann_pq(query: &mut SearchQuery) -> Result<IndexParams> {
    query.set_diskann_params(DiskannQueryParams::new(64))?;
    IndexParams::diskann(MetricType::L2, 32, 96, 8)
}
```

RaBitQ is a separate HNSW/IVF index family. It trains deterministic centers,
applies a four-round signed Hadamard rotation, and uses the compact code only
for candidate traversal or refinement. The authoritative vector remains the
source of public scores. HNSW defaults to seven bits and 16 centers; the typed
options constructor exposes bit width, center count, and sample count. IVF
uses `scale_factor * topk` as the bounded exact-refiner set:

```rust
use a3s_vec::{
    IndexParams, IvfRabitqQueryParams, MetricType, Result, SearchQuery,
};

fn configure_rabitq(query: &mut SearchQuery) -> Result<IndexParams> {
    let mut controls = IvfRabitqQueryParams::new(8, 0.0, false, true);
    controls.set_scale_factor(8.0)?;
    query.set_ivf_rabitq_params(controls)?;
    IndexParams::ivf_rabitq(MetricType::Cosine, 64, 7, 1_000)
}
```

Non-L2 Vamana/DiskANN, Vamana graph saturation/occlusion tuning, and standalone
Vamana quantization fail at schema validation until they have verified
execution implementations.

### Full-text search

The FTS pipeline analyzes documents and queries with the same ordered tokenizer
and filter configuration.

| Component | Supported values |
| --- | --- |
| Tokenizer | `standard`, `whitespace`, Unicode `ngram`, optional `jieba` / `jieba_accurate` |
| Token filter | `lowercase`, `ascii_folding`, `stemmer` |
| Query syntax | `AND`, `OR`, `NOT`, parentheses, `+` required, `-` prohibited, escaped characters, exact quoted phrases |
| Default operator | `OR` for compatibility, or explicit `AND` |

Omitting `filters` selects `lowercase`. Passing an explicit empty slice keeps
the standard, whitespace, and n-gram tokenizer output case-sensitive. Filters
run in declaration order on both indexed text and query text.

```rust
use a3s_vec::{IndexParams, Result};

fn workspace_text_index() -> Result<IndexParams> {
    IndexParams::fts(
        Some("standard"),
        Some(&["lowercase", "ascii_folding", "stemmer"]),
        Some(r#"{"max_token_length":255,"stemmer_lang":"english"}"#),
    )
}
```

The Snowball stemmer supports Arabic, Danish, Dutch, English, Finnish, French,
German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian,
Spanish, Swedish, Tamil, and Turkish. ASCII folding uses Unicode decomposition
plus common Latin compatibility mappings; it is not advertised as byte-for-byte
equivalent to every zvec folding table.

The n-gram tokenizer defaults to Unicode bigrams. `ngram_min`,
`ngram_max`, and `token_chars` configure its range and accepted Unicode
character classes:

```rust
use a3s_vec::{IndexParams, Result};

fn identifier_index() -> Result<IndexParams> {
    IndexParams::fts(
        Some("ngram"),
        None,
        Some(
            r#"{"ngram_min":2,"ngram_max":3,"token_chars":["letter","digit"]}"#,
        ),
    )
}
```

For selective identifier/path queries, `default_operator=AND` starts from the
shortest posting. Structured expressions build exact boolean candidate sets;
phrases verify token adjacency only for candidates. The planner falls back to
scan execution for broad expressions and keeps indexed refinement when a scalar
prefilter is available.

Wildcard terms, field-qualified terms, boosts, fuzzy/proximity suffixes, and
range syntax currently return `NotSupported` rather than changing query
meaning.

## How execution stays exact

```text
request
  → capture one immutable schema/document/index revision
  → validate route, type, dimension, limits, and syntax
  → derive scalar and FTS candidate ordinals when selective
  → run HNSW/IVF/RaBitQ/Vamana/DiskANN or the exact vector path
  → verify filters and phrases against authoritative documents
  → exact-score, deterministic top-k, projection, and optional fusion
```

- Flat vector and scan BM25 execution are always available as reference paths.
- Every derived index generation is immutable and tagged with its source
  revision.
- HNSW/IVF/RaBitQ/Vamana/DiskANN candidates are re-ranked with authoritative
  vectors.
- Indexed and scan FTS share `f64` corpus/scoring primitives and produce
  bit-identical public scores in differential fixtures.
- Equal scores use ascending primary key as the deterministic tie-break.

## Persistence and recovery

Documents, snapshots, and WAL records are authoritative. The current storage
format is version 4: checksummed MessagePack snapshots plus a manifest-committed
WAL boundary. Version-3 JSON snapshots remain readable and upgrade at the next
writable checkpoint.

ANN, scalar, FTS, and the shared ordinal table are persisted separately as a
non-authoritative derived cache. Cache format 10 includes RaBitQ rotations,
centers, compact codes, Vamana/DiskANN graphs, PQ codebooks/codes, parsed
tokenizer, and ordered filter state. A Vamana or
DiskANN generation additionally
requires `indexes/diskann-graph.bin`: an A3S-native 4 KiB-sector mirror bound to
the same revision, schema digest, and manifest identity. Its header, metadata,
padding, full vectors or PQ codes/codebooks, graph edges, and CRC are validated
before a cache hit. A missing, stale, corrupt, structurally invalid, or pre-v10
cache/sidecar pair
is ignored and rebuilt from recovered documents; read-only opens never repair
it.

The public API supports read-only handles, configurable durability, explicit
`flush`, targeted `rebuild_index`, whole-registry `optimize`, and per-handle
cache-hit/query/candidate plus DiskANN sector-read telemetry.

## Current boundaries

| Area | Status |
| --- | --- |
| Flat, HNSW, IVF | Implemented |
| HNSW/IVF RaBitQ | Implemented for L2, inner product, and cosine with 1-to-9-bit codes and exact re-ranking |
| L2 Vamana traversal and incremental overlays | Implemented in memory and through positioned sidecar reads after reopen |
| L2 DiskANN PQ/ADC and incremental overlays | Implemented in memory and through positioned PQ-code reads after reopen |
| Sector-aligned native Vamana/DiskANN file | Implemented |
| Scalar inverted index | Implemented |
| BM25 + structured boolean/phrase FTS | Implemented |
| FTS wildcard/field/boost/fuzzy/range syntax | Not implemented |
| On-demand DiskANN query reader | Implemented with portable positioned reads; mmap/async acceleration remains roadmap |
| Product quantization / RaBitQ | PQ implemented for DiskANN / RaBitQ implemented for HNSW and IVF |
| Binary vector query execution | Not implemented |
| Alibaba C++ binary-format compatibility | Requires an explicit future importer/exporter |

`a3s-vec` follows zvec's Rust vocabulary where it is useful, but it is not a
binary-compatible clone. `zvec-core` remains a private pure-Rust algorithm
dependency; callers use only A3S-owned collection, schema, document, query, and
error contracts.

## Quality gates

Run checks inside this crate:

```sh
cargo fmt --all -- --check
cargo test
cargo test --no-default-features
cargo test --all-features
cargo +1.75.0 test --locked
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Reproducible performance fixtures:

```sh
cargo bench --bench ann_recall
cargo bench --bench filtered_ann
cargo bench --bench scalar_filter
cargo bench --bench fts_index
cargo bench --bench ngram_fts
cargo bench --bench structured_fts
cargo bench --bench reopen_index
```

The default, no-default-feature, all-feature, strict Clippy, rustdoc, and Rust
1.75 gates are maintained separately. The optional Jieba dependency chain
currently requires a newer Cargo because one transitive package uses a Rust
2024 manifest.

## Platform and ownership

The portable correctness path targets Linux x86_64/aarch64, Windows x86_64,
and macOS arm64/x86_64, with macOS 12.0 as the Intel deployment target. It does
not require `io_uring`, a C/C++ runtime, or architecture-specific SIMD.

`a3s-vec` owns retrieval, persistence, and index execution. Workspace
scanning, embedding model runtimes, Agent sessions, and UI policy belong to
their callers. The cross-project boundary is documented in the
[A3S local retrieval platform architecture](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md).

This repository is `A3S-Lab/Vec`; the A3S monorepo consumes it as the
`crates/vec` submodule. Licensed under [MIT](LICENSE).
