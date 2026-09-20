<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec: process-local vector and full-text retrieval for Coding Agent workspaces">
</p>

<p align="center">
  <a href="https://crates.io/crates/a3s-vec"><img alt="crates.io" src="https://img.shields.io/crates/v/a3s-vec.svg"></a>
  <a href="https://docs.rs/a3s-vec"><img alt="docs.rs" src="https://img.shields.io/docsrs/a3s-vec"></a>
  <a href="https://github.com/A3S-Lab/Vec/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Vec/ci.yml?branch=main"></a>
  <img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.75-informational">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

`a3s-vec` is a native Rust, **process-local** retrieval engine: dense and sparse
vectors, scalar filters, and BM25 live in one durable collection—no server
process and no C/C++ runtime.

**`0.1.1` is published on [crates.io](https://crates.io/crates/a3s-vec).** Tag
`0.1.1`, hosted CI, and the published crate checksum bind to one revision
([RELEASE.md](RELEASE.md)). Exact execution stays the correctness oracle when
an index is missing, stale, or not selective enough. macOS 12 Monterey Intel
is unsupported.

[Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) ·
[Benchmarks](BENCHMARKS.md) · [Release](RELEASE.md) ·
[docs.rs](https://docs.rs/a3s-vec)

## Why it exists

Coding Agent workspaces need retrieval that is local, durable, and honest about
scores. `a3s-vec` owns the collection, WAL/snapshots, indexes, and query
planner. Embedding models, workspace scanners, and UI policy stay with the
caller.

| Need | What you get |
| --- | --- |
| Semantic search | Exact dense/sparse scan, HNSW, IVF/SOAR, RaBitQ, Vamana, PQ/ADC DiskANN, then **exact** full-vector re-rank |
| Workspace text | BM25, Unicode n-grams, structured boolean/phrase/wildcard/fuzzy/range syntax |
| Structured filters | Typed scalar indexes composed with ANN/FTS through one shared `u64` ordinal domain |
| Durability | WAL, checksummed snapshots, file locking, derived-index cache, typed resource limits |
| Predictable failure | Validation errors and exact fallbacks—no silent approximation |

## Proof vs zvec (honest harness)

Same host, same fixture, one worker, identical HNSW controls (`m=16`,
`ef_construction=96`, `ef=64`). a3s-vec keeps exact re-ranking and `f64` public
scores; the zvec harness sets `is_using_refiner=False`. Medians of three
independent processes. Methodology and CSVs:
[BENCHMARKS.md](BENCHMARKS.md).

### Apple Silicon · 100k × 128 (macOS arm64)

| Engine | Index build | Query p50 | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.1** | **22.4 s** | **99.5 µs** | **0.6000** |
| zvec 0.7.0 | 50.8 s | 146.0 µs | 0.5844 |

≈ **2.27×** faster build, ≈ **1.47×** lower query p50, higher stable recall.

### Windows Xeon · 100k × 128

| Engine | Index build | Query p50 | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.1** | **49.3 s** | 355 µs | **0.6000** |
| zvec 0.7.0 | 70.9 s | **349 µs** | 0.5875 |

≈ **1.44×** faster build; query p50 within noise (~2%).

These rows are directional evidence for one host and parameter point—not an
SLO. Flat keeps the same exact `f64` public score contract (lazy rebuild into a
contiguous promoted-`f64` base, bit-identical SIMD kernels). On the honest
1-worker harness Flat **insert** beats zvec; Flat **query** is typically about
**1.5–2×** slower—matching the `f64` vs native `f32` arithmetic floor, not a
missing index. Do not lower `ef`, drop exact re-ranking, or switch public
scores to `f32` to manufacture a Flat query win. HNSW remains the path that
beats zvec on build and query under identical controls.

## Install

```toml
[dependencies]
a3s-vec = "0.1.1"
```

Optional Tokio-safe query entry points:

```toml
a3s-vec = { version = "0.1.1", features = ["async"] }
```

In the A3S monorepo you can still use a path dependency:

```toml
a3s-vec = { path = "crates/vec" }
```

## Quick start

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
    let hits = collection.query(&SearchQuery::fts("body", &expression, 10)?)?;

    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

Async apps use the same planner on Tokio's blocking pool:

```rust
use a3s_vec::{Collection, Doc, Result, SearchQuery};

async fn search(collection: &Collection, query: &SearchQuery) -> Result<Vec<Doc>> {
    collection.query_async(query).await
}
```

`query_async` / `multi_query_async` / `group_by_async` match sync results and
telemetry. Dropping the future does not cancel work already running on
`spawn_blocking`.

## How a query stays exact

```text
request
  → freeze one schema / document / index revision
  → validate route, types, dimensions, limits
  → compose scalar + FTS candidates when selective
  → ANN or exact vector path
  → verify filters / phrases on authoritative docs
  → exact-score, deterministic top-k, projection
```

All indexes share one revisioned ordinal domain, so the planner composes
bitmaps without query-sized primary-key maps and resolves keys only for the
final top-k. Equal scores break ties by ascending primary key.

## Capabilities (index)

| Family | Surface |
| --- | --- |
| Vectors | FP16/32/64, INT4/8/16, Binary32/64; sparse FP16/32; L2, IP, cosine, MIPS-L2 |
| ANN | HNSW, IVF (+ optional SOAR), HNSW/IVF RaBitQ, Vamana, PQ DiskANN (positioned or mmap sidecar) |
| FTS | `standard` / `whitespace` / `ngram` / optional `jieba`; lowercase, ASCII fold, Snowball stem |
| Filters | Equality, range, `IN`, null, wildcard/prefix/suffix, boolean composition |
| Ops | Read-only opens, flush, targeted rebuild, optimize, health, owned maintenance scheduler |

Executable examples are part of CI:

```sh
cargo run --locked --example crud_operations
cargo run --locked --example vector_search
cargo run --locked --example retrieval_workflows
```

See [`examples/README.md`](examples/README.md). Deep contracts for DiskANN I/O,
RaBitQ controls, FTS analyzer options, resource limits, and recovery live in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Boundaries

- Not a binary-compatible clone of Alibaba zvec storage or C++ ABI.
- `zvec-core` is a private pure-Rust algorithm dependency; the public API is
  A3S-owned.
- Binary ANN and Alibaba C++ wire import/export are deliberate non-goals.
- Native async file reads and direct file-backed mmap stay refused until an
  invariant fails ([VEC-R2](ROADMAP.md)).

## Quality gates

```sh
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --all-features
cargo clippy --locked --all-targets -- -D warnings
cargo +1.75.0 test --locked
```

Hosted CI repeats quality, MSRV, recovery fuzz, and the smoke performance
matrix on Linux/Windows/macOS (arm64 + Intel, deployment target 15.0).
Reproducible benches: [BENCHMARKS.md](BENCHMARKS.md).

## Platform and ownership

Correctness targets Linux x86_64/aarch64, Windows x86_64, and macOS
arm64/x86_64 (macOS 15.0+). No `io_uring`, C/C++ runtime, or mandatory
arch-specific SIMD.

This repository is [`A3S-Lab/Vec`](https://github.com/A3S-Lab/Vec); the A3S
monorepo consumes it as `crates/vec`. Cross-project boundary:
[retrieval platform architecture](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md).

Licensed under [MIT](LICENSE).
