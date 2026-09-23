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

# a3s-vec

**Process-local retrieval for Coding Agent workspaces.**

A collection is a durable log of documents. The document snapshot and the WAL
are the source of truth. HNSW, IVF, RaBitQ, Vamana, DiskANN, scalar postings,
and BM25 are derived indexes: they propose candidates, and the public score is
the exact `f64` re-rank of the authoritative vector. A missing or stale index
falls back to that scan. Equal scores keep the ascending primary key.

Dense and sparse vectors, BM25 full-text, and typed scalar filters share one
revisioned ordinal domain. There is no server process and no C/C++ runtime.
`0.1.7` implements the filter, tokenizer, and quantization kernel in this crate.
`0.1.8` keeps that crate and replaces one test helper so Rust 1.75 can compile
the suite (`f32::next_up` is newer than the MSRV).

**[`0.1.8`](https://crates.io/crates/a3s-vec)** · this release. The registry
checksum is recorded in [RELEASE.md](RELEASE.md) after `cargo publish`.
`0.1.7` is published (tag `0.1.7` @ `57fc476` · SHA-256 `90254cfd…`).
Its library matches this ranking code; its unit tests do not build on Rust 1.75.
`0.1.6` remains tag `0.1.6` · SHA-256 `67c238a0…`.
`0.1.5` remains tag `0.1.5` · SHA-256 `bc42798f…`.
`0.1.4` remains tag `0.1.4` · SHA-256 `15c4220d…`.

[Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) ·
[Testing](TESTING.md) · [Benchmarks](BENCHMARKS.md) ·
[docs.rs](https://docs.rs/a3s-vec)

---

## Features

| Feature | What it does |
| --- | --- |
| **One collection** | Vectors, FTS, and scalar indexes share one revisioned `u64` ordinal domain—compose filters without building query-sized primary-key maps. |
| **Exact-first correctness** | ANN proposes candidates; authoritative vectors **exact re-rank** with public `f64` scores. Missing or stale indexes fall back to exact scan—no silent approximation. |
| **ANN depth** | HNSW, IVF (+ optional SOAR), HNSW/IVF RaBitQ, Vamana, PQ/ADC DiskANN (positioned I/O or mmap sidecar). |
| **Workspace text** | BM25 with `standard` / `whitespace` / `ngram` / optional `jieba`; boolean, phrase, wildcard, fuzzy, range; character-trigram prune before matcher expansion. |
| **Typed filters** | Equality, range, `IN`, null, wildcard/prefix/suffix, boolean composition—same planner as ANN/FTS. |
| **Durability** | WAL, checksummed snapshots, file locking, derived-index cache, typed resource limits; typed `StorageCeilings` (defaults 8 GiB / 8 GiB / 8 GiB / 512 MiB DiskANN)—explicit policy, never host autodetection. |
| **Fail-closed API** | Unsupported routes and bad dimensions fail with typed errors before mutation. |
| **Embed anywhere** | Embedding models stay with the caller; a3s-vec owns storage, indexes, and planning. |

Native encodings: FP16/32/64, INT4/8/16, Binary32/64, sparse FP16/32. Metrics:
L2, IP, cosine, MIPS-L2.

---

## Why teams pick it

1. **Runs inside the agent process** — no sidecar DB to operate; open a path,
   insert, query.
2. **Scores you can defend** — public ranking uses exact `f64` re-scoring of
   authoritative vectors; Flat recall is 1.0 by construction.
3. **Hybrid without glue code** — semantic + lexical + structured predicates
   in one planner and one durable generation.
4. **One revision, one checksum** — hosted CI, the git tag, and the crates.io
   artifact bind to the same commit. `0.1.6` keeps a stale DiskANN sidecar from
   dropping the index cache. `0.1.5` and `0.1.4` remain their own published
   bindings.
5. **Measured against zvec 0.7.0** — same corpus, cosine, top-10, `m=16`,
   `ef_construction=96`, `ef=64`, one worker, exact re-rank kept. See the proof
   below. Million-document flush stays inside the 8 GiB storage ceilings.

What it is **not**: a hosted vector cloud, a zvec C++ ABI clone, or a claim of
universal engine ranking.

---

## Install

```toml
[dependencies]
a3s-vec = "0.1.8"
```

Tokio-facing queries (same planner on `spawn_blocking`):

```toml
a3s-vec = { version = "0.1.8", features = ["async"] }
```

Monorepo path dependency: `a3s-vec = { path = "crates/vec" }`.

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

```rust
async fn search(collection: &a3s_vec::Collection, query: &a3s_vec::SearchQuery)
    -> a3s_vec::Result<Vec<a3s_vec::Doc>>
{
    collection.query_async(query).await
}
```

CI examples: `crud_operations`, `vector_search`, `retrieval_workflows` — see
[`examples/README.md`](examples/README.md).

---

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

Equal scores break ties by ascending primary key. Keys resolve only for the
final top-k.

---

## Ops surface

Read-only opens, flush, targeted rebuild, optimize, health, and an owned
maintenance scheduler. Deep contracts for DiskANN I/O, RaBitQ, FTS analyzers,
`CollectionResourceLimits`, `StorageCeilings`, and recovery:
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Proof vs zvec (honest, not a crown)

Same-host evidence from one fresh protocol run—not a capacity SLO.
Protocol: [docs/scale-compare-protocol.md](docs/scale-compare-protocol.md) ·
[BENCHMARKS.md](BENCHMARKS.md).

Controls: SplitMix64 corpus, cosine, top-10, 32×3 queries, batch 512, HNSW
`m=16` / `ef_construction=96` / `ef=64`, one worker. a3s-vec keeps exact
re-rank and public `f64` scores; zvec uses `is_using_refiner=False`.
2,000×32 and 100,000×128 are three-process medians. 1,000,000×128 is one
process. Measured on the `0.1.7` ranking code, unchanged in `0.1.8` except the
Rust 1.75 test helper. Apple M5 Max, zvec `0.7.0`, 2026-09-23.
Insert time includes the final flush.

| Fixture | a3s insert | zvec insert | a3s Flat p50 | zvec Flat p50 | a3s HNSW build | zvec HNSW build | a3s HNSW p50 | zvec HNSW p50 | a3s Recall@10 | zvec Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000×32 | 28.2 ms | 28.8 ms | 11.5 µs | 53.4 µs | 66.2 ms | 67.8 ms | 29.3 µs | 57.6 µs | 1.0000 | 1.0000 |
| 100,000×128 | 346 ms | 964 ms | 770 µs | 1,694 µs | 12.4 s | 45.3 s | 98.3 µs | 136 µs | 0.6000 | 0.5813 |
| 1,000,000×128 | 3.33 s | 10.0 s | 7.54 ms | 23.6 ms | 202 s | 559 s | 136 µs | 180 µs | 0.3063 | 0.2594 |

The 2026-09-20 million-document insert of `77,339.081` ms stays an unsplit
historical measurement. Protocol-default Recall@10 is not an accuracy SLO.
Do not lower `ef`, drop exact re-ranking, or switch public scores to `f32`
to manufacture a win.

---

## Boundaries

- Not binary-compatible with Alibaba zvec storage or C++ ABI.
- Filter parsing, FTS tokenization, and FP16/INT8/INT4 index quantization are
  Rust owned by this crate. The public API is A3S-owned.
- Binary ANN and C++ wire import/export are deliberate non-goals.
- Native async file reads and direct file-backed mmap stay refused until an
  invariant fails ([VEC-R2](ROADMAP.md)).
- macOS 12 Monterey Intel is unsupported.

## Quality gates

```sh
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --all-features
cargo clippy --locked --all-targets -- -D warnings
cargo +1.75.0 test --locked
```

Hosted CI: quality, MSRV, recovery fuzz, and smoke performance on
Linux/Windows/macOS (arm64 + Intel, deployment target 15.0).

## Platform and ownership

Correctness targets Linux x86_64/aarch64, Windows x86_64, and macOS
arm64/x86_64 (macOS 15.0+). No `io_uring`, C/C++ runtime, or mandatory
arch-specific SIMD.

Repository: [`A3S-Lab/Vec`](https://github.com/A3S-Lab/Vec). The A3S monorepo
consumes it as `crates/vec`. Cross-project boundary:
[retrieval platform architecture](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md).

Licensed under [MIT](LICENSE).
