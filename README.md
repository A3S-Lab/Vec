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
Dense and sparse vectors, BM25 full-text, and typed scalar filters live in one
durable Rust collection—no server process and no C/C++ runtime.

**[`0.1.4` on crates.io](https://crates.io/crates/a3s-vec)** · published
(tag `0.1.4` · SHA-256 `15c4220d…` · [RELEASE.md](RELEASE.md))

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
4. **Published Enterprise GA** — hosted multi-platform CI, versioned release
   candidate, and crates.io checksum bind to one revision (`0.1.4` typed
   `StorageCeilings`).
5. **Competitive HNSW under an honest harness** — same knobs, one worker,
   exact re-rank kept; see proof below (directional, not an SLO). Million-document
   flush is unblocked on workstation hosts (8 GiB storage ceilings).

What it is **not**: a hosted vector cloud, a zvec C++ ABI clone, or a claim of
universal engine ranking.

---

## Install

```toml
[dependencies]
a3s-vec = "0.1.4"
```

Tokio-facing queries (same planner on `spawn_blocking`):

```toml
a3s-vec = { version = "0.1.4", features = ["async"] }
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

Directional same-host evidence—not a capacity SLO and not “full dominance.”
Protocol: [docs/scale-compare-protocol.md](docs/scale-compare-protocol.md) ·
[BENCHMARKS.md](BENCHMARKS.md).

Controls: SplitMix64 corpus, cosine, top-10, 32×3 queries, batch 512, HNSW
`m=16` / `ef_construction=96` / `ef=64`, one worker. a3s-vec keeps exact
re-rank + `f64` scores; zvec uses `is_using_refiner=False`. Medians of three
processes · package `0.1.3` · Apple M5 Max / macOS 26.6.2 arm64 · zvec 0.7.0
(100k table). The million-document table is a single same-host process under
the same controls after the 8 GiB storage ceilings in `0.1.3`.

### HNSW · 100k × 128 (fairness harness)

| Engine | Index build | Query p50 | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.3** | **26.3 s** | **103 µs** | **0.6000** |
| zvec 0.7.0 | 46.2 s | 149 µs | 0.5813 |

≈ **1.75×** faster build, ≈ **1.44×** lower query p50, higher stable recall.

### Flat · same one-worker pin

| Engine | Query p50 | Recall@10 |
| --- | ---: | ---: |
| a3s-vec 0.1.3 | 3,550 µs | **1.0000** |
| zvec 0.7.0 | **1,841 µs** | **1.0000** |

Exact Flat with public `f64` scores is about **1.93×** slower here by
contract. With the host default Rayon pool (product default), a3s-vec Flat
p50 falls to **~651 µs** on this machine—report separately; do not mix into
the HNSW fairness table.

### HNSW · 1M × 128 (same controls, single process)

| Engine | Insert | Index build | Query p50 | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **a3s-vec 0.1.3** | 77.3 s | **422 s** | **159 µs** | **5954** | 0.3063 |
| zvec 0.7.0 | **13.5 s** | 628 s | 231 µs | 4244 | 0.2437 |

Directional only: a3s builds and queries HNSW faster at this scale; zvec
loads Flat faster. Protocol-default recall is **not** an accuracy claim—
raise `ef` / `ef_construction` before quoting million-scale recall.

Do not lower `ef`, drop exact re-ranking, or switch public scores to `f32` to
manufacture a win.

---

## Boundaries

- Not binary-compatible with Alibaba zvec storage or C++ ABI.
- `zvec-core` is a private pure-Rust algorithm kernel; the public API is
  A3S-owned.
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
