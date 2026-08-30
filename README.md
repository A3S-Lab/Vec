# a3s-vec

`a3s-vec` is the native Rust, in-process vector and full-text engine planned
for A3S Code's optional `vgrep` capability. It owns collections, typed
documents, schema validation, persistence, vector indexes, scalar indexes, and
FTS/BM25. It does not own workspace scanning, Embedding model runtimes, Agent
sessions, or UI policy.

The cross-project ownership contract is in the
[A3S local retrieval platform architecture](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md).
The engine-specific design and gates are in
[`ARCHITECTURE.md`](ARCHITECTURE.md) and [`ROADMAP.md`](ROADMAP.md).

## Scope

- Pure Rust API and portable CPU correctness path.
- Dense and sparse vectors with L2, inner-product, cosine, and MIPS-style
  scoring.
- Typed scalar/array/null values, filters, projections, FTS/BM25, hybrid and
  multi-query fusion, group-by, and iterators.
- WAL, checksummed snapshots, manifest generations, recovery, and locking.
- Caller-owned Embedding traits; no implicit network access or model download.

The current implementation is a prototype. The collection query path is an
exact-scan correctness oracle; HNSW, IVF, DiskANN, and indexed FTS are not yet
implemented and are not advertised as active indexes. Only Flat vector
configuration is reported as a ready index, because it directly executes over
the authoritative document snapshot; scan-time FTS tokenizer configuration is
not reported as a built index.

`zvec-core` is a private algorithm dependency. It is not re-exported through
the public API, so callers depend only on the A3S-owned collection, schema,
document, query, error, and configuration contracts.

## Verified baseline

As of 2026-08-30, the storage format is version 3 and the following behaviour
has deterministic test evidence:

- immutable generation-specific snapshots with the manifest as the only
  generation commit point;
- revisioned and checksummed WAL records for insert, update, upsert, delete,
  and schema/backfill operations;
- a manifest-committed WAL byte boundary, so an incomplete or complete
  uncommitted tail is ignored and replaced by the next commit;
- read-only create/open/close semantics that do not create missing files or
  checkpoint on close;
- bounded manifest, snapshot, WAL-frame, and total-WAL recovery reads;
- checksum, committed truncation, orphan-generation, replay, and snapshot-size
  failure tests.

Native vector payloads are also authoritative and lossless at rest. Dense and
sparse FP16 store raw IEEE 754 half-precision bits; INT4 accepts only signed
coordinates in `-8..=7`; INT8 and INT16 keep their native integer coordinates;
and Binary32/Binary64 validate complete 32-/64-bit chunks and bit dimensions.
A schema accepts only its exact physical vector type. These native integer
forms are not scale-bearing ANN quantizers and INT4 is not yet nibble-packed.
All numeric dense and sparse forms execute the four exact metrics. Scoring uses
`f64` intermediates—including FP64 source vectors—and narrows only the public
result score to `f32` after exact ranking and a checked range conversion.
Deterministic differential fixtures compare dense and sparse filtering,
radius, top-k, score, and primary-key ordering with an independent reference
scan. A dependency-free fixed-seed generator also checks a persisted and
reopened 256-document corpus across 100 dense metric/filter combinations.
Euclidean radius is non-negative by contract. Binary search remains explicitly
unsupported.

Scan-time FTS has independent BM25 fixtures, including 24 generated
query/filter combinations, and defines its corpus as the documents that
contain the queried text field; nullable documents without that field do not
change document frequency or average length. Plain token
`match_string` and `query_string` expressions are supported. Boolean, phrase,
wildcard, and fielded query-string syntax returns `NotSupported` until Phase 5
implements it, and providing both expression forms is rejected as ambiguous.

The in-process concurrency contract has deterministic public-API evidence.
Concurrent writers are serialized without losing disjoint patches, a document
iterator remains pinned to the revision it captured while a later batch is
published, and synchronized readers never observe a partially published
multi-document batch.

The public contract now also validates every query before exact execution:
dense dimensions are checked across all numeric vector types and metrics,
dense/sparse/FTS routes must match their schema field, and unsupported binary
or sparse source-ID routes fail with typed errors. JSON adapter values are
canonicalized to their declared scalar or array type at the write boundary;
incompatible, overflowing, and binary JSON values are rejected. Table-driven
contracts cover missing and explicit nulls for all 18 scalar/array types,
integer and floating-point extrema, array-member overflow, and durable null
updates. Schema backfills and replacement upserts are validated against the
resulting complete document.

The current baseline has 71 passing unit and integration tests plus four
compile-fail API-boundary doctests in each of the default, no-default-feature,
and all-feature configurations. Strict Clippy and rustdoc warning gates pass
for the same source revision. The default test suite also passes on the
declared Rust 1.75 MSRV; the optional Jieba dependency chain currently requires
a newer Cargo with Rust 2024-edition manifest support.

Format versions 1 and 2 were prototype-only and are not opened by the version
3 reader. Version 3 changes sparse FP16 persistence from misleading `f32`
coordinates to raw half-precision bits; rejecting older manifests prevents a
new payload from being silently reinterpreted by an older reader.
GitHub Actions runs the full default-feature suite on Linux x86_64/arm64,
Windows x86_64, and macOS arm64/Intel. The Intel job uses a macOS 12.0
deployment target, but GitHub's hosted runner executes a newer macOS release;
an actual macOS 12 Intel runtime smoke remains open. Per-handle fault injection
checks all 18 WAL/snapshot/manifest write, sync, rename, and prune boundaries;
lock contention reports recorded owner PID/time, and stale metadata never acts
as lock authority. A fixed-seed recovery mutation corpus and a separate
libFuzzer/AddressSanitizer target exercise manifest, snapshot, and WAL damage;
CI runs a bounded fuzz smoke. Real approximate indexes, indexed FTS,
scale-bearing index quantization, and binary search remain roadmap work.

## Configuration policy

Only controls with an execution consumer are public. `ConfigBuilder` sets the
process durability default and the WAL operation/byte checkpoint thresholds;
`CollectionOptions` exposes read-only mode and an optional per-collection
durability override. A collection without an override inherits the process
durability policy and captures it when the collection is created or opened;
later process reconfiguration does not change an active handle.

Memory/thread/logging controls, selectable I/O backends, mmap/buffer/segment
knobs, and their placeholder types are not exposed until distinct bounded
implementations exist. Compile-fail doctests guard this boundary. Future index
descriptors remain constructible for adapter compatibility, but attaching or
building them returns `NotSupported`. Future query setters return
`NotSupported` without mutating the query, and deserialized future or unknown
parameters are rejected again at execution. Non-zero schema segment sizing and
schema-evolution concurrency are rejected until they have execution owners.

## Platform policy

The baseline must compile and run on Linux x86_64/aarch64, Windows x86_64, and
macOS arm64/x86_64 with macOS 12.0 as the Intel minimum. C/C++ runtimes,
Linux-only `io_uring`, and required architecture-specific SIMD are not part of
the correctness path.

The default feature set is pure Rust and is tested on Rust 1.75. `zvec-core`
currently permits any Rayon 1.x release, so this crate pins Rayon to the latest
line compatible with that MSRV. Chinese Jieba tokenization is opt-in with
`--features jieba`; that feature currently pulls Jieba's compressed dictionary
build chain, including Rust 2024-edition manifests, `zstd-sys`, and a C
compiler. Requesting the `jieba` tokenizer without the feature returns
`NotSupported` instead of silently using a different tokenizer.

## Development

Run checks from this crate directory or with its manifest:

```sh
cargo fmt --manifest-path crates/vec/Cargo.toml -- --check
cargo test --manifest-path crates/vec/Cargo.toml
cargo test --manifest-path crates/vec/Cargo.toml --no-default-features
cargo test --manifest-path crates/vec/Cargo.toml --all-features
cargo +1.75.0 test --manifest-path crates/vec/Cargo.toml --locked
cargo clippy --manifest-path crates/vec/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path crates/vec/Cargo.toml --all-targets --all-features -- -D warnings
```

This repository is `A3S-Lab/Vec`; the A3S monorepo consumes it as the
`crates/vec` git submodule. The monorepo root remains a composition repository
and is not a Rust workspace.
