# a3s-vec

`a3s-vec` is the native Rust, in-process vector and full-text engine planned
for A3S Code's optional `vgrep` capability. It owns collections, typed
documents, schema validation, persistence, vector indexes, scalar indexes, and
FTS/BM25. It does not own workspace scanning, Embedding model runtimes, Agent
sessions, or UI policy.

The cross-project ownership contract is in
[`../../docs/retrieval-platform-architecture.md`](../../docs/retrieval-platform-architecture.md).
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

The current implementation is a prototype. HNSW, IVF, and DiskANN modules may
initially use an exact-flat correctness facade until their independent recall,
latency, corruption, and portability gates pass.

## Platform policy

The baseline must compile and run on Linux x86_64/aarch64, Windows x86_64, and
macOS arm64/x86_64 with macOS 12.0 as the Intel minimum. C/C++ runtimes,
Linux-only `io_uring`, and required architecture-specific SIMD are not part of
the correctness path.

## Development

Run checks from this crate directory or with its manifest:

```sh
cargo fmt --manifest-path crates/vec/Cargo.toml -- --check
cargo test --manifest-path crates/vec/Cargo.toml
cargo clippy --manifest-path crates/vec/Cargo.toml --all-targets -- -D warnings
```

Before this crate is published, it must be moved to the external
`A3S-Lab/Vec` repository and added as a submodule. The monorepo root remains a
composition repository and is not a Rust workspace.

