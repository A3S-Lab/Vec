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
implemented and are not advertised as active indexes.

## Verified baseline

As of 2026-08-30, the storage format is version 2 and the following behaviour
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

Format version 1 was never released and is not opened by the version 2 reader.
Fault-injection hooks, WAL-pruning crash tests, real approximate indexes,
indexed FTS, fuzzing, and the full platform matrix remain roadmap work.

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
cargo test --manifest-path crates/vec/Cargo.toml --no-default-features
cargo test --manifest-path crates/vec/Cargo.toml --all-features
cargo clippy --manifest-path crates/vec/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path crates/vec/Cargo.toml --all-targets --all-features -- -D warnings
```

This repository is `A3S-Lab/Vec`; the A3S monorepo consumes it as the
`crates/vec` git submodule. The monorepo root remains a composition repository
and is not a Rust workspace.
