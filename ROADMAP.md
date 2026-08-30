# A3S Vec Roadmap

This is the engine-only roadmap. The dependency order for integrating the
engine into `a3s-code`, exposing `vgrep`, and removing the duplicate SQLite/
BM25 workspace paths is maintained in the
[A3S local retrieval platform roadmap](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-roadmap.md).

The roadmap is ordered by dependency and by the cost of being wrong. Each
phase has an explicit exit gate; a later approximate or optimized feature does
not replace an earlier correctness gate.

## Current implementation status

**2026-08-30:** Phase 1's query/write contract hardening and Phase 3's core
recovery transaction are implemented. Query routes, dense dimensions, sparse
indices, JSON adapter values, schema defaults, and replacement upserts are
validated before execution or persistence. Storage format version 2 provides
generation-specific snapshots, manifest-committed WAL byte boundaries,
monotonic DML/schema revisions, read-only lifecycle semantics, and bounded
recovery reads. The external algorithm kernel is private and has a negative
compile-time API fixture. The baseline has 26 passing unit/integration tests
plus that compile-fail doctest in each of the default, no-default-feature, and
all-feature configurations. Formatting, default/all-feature Clippy with
`-D warnings`, and rustdoc are green.

Phase 3 is not complete: deterministic fault-injection hooks, crash tests at
every fsync/rename/prune boundary, stale-lock diagnostics, fuzzing, and the
platform matrix remain open. Phase 2 also still needs differential and
concurrency evidence. Approximate index and indexed-FTS names are schema/query
contracts only; the unused exact-facade implementations were removed so they
cannot be mistaken for shipped ANN behaviour.

## Phase 0 — Contract and compatibility baseline

**Deliverables**

- Freeze the zvec Rust capability matrix and the `a3s_vec` naming policy.
- Add the crate skeleton, license/attribution, feature flags, and API examples.
- Define the on-disk format version, error/status mapping, and platform matrix.

**Exit gate**

- `cargo check`, rustdoc, and the basic zvec-style example compile on Linux and
  macOS arm64; the public API contains no C/C++ or Python dependency.

## Phase 1 — Types, schema, and document contract

**Deliverables**

- Complete scalar, array, dense, sparse, binary, and quantized data types.
- Builders and validation for collection/field/vector schemas and index params.
- Typed `Doc` setters/getters, null handling, serde round trips, and projection.
- Typed error taxonomy with stable status codes.

**Exit gate**

- Every supported type has positive and negative tests, including dimension,
  nullability, duplicate-name, and overflow cases.

**Evidence landed on 2026-08-30**

- Completed: dense query dimension errors for every current numeric vector
  type and L2/IP/cosine/MIPS-L2 metric; route/type checks for dense, sparse,
  scalar, and FTS fields; explicit unsupported errors for binary search and
  sparse source-ID search.
- Completed: schema-aware JSON adapter coercion for every supported scalar and
  non-binary array type, with incompatible and binary values rejected before
  WAL append. Recovered documents use the same normalization and validation.
- Completed: typed schema-default backfill validation and complete-document
  validation for replacement upserts.
- Completed: the `zvec-core` implementation dependency is no longer re-exported
  through the A3S public surface; a compile-fail doctest guards the boundary.
- Open: encoded binary and quantized round trips/error bounds, broader
  nullability/overflow fixtures, and the supported-platform matrix required by
  the full Phase 1 exit gate.

## Phase 2 — Correct in-memory collection

**Deliverables**

- Thread-safe collection handle and deterministic CRUD semantics.
- Exact flat dense/sparse search with L2, inner-product, cosine, and MIPS-L2.
- Filter parser/evaluator, radius filtering, top-k, fetch, and snapshot iterator.
- Result projection and per-document write results.

**Exit gate**

- Differential tests compare every query result with a simple reference scan;
  concurrent readers see a coherent revision while writes are serialized.

## Phase 3 — Durability and recovery

**Deliverables**

- Framed CRC WAL, atomic snapshots, manifest generations, and replay.
- `always`, `interval`, and `manual` durability policies.
- POSIX single-writer lock, read-only opens, stale-lock diagnostics.
- Fault-injection hooks for interrupted writes and checkpoints.

**Exit gate**

- Restart and crash-recovery tests cover insert/update/upsert/delete,
  schema changes, partial final frames, checksum failures, and WAL pruning.

**Evidence landed on 2026-08-30**

- Completed: versioned CRC WAL records with monotonic operation identity,
  immutable snapshot generations, one manifest commit point, and replay to the
  manifest revision.
- Completed: read-only create rejection, side-effect-free close, existing-lock
  requirement, and explicit manual flush synchronization.
- Completed: restart tests for every DML operation and schema add/backfill,
  rename, and drop; corruption tests for checksum mismatch, committed
  truncation, partial uncommitted tails, orphan snapshots, and oversized
  snapshots.
- Open: fault-injection hooks, every-boundary crash matrix, explicit WAL-prune
  crash evidence, stale-lock owner diagnostics, recovery fuzzing, and platform
  CI.

## Phase 4 — Memory ANN indexes

**Deliverables**

- HNSW build/search/update with bounded `ef` and deterministic seeds.
- IVF centroids/postings with `nprobe` and configurable training iterations.
- Runtime create/drop/rebuild/optimize and generation-safe publication.
- FP16, INT8, and INT4 scalar quantizers with exact re-ranking.

**Exit gate**

- ANN results always match the flat reference when configured for exhaustive
  search; recall/latency benchmark fixtures and stale-index fallback pass.

## Phase 5 — Structured and full-text retrieval

**Deliverables**

- Equality/range scalar inverted indexes and bitmap pre-filtering.
- Standard, whitespace, n-gram, and optional jieba tokenizers.
- BM25 FTS with boolean operators, phrases, prefix/suffix, and filters.
- Dense + sparse + FTS hybrid query planning.

**Exit gate**

- FTS ranking and filter semantics have golden fixtures; index and fallback
  paths return the same eligible document set.

## Phase 6 — DiskANN family and compression

**Deliverables**

- Vamana graph construction and sector-aligned DiskANN files.
- PQ codebook training/ADC, optional RaBitQ, mmap/pread readers, and delta
  documents for post-build writes.
- Query-time beam/list parameters and full-vector refinement.

**Exit gate**

- Index files survive reopen and checksum validation; Linux and macOS Intel
  use the correct I/O backend; recall and corruption tests pass.

## Phase 7 — Advanced collection API

**Deliverables**

- Multi-query routes, RRF/weighted reranking, score normalization, and
  group-by top-k.
- Add/alter/rename/drop columns with backfill and schema revisions.
- Background compaction, index progress, collection statistics, and health.
- Caller-owned dense/sparse embedding traits and an optional query executor.

**Exit gate**

- zvec-style Rust examples for CRUD, vector/FTS/hybrid search, multi-query,
  group-by, iterator, and schema evolution run unchanged after mechanical
  namespace replacement (`zvec_rust` → `a3s_vec`).

## Phase 8 — A3S integration and release hardening

**Deliverables**

- Integrate the crate behind an explicit A3S Code/Memory adapter; keep the
  collection API independent of CLI policy.
- Add benchmarks, fuzz targets, memory/CPU limits, and observability hooks.
- Add Linux/macOS arm64/macOS x86_64/Windows CI, including macOS 12 Intel.
- Publish README, migration notes, API docs, and a versioned release artifact.

**Release gate**

- `cargo fmt --check`, `cargo clippy -- -D warnings`, unit/integration/fuzz
  smoke tests, recovery suite, benchmark report, and Intel macOS 12 runtime
  smoke all pass. No feature is advertised unless its gate has evidence.

## Deliberate boundaries

- Python, Node, Go, Dart, and C ABI bindings are not part of `a3s-vec`.
- Network-backed embedding providers and implicit model downloads are adapters,
  not core database behavior.
- Binary compatibility with Alibaba's C++ storage is an explicit import/export
  task, not an assumption of the native A3S format.

## Immediate implementation order

1. Land the contract/types and reference flat engine.
2. Land WAL/snapshot recovery before ANN optimization.
3. Add HNSW/IVF and FTS/scalar indexes behind the same planner contracts.
4. Add DiskANN/PQ/RaBitQ only after the exact reference and corruption tests are
   green.
5. Finish API compatibility, Intel validation, and A3S integration as release
   work rather than mixing them into the storage core.
