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
validated before execution or persistence. Storage format version 3 provides
generation-specific snapshots, manifest-committed WAL byte boundaries,
monotonic DML/schema revisions, read-only lifecycle semantics, and bounded
recovery reads. Native FP16, INT4, INT8, INT16, and binary payloads now have
strict physical-type validation and lossless persistence; dense and sparse
numeric vectors share exact L2/IP/cosine/MIPS-L2 scoring with `f64`
intermediates and are ranked before the public score is narrowed to `f32`.
Independent differential fixtures now cover deterministic dense/sparse scores,
filters, radius/top-k ordering, and scan BM25 corpus statistics. A fixed-seed
256-document corpus adds 100 dense metric/filter combinations and 24
BM25/filter combinations, including a flush/reopen boundary. Advanced FTS
syntax fails explicitly instead of being approximated. Concurrent public-API
fixtures prove serialized disjoint updates, revision-pinned iterators, and
atomic multi-document publication to readers. The external algorithm kernel
is private and has a negative compile-time API fixture. Inert process/
collection controls have been removed;
the retained durability policy and WAL checkpoint limits are connected and
tested. Missing, typed-null, and JSON-null behavior is checked for every scalar
and array type, and numeric scalar/array conversion is checked at both extrema
and beyond each representable boundary. Future index/query/schema tuning now
fails explicitly unless it has an exact execution consumer; Flat and scan FTS
telemetry no longer claim ANN or a built FTS index. The baseline has 71 passing
unit/integration tests plus four
compile-fail doctests in each of the default, no-default-feature, and all-
feature configurations. Formatting, default/all-feature Clippy with
`-D warnings`, and rustdoc are green. The full default-feature suite also
passes on the declared Rust 1.75 MSRV after constraining `zvec-core`'s broad
Rayon dependency to its compatible release line; optional Jieba still requires
a newer Cargo because its current compressed-dictionary chain uses Rust 2024
manifests. GitHub Actions runs the full default-feature suite on Linux x86_64
and arm64, Windows x86_64, and macOS arm64 and Intel, while separate jobs gate
Rust 1.75, formatting, all-feature Clippy/tests, and rustdoc. The Intel build
uses a macOS 12.0 deployment target; an actual macOS 12 runtime smoke still
requires a self-hosted or external runner.

Phase 3's portable implementation gate is complete: per-handle deterministic
fault injection covers all 18 write/sync/rename/prune boundaries, including
WAL and snapshot cleanup; lock conflicts include bounded owner metadata; and
both fixed-seed mutation fuzzing and a libFuzzer/AddressSanitizer smoke target
exercise recovery. The actual macOS 12 Intel runtime smoke remains external
runner work. Approximate index and indexed-FTS names are schema/query contracts
only; the unused exact-facade implementations were removed so they cannot be
mistaken for shipped ANN behaviour.

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
- Completed: table-driven nullability contracts cover absent fields, typed and
  JSON nulls, durable null updates, and required-field rejection for all 18
  scalar/array types. Numeric JSON scalar and array fixtures cover signed and
  unsigned extrema, wrong signs, fractions, overflowing members, and finite
  floating-point limits without narrowing or wraparound.
- Completed: the `zvec-core` implementation dependency is no longer re-exported
  through the A3S public surface; a compile-fail doctest guards the boundary.
- Partially completed: removed inert memory/thread/logging/I/O/mmap/buffer/
  segment controls, fixed process-versus-collection durability precedence, and
  added execution tests for WAL operation/byte checkpoint thresholds.
- Completed: future physical index descriptors, query tuning parameters,
  segment sizing, and non-zero schema-evolution concurrency fail with typed
  errors before mutation. Flat remains the sole ready exact index; scan FTS
  tokenizer configuration and multi-query fusion never increment ANN
  telemetry.
- Completed: dense/sparse FP16 bit encodings, signed INT4 range checks, native
  INT8/INT16 coordinates, and Binary32/Binary64 chunk/dimension contracts have
  positive, negative, typed-getter, and storage round-trip evidence. Every
  native numeric type matches the exact four-metric reference, FP16 conversion
  is round-to-nearest-even with bounded error, and FP64 scoring is not narrowed
  before accumulation.
- Completed: deterministic independent-reference fixtures for dense and sparse
  exact scan across all four metrics, filters, L2/similarity radius, top-k, and
  primary-key tie-breaking. FP64 close-score ordering is tested before the
  public `f32` narrowing boundary; negative L2 radius is rejected.
- Completed: independent scan-BM25 ranking over the text-bearing corpus,
  including a nullable missing-field case. Ambiguous expression forms are
  rejected, and unimplemented boolean, phrase, wildcard, and fielded syntax
  returns `NotSupported` rather than silently changing query meaning.
- Completed: a dependency-free fixed-seed generator produces 256 mixed
  documents and checks 100 dense metric/filter combinations plus 24
  BM25/filter combinations against independent references after persistence
  and reopen.
- Completed for hosted runners: the default suite runs on Linux x86_64/arm64,
  Windows x86_64, and macOS arm64/Intel. The Intel job compiles with a 12.0
  deployment target, but an actual macOS 12 runtime smoke remains open.
- Open: the macOS 12 Intel runtime required by the full Phase 1 exit gate.
  Scale-bearing FP16/INT8/INT4 index quantization with exact re-ranking remains
  Phase 4 work; binary query execution remains unsupported.

## Phase 2 — Correct in-memory collection

**Deliverables**

- Thread-safe collection handle and deterministic CRUD semantics.
- Exact flat dense/sparse search with L2, inner-product, cosine, and MIPS-L2.
- Filter parser/evaluator, radius filtering, top-k, fetch, and snapshot iterator.
- Result projection and per-document write results.

**Exit gate**

- Differential tests compare every query result with a simple reference scan;
  concurrent readers see a coherent revision while writes are serialized.

**Evidence landed on 2026-08-30**

- Completed for the current exact surface: independent dense and sparse
  references cover all four metrics, deterministic filtering, radius/top-k,
  score comparison, and primary-key ordering.
- Completed for the current in-process surface: concurrent disjoint updates
  preserve both patches and monotonic revisions; iterators retain one captured
  revision; synchronized readers racing repeated two-document upserts observe
  only a complete previous or next batch.
- Completed: the fixed-seed 256-document differential corpus and concurrency
  fixtures run in the hosted Linux x86_64/arm64, Windows x86_64, and macOS
  arm64/Intel matrix.

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
- Completed: a handle-local injector names all 18 WAL, snapshot, manifest, and
  cleanup boundaries. Crash-equivalent tests prove the manifest commit point,
  replacement of uncommitted WAL tails, orphan-candidate isolation, and safe
  interruption before/after WAL and snapshot removal. Pruned directories are
  synchronized where the platform supports directory fsync.
- Completed: a sidecar records bounded PID/acquisition-time diagnostics after
  the kernel lock succeeds. Keeping metadata separate from the locked handle
  makes it readable on Windows. Contention reports that record, while stale or
  malformed metadata never becomes lock authority and is replaced by the next
  exclusive owner.
- Completed: fixed-seed recovery mutation fuzzing flips every persisted byte,
  truncates and appends structural cases, and runs 256 combined mutations per
  manifest/snapshot/WAL file. A separate cargo-fuzz target exercises the same
  public recovery boundary under libFuzzer and AddressSanitizer; CI runs 256
  smoke iterations.
- Open: the actual macOS 12 Intel runtime smoke requires a self-hosted or
  external runner.

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
