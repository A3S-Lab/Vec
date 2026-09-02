# A3S Vec Roadmap

This is the engine-only roadmap. The dependency order for integrating the
engine into `a3s-code`, exposing `vgrep`, and removing the duplicate SQLite/
BM25 workspace paths is maintained in the
[A3S local retrieval platform roadmap](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-roadmap.md).

The roadmap is ordered by dependency and by the cost of being wrong. Each
phase has an explicit exit gate; a later approximate or optimized feature does
not replace an earlier correctness gate.

## Current implementation status

**2026-09-02:** Phase 1's query/write contract hardening, Phase 3's core
recovery transaction, and Phase 4's in-memory ANN gate are implemented. Query
routes, dense dimensions, sparse
indices, JSON adapter values, schema defaults, and replacement upserts are
validated before execution or persistence. Storage format version 4 provides
compact MessagePack generation snapshots, manifest-committed WAL byte
boundaries, monotonic DML/schema revisions, read-only lifecycle semantics, and
bounded recovery reads. Version-3 JSON snapshots remain readable and upgrade
atomically at the next writable checkpoint. Native FP16, INT4, INT8, INT16,
and binary payloads now have strict physical-type validation and lossless
persistence. Dense and sparse numeric vectors share exact
L2/IP/cosine/MIPS-L2 scoring; Binary32/Binary64 exact L2 uses the XOR Hamming
count as squared distance. All routes rank with `f64` intermediates before the
public score is narrowed to `f32`. Dense, sparse, and binary queries can resolve
their authoritative payload from a source document ID; missing documents and
missing source vectors have distinct typed errors.
The fluent `SearchQueryBuilder` constructs dense, binary, or pure FTS routes,
rejects ambiguous route combinations, and query results can expose the
generation ordinal through the explicit `include_doc_id` control.
Independent differential fixtures now cover deterministic dense/sparse scores,
filters, radius/top-k ordering, and scan BM25 corpus statistics. A fixed-seed
256-document corpus adds 100 dense metric/filter combinations and 24
BM25/filter combinations, including a flush/reopen boundary. Structured FTS
now executes explicit boolean groups, required/prohibited clauses, wildcard
and fuzzy terms, field qualifiers, finite boosts, lexical ranges, and exact or
ordered-proximity phrases with shared index/scan semantics. Concurrent public-API
fixtures prove serialized disjoint updates, revision-pinned iterators, and
atomic multi-document publication to readers. The external algorithm kernel
is private and has a negative compile-time API fixture. Inert process/
collection controls have been removed;
the retained durability policy and WAL checkpoint limits are connected and
tested. Missing, typed-null, and JSON-null behavior is checked for every scalar
and array type, and numeric scalar/array conversion is checked at both extrema
and beyond each representable boundary. Future index/query/schema tuning now
fails explicitly unless it has an execution consumer; Flat and unindexed scan
FTS telemetry do not claim approximate or physical-index execution. HNSW, IVF,
HNSW/IVF RaBitQ, metric-aware Vamana and DiskANN/PQ (L2, inner product, cosine,
and MIPS-L2) with in-memory, positioned, or immutable mmap-snapshot traversal,
and FTS are live derived indexes with dedicated telemetry.
Document generations
share unchanged `Arc<Doc>` values through a persistent ordered tree, so ordinary
writes copy only an O(log N) tree path. Indexed mutations share an immutable ANN
base plus bounded delta/tombstone overlays instead of rebuilding every graph on
every write. Phase 5's scalar-inverted slice is also live: revisioned persistent
posting dictionaries and Roaring bitmaps prefilter vector, FTS, multi-query,
and delete-by-filter execution while a final AST scan preserves exact
eligibility. Revisioned FTS generations maintain term frequencies, document
lengths, and corpus totals incrementally while preserving scan-oracle BM25
scores. Scalar bitmaps are now pushed into HNSW/IVF/RaBitQ/Vamana/DiskANN: rejected graph
nodes remain navigation bridges, IVF intersects ranked centroid ordinal
postings and
expands filtered probes to fill top-k, and an underfilled or costlier traversal
falls back to exact eligible scoring. Vector, scalar, and FTS indexes share one
persistent ordinal generation; vector membership and IVF postings stay as
Roaring bitmaps through the bounded exact-executor handoff. Vector base/delta
maps, HNSW/Vamana/DiskANN nodes and edges, IVF postings, tombstones, and candidate
selections all
use shared `u64` ordinals; direct-address ordinal arrays back vector slots and
HNSW/Vamana/DiskANN graph layers. The persistent ordinal registry keeps ID-to-ordinal lookup in
an ordered map and its dense append-only ordinal-to-ID lookup in an indexed
vector, so immutable generations share the reverse mapping without tree-key
comparisons during result resolution. HNSW uses deterministic frontier/result
heaps with primary keys borrowed only for equal-score ordering instead of
repeated candidate-vector sorting. A versioned, checksummed, manifest-bound
derived-index cache now restores the shared ordinal plus
HNSW/IVF/RaBitQ/Vamana/DiskANN, PQ, scalar, and FTS generations across process
restarts. Cache format 10 records deterministic RaBitQ rotation/center/code
state, Vamana/DiskANN graph and PQ state, parsed tokenizer, and ordered filter
state beside the contiguous FTS layouts. A
Vamana or DiskANN cache hit also requires the matching native 4 KiB-sector
graph/vector-or-code sidecar; format-2/3/4/5/6/7/8/9, stale, missing, or
corrupt bytes fall back to document-derived rebuilds,
and read-only opens never repair the cache. Indexed BM25 scores now
remain in the shared ordinal domain through document lookup, eliminating
query-sized owned primary-key maps and duplicate candidate bitmaps. Exact vector, ANN re-rank,
and BM25 execution now retain only bounded top-k borrowed document references
before cloning and projection. BM25 query scores use contiguous ordinal slices,
direct single-posting evaluation, and bounded adaptive direct-address
multi-term accumulation. Safe indexed-BM25 plans now retain only the best
`topk` ordinal scores before document resolution while preserving full
candidate telemetry; conservative filters retain every score for final AST
evaluation. Posting entries carry document length beside frequency so BM25
scoring does not perform a second persistent-tree lookup for each term hit.
The term dictionary, each term posting, and the direct-address document-length
table use contiguous immutable bases plus bounded persistent change maps,
retaining cheap generation clones and incremental writes while scanning stable
bases sequentially. Unicode n-gram tokenization, ordered lowercase/folding/
stemmer filters, OR/AND analyzed-term execution, and structured boolean/phrase
queries are live. Selective conjunctions start with the shortest posting;
broad structured expressions use a cost-aware exact scan fallback. The
all-feature baseline has 263 passing unit/integration tests plus four doctests;
the default and no-default feature suites each pass 260 unit/integration tests
plus four doctests, and the feature gates remain separate. Formatting,
default/all-feature Clippy with `-D warnings`, and rustdoc are green. The full
default-feature suite
also passes on the declared Rust 1.75 MSRV after constraining the broad Rayon
and `rmp` dependency ranges to compatible release lines; optional Jieba still
requires a newer Cargo because its current compressed-dictionary chain uses
Rust 2024 manifests. GitHub Actions runs the full default-feature suite on Linux x86_64
and arm64, Windows x86_64, and macOS arm64 and Intel, while separate jobs gate
Rust 1.75, formatting, all-feature Clippy/tests, and rustdoc. The Intel build
uses a macOS 12.0 deployment target; an actual macOS 12 runtime smoke still
requires a self-hosted or external runner.

**Verification refresh (2026-09-03):** Vec revision `03e4031` independently
passes the all-feature and no-default suites (debug and release), the Rust
1.75 no-default release suite, all compatibility examples, formatting,
all-target checks, all-feature Clippy, locked packaging, and the complete
local benchmark sweep: the 53-row feature matrix, concurrent readers, mixed
read/write contention, ANN recall, filtered ANN, scalar filters, indexed FTS,
incremental writes, reopen, n-gram FTS, and structured FTS. Their CSVs contain
finite metrics and the three gate validators pass on the local Windows
x86_64 host. Hosted revision-bound artifacts are recorded in
[CI run 33656188603](https://github.com/A3S-Lab/Vec/actions/runs/33656188603),
which passed all ten jobs. These checks do not replace the actual macOS 12
Intel runtime gate described below.

Phase 3's portable implementation gate is complete: per-handle deterministic
fault injection covers all 18 write/sync/rename/prune boundaries, including
WAL and snapshot cleanup; lock conflicts include bounded owner metadata; and
both fixed-seed mutation fuzzing and a libFuzzer/AddressSanitizer smoke target
exercise recovery. The actual macOS 12 Intel runtime smoke remains external
runner work. HNSW/IVF/RaBitQ/Vamana/DiskANN schema and query controls execute against
immutable, revision-tagged generations; scalar and FTS indexes publish matching
immutable generations. Native sector-aligned Vamana/DiskANN files, bounded
positioned or immutable mmap-snapshot query traversal, PQ/ADC compression, and
portable multi-bit RaBitQ are live. The optional `async` feature moves query,
multi-query, and group-by snapshot/planner/sidecar-I/O work to Tokio's blocking
pool with identical results, fallbacks, and telemetry. Native async file reads
and direct file-backed mmap remain future work.

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

**Evidence landed through 2026-09-02**

- Completed: dense query dimension errors for every current numeric vector
  type and L2/IP/cosine/MIPS-L2 metric, plus byte-length and route/type checks
  for Binary32/Binary64, dense, sparse, scalar, and FTS fields. Binary exact
  search supports only Flat L2/Hamming; non-L2 metrics and ANN descriptors fail
  explicitly.
- Completed: dense, sparse, and binary source-ID queries resolve the stored
  authoritative vector before exact scoring. FP16/FP32 sparse fixtures cover
  all four metrics, filters, vector projection, flush/reopen, and optional
  Tokio execution against the explicit-payload oracle. Missing source
  documents return `NotFound`; an absent sparse source vector returns
  `FailedPrecondition`; empty or NUL-bearing source IDs are rejected.
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
  segment controls, then restored only the typed I/O choice after both bounded
  backends existed; fixed process-versus-collection durability precedence, and
  added execution tests for WAL operation/byte checkpoint thresholds.
- Completed: unsupported physical index descriptors, query tuning parameters,
  segment sizing, and non-zero schema-evolution concurrency fail with typed
  errors before mutation. Flat remains the exact vector index and scan BM25 is
  the FTS fallback; neither increments ANN telemetry. HNSW/IVF became live in
  Phase 4.
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
  rejected. Boolean groups, exact phrases, required/prohibited modifiers,
  wildcard, fielded, boosted, fuzzy, and range syntax execute with shared
  indexed/scan semantics in Phase 5.
- Completed: fluent `SearchQueryBuilder` dense, binary, FTS query-string, and
  FTS match-string routes execute against their respective exact or BM25
  oracle; ambiguous route and dual-expression combinations fail at build time.
  `include_doc_id` resolves the shared
  generation ordinal and remains deterministic across flush/reopen.
- Completed: a dependency-free fixed-seed generator produces 256 mixed
  documents and checks 100 dense metric/filter combinations plus 24
  BM25/filter combinations against independent references after persistence
  and reopen.
- Completed for hosted runners: the default suite runs on Linux x86_64/arm64,
  Windows x86_64, and macOS arm64/Intel. The Intel job compiles with a 12.0
  deployment target, but an actual macOS 12 runtime smoke remains open.
- Open: the macOS 12 Intel runtime required by the full Phase 1 exit gate.
  Scale-bearing FP16/INT8/INT4 index quantization and exact re-ranking are
  completed in Phase 4. Binary exact query execution is complete; binary ANN
  remains an explicit non-goal until a metric/index contract is justified.

## Phase 2 — Correct in-memory collection

**Deliverables**

- Thread-safe collection handle and deterministic CRUD semantics.
- Exact flat dense/sparse numeric search with L2, inner-product, cosine, and
  MIPS-L2, plus Binary32/Binary64 L2/Hamming search.
- Filter parser/evaluator, radius filtering, top-k, fetch, and snapshot iterator.
- Result projection and per-document write results.

**Exit gate**

- Differential tests compare every query result with a simple reference scan;
  concurrent readers see a coherent revision while writes are serialized.

**Evidence landed through 2026-09-02**

- Completed for the current exact surface: independent dense and sparse
  references cover all four numeric metrics, while fixed-seed Binary32 and
  Binary64 references cover XOR Hamming scoring. Deterministic filtering,
  radius/top-k, source-ID, score comparison, primary-key ordering, projection,
  multi-query, group-by, optional Tokio execution, and persistence are gated.
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
- Completed: format-4 MessagePack snapshots cut the 5,000-document fixture from
  2,950,487 to 1,420,700 bytes and its documents-only warm reopen median from
  9.88 to 5.93 milliseconds. The reader still opens format-3 JSON generations;
  checkpoint upgrade, matching-checksum truncation/trailing payloads, bounds,
  and interruption at every publication boundary have deterministic tests.
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

**Evidence landed on 2026-08-30**

- Completed: deterministic multi-layer HNSW construction consumes both `m`
  and `ef_construction`; query traversal bounds the returned candidate set by
  `ef`. Setting `ef` to the indexed document count matches Flat ranking and
  scores exactly.
- Completed: graph construction and queries share a best-first binary frontier
  heap and a bounded worst-first result heap. Entries carry compact ordinals,
  visited hashes do not affect order, and equal scores retain ascending
  primary-key ties through the immutable ordinal table. A paired 2,000-vector
  run reduced HNSW query
  latency from 254.42 to 121.02 microseconds and full rebuild time from 932.17
  to 475.96 milliseconds while recall@10 remained 1.0000.
- Completed: deterministic farthest-first IVF training consumes configurable
  iteration counts, stores centroids/postings, and probes with `nprobe`.
  Probing every actual centroid matches Flat exactly.
- Completed on 2026-09-02: `use_soar=true` assigns each base vector to its
  nearest primary centroid plus one deterministic secondary centroid minimizing
  the SOAR residual objective with lambda one. Probe unions and filtered window
  expansion count unique ordinals, cache validation requires exactly the
  configured one-or-two assignments, and exhaustive search plus cache reopen
  match the Flat oracle.
- Completed: ANN vector maps, graph nodes/edges, IVF postings, tombstones, and
  candidate selections remain in the shared `u64` ordinal domain. A paired
  2,000-vector run reduced HNSW p50/p95/p99 by 13.9/16.6/13.1 percent and its
  deterministic payload estimate by 7.9 percent; IVF reduced those percentiles
  by 40.1/38.1/34.9 percent and payload by 6.5 percent. Recall remained 1.0000
  and 0.9083 respectively. The payload estimate excludes allocator/map-node
  and authoritative-document overhead and is not RSS.
- Completed: vector generations and HNSW graph layers replaced ordered ordinal
  maps with validated direct-address slots. In the paired 2,000-vector fixture,
  HNSW p50/p95/p99 changed from 97.04/116.38/135.67 to
  81.29/109.54/127.25 microseconds and estimated payload changed from 820,748
  to 795,592 bytes; recall@10 remained 1.0000. IVF payload changed from 290,028
  to 276,028 bytes, while its latency tails were noisy and are not claimed as
  an improvement.
- Completed: the dense append-only ordinal-to-primary-key reverse map moved
  from a persistent ordered map to a persistent indexed vector while the
  ID-to-ordinal map remains ordered. Two order-reversed FTS A/B pairs reduced
  component, sparse multi-term, common-term, and mixed-term medians by 50.9,
  21.7, 42.6, and 33.8 percent. Four incremental-write observations showed no
  coherent regression; ANN recall and estimated payload were unchanged.
- Completed: a bounded binary cache persists the shared ordinal table plus ANN,
  scalar, and FTS generations as optional derived state. Format 4 introduced
  the contiguous term/posting/document-length layouts, format 5 added parsed
  tokenizer state, format 6 added the ordered filter pipeline, and format 7
  added Vamana graph generations. Format 8 required the matching native
  sector-aligned Vamana sidecar, and format 9 adds DiskANN generations plus PQ
  codebooks/codes and requires the matching vector-or-code sidecar; exact legacy
  format-2/3 fixtures and older version bytes are ignored safely. Reuse is
  gated by format, CRC, schema, revision,
  exact manifest/WAL/snapshot identity, structural/live-membership validation,
  and equality with vectors/scalar values derived from authoritative documents.
  Corrupt, stale, read-only, scalar/FTS delta, ANN overlay, and tombstone
  lifecycle fixtures pass. On a 5,000-document, 32-dimensional workspace-shaped
  collection with HNSW, scalar-inverted, and FTS indexes, five read-only warm
  reopens produced a 12.06 ms cache-hit median versus 486.33 ms after forcing
  rebuilds in the same run: 40.33 times faster (97.52 percent lower), with a
  2,982,216-byte cache and a 1,420,899-byte authoritative snapshot. A separate
  100,000-document scalar+FTS fixture produced 210.30 ms cache-hit versus
  259.05 ms forced-rebuild medians across two observations (-18.8 percent);
  after subtracting each documents-only control, cache restore reduced the
  derived portion by 40.9 percent.
- Completed: create, drop, targeted rebuild, optimize, insert, update, upsert,
  delete, and reopen publish complete revision-tagged index generations.
  Targeted rebuilds preserve the ordinal table and unrelated immutable
  generations instead of rebuilding the whole registry: on the mixed
  5,000-document fixture, scalar and FTS rebuilds took 1.63 and 5.71 ms versus
  497.75 ms for `optimize()`; rebuilding HNSW remained 477.13 ms as expected.
  Scalar/FTS rebuilds also skip rewriting the logically equivalent derived
  cache generation.
  Document generations use a persistent ordered tree: clones share the root,
  writes copy only an O(log N) path, and unchanged `Arc<Doc>` values remain
  shared. Indexed writes share the immutable graph/posting base, shadow
  replacement/deletion entries with tombstones, scan a bounded changed-vector
  delta, and compact at a bounded threshold. Construction leaves the previous
  generation readable, and a revision mismatch selects the exact fallback
  instead of stale candidates.
- Completed: index-only FP16, symmetric INT8, and packed symmetric INT4
  encodings reduce candidate-vector storage while authoritative vectors remain
  lossless. Every ANN result is re-ranked with the existing f64 exact oracle.
- Completed: fixed-seed HNSW, IVF, HNSW/IVF RaBitQ, Vamana, and DiskANN/PQ
  recall tests enforce bounded candidate counts and recall@10 thresholds.
  `cargo bench --bench ann_recall` provides a
  2,000-document, 32-dimension, five-round median latency/recall fixture.
  `cargo bench --bench incremental_write` compares single-document delta
  publication with a complete HNSW rebuild and checks document-generation
  scaling at 2,000, 20,000, and 100,000 documents; current machine-local
  evidence is recorded in `BENCHMARKS.md`.
- Completed: revision-matched scalar sets are pushed into ANN rather than
  intersected after a fixed unfiltered search. HNSW traverses rejected nodes
  while retaining an eligible result heap and scales its navigation budget by
  selectivity. IVF intersects shared-ordinal centroid postings with the scalar
  and tombstone bitmaps, then extends the initial `nprobe` window until top-k is
  available. Filtered delta vectors are merged before the final `ef`/scale-
  factor bound. Costly or underfilled searches use exact eligible scoring.
  Public fixtures cover partial bitmap refinement, shared-ordinal compaction,
  scalar mutation, reopen, concurrent generation publication, exact result
  equivalence, and bounded candidates. Across the eligibility-bitmap and full
  ANN-ordinal milestones, a paired 8,400-document run reduced filtered IVF
  latency from 671.41 to 54.97 microseconds and filtered HNSW latency from
  738.88 to 79.50 microseconds with recall@10 unchanged at
  1.0000. `cargo bench --bench filtered_ann` records the post-filter underfill
  counterexample and filter-aware latency/recall.

## Phase 5 — Structured and full-text retrieval

**Status:** complete. Scalar indexing, bitmap prefiltering, indexed BM25,
Unicode n-gram tokenization, ordered token filters, and structured boolean,
wildcard, field-qualified, boosted, fuzzy, range, and phrase-proximity
execution are implemented.

**Deliverables**

- Equality/range scalar inverted indexes and bitmap pre-filtering.
- Standard, whitespace, n-gram, and optional jieba tokenizers.
- BM25 FTS with boolean operators, phrases, prefix/suffix, and filters.
- Dense + sparse + FTS hybrid query planning.

**Exit gate**

- FTS ranking and filter semantics have golden fixtures; index and fallback
  paths return the same eligible document set.

**Evidence**

- Completed: scalar `Invert` descriptors build immutable, revision-tagged
  persistent value dictionaries with copy-on-write Roaring postings. Equality,
  ordered range, `IN`, null, wildcard, prefix, suffix, and boolean composition
  use conservative prefilters followed by authoritative AST verification.
  `NOT` refuses a partial subtree, preventing false-negative complements.
- Completed: insert, update, upsert, delete, delete-by-filter, drop/recreate,
  explicit rebuild, flush/reopen, ordinal-tombstone compaction, and concurrent
  publication keep scalar postings at the document revision.
- Completed: one persistent registry ordinal generation is shared by scalar,
  vector, IVF, and FTS postings. Vector membership and large scalar candidate
  sets remain Roaring bitmaps through ANN planning; primary keys are resolved
  only for exact execution. Scan-FTS restricts eligible scoring while retaining
  whole-corpus BM25 statistics. Each multi-query branch receives its own plan.
  Selective or costlier scalar sets bypass ANN for exact scoring; larger exact
  sets are pushed into filter-aware HNSW/IVF/RaBitQ/Vamana/DiskANN. Conservative boolean supersets
  are refined against the authoritative AST before ANN, preventing unindexed
  conjuncts from spending the bounded candidate budget. Dedicated telemetry
  reports scalar-index use and exact re-rank candidate counts.
- Completed: differential fixtures compare bitmap and scan execution across
  range/boolean/null/wildcard and mixed indexed/unindexed expressions. A
  100,000-document benchmark covers language equality, modification-time
  range, path prefix, and workspace-style conjunction filters; current
  machine-local evidence is recorded in `BENCHMARKS.md`.
- Completed: FTS descriptors build persistent term-frequency postings and
  exact document/corpus length statistics. Queries traverse only matching
  postings, optionally intersect a scalar prefilter, and fall back to scan BM25
  when the generation is missing or stale. Index and scan scores are identical
  for repeated terms, missing/null/empty text, and filtered queries.
- Completed: FTS insert/update/upsert/delete, reopen, drop/recreate, rebuild,
  ordinal compaction, and concurrent generation fixtures. A 50,000-document
  query benchmark plus incremental-update/full-rebuild measurements are in
  `BENCHMARKS.md`.
- Completed: indexed BM25 score generations remain ordinal-keyed through exact
  result execution. A paired 50,000-document run reduced indexed component,
  common-term, and mixed-term latency by 23.9, 32.1, and 28.2 percent without
  changing scan-equivalent scores or candidate telemetry.
- Completed: exact execution uses a deterministic bounded top-k heap instead of
  cloning and sorting every eligible document. A paired 2,000-document exact
  vector run reduced median latency from 507.40 to 136.44 microseconds, while
  common and mixed indexed-BM25 queries fell another 38.4 and 50.9 percent.
- Completed: query-time BM25 scores are compact ordinal-score slices.
  Single-term postings bypass tree insertion, while high-cardinality multi-term
  queries use bounded direct-address scratch and selective queries retain sparse
  accumulation. Paired runs reduced component, common-term, and mixed-term
  latency by another 13.7, 20.5, and 14.5 percent.
- Completed: unfiltered indexed BM25 and filters with exact scalar-index
  coverage apply deterministic `topk` retention at the ordinal-score generation
  boundary. Full scored-candidate telemetry is preserved, and conservative or
  unindexed filters retain every score for final authoritative evaluation. A
  paired 50,000-document run reduced component, sparse multi-term, common-term,
  and mixed-term latency by another 75.4, 64.4, 75.6, and 73.1 percent.
- Completed: FTS posting entries carry document length beside term frequency,
  removing one persistent document-length lookup per BM25 contribution. The
  aligned B-tree entry remains 16 bytes on the supported 64-bit targets, and
  two order-reversed A/B pairs reduced component, sparse multi-term,
  common-term, and mixed-term latency by median-pair deltas of 39.4, 34.7,
  44.1, and 41.1 percent without regressing incremental update latency.
- Completed: replacing the shared ordinal reverse ordered map with a persistent
  indexed vector reduced the same component, sparse multi-term, common-term,
  and mixed-term queries by another 50.9, 21.7, 42.6, and 33.8 percent across
  two order-reversed A/B pairs. Candidate counts and public results were
  unchanged, and four write-path observations showed no coherent regression.
- Completed: term postings now combine a sorted contiguous immutable base with
  a persistent ordered change map that compacts at one eighth of the base,
  bounded to 64..=2,048 changes. Two order-reversed A/B pairs reduced sparse
  multi-term, common-term, and mixed-term medians by another 14.0, 13.2, and
  11.9 percent. Incremental generations retain base sharing, and the paired
  indexed-update/scan-update ratio showed no write-path regression.
- Completed: the FTS term dictionary and direct-address document-length table
  now use the same shared contiguous-base/bounded-delta design. Term lookup
  binary-searches the stable base, document-length validation is direct, and a
  document batch compacts each outer structure at most once. At 100,000
  documents, the documents-control-adjusted derived rebuild portion fell 16.6
  percent, while the then-format-4 cache reduced that derived portion by
  another 40.9 percent. The paired FTS query/update controls retained identical
  results and showed no coherent latency regression.
- Completed: an initial insert whose changed-ID set covers the whole new corpus
  now uses the mutable bulk FTS builder directly; partial and non-empty batches
  retain persistent deltas. In an order-reversed 100,000-document pair, raw
  scalar+FTS fixture insertion fell 38.4 percent, while the derived portion
  after subtracting each documents-only control fell 48.7 percent.
- Completed: full scalar and FTS builds aggregate in mutable ordered maps and
  Roaring bitmaps before freezing one persistent generation. Paired rebuilds
  fell from 161.53 to 157.03 milliseconds for three scalar fields over 100,000
  documents, and from 145.37 to 120.76 milliseconds for one FTS field over
  50,000 documents. Incremental copy-on-write mutation remains unchanged.
- Completed: the zvec-compatible `ngram` tokenizer defaults to Unicode
  bigrams, accepts at most two adjacent gram sizes and five Unicode character classes,
  and shares one parsed configuration across an FTS generation. Independent
  BM25, mutation, cache-reopen, and schema-error fixtures cover the contract.
  `FtsQueryParams.default_operator` now executes OR or AND for both indexed and
  scan BM25; AND uses the shortest posting as its driver. In two 10,000-document
  observations, a selective workspace identifier query fell from 10,000 to 11
  scored candidates and from about 520 to 16 microseconds.
- Completed: tokenizer output passes through a serializable ordered filter
  pipeline. Omitted filters select Unicode lowercase, while an explicit empty
  pipeline preserves native tokenizer case. NFKD-based ASCII folding and 18
  Snowball stemming languages apply to both document and query tokens. The
  standard tokenizer enforces a configurable Unicode-character length limit.
  Schema validation rejects unknown filters, languages, orphaned parameters,
  and out-of-range lengths before mutation. Mutation and current-cache reopen
  fixtures retain identical analysis.
- Completed: `query_string` parses AND-before-OR boolean groups, parentheses,
  exact phrases, escapes, and `+`/`-` required/prohibited modifiers into one
  AST shared by indexed and scan execution. Simple term queries retain their
  specialized posting paths. Structured indexed execution drives from the
  smallest required posting/subtree and verifies phrase adjacency only for
  candidates. A selectivity heuristic routes broad expressions to exact scan
  when bitmap/refinement work is unlikely to win. A generated 256-document
  differential matrix compares IDs and score bits across twelve expression
  shapes, with mutation, cache reopen, malformed syntax, and planner telemetry
  coverage. In a 25,000-document benchmark, the selective phrase and required+
  optional cases each scored one candidate instead of 25,000; broad phrase and
  boolean-NOT cases selected the scan fallback.
- Completed: wildcard `*`/`?` patterns, same-field qualifiers, finite boosts,
  transposition-aware fuzzy distances 1 and 2, independently inclusive lexical
  term ranges, and ordered phrase slop from 0 through 1,024 expand and evaluate
  through one AST on both index and scan paths. Differential fixtures compare
  IDs and public score bits across mutation and cache reopen boundaries. Broad
  dynamic leaves still use the cost-aware exact scan fallback. Positional
  postings remain an optional performance optimization; phrase proximity
  currently retokenizes only candidate documents. In the 25,000-document
  benchmark, wildcard and exact-range queries scored one candidate, fuzzy
  distance 1 scored 36, and ordered proximity scored one; every result and
  public score bit matched the scan control.

## Phase 6 — DiskANN family and compression

**Deliverables**

- Vamana graph construction and sector-aligned DiskANN files.
- PQ codebook training/ADC, optional RaBitQ, mmap/pread readers, and delta
  documents for post-build writes.
- Query-time beam/list parameters and full-vector refinement.

**Exit gate**

- Index files survive reopen and checksum validation; Linux and macOS Intel
  use the correct I/O backend; recall and corruption tests pass.

**Progress**

- Completed: deterministic in-memory metric-aware Vamana construction follows the
  two-pass [DiskANN sequence](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf): seeded R-regular initialization, centroid medoid,
  greedy search, RobustPrune at alpha 1 then the configured alpha, and bounded
  backward edges. `list_size` controls SearchQuery, group-by, and multi-query
  branches; candidates receive authoritative full-vector refinement. L2 uses
  squared distance, cosine uses angular distance, and inner-product/MIPS-L2
  use a norm-augmentation transform with an immutable-base bound.
- Completed: immutable Vamana bases participate in incremental delta/tombstone
  overlays, scalar-filter planning, targeted rebuilds, telemetry, and validated
  cache-format-10 reopen. Unit, exhaustive-oracle, bounded-candidate recall,
  mutation, rebuild, and cache-hit tests cover the slice for all four numeric
  metrics. The public matrix checks eight deterministic bounded queries for
  non-L2 graph recall and enforces the candidate budget.
- Completed: every Vamana and DiskANN base is mirrored in the A3S-native
  `indexes/diskann-graph.bin` format. Its versioned header and field metadata
  bind the schema digest, revision, and manifest-derived source identity;
  full-vector or PQ-code neighbor records use fixed lengths, pack without
  crossing 4 KiB sectors, and use whole-sector strides when a record is larger
  than one sector. PQ codebooks live in field metadata. A CRC covers metadata,
  padding, and data. Recovery uses bounded
  positional reads on Unix and Windows, validates canonical padding and graph
  contents, and treats missing, truncated, corrupt, or mismatched bytes as a
  cache miss. Read-only opens do not repair; writable opens atomically refresh
  the sidecar before publishing cache format 10. Small-, PQ-, and multi-sector unit
  fixtures plus public lifecycle tests cover those paths on Windows. The
  default-feature crate also cross-checks the Unix `read_at` branch for the
  installed Linux x86_64/aarch64 and macOS arm64/x86_64 targets, including a
  macOS 12 deployment target; the Intel macOS 12 runtime gate remains external.
- Completed: a validated cache reopen attaches one immutable positioned reader
  per Vamana or DiskANN field. Bounded queries load packed 4 KiB sectors or multi-sector
  node strides into a request-local extent/node cache, while incremental
  overlays retain the reader and complete rebuilds invalidate it until reopen.
  A query-time short read or malformed record falls back to the equivalent
  in-memory full-vector or ADC graph. Packed/oversized/PQ,
  filtered/unfiltered parity, corruption,
  overlay, rebuild, and telemetry fixtures cover the contract.
- Completed: `IndexType::Diskann` accepts L2, inner product, cosine, and MIPS-L2
  with `pq_chunk_num` in `0..=dimension`. Positive values split dimensions into balanced contiguous
  chunks, train up to 256 deterministic centroids per chunk with eight Lloyd
  iterations, encode one byte per chunk, and build one query-local metric-aware
  ADC table (distance, inner-product, or cosine scoring).
  In-memory, positioned, and mmap-snapshot traversals use identical
  codes/tables; full vectors remain authoritative for exact final ranking and
  for delta documents until a
  rebuild retrains the generation. Cache and sidecar validation cover
  codebooks, codes, graph membership, deterministic training, filtered and
  unfiltered parity, all numeric metrics, lifecycle, corruption, and query-time
  fallback.
- Completed: `IndexType::HnswRabitq` and `IndexType::IvfRabitq` execute for
  L2, inner product, and cosine using the
  [RaBitQ estimator](https://arxiv.org/abs/2405.12497) and the official
  [multi-bit quantizer contract](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/quantizer/).
  The portable scalar implementation trains
  deterministic centers, applies four fixed-seed signed normalized Hadamard
  rounds, and compactly packs one through nine bits per padded dimension. The
  sign bit plus extended magnitude bits use an iteratively optimized rescale;
  traversal evaluates the RaBitQ unbiased residual estimator, while
  authoritative vectors retain exact public scores. HNSW exposes bounded
  `ef`; IVF exposes `nprobe`, linear fallback, radius, and a bounded
  `scale_factor * topk` refiner set. Empty builds, delta/tombstone overlays,
  scalar-filter navigation, targeted rebuild, optimize, deterministic
  training, exhaustive-oracle ranking, fixed-seed recall, all bit widths,
  cache corruption fallback, and cache-format-10 reopen are covered.
- Completed: the `async` feature exposes `query_async`, `multi_query_async`,
  and `group_by_async`. Each method requires an active Tokio runtime and moves
  the complete synchronous snapshot, planner, selected sidecar traversal,
  corruption fallback, exact refinement, and telemetry path to its blocking
  pool. A cache-reopened PQ DiskANN integration fixture proves bit-identical
  single, fused, and grouped results and then truncates the sidecar to prove
  identical in-memory fallback; unit coverage proves work leaves the runtime
  thread and missing-runtime use returns `FailedPrecondition`.
- Completed: public `IoBackend` selection resolves from the process default or
  a per-collection override, with portable positioned reads remaining the
  default. `IoBackend::Mmap` copies the fully validated sidecar into a read-only
  anonymous map and serves the existing bounded random-access reader from that
  immutable snapshot without retaining dependence on the source file. Exact
  ID/score-bit parity covers in-memory, positioned, and mapped PQ traversal;
  query telemetry distinguishes the mmap subset. A live mapped handle survives
  later sidecar truncation, while a subsequent reopen rejects the truncated
  artifact and rebuilds safely in memory. The design retains `unsafe_code =
  "deny"`; its explicit cost is an open-time full copy and sidecar-sized
  retained mapping.
- Completed: the repeated 2,000-vector cosine benchmark now includes HNSW and
  IVF RaBitQ7. On the latter 2026-09-01 Windows run, HNSW RaBitQ retained
  recall@10 1.0000 at a 122.96 microsecond median versus 123.56 for HNSW; IVF
  RaBitQ retained the same 0.9083 recall as IVF at 91.72 versus 68.00
  microseconds. Estimated payloads were 949,664 and 436,244 bytes because the
  correctness-first derived generation deliberately retains refinement
  vectors in addition to compact codes. These rows are regression evidence,
  not a compression or speedup claim.
- Completed: the fixed 2,000-vector benchmark includes exact L2, in-memory
  Vamana/DiskANN PQ8, and cache-reopened positioned and mmap-snapshot paths at
  `list_size=64`. On the post-change 2026-09-01 Windows fixture all seven modes
  produced recall@10 1.0000. PQ reduced sidecar size from 823,296 to 622,592
  bytes and staged extents from 138.04 to 108.16 sectors/query. Vamana medians
  were 311.75, 1,283.52, and 846.68 microseconds for memory, positioned, and
  mmap; PQ8 medians were 269.32, 1,036.52, and 702.38 microseconds. The mmap
  snapshot reduced measured query median by 34.0% and 32.2% versus the matching
  positioned rows, excluding its full-copy open cost. Exact L2 remained faster
  at 128.28 microseconds, so no exact-scan speedup is claimed.
- Completed: non-L2 Vamana and DiskANN/PQ now use metric-aware graph pruning
  and ADC for inner product, cosine, and MIPS-L2. Exact-oracle tests cover
  in-memory and sidecar reopen paths; a deterministic eight-query fixture
  requires at least 0.50 recall@10 while capping exact candidate work at 96.
  The public feature benchmark records p50/p95/p99 latency and throughput for
  all four metrics in both graph families.
- Remaining: native async file reads or a sound direct file-backed mmap
  backend, plus the Linux/macOS Intel runtime portability gate.

## Phase 7 — Advanced collection API

**Deliverables**

- Multi-query routes, RRF/weighted reranking, score normalization, and
  group-by top-k.
- Add/alter/rename/drop columns with backfill and schema revisions.
- Background compaction, index progress, collection statistics, and health.
- Caller-owned dense/sparse embedding traits and an optional query executor.

**Progress**

- Completed: the pinned upstream CRUD, vector-search, and schema-builder
  fixtures from
  `zvec-ai/zvec-rust@0d40cb1aef081bae175061fef35c89269e6a80f4` differ only by
  the mechanical `zvec_rust` to `a3s_vec` namespace replacement. Wrappers
  apply only upstream-style lint allowances; all three build and run in CI. The
  schema fixture exercises the now-live IVF SOAR parameter. The CRUD fixture's
  two incomplete replacement upserts are retained as an auditable upstream
  fixture defect; both official zvec and A3S correctly reject the missing
  required `id` field.
- Completed: asserted executable gates cover vector and FTS search, RRF and
  weighted hybrid fusion with normalization, group-by top-k, snapshot-isolated
  iteration, add/rename/alter/drop schema evolution, flush, and reopen. CI
  executes every gate rather than treating compilation as sufficient.
- Completed: collection and per-index statistics, ready/missing completeness,
  query telemetry, automatic bounded ordinal/index-delta compaction, and
  caller-owned dense/sparse embedding and query-executor traits are live.
- Completed: an explicitly owned standard-thread runtime schedules full
  derived-registry rebuild plus same-revision checkpoints, skips already
  maintained revisions, coalesces immediate triggers, records bounded worker
  diagnostics, rejects duplicate/read-only ownership, and joins on close or
  drop. Public collection health distinguishes healthy, degraded, unhealthy,
  and closed state from normal WAL checkpoint lag by checking authoritative
  revision agreement and every index generation.
- Remaining exit-gate provenance: the pinned upstream Rust SDK does not yet
  publish standalone FTS/hybrid, multi-query, group-by, iterator, or
  schema-evolution examples, so those project-owned gates cannot honestly claim
  namespace-only upstream provenance yet.

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

**Progress through 2026-09-02**

- Completed in the engine: typed per-handle limits for retained document count,
  deterministic authoritative-plus-derived accounted bytes, cumulative query
  refinement candidates, and write-batch size. Admission precedes WAL/state
  publication, filtered deletes share the write budget, multi-query branches
  share one candidate budget, and statistics expose only aggregate accounting,
  active limits, and rejection counts.
- Completed in the engine: ANN/filtered/DiskANN benchmarks, deterministic
  recovery fuzzing plus a libFuzzer/AddressSanitizer smoke target, collection
  health, query/index/WAL telemetry, and hosted Linux arm64/x86_64, Windows
  x86_64, and macOS arm64/Intel CI. The Intel hosted job uses a macOS 12.0
  deployment target. The platform matrix also runs and validates the feature,
  concurrent-reader, and mixed-read/write smoke CSVs on every hosted OS and
  architecture, retaining one revision-bound artifact per platform.
- Completed in the engine: a public feature matrix (`tests/feature_matrix.rs`)
  covers CRUD, projection, exact dense/sparse/binary and source-ID queries, scalar/
  FTS/hybrid/group-by execution, iterator and schema evolution, flush/reopen,
  health, cache/sidecar readers, all six ANN families, and explicit unsupported
  binary-ANN boundaries. `benches/feature_matrix.rs` adds 53 asserted
  p50/p95/p99 and throughput rows, including Binary32/64 exact, radius,
  projection, source-ID, scalar-filter, multi-query, group-by, and Tokio paths.
  The smoke-scale CSV is
  uploaded by CI and is a correctness
  gate; same-host default-scale values are recorded in `BENCHMARKS.md`.
- Completed in the engine: `benches/concurrent_queries.rs` starts synchronized
  1/2/4/8-worker HNSW readers and records index-build time, flat-oracle
  Recall@10, nearest-rank p50/p95/p99 query latency, and wall-clock QPS. The
  smoke CSV is validated by `.github/check_concurrent.awk` and uploaded with
  the public performance artifact. This closes concurrent read-tail evidence.
- Completed in the engine: `benches/mixed_workload.rs` starts one scalar-update
  writer alongside synchronized 1/2/4/8-worker HNSW readers and records read and
  write p50/p95/p99, wall-clock QPS, Recall@10, final revision, and logical
  accounted bytes. The 19-column smoke CSV is validated by
  `.github/check_mixed.awk` and uploaded with the public performance artifact.
  Vector-index mutation remains covered by `incremental_write`; process RSS,
  allocator attribution, and cross-project large-corpus scale remain separate
  external measurements.
- Completed cross-project integration on 2026-09-02: A3S Code commit
  `4163d8e3a1a96bbae430dc987005acaa362efb30` pins Vec commit
  `019fdb929a57dee1803691e6def60df3946d9561` behind a Memory-authoritative
  workspace-retrieval shadow. Code admits one embedding batch, mirrors it once
  into a session-local temporary Vec collection, compares IDs, partitions,
  `f32` scores, and search accounting behind one publication gate, and exposes
  bounded lifecycle/resource/parity diagnostics through Rust, Node.js, Python,
  and Go. Vec failures degrade the shadow only; Memory remains the sole result
  authority and close releases both engines. The change-scoped schema-4
  `workspace-retrieval-v3` run retained 25,000 records in each hybrid arm and
  matched all 120 queries with zero failures or mismatches.
- Completed release-candidate hardening: the final public API review keeps the
  external kernel private, preserves `unsafe_code = "deny"`, and compile-checks
  `Send + Sync` across the owned public contract. `cargo package --locked` and
  `cargo publish --dry-run --locked` verify version `0.1.0`. After every hosted
  gate passes on `main`, CI uploads the verified crate, SHA-256 checksum, and a
  source-revision manifest as one versioned release-candidate artifact.
- Completed external-gate automation: a manual workflow accepts an exact
  revision only on a self-hosted runner labeled `a3s-macos-12`. Its reusable
  script rejects any host that is not actual macOS 12 on Intel x86-64, runs the
  locked format, Clippy, default/all-feature, recovery, async, DiskANN, example,
  rustdoc, package, and all three smoke-scale performance fixtures with the
  hosted CSV validators offline, and emits checksummed machine-readable
  evidence. No qualifying runner is currently registered, so this automation
  does not close the hardware gate by itself.
- Remaining release work: an actual macOS 12 Intel runtime result from an
  external runner and publication of the formal tagged artifact against that
  same revision. Logical collection accounting is not advertised as a process-RSS
  or hard CPU-time limit. The adapter contract and evidence are maintained in
  [Code's migration note](https://github.com/A3S-Lab/Code/blob/main/manual/WORKSPACE_RETRIEVAL_VEC_MIGRATION.md).

**Release gate**

- `cargo fmt --check`, `cargo clippy -- -D warnings`, unit/integration/fuzz
  smoke tests, recovery suite, `cargo bench --locked --bench feature_matrix
  --features async`, `concurrent_queries`, and `mixed_workload` (smoke scale
  in CI), benchmark report, and Intel macOS 12 runtime smoke all pass. No
  feature is advertised unless its gate has evidence.

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
4. Extend the completed DiskANN/PQ/RaBitQ path beyond scheduler-safe Tokio and
   immutable mmap-snapshot query offload with native async reads or direct
   file-backed mmap only after equivalent
   exact-reference, recall, and corruption tests are green.
5. Finish API compatibility, Intel validation, and the versioned release
   artifact as release work; keep the completed Code/Memory adapter outside the
   storage core and governed by its migration contract.
