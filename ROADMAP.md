# A3S Vec Roadmap

This is the engine-only roadmap. The dependency order for integrating the
engine into `a3s-code`, exposing `vgrep`, and removing the duplicate SQLite/
BM25 workspace paths is maintained in the
[A3S local retrieval platform roadmap](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-roadmap.md).

The roadmap is ordered by dependency and by the cost of being wrong. Each
phase has an explicit exit gate; a later approximate or optimized feature does
not replace an earlier correctness gate.

## Current implementation status

**2026-08-31:** Phase 1's query/write contract hardening, Phase 3's core
recovery transaction, and Phase 4's in-memory ANN gate are implemented. Query
routes, dense dimensions, sparse
indices, JSON adapter values, schema defaults, and replacement upserts are
validated before execution or persistence. Storage format version 4 provides
compact MessagePack generation snapshots, manifest-committed WAL byte
boundaries, monotonic DML/schema revisions, read-only lifecycle semantics, and
bounded recovery reads. Version-3 JSON snapshots remain readable and upgrade
atomically at the next writable checkpoint. Native FP16, INT4, INT8, INT16,
and binary payloads now have
strict physical-type validation and lossless persistence; dense and sparse
numeric vectors share exact L2/IP/cosine/MIPS-L2 scoring with `f64`
intermediates and are ranked before the public score is narrowed to `f32`.
Independent differential fixtures now cover deterministic dense/sparse scores,
filters, radius/top-k ordering, and scan BM25 corpus statistics. A fixed-seed
256-document corpus adds 100 dense metric/filter combinations and 24
BM25/filter combinations, including a flush/reopen boundary. Structured FTS
now executes explicit boolean groups, required/prohibited clauses, and exact
phrases; wildcard, fielded, boosted, fuzzy, and range syntax still fails
explicitly instead of being approximated. Concurrent public-API
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
and FTS are live derived indexes with dedicated telemetry. Document generations
share unchanged `Arc<Doc>` values through a persistent ordered tree, so ordinary
writes copy only an O(log N) tree path. Indexed mutations share an immutable ANN
base plus bounded delta/tombstone overlays instead of rebuilding every graph on
every write. Phase 5's scalar-inverted slice is also live: revisioned persistent
posting dictionaries and Roaring bitmaps prefilter vector, FTS, multi-query,
and delete-by-filter execution while a final AST scan preserves exact
eligibility. Revisioned FTS generations maintain term frequencies, document
lengths, and corpus totals incrementally while preserving scan-oracle BM25
scores. Scalar bitmaps are now pushed into HNSW/IVF: rejected HNSW nodes remain
navigation bridges, IVF intersects ranked centroid ordinal postings and
expands filtered probes to fill top-k, and an underfilled or costlier traversal
falls back to exact eligible scoring. Vector, scalar, and FTS indexes share one
persistent ordinal generation; vector membership and IVF postings stay as
Roaring bitmaps through the bounded exact-executor handoff. Vector base/delta
maps, HNSW nodes/edges, IVF postings, tombstones, and candidate selections all
use shared `u64` ordinals; vector slots and HNSW graph layers use direct-address
ordinal arrays. The persistent ordinal registry keeps ID-to-ordinal lookup in
an ordered map and its dense append-only ordinal-to-ID lookup in an indexed
vector, so immutable generations share the reverse mapping without tree-key
comparisons during result resolution. HNSW uses deterministic frontier/result
heaps with primary keys borrowed only for equal-score ordering instead of
repeated candidate-vector sorting. A versioned, checksummed, manifest-bound
derived-index cache now restores the shared ordinal plus HNSW/IVF, scalar, and
FTS generations across process restarts. Cache format 6 records parsed
tokenizer and ordered filter state beside the contiguous FTS layouts;
format-2/3/4/5, stale, or corrupt bytes fall back to document-derived rebuilds,
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
all-feature baseline has 159 passing unit/integration tests plus four
compile-fail doctests; default and
no-default-feature suites are separate gates. Formatting, default/all-feature
Clippy with `-D warnings`, and rustdoc are green. The full default-feature suite
also passes on the declared Rust 1.75 MSRV after constraining the broad Rayon
and `rmp` dependency ranges to compatible release lines; optional Jieba still
requires a newer Cargo because its current compressed-dictionary chain uses
Rust 2024 manifests. GitHub Actions runs the full default-feature suite on Linux x86_64
and arm64, Windows x86_64, and macOS arm64 and Intel, while separate jobs gate
Rust 1.75, formatting, all-feature Clippy/tests, and rustdoc. The Intel build
uses a macOS 12.0 deployment target; an actual macOS 12 runtime smoke still
requires a self-hosted or external runner.

Phase 3's portable implementation gate is complete: per-handle deterministic
fault injection covers all 18 write/sync/rename/prune boundaries, including
WAL and snapshot cleanup; lock conflicts include bounded owner metadata; and
both fixed-seed mutation fuzzing and a libFuzzer/AddressSanitizer smoke target
exercise recovery. The actual macOS 12 Intel runtime smoke remains external
runner work. HNSW/IVF schema and query controls execute against immutable,
revision-tagged generations; scalar and FTS indexes publish matching immutable
generations. DiskANN-family names remain contracts only.

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
  rejected. Boolean groups, exact phrases, and required/prohibited modifiers
  now execute in Phase 5; wildcard, fielded, boosted, fuzzy, and range syntax
  returns `NotSupported` rather than silently changing query meaning.
- Completed: a dependency-free fixed-seed generator produces 256 mixed
  documents and checks 100 dense metric/filter combinations plus 24
  BM25/filter combinations against independent references after persistence
  and reopen.
- Completed for hosted runners: the default suite runs on Linux x86_64/arm64,
  Windows x86_64, and macOS arm64/Intel. The Intel job compiles with a 12.0
  deployment target, but an actual macOS 12 runtime smoke remains open.
- Open: the macOS 12 Intel runtime required by the full Phase 1 exit gate.
  Scale-bearing FP16/INT8/INT4 index quantization and exact re-ranking are
  completed in Phase 4; binary query execution remains unsupported.

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
  tokenizer state, and format 6 adds the ordered filter pipeline; exact legacy
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
- Completed: fixed-seed HNSW and IVF recall tests enforce bounded candidate
  counts and recall@10 thresholds. `cargo bench --bench ann_recall` provides a
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

**Status:** scalar indexing, bitmap prefiltering, indexed BM25, Unicode n-gram
tokenization, ordered token filters, and structured boolean/phrase execution
are implemented. Wildcard, fielded, boosted, fuzzy, and range FTS syntax
remains open.

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
  sets are pushed into filter-aware HNSW/IVF. Conservative boolean supersets
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
  and out-of-range lengths before mutation. Mutation and format-6 cache-reopen
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
- Remaining: wildcard terms, field-qualified terms, boosts, fuzzy/proximity
  suffixes, and FTS range syntax. Positional postings remain an optional future
  optimization; exact phrase semantics currently retokenize the candidate
  documents.

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
