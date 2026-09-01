# In-process index benchmark evidence

These dependency-light, deterministic fixtures provide machine-local
regression evidence for the in-process ANN, scalar-filter, and full-text paths.
They are not a zvec comparison and should not be generalized to other
hardware, corpora, dimensions, build parameters, durability policies, or
concurrency levels.

Environment for the 2026-08-30 and 2026-08-31 measurements: Apple M5, 16 GiB
memory, Darwin arm64, Rust/Cargo 1.98.0.

## Query recall and latency

`benches/ann_recall.rs` creates 2,000 vectors with 32 dimensions and runs 48
top-10 queries for cosine exact/HNSW/IVF and L2 exact/Vamana/DiskANN-PQ modes. One untimed
warmup precedes each mode. `Median round` is the
median of five timed rounds divided by 48; p50/p95/p99 use nearest-rank
percentiles over all 240 individual queries. Index construction is excluded.

```sh
cargo bench --bench ann_recall
```

| Mode | Metric | Recall@10 | Median round (µs/query) | p50 (µs) | p95 (µs) | p99 (µs) | Estimated payload (bytes) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | Cosine | 1.0000 | 370.86 | 360.88 | 435.33 | 462.58 | n/a |
| HNSW (`m=16`, `ef_construction=96`, `ef=64`) | Cosine | 1.0000 | 83.16 | 81.29 | 109.54 | 127.25 | 795,592 |
| IVF (`nlist=64`, 8 iterations, `nprobe=8`, scale factor 8) | Cosine | 0.9083 | 48.95 | 47.04 | 73.42 | 133.04 | 276,028 |

`estimated_payload_bytes` is a deterministic lower-bound estimate for encoded
ANN vectors, ordinal slot membership, graph edges, centroids, and postings. It
excludes allocator/container overhead plus authoritative document storage; it
is not process RSS or a heap-profiler measurement.

The positioned-reader Vamana/PQ baseline was recorded on 2026-09-01 on Windows
x86_64 with an Intel Xeon w5-2445, 128 GiB memory, and Rust/Cargo 1.97.1. It
uses the same deterministic vectors, query count, rounds, and percentile
methodology. Both graph families validate L2 only, so their reference is a
separate exact L2 run. Each positioned row closes and reopens the collection
read-only, asserts a validated cache hit, performs one untimed warmup, and then
traverses the A3S-native sidecar. The PQ row uses eight balanced chunks and up
to 256 centroids per chunk. Open-time full-file validation is excluded;
operating-system page-cache effects are not controlled.

| Mode | Metric | Recall@10 | Median round (µs/query) | p50 (µs) | p95 (µs) | p99 (µs) | Estimated payload (bytes) | Sidecar (bytes) | 4 KiB sectors/query |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | L2 | 1.0000 | 135.29 | 129.90 | 145.70 | 241.60 | n/a | n/a | 0.00 |
| In-memory Vamana (`R=32`, build list 96, alpha 1.2, query list 64) | L2 | 1.0000 | 302.42 | 300.30 | 345.20 | 429.90 | 681,828 | n/a | 0.00 |
| Positioned Vamana after reopen (same parameters) | L2 | 1.0000 | 1,319.95 | 1,330.70 | 1,512.10 | 1,616.10 | 681,828 | 823,296 | 138.04 |
| In-memory DiskANN PQ8 (same graph/query controls) | L2 | 1.0000 | 256.45 | 258.20 | 290.50 | 321.50 | 732,668 | n/a | 0.00 |
| Positioned DiskANN PQ8 after reopen | L2 | 1.0000 | 959.81 | 961.60 | 1,094.40 | 1,232.60 | 732,668 | 622,592 | 108.16 |

This is correctness, compression, and I/O-volume evidence, not an exact-scan
speedup claim. Against the matching Vamana rows, PQ8 reduced the in-memory
median by 15.2 percent, the positioned median by 27.3 percent, sidecar bytes by
24.4 percent, and loaded sectors/query by 21.6 percent while retaining
recall@10 1.0000. It remained slower than the 135.29 µs exact L2 scan at this
small corpus. The estimated in-memory payload is 7.5 percent larger because
the derived cache deliberately retains authoritative-equivalent full vectors
for validation, fallback, and final refinement in addition to PQ state. Each
query owns a bounded extent/node cache, so repeated graph edges within one
query do not repeat file reads. Cross-query caching is left to the operating
system. RaBitQ, mmap, and asynchronous I/O remain open optimizations.

Immediately before the exact executor replaced full result materialization and
sorting with a deterministic bounded top-k heap, the same exact fixture produced
the paired result below. Every eligible score is still computed in `f64`, while
only the ten retained document references are cloned and projected.

| Exact path | Median round (µs/query) | p50 (µs) | p95 (µs) | p99 (µs) |
| --- | ---: | ---: | ---: | ---: |
| Materialize and sort all 2,000 | 507.40 | 462.38 | 781.83 | 991.50 |
| Bounded top-10 heap | 136.44 | 133.04 | 193.46 | 236.96 |
| Change | -73.1% | -71.2% | -75.3% | -76.1% |

Immediately before the remaining ANN structures were converted from owned
primary-key strings to shared `u64` ordinals, the same warm fixture produced:

| Mode | String-key p50/p95/p99 (µs) | Ordinal p50/p95/p99 (µs) | Tail change | Payload before/after | Payload change |
| --- | ---: | ---: | ---: | ---: | ---: |
| HNSW | 114.29 / 140.17 / 168.25 | 98.38 / 116.88 / 146.29 | -13.9% / -16.6% / -13.1% | 890,836 / 820,748 | -7.9% |
| IVF | 78.75 / 93.12 / 109.17 | 47.17 / 57.67 / 71.12 | -40.1% / -38.1% / -34.9% | 310,028 / 290,028 | -6.5% |

Recall stayed at 1.0000 for HNSW and 0.9083 for IVF. Base and delta vector
maps, HNSW nodes/edges, IVF postings, tombstones, and candidate selections now
remain ordinal-valued. Primary keys are borrowed only for deterministic
equal-score comparisons and resolved lazily when the exact executor reads the
bounded candidate set.

The remaining ordered maps on the dense-vector and HNSW graph hot paths were
then replaced with validated direct-address ordinal slots. The immediately
paired measurements were:

| Mode | Ordered-map p50/p95/p99 (µs) | Direct-address p50/p95/p99 (µs) | Payload before/after (bytes) |
| --- | ---: | ---: | ---: |
| HNSW | 97.04 / 116.38 / 135.67 | 81.29 / 109.54 / 127.25 | 820,748 / 795,592 |
| IVF | 47.67 / 58.58 / 86.33 | 47.04 / 73.42 / 133.04 | 290,028 / 276,028 |

HNSW p50/p95/p99 fell by 16.2/5.9/6.2 percent and its estimated payload fell
by 3.1 percent with recall unchanged. IVF payload fell by 4.8 percent and its
median was effectively unchanged; the slower tail in this short run is
reported rather than treated as a speedup. The direct-address representation
removes tree lookup and per-node key storage, but does not change IVF's
centroid probing or exact re-rank work.

The HNSW search/build kernel originally maintained its frontier and retained
set by repeatedly sorting `Vec` values and removing the first element. A paired
run immediately before and after replacing that work with a max frontier heap,
a bounded worst-first result heap, borrowed graph identifiers, and a membership-
only hash set produced:

| HNSW operation | Repeated-sort baseline (µs) | Heap implementation (µs) | Speedup |
| --- | ---: | ---: | ---: |
| Query, per operation | 254.42 | 121.02 | 2.10× |
| Complete 2,000-vector rebuild | 932,172.00 | 475,962.00 | 1.96× |

Recall@10 remained 1.0000. Equal scores retain deterministic primary-key order,
and the same heap traversal is used by graph construction and querying.

## Filter-aware ANN latency and completeness

`benches/filtered_ann.rs` creates 8,400 two-dimensional documents. One filter
selects 4,200 vectors that occupy centroids far from every query, which makes a
fixed unfiltered IVF result deterministically underfill after filtering. A
second filter selects alternating document identifiers, representing metadata
that is distributed through vector space. Each mode has one untimed warmup and
32 queries in each of five measured rounds. Index construction is excluded.

```sh
cargo bench --bench filtered_ann
```

| Mode | Results/query | Exact re-rank candidates | Recall@10 | Microseconds/query | Exact speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact clustered-scope filter | 10 | 8,400 | 1.0000 | 2,477.06 | 1.00× |
| Fixed IVF then external post-filter | 0 | 80 | 0.0000 | 53.66 | invalid result |
| Filter-aware IVF | 10 | 80 | 1.0000 | 54.97 | 45.06× |
| Exact distributed-shard filter | 10 | 8,400 | 1.0000 | 2,302.84 | 1.00× |
| Filter-aware HNSW | 10 | 64 | 1.0000 | 79.50 | 28.97× |

The external post-filter row is a counterexample for the removed planning
strategy, not a selectable engine mode: its low latency accompanies an empty,
incorrect page. Filter-aware IVF starts at `nprobe=2`, intersects each ranked
centroid's ordinal posting bitmap with the scalar eligibility bitmap, and
expands through the first eligible buckets before limiting exact re-ranking to
80 vectors. Filter-aware HNSW keeps rejected nodes available as graph bridges
while retaining at most 64 eligible re-rank candidates. Both filter-aware rows
matched the exact top-10 identifiers for all 32 queries. Candidate counts
report the authoritative exact re-rank set, not internal centroid or graph-
navigation work. The deliberately simple, clustered corpus is a regression
fixture for completeness and should not be treated as a general ANN
performance model.

Previously, scalar evaluation converted every eligible ordinal into an owned
primary-key string and inserted it into a `BTreeSet` before ANN. A paired run
immediately before and after retaining the Roaring bitmap through vector
membership counting and ANN traversal produced:

| Filter-aware operation | String-set baseline (µs) | Shared eligibility bitmap (µs) | Fully ordinal ANN state (µs) | Total speedup |
| --- | ---: | ---: | ---: | ---: |
| IVF query | 671.41 | 99.38 | 54.97 | 12.21× |
| HNSW query | 738.88 | 116.91 | 79.50 | 9.29× |

The middle column records the earlier milestone that removed the scalar
candidate string set. The final column also removes owned strings from vector
maps, HNSW graph edges, IVF postings, tombstones, and the ANN/exact handoff.
The registry owns one persistent ID/ordinal generation shared by vector,
scalar, and FTS indexes; cardinality and eligibility use bitmap intersections.
Recall@10 remained 1.0000 in both filtered paths.

## Indexed collection reopen latency

`benches/reopen_index.rs` creates three 5,000-document, 32-dimensional
workspace-shaped fixtures: documents without physical indexes, scalar plus
BM25 indexes, and all of those indexes plus HNSW. It reports fixture insert and
close costs separately; those costs are excluded from reopen rows. Each reopen
row is the median of five read-only opens in the same process, so it represents
warm operating-system page-cache conditions. A forced-rebuild row removes only
the optional derived-index cache, and a read-only handle cannot refresh it.

```sh
cargo bench --bench reopen_index
```

| Open mode | Median (ms) | Cache file (bytes) | Snapshot (bytes) |
| --- | ---: | ---: | ---: |
| Documents only | 5.87 | 0 | 1,420,700 |
| Valid scalar + FTS cache | 9.24 | 882,259 | 1,420,841 |
| Forced scalar + FTS rebuild | 11.76 | 0 | 1,420,841 |
| Valid cache, all indexes | 12.39 | 2,982,245 | 1,420,909 |
| Forced rebuild, all indexes | 479.33 | 0 | 1,420,909 |

The format-6 cache reduced the paired all-index reopen by 97.42 percent, or
38.69 times. Scalar+FTS cache restore was 21.4 percent faster than rebuilding
those indexes at this small scale. The phase-isolated rows show that 5.87 ms
belongs to authoritative snapshot recovery and document normalization; the
remaining differences approximate derived-index work, but small schema
differences mean they are not a component profiler.

The benchmark accepts positive `A3S_VEC_REOPEN_DOCUMENTS` and
`A3S_VEC_REOPEN_ROUNDS` values. Setting
`A3S_VEC_REOPEN_SCALAR_FTS_ONLY=1` skips the HNSW fixture, which permits larger
workspace-scale measurements without ANN construction dominating the run:

```sh
A3S_VEC_REOPEN_DOCUMENTS=100000 \
A3S_VEC_REOPEN_ROUNDS=5 \
A3S_VEC_REOPEN_SCALAR_FTS_ONLY=1 \
cargo bench --bench reopen_index
```

Two prior format-5 100,000-document observations produced medians of 210.30 ms for a
cache hit and 259.05 ms after forcing a scalar+FTS rebuild, an 18.8 percent
reduction. The cache was 17,107,398 bytes beside a 28,523,491-byte authoritative
snapshot. The pre-change benchmark binary produced a 270.95 ms rebuild median;
its documents-only control was also 9.3 percent faster than the final control,
so raw wall time understates the layout gain. After subtracting each
documents-only control, the derived rebuild portion fell by about 16.6 percent,
and cache restore was another 40.9 percent cheaper than the current derived
rebuild portion. In the two final observations, cache persistence added about
27.9 ms to the initial close, less than the measured saving from one later
open.

As historical storage-codec evidence, immediately before format-4 MessagePack
snapshots replaced format-3 JSON—and before the later bulk-freeze and
derived-cache changes—the then-current ANN-only cache produced this paired run:

| Open mode | JSON v3 (ms) | MessagePack v4 (ms) | Change | Snapshot v3/v4 (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Documents only | 9.88 | 5.93 | -40.0% | 2,950,487 / 1,420,700 |
| Scalar + FTS rebuild | 16.12 | 12.04 | -25.3% | 2,950,752 / 1,420,831 |
| Valid ANN cache | 19.86 | 16.18 | -18.5% | 2,950,888 / 1,420,899 |
| Forced ANN rebuild | 567.93 | 570.26 | +0.4% | 2,950,888 / 1,420,899 |

The authoritative snapshot shrank by 51.8 percent. The ANN rebuild dominates
its row, so its 0.4 percent difference is treated as measurement noise rather
than a binary-decoding regression.

A cache hit still performs snapshot/WAL recovery, document normalization and
validation, cache decoding, and structural/content validation. Cache format 9
contains the shared ordinal table plus HNSW/IVF/Vamana/DiskANN, PQ state,
scalar, FTS, parsed tokenizer, and ordered token-filter generations. A Vamana
or DiskANN hit also validates the native 4 KiB-sector graph/vector-or-code
sidecar. Both artifacts are
non-authoritative and bound to the format, schema, revision, and exact manifest
snapshot/checkpoint/committed-WAL identity. Exact format-2/3 fixtures, older
version bytes, and missing, stale, corrupt, oversized, or structurally invalid
files all fall back to rebuilding from documents; a read-only open never writes
or repairs them. This reopen fixture does not issue queries. Cache-restored
Vamana and DiskANN generations now use positioned sidecar traversal; their
separate recall, latency, capacity, and sector-volume evidence appears above.

The same benchmark then opens the all-index fixture for writes and measures one
derived-index rebuild per call. ANN and full rebuilds refresh the cache after
publication; exact scalar/FTS rebuilds preserve its logically equivalent
generation and avoid rewriting it:

| Rebuild operation | Milliseconds | Speedup vs all indexes |
| --- | ---: | ---: |
| Target scalar field `language` | 1.56 | 309.49× |
| Target FTS field `body` | 5.55 | 86.99× |
| Target HNSW field `embedding` | 476.47 | 1.01× |
| `optimize()` all indexes | 482.80 | 1.00× |

Previously, every `rebuild_index(field)` call rebuilt the whole registry and
was equivalent to the `optimize()` row. The targeted path now shares the
ordinal table and unrelated immutable generations. HNSW remains dominant when
it is itself the requested field; scalar and FTS maintenance no longer pays
that cost. Those exact targeted rebuilds also avoid rewriting the logically
equivalent 2.98 MB derived cache.

## Incremental write latency

`benches/incremental_write.rs` builds the same 2,000-vector HNSW collection,
performs one untimed upsert, then measures 48 distinct single-document upserts.
It also measures 64 single-document scalar patches at 2,000, 20,000, and
100,000 documents. Manual durability isolates in-process document/index
maintenance from fsync latency. A query verifies the last delta vector before
timing a complete HNSW rebuild on the same resulting collection.

```sh
cargo bench --bench incremental_write
```

| Operation | Documents | Samples | Microseconds/operation |
| --- | ---: | ---: | ---: |
| Incremental HNSW upsert | 2,000 | 48 | 151.15 |
| Full HNSW rebuild | 2,000 | 1 | 302,156.00 |
| Persistent-tree scalar update | 2,000 | 64 | 169.12 |
| Persistent-tree scalar update | 20,000 | 64 | 153.34 |
| Persistent-tree scalar update | 100,000 | 64 | 148.94 |

The full rebuild cost was roughly 2,000 times the average incremental upsert in
this fixture. This demonstrates the cost avoided by sharing the immutable base
and publishing a bounded delta.

The same scalar-update fixture was run immediately before replacing the copied
`BTreeMap` generation with the persistent tree:

| Documents | Copied `BTreeMap` (µs) | Persistent tree (µs) | Speedup |
| ---: | ---: | ---: | ---: |
| 2,000 | 170.80 | 141.27 | 1.21× |
| 20,000 | 536.20 | 144.50 | 3.71× |
| 100,000 | 2,096.39 | 143.12 | 14.65× |

The old path increased 12.3 times across a 50-times-larger collection, while
the persistent-tree measurements stayed within 2.3 percent of one another.
These are not end-to-end durability or cross-project performance claims.

## Scalar bitmap prefilter latency

`benches/scalar_filter.rs` creates 100,000 workspace-shaped documents with a
four-dimensional vector plus language, modification-time, and path metadata.
It compares the exact scan oracle with revisioned scalar indexes. Each mode has
one untimed warmup followed by 32 queries in each of five rounds; latency is the
median round. Every indexed result and score is checked against the scan path.

```sh
cargo bench --bench scalar_filter
```

| Filter | Mode | Candidates/query | Microseconds/query | Speedup |
| --- | --- | ---: | ---: | ---: |
| Language equality | Scan | 100,000 | 35,003.25 | 1.00× |
| Language equality | Bitmap | 12,500 | 14,570.56 | 2.40× |
| Recent modification range | Scan | 100,000 | 27,796.28 | 1.00× |
| Recent modification range | Bitmap | 100 | 64.50 | 430.95× |
| Source path prefix | Scan | 100,000 | 28,951.00 | 1.00× |
| Source path prefix | Bitmap | 100 | 113.75 | 254.51× |
| Workspace conjunction | Scan | 100,000 | 30,795.88 | 1.00× |
| Workspace conjunction | Bitmap | 100 | 66.47 | 463.30× |

The conjunction combines language, path prefix, and modification time. Its
planner stops once a safe 100-document superset is selective enough and lets
the final AST check reduce it to the 10 eligible documents, avoiding a larger
range-posting union without weakening correctness. Equality speedup is lower
because one language intentionally selects one eighth of the corpus and all
12,500 candidates still receive exact vector scoring.

The benchmark also rebuilds all three scalar fields after the query fixture.
Replacing per-document writes into persistent posting maps with mutable bulk
aggregation followed by one freeze produced this paired result:

| Operation | Persistent incremental builder (µs) | Bulk freeze (µs) | Change |
| --- | ---: | ---: | ---: |
| Full three-field scalar rebuild, 100,000 documents | 161,526.71 | 157,029.58 | -2.8% |

## Indexed BM25 query and write latency

`benches/fts_index.rs` creates 50,000 workspace-shaped source documents and
compares scan BM25 with the revisioned term-frequency index. Each mode has one
untimed warmup followed by 16 queries in each of five rounds; latency is the
median round. The benchmark checks every indexed result and public score
against scan BM25 before measuring 64 single-document body updates and one
complete index rebuild.

```sh
cargo bench --bench fts_index
```

| Query | Mode | Candidates/query | Microseconds/query | Speedup |
| --- | --- | ---: | ---: | ---: |
| Unique symbol | Scan | 50,000 | 17,131.00 | 1.00× |
| Unique symbol | Indexed | 1 | 0.69 | 24,827.54× |
| Component scope | Scan | 50,000 | 17,449.06 | 1.00× |
| Component scope | Indexed | 781 | 16.31 | 1,069.84× |
| Sparse multi-term | Scan | 50,000 | 18,251.69 | 1.00× |
| Sparse multi-term | Indexed | 782 | 32.38 | 563.67× |
| Common language | Scan | 50,000 | 17,577.94 | 1.00× |
| Common language | Indexed | 25,000 | 292.00 | 60.20× |
| Mixed terms | Scan | 50,000 | 19,582.25 | 1.00× |
| Mixed terms | Indexed | 25,782 | 496.00 | 39.48× |

| Write operation | Documents | Samples | Microseconds/operation |
| --- | ---: | ---: | ---: |
| Body update without FTS index | 50,000 | 64 | 143.16 |
| Incremental indexed-FTS update | 50,000 | 64 | 161.33 |
| Full FTS rebuild | 50,000 | 1 | 117,249.00 |

The selective symbol and component cases avoid corpus tokenization and visit
only relevant postings. Common-term queries still perform exact BM25 scoring
for tens of thousands of candidates, but safe plans resolve documents only for
the retained top-k ordinals. Incremental posting maintenance avoids a full
rebuild that costs roughly 727 times as much as the measured indexed update.

Immediately before indexed score ownership changed from primary-key strings to
the shared ordinal generation, the same fixture produced the following paired
query result. Candidate counts and public results were unchanged; the new path
also avoids building a second candidate bitmap solely for telemetry.

| Query | Owned primary-key scores (µs) | Ordinal scores (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.94 | 0.88 | -6.4% |
| Component scope | 473.00 | 360.12 | -23.9% |
| Common language | 13,834.44 | 9,397.75 | -32.1% |
| Mixed terms | 15,986.38 | 11,474.31 | -28.2% |

The next paired run bounded exact result retention to `topk=20` instead of
cloning and sorting every scored document. The unique-symbol case is below the
timer's stable microsecond range; higher-cardinality cases show the retained
effect.

| Query | Full materialization (µs) | Bounded top-k (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.69 | 0.81 | +17.4% |
| Component scope | 312.00 | 262.19 | -16.0% |
| Common language | 9,370.75 | 5,775.38 | -38.4% |
| Mixed terms | 11,535.00 | 5,667.50 | -50.9% |

Query score construction was then changed from an ordinal-keyed tree payload
to a validated contiguous ordinal-score slice. A single term streams its
posting directly into that slice. Multi-term queries use direct-address score
scratch only for at least 4,096 estimated visits and when the ordinal span is no
more than eight times the estimate; sparse queries retain tree accumulation.

| Query | Tree score accumulation (µs) | Adaptive ordinal scores (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.81 | 0.75 | -7.4% |
| Component scope | 245.06 | 211.38 | -13.7% |
| Common language | 5,967.12 | 4,741.31 | -20.5% |
| Mixed terms | 5,695.56 | 4,868.69 | -14.5% |

The unique-symbol result remains below the timer's stable microsecond range.
The benchmark now includes a 782-candidate sparse multi-term case to ensure the
adaptive policy does not force direct-address scratch onto selective queries.

Indexed score planning was then changed to retain only the best `topk=20`
ordinals before document resolution when there is no final filter, or when the
scalar candidate generation proves exact coverage of that filter. Equal scores
use ascending primary key, matching public result ordering even when ordinal
order differs. Validation and `candidates_scanned` still cover the complete
score generation. Conservative or unindexed filters deliberately bypass this
pushdown so the authoritative document filter sees every scored candidate.

| Query | Full score handoff (µs) | Ordinal top-k handoff (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.75 | 0.75 | 0.0% |
| Component scope | 222.88 | 54.75 | -75.4% |
| Sparse multi-term | 242.06 | 86.25 | -64.4% |
| Common language | 4,647.44 | 1,132.38 | -75.6% |
| Mixed terms | 4,944.44 | 1,323.50 | -73.1% |

The paired run retained the same public results and full candidate counts of
1, 781, 782, 25,000, and 25,782 respectively. The pushdown removes redundant
document lookups and result-heap work; posting traversal and exact BM25 score
calculation still cover all matching ordinals.

BM25 scoring previously performed a separate persistent-map lookup for the
document length of every posting hit. Each posting entry now carries its
document length beside term frequency. On the measured 64-bit target,
`(u64, PostingEntry)` and the former `(u64, u32)` both occupy 16 bytes because
the extra `u32` reuses alignment padding, so the B-tree entry slot does not
grow. The document-length map remains the source for corpus accounting and
incremental insert/remove validation.

To control machine drift, the pre-change benchmark binary was saved and run in
two A/B pairs with the execution order reversed for the second pair. The table
uses the median of the two observations for each binary; public results and
candidate telemetry were identical in every run.

| Query | Separate length lookup (µs) | Posting-local length (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.69 | 0.69 | 0.0% |
| Component scope | 57.00 | 34.56 | -39.4% |
| Sparse multi-term | 76.09 | 49.69 | -34.7% |
| Common language | 1,151.00 | 643.69 | -44.1% |
| Mixed terms | 1,384.28 | 815.91 | -41.1% |

The two indexed-update observations changed from 179.70/168.25 microseconds
before to 176.08/171.50 microseconds after, showing no measured write-latency
regression. Full rebuild timings remained within the run-to-run spread.

The shared ordinal table then replaced its dense, append-only
ordinal-to-primary-key ordered map with a persistent indexed vector. The
ID-to-ordinal direction remains a persistent ordered map, and immutable
generations structurally share both directions. This removes integer B-tree
key comparisons from primary-key resolution without copying the reverse map on
ordinary writes.

The pre-change FTS binary was retained and run in two A/B pairs, reversing the
execution order for the second pair. The table reports the median of the two
observations for each binary. Public results and candidate counts were
unchanged in every run.

| Query | Reverse ordered map (µs) | Persistent indexed vector (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.91 | 0.78 | -13.8% |
| Component scope | 41.34 | 20.28 | -50.9% |
| Sparse multi-term | 58.85 | 46.06 | -21.7% |
| Common language | 748.88 | 429.91 | -42.6% |
| Mixed terms | 975.06 | 645.79 | -33.8% |

The unique-symbol row is below the timer's stable microsecond range. Across
four observations of each ANN binary, exact and HNSW medians shifted by +1.9
and +2.4 percent, which is treated as run noise, while IVF fell from 48.21 to
42.26 microseconds per query (-12.4 percent). Recall and deterministic payload
estimates were unchanged. Four observations of the incremental-write fixture
showed medians of 190.62 versus 187.17 microseconds for HNSW upsert and
191.86/184.45/191.60 versus 181.52/188.24/179.05 microseconds for scalar
updates at 2,000/20,000/100,000 documents. The mixed directions and small
magnitudes show no coherent write-path regression.

At that milestone, the optional ANN-only derived cache became format 3. A
deterministic 128-document fixture proved that the equivalent format-3 bytes
were smaller than the legacy format-2 encoding and that format-2 bytes were
ignored safely for document-derived rebuild.

Term postings then moved from a persistent ordered map to a sorted contiguous
immutable base plus a small persistent ordered change map. Index generations
share the base; insertions, replacements, and tombstones copy only change-map
paths. A posting compacts into a new base at one eighth of its base cardinality,
bounded to 64..=2,048 changes, while readers retain the previous generation.

The ordered-map benchmark binary was retained and run in two A/B pairs, with
the second pair reversing execution order. The table reports the median of the
two observations for each binary. Public results and candidate counts were
identical in every run.

| Query | Persistent ordered posting (µs) | Contiguous base + changes (µs) | Change |
| --- | ---: | ---: | ---: |
| Unique symbol | 0.66 | 0.62 | -5.3% |
| Component scope | 17.03 | 16.69 | -2.0% |
| Sparse multi-term | 33.81 | 29.07 | -14.0% |
| Common language | 329.66 | 286.13 | -13.2% |
| Mixed terms | 519.81 | 457.85 | -11.9% |

The unique-symbol row is below the timer's stable microsecond range. One of the
after observations experienced a machine-wide update slowdown that also moved
the no-index scan control. The paired indexed-update/scan-update ratio changed
from 1.17 before to 1.10 after, providing no evidence of posting-specific write
regression. Full rebuild medians were effectively unchanged at 117.27 versus
116.83 milliseconds.

For full rebuilds, mutable term/posting aggregation followed by one persistent
generation freeze replaced per-document mutation of persistent maps. A paired
run reduced the 50,000-document rebuild from 145,366 to 120,760 microseconds,
or 16.9 percent; incremental-update paths are unchanged.

The outer FTS term dictionary and document-length table then adopted the same
contiguous immutable base plus bounded persistent-delta design. Term lookup
checks the small change map before binary-searching the sorted base; document
lengths use ordinal-addressed slots, and each structure compacts at most once
after a document batch. Two order-reversed term-dictionary observations kept
the indexed-update median effectively unchanged at 155.96 versus 155.38
microseconds and the full-rebuild median within noise at 112.97 versus 111.56
milliseconds. Indexed/scan query ratios and all public results/candidate counts
were unchanged.

Because scalar/FTS generations and these layouts became serialized at that
milestone, the derived cache became format 4. Exact format-2 and format-3 byte
fixtures are safe misses. The 100,000-document reopen measurements above
provide the retained effect: the documents-control-adjusted derived rebuild
portion fell about 16.6
percent, and cache restore reduced the new full reopen median by another 18.8
percent (40.9 percent of the derived portion). Mutation/reopen tests also
compare cached scalar/FTS deltas bit-for-bit with a cache-free authoritative
rebuild.

Finally, inserting into an empty collection now detects that the mutation batch
covers the whole corpus and uses the mutable bulk FTS build directly instead of
first constructing a large persistent delta. In an order-reversed pair at
100,000 documents, scalar+FTS fixture insertion fell from 1,083.03 to 667.25
milliseconds (-38.4 percent). Subtracting each binary's documents-only control,
the derived-index portion fell from 613.21 to 314.77 milliseconds (-48.7
percent). Ordinary non-empty and partial batches continue to use persistent
deltas.

## Workspace n-gram FTS latency

`benches/ngram_fts.rs` creates 10,000 workspace-shaped source/path documents
with the Unicode n-gram tokenizer configured for 2..3 letter/digit grams. It
measures initial indexed insertion, one untimed query warmup followed by 16
queries in each of five rounds, 64 single-document updates, a complete FTS
rebuild, close, and a read-only cache reopen. Manual durability isolates index
maintenance from per-write fsync latency.

```sh
cargo bench --bench ngram_fts
```

The benchmark accepts positive `A3S_VEC_NGRAM_DOCUMENTS` and
`A3S_VEC_NGRAM_ROUNDS` values. The second of two consecutive default runs
produced:

| Query | Default operator | Candidates/query | Microseconds/query |
| --- | --- | ---: | ---: |
| Selective identifier suffix | OR | 10,000 | 527.31 |
| Selective identifier suffix | AND | 11 | 17.81 |
| Common path fragment | AND | 10,000 | 7,229.12 |
| Common CJK phrase | AND | 10,000 | 1,026.06 |

For the selective identifier, shortest-posting AND reduced scored candidates
909 times and latency 29.61 times while retaining the same first result. The
common rows intentionally show the opposite boundary: AND is not a universal
speed switch when every document contains every gram, because all candidate
documents still require intersection checks and BM25 scoring. OR remains the
compatibility default; callers should select AND when all analyzed grams are
semantically required.

| Operation | Samples | Result |
| --- | ---: | ---: |
| Initial indexed insert | 1 | 192.71 ms |
| Incremental indexed body update | 64 | 218.08 µs/update |
| Complete n-gram FTS rebuild | 1 | 188.08 ms |
| Checkpoint/cache close | 1 | 52.35 ms |
| Format-6 cache reopen | 1 | 19.64 ms |
| Derived cache size | 1 | 18,701,015 bytes |

The first run measured 196.10 ms insert, 537.50/18.38 microseconds for the
identifier OR/AND pair, 244.53 microseconds per update, 199.34 ms rebuild,
52.55 ms close, and 19.97 ms reopen. The close/reopen contract also verifies
that the parsed tokenizer, ordered filters, and n-gram postings restore from
cache. Cache format 6 supersedes format 5 because filter state is part of the
validated derived FTS generation; older cache bytes remain non-authoritative
safe misses.

## Structured FTS planner latency

`benches/structured_fts.rs` builds paired indexed and scan collections over
25,000 workspace-shaped documents. Each case performs one warmup followed by
eight queries in each of five rounds and reports the median round. The fixture
asserts identical result IDs and score bits between both collections.

```sh
cargo bench --bench structured_fts
```

Positive `A3S_VEC_STRUCTURED_FTS_DOCUMENTS` and
`A3S_VEC_STRUCTURED_FTS_ROUNDS` values override the defaults. The second of two
consecutive default runs produced:

| Query | Planner path | Candidates/query | Planner µs/query | Explicit scan µs/query |
| --- | --- | ---: | ---: | ---: |
| Selective unique term + phrase | Indexed | 1 | 3.50 | 13,218.88 |
| Selective required + optional | Indexed | 1 | 1.88 | 13,472.00 |
| Common phrase | Planner scan | 25,000 | 13,519.25 | 13,600.25 |
| Broad boolean + NOT | Planner scan | 25,000 | 14,697.00 | 14,661.12 |

The two selective expressions reduced scored candidates 25,000 times and were
3,777 and 7,166 times faster than their scan controls in this observation. The
first run measured 3.75/15,257.00 and 2.25/15,275.12 microseconds for those
pairs, preserving the same candidate counts. Phrase verification does not yet
store token positions, so broad phrases can make indexed retokenization more
expensive than a sequential scan. The planner therefore selects scan when a
phrase estimate reaches half the corpus, or another structured estimate reaches
three quarters, unless a scalar bitmap already narrows the eligible documents.
Across both runs, the two automatic-fallback rows stayed within 0.6 percent of
their explicit scan controls.

A representative cross-project benchmark suite, larger corpora, sustained
mixed read/write workloads, allocator-aware memory accounting, and concurrent
tail-latency reporting remain Phase 7 work. The ANN fixture above provides a
single-process latency distribution and deterministic encoded-payload estimate,
not those broader claims.
