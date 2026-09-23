# In-process index benchmark evidence

These dependency-light, deterministic fixtures provide machine-local
regression evidence for the in-process ANN, scalar-filter, and full-text paths.
They are not a zvec comparison and should not be generalized to other
hardware, corpora, dimensions, build parameters, durability policies, or
concurrency levels.

Environment for the 2026-08-30 and 2026-08-31 measurements: Apple M5, 16 GiB
memory, Darwin arm64, Rust/Cargo 1.98.0.

## First-principles a3s-vec ↔ zvec comparison (current)

Protocol: [`docs/scale-compare-protocol.md`](docs/scale-compare-protocol.md).
Runner: `scripts/run_fp_compare.sh`. Companion harnesses:
`benches/scale_compare.rs` and `scripts/scale_compare_zvec.py`.

Claims in this section come from one fresh same-host run of that protocol.
Each numeric column is the median of the processes named below. The
2026-09-20 tables, including the unsplit `77,339.081` ms million-document
insert, stay historical and are not a split of this run.

### Host and controls (fairness harness)

| Item | Value |
| --- | --- |
| Host | Apple M5 Max, macOS arm64 |
| a3s-vec | `0.1.8` (proof measured on the `0.1.7` ranking code), `RAYON_NUM_THREADS=1` |
| zvec | `0.7.0`, `IndexOption(concurrency=1)`, `init(query_threads=1)`, `is_using_refiner=False` |
| Fixture | Cosine, top-10, 32 queries × 3 rounds, batch 512, HNSW `m=16`, `ef_construction=96`, `ef=64` |
| Processes | 2,000×32 and 100,000×128: three. 1,000,000×128: one |
| Stamp | 2026-09-23 |

Asymmetries that remain material: portable Rust crate vs native C++ wheel,
a3s exact `f64` re-rank vs zvec without refiner, and a3s Flat
`rebuild_index` time (reported in `index_build_ms`) vs zvec Flat `0`.
Insert time still includes batch insert and the final flush.

### 2,000 × 32 (three-process median)

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 28.220 | 0.721 | 28.975 | 11.541 | 13.167 | 16.167 | 82,341.59 | 1.0000 |
| zvec 0.7.0 flat | 28.817 | 0.000 | 28.817 | 53.375 | 74.833 | 93.333 | 17,390.65 | 1.0000 |
| a3s-vec HNSW | 28.220 | 66.201 | 94.422 | 29.250 | 34.875 | 36.667 | 32,757.98 | 1.0000 |
| zvec 0.7.0 HNSW | 28.817 | 67.831 | 96.639 | 57.625 | 69.917 | 85.625 | 16,688.03 | 1.0000 |

### 100,000 × 128 (three-process median)

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 345.696 | 48.013 | 393.709 | 770.416 | 817.833 | 854.458 | 1,287.79 | 1.0000 |
| zvec 0.7.0 flat | 964.416 | 0.000 | 964.416 | 1,694.083 | 1,910.208 | 2,332.167 | 583.18 | 1.0000 |
| a3s-vec HNSW | 345.696 | 12,405.774 | 12,750.889 | 98.291 | 127.667 | 145.000 | 9,842.96 | 0.6000 |
| zvec 0.7.0 HNSW | 964.416 | 45,275.918 | 46,240.335 | 136.333 | 181.542 | 226.084 | 6,928.93 | 0.5813 |

### 1,000,000 × 128 (one process)

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 3,332.440 | 533.593 | 3,866.032 | 7,538.708 | 7,708.417 | 8,454.000 | 132.47 | 1.0000 |
| zvec 0.7.0 flat | 9,999.630 | 0.000 | 9,999.630 | 23,638.417 | 24,077.792 | 24,581.792 | 42.30 | 1.0000 |
| a3s-vec HNSW | 3,332.440 | 201,843.549 | 205,175.989 | 135.666 | 220.583 | 279.292 | 6,777.27 | 0.3063 |
| zvec 0.7.0 HNSW | 9,999.630 | 558,963.467 | 568,963.097 | 180.292 | 230.584 | 249.667 | 5,448.29 | 0.2594 |

On every fixture in this run, a3s-vec insert and one-worker Flat p50 are lower
than zvec, and a3s-vec HNSW build, HNSW p50, and Recall@10 are not worse.
Public scores stay `f64` from exact re-rank. Default `ef` stays 64.

## Historical fairness harness (2026-09-20)

The rows below are the previous same-host record. The million-document insert
`77,339.081` ms is that run's unsplit measurement. It is not decomposed here.

### Host and controls

| Item | Value |
| --- | --- |
| Host | Apple M5 Max, 64 GiB, macOS 26.6.2 arm64 |
| a3s-vec | `0.1.3` on Apple M5 Max, `RAYON_NUM_THREADS=1` |
| zvec | `0.7.0` macOS arm64 wheel, Python 3.13.15, `IndexOption(concurrency=1)`, `init(query_threads=1)`, `is_using_refiner=False` |
| Fixture | Cosine, top-10, 32 queries × 3 rounds, batch 512, HNSW `m=16`, `ef_construction=96`, `ef=64` |
| Artifacts | `target/fp-compare-20260920/{small,scale}/` (local; not committed) |
| Stamp | `20260920T093805Z` (scale), `20260920T093757Z` (small) |

### 100,000 × 128 (2026-09-20)

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 1,278.853 | 52.696 | 1,331.549 | 3,550.375 | 3,895.000 | 4,226.208 | 277.31 | 1.0000 |
| zvec 0.7.0 flat | 1,036.483 | 0.000 | 1,036.483 | 1,841.084 | 2,285.334 | 2,315.667 | 534.10 | 1.0000 |
| a3s-vec HNSW | 1,278.853 | 26,341.207 | 27,733.976 | 103.417 | 139.917 | 154.458 | 9,255.35 | 0.6000 |
| zvec 0.7.0 HNSW | 1,036.483 | 46,167.887 | 47,194.801 | 149.000 | 201.584 | 210.583 | 6,507.28 | 0.5813 |

Relative medians (same controls):

- HNSW index build: a3s-vec ≈ **1.75×** shorter than zvec.
- HNSW query p50: a3s-vec ≈ **1.44×** lower than zvec, with exact re-ranking kept.
- HNSW Recall@10: a3s-vec stable **0.6000**; zvec **0.5656–0.6031** (median 0.5813).
- Flat query under one Rayon worker: a3s-vec ≈ **1.93×** higher p50 than zvec
  (`f64` public score / exact scan vs native `f32` path). Insert is about
  **1.23×** longer on a3s-vec in this run.

### Product-default Flat (not the fairness harness)

Exact Flat scan is embarrassingly parallel. With Rayon **unset** (host
reports 18 logical CPUs) on the same 100k×128 fixture, three-process a3s-vec
Flat median p50 is **650.584 µs** (Recall@10 = 1.0) versus zvec’s one-worker
Flat median **1,841.084 µs** — about **2.83×** lower p50. Do **not** mix this
row into the one-worker HNSW fairness table.

### 1,000,000 × 128 (2026-09-20, single process)

Same fairness controls (`RAYON_NUM_THREADS=1`, protocol HNSW knobs).
One process per engine. Not a three-process median. The insert column
`77,339.081` ms is unsplit: it is not assigned to WAL, snapshot, or index work.

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 77,339.081 | 651.834 | 77,990.915 | 42,871.000 | 46,497.917 | 48,390.625 | 23.37 | 1.0000 |
| zvec 0.7.0 flat | 13,507.130 | 0.000 | 13,507.130 | 26,201.459 | 28,228.583 | 29,535.583 | 37.94 | 1.0000 |
| a3s-vec HNSW | 77,339.081 | 422,410.953 | 499,750.034 | 158.750 | 238.708 | 279.375 | 5,953.86 | 0.3063 |
| zvec 0.7.0 HNSW | 13,507.130 | 627,723.432 | 641,230.562 | 231.417 | 283.458 | 334.167 | 4,243.70 | 0.2437 |

Directional: a3s HNSW builds and queries faster; zvec Flat insert/query is
faster under one worker. Protocol-default Recall@10 is low on both engines—
do not quote it as an accuracy SLO without raising `ef` / `ef_construction`.

### 2,000 × 32 (2026-09-20)

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 30.947 | 0.718 | 31.673 | 19.667 | 23.542 | 36.167 | 48,104.22 | 1.0000 |
| zvec 0.7.0 flat | 33.379 | 0.000 | 33.379 | 55.750 | 111.791 | 200.458 | 15,257.27 | 1.0000 |
| a3s-vec HNSW | 30.947 | 94.657 | 124.797 | 30.709 | 37.833 | 40.542 | 31,102.77 | 1.0000 |
| zvec 0.7.0 HNSW | 33.379 | 69.722 | 101.882 | 57.333 | 113.125 | 163.250 | 15,231.55 | 1.0000 |

In that run, zvec HNSW built faster while a3s-vec HNSW query p50 was lower.
Recall@10 was 1.0 for both. The current section above replaces this row.

## Historical: Xeon smoke comparison (2026-09-03)

This is a supplemental, same-host API comparison refreshed on 2026-09-03. It
is not the release gate and is not a substitute for a VectorDBBench run at
production scale. The a3s-vec side uses its Rust API and the zvec side uses
the zvec 0.7.0 Python/native binding. Unless noted otherwise, a3s-vec is built
with Cargo's portable default target features, while the zvec wheel contains
its release native C++ backend. That shipped-portability versus native-wheel
asymmetry is material for CPU-bound distance work, so these rows are a
directional product comparison rather than a compiler-level apples-to-apples
claim. Both use a vector-only schema, the same deterministic vector generator
as `ann_recall`, 2,000 documents, 32 dimensions, 32 top-10 queries, three
measured rounds, batches of 512, cosine distance, and one query worker. zvec
index creation is explicitly set to
`IndexOption(concurrency=1)`; the query and Rust Rayon worker controls are
also pinned to one. Index construction is timed separately from query
execution; queries include one warmup and result decoding is limited to
primary-key collection. The host was Windows x86_64 with an Intel Xeon w5-2445,
128 GiB RAM, Rust/Cargo 1.97.1, and Python 3.13. Each row is the median of
three independent process runs, so the values are regression evidence rather
than an SLO.

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 33.564 | 0 | 33.564 | 88.6 | 93.7 | 136.3 | 11,048.32 | 1.0000 |
| zvec 0.7.0 flat | 58.135 | 0 | 58.135 | 161.1 | 275.7 | 435.5 | 5,394.47 | 1.0000 |
| a3s-vec HNSW (`m=16`, `ef_construction=96`, `ef=64`) | 33.564 | 394.229 | 429.689 | 187.4 | 213.9 | 264.5 | 5,178.44 | 1.0000 |
| zvec 0.7.0 HNSW (same nominal controls) | 58.135 | 129.571 | 189.094 | 159.6 | 289.5 | 436.0 | 5,284.57 | 1.0000 |

On this current small fixture, a3s-vec's median Flat query p50 is about 1.8x
lower, while zvec's HNSW query p50 is about 1.2x lower. zvec's HNSW
index-build time is about 3.0x lower and its total build time about 2.3x
lower. Recall is identical at this scale. The a3s-vec rows include the
borrowed exact-score kernel and one-query-norm path added after the earlier
2026-09-02 table is intentionally superseded: it
used implicit zvec index concurrency and therefore did not enforce the
one-worker comparison. The two engines also have different persistence and
index lifecycle semantics, and process-to-process timing variance remains
material. The a3s-vec executor always exact-reranks authoritative vectors as
a correctness invariant. The zvec harness sets `is_using_refiner=False`; zvec's
optional refiner requires a separate flat reference index and is not enabled
by this fixture. Consequently the HNSW rows are not a strict same-refiner
comparison and should be treated as directional. Flat is closer to an
exact-vs-exact comparison, but its numeric kernels and storage layouts still
differ. The a3s-vec release `ann_recall` gate uses its authoritative HNSW
refiner; this cross-project smoke run must not be read as a replacement for
the release-gate numbers above.

## Larger-corpus scale comparison

`benches/scale_compare.rs` and `scripts/scale_compare_zvec.py` provide a
repeatable larger-corpus comparison. They share a SplitMix64 `f32` generator,
document IDs, query schedule, batch size, cosine metric, HNSW `m`,
`ef_construction`, `ef`, nearest-rank percentiles, and Recall@10 calculation.
The Rust benchmark is dependency-free and is included in the smoke CI gate;
the Python companion is an opt-in local tool because zvec wheels are
platform-specific. Both tools emit the same 20-column CSV contract, including
all HNSW construction and search controls.

The following three-process median was collected on 2026-09-03 on the same
Windows x86_64 Intel Xeon w5-2445 host (128 GiB RAM). The fixture has 100,000
documents x 128 dimensions, 32 queries, three measured rounds, batches of
512, one query/build worker, cosine distance, and HNSW `m=16`,
`ef_construction=96`, `ef=64`. zvec post-optimize and its optional
`is_using_refiner` path were disabled. The zvec harness passes
`IndexOption(concurrency=1)`; the a3s-vec harness uses `RAYON_NUM_THREADS=1`.
The a3s-vec executor still exact-reranks authoritative vectors after
candidate generation, so HNSW latency is directional rather than a strict
same-refiner comparison. The a3s-vec run uses manual durability;
`insert_ms` includes the initial insert-plus-flush/checkpoint boundary in
both harnesses, while index lifecycle timings are reported separately.
Percentiles are based on 96 samples per process. zvec's HNSW graph is not
bit-for-bit deterministic across processes, so the recorded recall range is
reported below alongside the median. These are scale indicators rather than
SLOs.

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 2,767.024 | 0 | 2,767.024 | 34,581.500 | 36,223.000 | 37,494.600 | 28.94 | 1.0000 |
| zvec 0.7.0 flat | 2,720.001 | 0 | 2,720.001 | 6,206.500 | 6,769.200 | 9,067.300 | 159.96 | 1.0000 |
| a3s-vec HNSW | 2,767.024 | 185,429.317 | 188,288.828 | 1,725.400 | 2,499.600 | 2,666.600 | 550.80 | 0.6000 |
| zvec 0.7.0 HNSW | 2,720.001 | 82,119.333 | 86,154.840 | 340.300 | 464.400 | 571.100 | 2,722.18 | 0.5719 |

### HNSW candidate refresh (2026-09-20, pre-tag)

The uncommitted-then-landed 0.1.1 candidate keeps exact re-ranking, `ef=64`,
and `f64` public scores. Navigation uses the packed `f32` slab, cosine-norm
cache, heap-skip, visited TLS, and construction-time edge-score reuse that
preserves the rescored neighbor lists. Values below are three-process medians
on the same Xeon w5-2445 host and controls as the 2026-09-03 table. zvec
Recall@10 ranged 0.5625–0.6219; a3s-vec was 0.6000 in every process.

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec 0.1.1 candidate HNSW | 2,913.047 | 49,316.543 | 52,364.205 | 355.000 | 415.700 | 486.900 | 2,736.66 | 0.6000 |
| zvec 0.7.0 HNSW (same day) | 2,490.961 | 70,919.508 | 73,410.469 | 348.500 | 461.300 | 718.100 | 2,759.95 | 0.5875 |

Relative to same-day zvec: index build is about 1.44× shorter, Recall@10 is
higher, and query p50 is within about 2% (noise). Relative to the 2026-09-03
a3s-vec HNSW row above, query p50 falls from 1,725.4 µs to 355.0 µs (−79.4%)
and index build from 185,429 ms to 49,317 ms (−73.4%) without lowering `ef`
or dropping exact re-ranking.

The matching 2,000 × 32 three-process medians on the same day:

| Engine / mode | Index build (ms) | p50 (µs) | Recall@10 |
| --- | ---: | ---: | ---: |
| a3s-vec 0.1.1 candidate HNSW | 145.869 | 62.800 | 1.0000 |
| zvec 0.7.0 HNSW | 123.652 | 206.800 | 1.0000 |

At this corpus size and configuration, zvec's median flat query p50 is about
5.6x lower and its historical 2026-09-03 HNSW query p50 about 5.1x lower than
the then-current a3s-vec row. The 2026-09-20 candidate closes the HNSW query
gap to noise while remaining faster to build. The
three zvec HNSW processes on 2026-09-03 produced Recall@10 values from 0.5625 to 0.5781
(median 0.5719); a3s-vec produced 0.6000 in all three, or 2.81 percentage
points higher than the zvec median at this parameter point. Both recall
values are too low to serve as a production target without increasing
`ef`. These results do not establish a universal engine ranking: they are
one host, one corpus, one worker, one compiler/runtime pairing, and one
parameter point. The HNSW build
lifecycle also differs: a3s-vec maintains and checkpoints its graph during
the index-creation transaction, while the zvec call has different persistence
and post-insert lifecycle semantics. Repeat the run with identical durability,
optimize, and recall targets before using it for a capacity decision.

### Historical: stacked Apple Silicon notes (superseded by FP redo above)

The following same-day progressive tables (small fixture, 100k, prefetch,
ordinal rerank) are retained for archaeology only. Prefer the
**First-principles** section for current HEAD claims.

#### Archived local Apple Silicon refresh (2026-09-20, pre-FP redo)

Same harness and controls as the tables above (`RAYON_NUM_THREADS=1`, zvec
`IndexOption(concurrency=1)`, `is_using_refiner=False`, cosine, `m=16`,
`ef_construction=96`, `ef=64`, 32 queries × 3 rounds, batch 512). Host:
macOS 26.6.2 arm64, Rust 1.98.1, Python 3.13.15, zvec 0.7.0 macOS arm64
wheel. Vec checkout `e6d067f` (package `0.1.1`; candidate bytes bound to
`a08413a`). Re-run at `2026-09-20T02:50:23Z`. Raw CSVs and three-process
medians: `target/local-compare/` (not committed). Each cell is the median of
three independent processes.

#### 2,000 documents × 32 dimensions

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 30.022 | 0.000 | 30.022 | 38.708 | 64.125 | 154.541 | 23,376.39 | 1.0000 |
| zvec 0.7.0 flat | 28.668 | 0.000 | 28.668 | 54.750 | 84.708 | 114.500 | 16,679.34 | 1.0000 |
| a3s-vec HNSW | 30.022 | 93.073 | 123.095 | 35.667 | 42.042 | 45.791 | 27,197.71 | 1.0000 |
| zvec 0.7.0 HNSW | 28.668 | 66.635 | 95.574 | 52.083 | 66.917 | 85.000 | 18,165.05 | 1.0000 |

On this small fixture a3s-vec HNSW query p50 is about **1.46×** lower than
zvec; zvec HNSW index build is about **1.40×** shorter. Recall@10 is 1.0000
for both.

#### 100,000 documents × 128 dimensions

| Engine / mode | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a3s-vec flat | 1,007.213 | 0.000 | 1,007.213 | 9,283.875 | 10,328.542 | 11,898.375 | 107.82 | 1.0000 |
| zvec 0.7.0 flat | 1,058.536 | 0.000 | 1,058.536 | 2,019.750 | 2,447.375 | 2,775.750 | 470.02 | 1.0000 |
| a3s-vec HNSW | 1,007.213 | 33,608.264 | 34,614.808 | 232.250 | 338.792 | 390.541 | 4,055.37 | 0.6000 |
| zvec 0.7.0 HNSW | 1,058.536 | 49,333.605 | 50,415.331 | 152.083 | 208.667 | 228.333 | 6,250.29 | 0.5750 |

On this host a3s-vec HNSW builds about **1.47×** faster than zvec and holds
Recall@10 0.6000 in every process (zvec 0.5719–0.6062, median 0.5750). HNSW
query p50 remains about **1.53×** higher than zvec while retaining exact
re-ranking. Flat exact `f64` stays slower than zvec's native path (~4.6×
p50), as expected from the public score contract. These arm64 rows are not
substitutes for the Windows Xeon release-candidate table. macOS 12 Monterey
Intel is unsupported.

#### After parallel Flat Cosine scan (2026-09-20, same host class)

Exact Flat scan is embarrassingly parallel. With the default Rayon pool
(host core count) and the same 100k×128 fixture, three-process medians show
a3s-vec Flat query p50 **below** zvec Flat while keeping public `f64`
re-score and Recall@10 = 1.0. Pinning `RAYON_NUM_THREADS=1` restores the
prior ~1.5–2× `f64`-vs-`f32` floor. Flat insert beats zvec either way.
See README for the current product summary.

#### After aarch64 navigation prefetch (same day, HNSW-only remeasure)

Navigation prefetch previously existed only for x86_64. Adding a stable
`prfm pldl1keep` hint on aarch64, then lengthening the aarch64 neighbor
prefetch runway to 8 (x86_64 stays at 4), does not change scores or recall.
Three-process HNSW-only medians on the same host and controls:

| Engine / mode | Index build (ms) | p50 (µs) | Recall@10 |
| --- | ---: | ---: | ---: |
| a3s-vec HNSW | 21,393.174 | 149.167 | 0.6000 |
| zvec 0.7.0 HNSW | 49,141.421 | 146.875 | 0.5938 |

Relative to same-day zvec: index build about **2.30×** faster; query p50 within
about **1.6%** (noise band on this host; a3s process p50s were 124–176 µs);
Recall@10 still higher and stable at 0.6000.

#### After ordinal exact re-rank (same day, HNSW-only interleaved)

Exact re-ranking still uses the authoritative `f64` promotion. It now scores
ANN ordinals directly and resolves primary keys only for competitive top-k
candidates, removing a redundant ordinal→id→ordinal round trip. Three-process
interleaved HNSW-only medians on the same host and controls
(`target/local-compare/rerank-ordinal/`):

| Engine / mode | Index build (ms) | p50 (µs) | Recall@10 |
| --- | ---: | ---: | ---: |
| a3s-vec HNSW | 22,375.451 | 99.500 | 0.6000 |
| zvec 0.7.0 HNSW | 50,761.445 | 146.000 | 0.5844 |

Relative to same-day zvec: index build about **2.27×** faster; query p50 about
**1.47×** lower; Recall@10 still higher and stable at 0.6000. This closes the
Apple Silicon HNSW query gap under the honest harness without lowering `ef`
or dropping exact re-ranking.

### HNSW kernel follow-up (qualified revision 13585ccd)

The qualified candidate keeps graph construction and final document re-ranking on
the authoritative `f64` scorer, but uses the runtime-dispatched `f32` SIMD
kernel only while traversing an unquantized HNSW graph. It also resolves
primary keys lazily (only for exact score ties) and uses a bounded ordinal
bitset for dense visited sets. Encoded-vector paths retain their existing
representation-aware scorer, and non-finite SIMD accumulators fall back to
the `f64` implementation. This is therefore a query-kernel optimization, not
a change to the public score or Recall contract.

Three independent 100,000-document observations were collected on the same
Windows x86_64 Intel Xeon w5-2445 host (128 GiB RAM), with 128 dimensions,
32 queries, three rounds, batch size 512, one worker, cosine distance, and
HNSW `m=16`, `ef_construction=96`, `ef=64`. Values below are medians across
the three runs; Recall@10 was 0.6000 in every run.

| a3s-vec HNSW candidate | Insert (ms) | Index build (ms) | Total build (ms) | p50 (µs) | p95 (µs) | p99 (µs) | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous recorded revision | 2,767.024 | 185,429.317 | 188,288.828 | 1,725.400 | 2,499.600 | 2,666.600 | 550.80 | 0.6000 |
| Qualified revision 13585ccd | 2,624.877 | 141,622.863 | 144,213.026 | 923.400 | 1,143.200 | 1,511.900 | 1,061.96 | 0.6000 |
| Change | -5.1% | -23.6% | -23.4% | -46.5% | -54.3% | -43.3% | +92.8% | unchanged |

The old and new rows use the same Rust harness and controls. The new query
median is still about 2.71x slower than the zvec 0.7.0 HNSW row above
(340.300 µs), and its build median is about 1.72x longer than zvec
(82,119.333 ms). The remaining gap is expected from the native zvec wheel's
CPU-specialized implementation and different graph/storage lifecycle; this
optimization narrows the A3S query gap without trading away the exact
re-ranking or the measured Recall.

To isolate compiler target selection from engine design, three additional Flat
runs used the same host and fixture with `RUSTFLAGS="-C target-cpu=native"`.
Their median p50 was 29,988.7 microseconds, 13.3% below the portable a3s-vec
row's 34,581.5 microseconds, but still 4.83x above zvec's 6,206.5 microseconds.
Native code generation therefore explains part, not most, of the observed Flat
gap. This control is diagnostic only and is not the portable release baseline.

Run the Rust side at smoke or selected scale with:

```sh
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench scale_compare --quiet
A3S_VEC_SCALE_DOCUMENTS=100000 \
A3S_VEC_SCALE_DIMENSIONS=128 \
A3S_VEC_SCALE_QUERIES=32 \
A3S_VEC_SCALE_ROUNDS=3 \
A3S_VEC_SCALE_BATCH_SIZE=512 \
A3S_VEC_SCALE_MODE=both \
RAYON_NUM_THREADS=1 \
cargo bench --locked --bench scale_compare --quiet
```

Run the zvec companion from an environment containing zvec 0.7.0 and NumPy:

```sh
python scripts/scale_compare_zvec.py \
  --documents 100000 --dimensions 128 --queries 32 --rounds 3 \
  --batch-size 512 --ef-search 64 --hnsw-m 16 \
  --ef-construction 96 --mode both
```

On Unix, validate a Rust CSV before recording it:

```sh
awk -F, -f .github/check_scale_compare.awk scale-compare.csv
```

On Windows PowerShell, write the native command output as UTF-8 first; the
default `>` redirection in Windows PowerShell is UTF-16 and cannot be parsed by
the Unix validator:

```powershell
$env:A3S_VEC_BENCH_SCALE = "smoke"
$env:RAYON_NUM_THREADS = "1"
cargo bench --locked --bench scale_compare --quiet |
  Set-Content -Encoding utf8 target/scale-compare.csv
wsl bash -lc 'cd /mnt/d/code/a3s-vec && awk -F, -f .github/check_scale_compare.awk target/scale-compare.csv'
```

The hosted Windows job uses Bash redirection and therefore already emits
UTF-8. The same capture rule applies to the feature, concurrent, and mixed
workload CSVs when they are validated through WSL.

The same validator accepts the zvec companion CSV (`engine` must be either
`a3s-vec` or `zvec`), so both sides can be checked before a comparison is
recorded.

For scale context, zvec's published benchmark page evaluates Cohere 1M and
10M (768-dimensional) with QPS, recall, and load duration using VectorDBBench;
its reproduction commands pin the historical `zvec==v0.1.1` package. The
published results and environment are documented in the
[official zvec benchmark report](https://zvec.org/en/docs/db/benchmarks/).
Zvec's newer DiskANN report uses a 16-vCPU/64-GiB host and reports Cohere 1M
single-thread QPS of 238.2/142.9/94.3 at list sizes 100/300/500, with Recall@10
of 98.30/99.49/99.67%; those disk-backed numbers are not comparable to this
in-memory 2,000-vector smoke test. See the
[official DiskANN analysis](https://zvec.org/en/blog/2026-08-04-zvec-diskann/).

## Public feature matrix and performance gate

The release gate now has one small, deterministic matrix for the public
collection surface. `tests/feature_matrix.rs` contains three integration tests:

* the CRUD, projection, exact dense/sparse, source-ID, scalar/FTS, hybrid,
  group-by, iterator, schema-evolution, flush, reopen, and health routes are
  checked against a bounded fixture;
* HNSW, IVF/SOAR, HNSW RaBitQ, IVF RaBitQ, and metric-aware Vamana and
  DiskANN/PQ (L2, inner product, cosine, and MIPS-L2) are compared with the
  exact oracle at exhaustive controls; and
* Binary32/Binary64 exact L2/Hamming execution is checked across direct,
  source-ID, filtered, radius, projection, multi-query, group-by, persistence,
  and optional Tokio paths; unsupported binary ANN controls remain explicit;
  and
* the read-only cache/sidecar path and health state are asserted explicitly.

Run the focused matrix or the complete suite with:

```sh
cargo test --locked --test feature_matrix
cargo test --locked --all-features
```

`benches/feature_matrix.rs` executes every successful query route (all four
dense metrics, include-doc-id, dense/sparse/Binary32/Binary64 source-ID, sparse,
Binary32/Binary64 exact and radius/projection combinations, indexed FTS,
scalar-filtered dense/Binary32/Binary64, hybrid RRF,
  Binary32/Binary64 multi-query, dense/Binary32/Binary64 group-by,
fetch, iterator, and statistics/health), mutation and flush controls, all six
ANN families across their supported metrics, both reopened DiskANN sidecar
readers, and the Tokio query wrappers, including Binary32/Binary64 exact
execution. Each sample asserts a
non-empty or exactly-sized result, so a disconnected implementation fails the
benchmark.
The gate also requires non-zero monotonic percentiles and finite positive
throughput. It emits CSV with nearest-rank p50/p95/p99 latency, total work, and
work per second. The default fixture is 512 documents x 16 dimensions, 16
queries, and three rounds; CI uses the smaller smoke scale:

```sh
cargo bench --locked --bench feature_matrix --features async
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench feature_matrix --features async
```

The same CSV contract used by CI can be checked locally with
`.github/check_feature_matrix.awk`; it requires all 53 operation names to be
unique and every dimension, work, percentile, throughput, and work-per-sample
column to be a finite positive number with monotonic p50/p95/p99 values.

## Lifecycle, resource, and maintenance matrix

`benches/lifecycle_matrix.rs` covers the management plane that is not a query
route: collection creation, batch insert, update, upsert, delete, filtered
delete, schema add/rename/alter/drop, index create/drop/rebuild/optimize,
flush, cache-hit reopen, resource-limit rejection, stats/health, and explicit
maintenance ownership. Each of its 16 rows asserts the operation's result and
reports nearest-rank p50/p95/p99 latency, total work, and work per second in a
10-column CSV. The smoke output is validated by
`.github/check_lifecycle_matrix.awk` and is uploaded with the other hosted
performance artifacts.

```sh
cargo bench --locked --bench lifecycle_matrix
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench lifecycle_matrix
```

A representative local Windows x86_64 smoke run (96 documents × 8 dimensions, two
samples, one process) produced the following p50 values; the checked CSV also
contains p95/p99 and throughput for every row:

| Operation | p50 (µs) | Work/sample |
| --- | ---: | ---: |
| create_collection | 15,614 | 1 |
| insert_batch | 4,584 | 96 |
| update | 3,491 | 1 |
| upsert | 3,780 | 1 |
| delete | 3,086 | 1 |
| delete_by_filter | 3,770 | 48 |
| create_index | 19,120 | 96 |
| drop_index | 10,576 | 96 |
| rebuild_index | 12,996 | 96 |
| optimize | 16,440 | 96 |
| schema_evolution | 41,765 | 4 |
| flush | 11,135 | 1 |
| reopen | 1,509 | 96 |
| resource_rejection | 19 | 1 |
| stats_health | 10 | 1 |
| maintenance_start_close | 225 | 1 |

These values are management-plane regression indicators, not portable SLOs;
repeat them on the same host when comparing revisions. Logical accounted bytes
remain an engine-owned estimate and are not a process-RSS measurement.

## Cross-platform smoke performance evidence

The CI platform matrix runs the same smoke-scale feature, concurrent-reader,
mixed-read/write, scale, and lifecycle benches on Linux x86_64, Linux arm64,
Windows x86_64, macOS arm64, and macOS Intel. Each platform validates its five CSVs
with the same AWK gates and uploads them as a revision-bound artifact named
`a3s-vec-platform-performance-<platform>-<revision>`. This catches platform-
specific regressions in latency, throughput, ANN recall, and mixed-workload
revision/accounting behavior instead of treating a single Linux run as
portable evidence.

The hosted `macOS Intel` row is an Intel macOS 15 image compiled with a 15.0
deployment target. It is useful portability evidence. macOS 12 Monterey Intel
is unsupported and is not a release gate. The platform smoke fixture is
intentionally small; the default-scale
same-host measurements below remain the source for trend comparisons, and
process RSS/allocator attribution still requires an OS-specific harness.

The latest complete hosted run is [CI run 33772179017](https://github.com/A3S-Lab/Vec/actions/runs/33772179017)
for revision `13585ccd3f956f6cb7d669b2ee6acc7096fca03d`. All ten jobs
passed, including the versioned release-candidate package. The following compact
extraction comes from its revision-bound platform artifacts. It uses the smoke
fixture (96 documents, 8 dimensions, 6 feature-matrix queries, 2 rounds; the
contention fixtures use 8 queries and 2 rounds), so it is a portability
snapshot rather than a production SLO. Latencies are p50 microseconds; the
last columns are the 8-reader mixed-workload read/write p50 and the parallel
schema-evolution lifecycle p50.

| Platform | Dense cosine p50 | HNSW p50 | Indexed FTS p50 | 8-reader HNSW QPS | Mixed read/write p50 | Schema evolution p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linux arm64 | 15.257 | 26.385 | 23.001 | 62,259.60 | 35.30 / 129.98 | 3,552.455 |
| Linux x86_64 | 16.090 | 31.218 | 26.359 | 40,215.50 | 62.23 / 145.20 | 4,425.980 |
| macOS arm64 | 14.542 | 24.666 | 19.875 | 49,420.07 | 32.42 / 228.38 | 11,748.875 |
| macOS Intel (hosted macOS 15) | 52.375 | 52.133 | 63.000 | 40,652.36 | 717.19 / 751.23 | 31,467.990 |
| Windows x86_64 | 25.300 | 31.300 | 31.900 | 40,067.61 | 967.40 / 1,045.60 | 20,194.200 |

Every contention row retained Recall@10 = 1.0000. The artifact also contains
the full 53-row feature matrix and all percentile/throughput columns; the
table only shows representative routes to keep platform differences legible.

## Concurrent query tail latency

`benches/concurrent_queries.rs` complements the feature matrix with a shared
HNSW collection queried concurrently by 1, 2, 4, and 8 workers. A barrier starts
all workers together; every worker executes the same deterministic top-10 query
set for the configured number of rounds. The report includes index build time,
Recall@10 against a flat oracle, nearest-rank p50/p95/p99 per-query latency,
and wall-clock QPS. The default fixture is 2,000 documents x 32 dimensions,
48 queries, and five rounds. CI uses the smoke scale (96 x 8, eight queries,
two rounds), requires Recall@10 >= 0.80, and validates the CSV with
`.github/check_concurrent.awk`.

```sh
cargo bench --locked --bench concurrent_queries --quiet
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench concurrent_queries --quiet
```

Set `A3S_VEC_CONCURRENCY` to a comma-separated positive worker list (for
example, `1,2,4,8`) and use the `A3S_VEC_CONCURRENT_*` variables to select a
larger fixture. This is read-only query contention evidence. It does not claim
mixed read/write behavior, process RSS, or a portable latency SLO.

## Mixed read/write contention

`benches/mixed_workload.rs` runs one scalar-update writer alongside 1, 2, 4, or
8 synchronized HNSW readers. Vectors remain unchanged so Recall@10 is compared
with a stable flat oracle; vector-index mutation is measured separately by the
`incremental_write` fixture. Each row reports index-build time, read and write
p50/p95/p99 latency, read/write wall-clock QPS, Recall@10, final revision, and
logical accounted bytes. CI uses the smoke scale and validates the 19-column
CSV with `.github/check_mixed.awk`; both CI gates require Recall@10 >= 0.80.

```sh
cargo bench --locked --bench mixed_workload --quiet
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench mixed_workload --quiet
```

The default fixture is 2,000 documents x 32 dimensions, 48 queries, five
rounds, and 192 scalar updates. Set `A3S_VEC_MIXED_READERS` and the
`A3S_VEC_MIXED_*` variables to repeat a larger or smaller deterministic run.
Logical accounted bytes are an engine-owned estimate, not process RSS; an
allocator/process-memory result still requires an OS-specific external harness.

A repeated default-scale run on the same Windows host (one writer, three
process runs, median per reader count, `RAYON_NUM_THREADS=1`) produced:

| Readers | HNSW build (ms) | Recall@10 | Read p50/p95/p99 (µs) | Write p50/p95/p99 (µs) | Read QPS | Write QPS | Accounted bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 400.8 | 1.0000 | 1,762.9 / 2,098.7 / 2,392.7 | 1,782.5 / 2,157.5 / 2,379.1 | 661 | 529 | 1,263,600 |
| 2 | 404.1 | 1.0000 | 1,722.7 / 1,986.4 / 2,655.7 | 1,751.8 / 2,027.5 / 2,903.5 | 1,339 | 536 | 1,263,600 |
| 4 | 406.6 | 1.0000 | 1,781.4 / 2,204.1 / 2,465.6 | 1,823.7 / 2,232.1 / 2,759.8 | 2,590 | 518 | 1,263,600 |
| 8 | 406.1 | 1.0000 | 1,703.2 / 2,185.2 / 2,530.4 | 1,741.3 / 2,266.0 / 2,686.2 | 5,264 | 526 | 1,263,600 |

These values are local contention indicators, not portable SLOs; compare
repeated runs on the same host and retain recall/correctness as the gate.

A repeated default-scale run on the Windows host above (three process runs,
median per worker count, `RAYON_NUM_THREADS=1`) produced:

| Workers | HNSW build (ms) | Recall@10 | p50 (µs) | p95 (µs) | p99 (µs) | QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 395.7 | 1.0000 | 125.7 | 168.3 | 225.0 | 7,553 |
| 2 | 395.7 | 1.0000 | 130.8 | 174.3 | 257.8 | 14,331 |
| 4 | 395.7 | 1.0000 | 131.9 | 176.8 | 215.6 | 28,131 |
| 8 | 395.7 | 1.0000 | 139.6 | 213.9 | 296.7 | 45,231 |

The following is a representative default-scale run on 2026-09-02 (Windows
11 x86_64, Intel Xeon w5-2445, 128 GiB, Rust/Cargo 1.97.1). Values are local
regression indicators, not portable SLOs; compare repeated runs on the same
host and keep recall/correctness as the acceptance gate. All latency columns
are microseconds per operation, and the benchmark's CSV also reports
`work_per_second`. Work is the asserted result/write unit count rather than
query operations; the CSV's `work_per_sample` makes that distinction explicit.

| Operation | Samples | p50_us | p95_us | p99_us | Work/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_l2 | 48 | 94.3 | 125.3 | 137.8 | 106,044.5 |
| dense_ip | 48 | 92.4 | 95.7 | 111.4 | 108,225.1 |
| dense_cosine | 48 | 98.5 | 102.4 | 113.2 | 101,522.8 |
| dense_mips_l2 | 48 | 48.9 | 49.9 | 50.9 | 204,499.0 |
| dense_include_doc_id | 48 | 54.5 | 55.7 | 57.1 | 183,486.2 |
| dense_source_id_l2 | 48 | 50.0 | 50.9 | 53.6 | 200,000.0 |
| dense_source_id_ip | 48 | 48.2 | 49.2 | 59.1 | 207,468.9 |
| dense_source_id_cosine | 48 | 52.6 | 53.1 | 53.6 | 190,114.1 |
| dense_source_id_mips_l2 | 48 | 48.9 | 91.6 | 93.0 | 204,499.0 |
| sparse | 48 | 94.4 | 137.0 | 141.3 | 105,932.2 |
| sparse_source_id | 48 | 89.6 | 103.0 | 108.8 | 111,607.1 |
| binary32_exact | 48 | 30.4 | 32.1 | 38.4 | 328,947.4 |
| binary32_radius | 48 | 15.1 | 20.9 | 23.0 | 132,450.3 |
| binary32_projection | 48 | 36.8 | 54.9 | 68.0 | 271,739.1 |
| binary64_exact | 48 | 33.0 | 42.9 | 46.3 | 303,030.3 |
| binary64_radius | 48 | 18.1 | 22.6 | 27.6 | 110,497.2 |
| binary64_projection | 48 | 38.6 | 62.6 | 66.9 | 259,067.4 |
| binary_source_id | 48 | 28.1 | 28.7 | 29.0 | 355,871.9 |
| binary64_source_id | 48 | 32.0 | 47.1 | 101.7 | 312,500.0 |
| binary_scalar_filter | 48 | 274.5 | 323.5 | 327.4 | 36,429.9 |
| binary64_scalar_filter | 48 | 278.8 | 367.7 | 396.4 | 35,868.0 |
| fts_indexed | 48 | 57.1 | 74.6 | 123.5 | 175,131.3 |
| dense_scalar_filter | 48 | 300.7 | 506.1 | 574.0 | 33,255.7 |
| multi_rrf | 48 | 147.2 | 181.6 | 208.0 | 67,934.8 |
| binary_multi | 48 | 43.6 | 56.1 | 61.0 | 229,357.8 |
| binary64_multi | 48 | 47.7 | 57.6 | 67.3 | 209,643.6 |
| group_by | 48 | 50.5 | 56.5 | 59.3 | 118,811.9 |
| binary_group_by | 48 | 22.7 | 31.1 | 71.8 | 132,158.6 |
| binary64_group_by | 48 | 26.1 | 34.4 | 171.7 | 114,942.5 |
| fetch_projection | 48 | 1.9 | 2.1 | 2.3 | 1,052,631.6 |
| snapshot_iterator | 3 | 544.0 | 594.3 | 594.3 | 941,176.5 |
| stats_health | 48 | 31.5 | 32.5 | 166.9 | 31,746.0 |
| partial_update | 3 | 1,913.4 | 1,915.4 | 1,915.4 | 522.6 |
| flush | 3 | 17,136.5 | 17,976.5 | 17,976.5 | 58.4 |
| dense_async | 48 | 91.3 | 162.1 | 244.1 | 109,529.0 |
| binary_async | 48 | 69.7 | 94.4 | 108.8 | 143,472.0 |
| binary64_async | 48 | 75.4 | 113.5 | 140.6 | 132,626.0 |
| multi_async | 48 | 115.9 | 184.6 | 208.8 | 86,281.3 |
| group_by_async | 48 | 117.1 | 160.8 | 167.9 | 51,238.3 |
| ann_hnsw | 48 | 61.9 | 69.7 | 82.8 | 161,550.9 |
| ann_ivf_soar | 48 | 47.2 | 69.7 | 121.3 | 211,864.4 |
| ann_hnsw_rabitq | 48 | 62.0 | 69.5 | 77.1 | 161,290.3 |
| ann_ivf_rabitq | 48 | 42.1 | 58.9 | 63.4 | 237,529.7 |
| ann_vamana | 48 | 118.3 | 201.8 | 218.1 | 84,530.9 |
| ann_vamana_ip | 48 | 106.3 | 119.3 | 131.6 | 94,073.4 |
| ann_vamana_cosine | 48 | 116.1 | 128.5 | 133.9 | 86,132.6 |
| ann_vamana_mips_l2 | 48 | 108.2 | 142.2 | 272.6 | 92,421.4 |
| ann_diskann_pq | 48 | 136.2 | 159.7 | 293.9 | 73,421.4 |
| ann_diskann_ip_pq | 48 | 132.8 | 137.3 | 152.6 | 75,301.2 |
| ann_diskann_cosine_pq | 48 | 137.9 | 152.8 | 158.0 | 72,516.3 |
| ann_diskann_mips_l2_pq | 48 | 133.6 | 140.5 | 166.0 | 74,850.3 |
| diskann_positioned_reopen_query | 3 | 6,769.0 | 7,501.3 | 7,501.3 | 1,477.3 |
| diskann_mmap_reopen_query | 3 | 6,829.3 | 7,072.6 | 7,072.6 | 1,464.3 |

The metric-aware Vamana and DiskANN rows use the same 512-document, 16-
dimension fixture and `list_size=64` controls as the original ANN rows. Their
correctness companion checks use eight deterministic query points, require at
least 0.50 recall@10, and cap exact candidate work at 96 on a 256-document
fixture; the benchmark rows are latency/throughput observations rather than a
portable latency SLO.

The smoke job uploads this CSV as a CI artifact and gates the release-candidate
job. It intentionally checks correctness and captures metrics without applying
a hardware-specific latency threshold. `ann_recall`, `filtered_ann`,
`structured_fts`, and the other historical fixtures below remain the sources
for recall, candidate-bound, planner, reopen, and larger-corpus comparisons.

## Query recall and latency

`benches/ann_recall.rs` creates 2,000 vectors with 32 dimensions and runs 48
top-10 queries for cosine exact/HNSW/IVF, HNSW/IVF RaBitQ, and L2
exact/Vamana/DiskANN-PQ modes. One untimed warmup precedes each mode. `Median round` is the
median of five timed rounds divided by 48; p50/p95/p99 use nearest-rank
percentiles over all 240 individual queries. Index construction is excluded.
The historical IVF rows use `use_soar=false`; they are not performance or
recall measurements for the optional SOAR dual-assignment path.

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

The RaBitQ rows were added on 2026-09-01 and measured on Windows x86_64 with an
Intel Xeon w5-2445, 128 GiB memory, and Rust/Cargo 1.97.1. The table below is
the final post-change run after numeric hardening. Both variants use seven bits;
HNSW uses 16 centers and `ef=64`, while IVF uses 64 lists/centers,
`nprobe=8`, a 1,000-vector training sample, and an 80-vector exact-refiner
limit. All modes keep authoritative vectors for final public scoring.

| Mode | Metric | Recall@10 | Median round (µs/query) | p50 (µs) | p95 (µs) | p99 (µs) | Estimated payload (bytes) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | Cosine | 1.0000 | 180.37 | 169.70 | 228.60 | 244.90 | n/a |
| HNSW | Cosine | 1.0000 | 123.56 | 120.60 | 170.80 | 236.80 | 795,592 |
| HNSW RaBitQ7 | Cosine | 1.0000 | 122.96 | 120.70 | 170.40 | 236.60 | 949,664 |
| IVF | Cosine | 0.9083 | 68.00 | 67.20 | 88.80 | 129.30 | 276,028 |
| IVF RaBitQ7 | Cosine | 0.9083 | 91.72 | 84.00 | 149.40 | 243.30 | 436,244 |

The fixed fixture shows no recall regression from RaBitQ: HNSW remains 1.0000
and both IVF variants remain 0.9083, where the shared eight-list probe window
is the limiting factor. It is not a memory or speedup claim. The deterministic
payload estimate intentionally counts full refinement vectors plus RaBitQ
centers/codes. At only 32 dimensions the HNSW medians are within run-to-run
noise, while the IVF estimator adds measurable work. The rows are a
regression baseline for compact-code execution; SIMD and code-only persisted
layouts remain possible later optimizations.

The sidecar-reader Vamana/PQ baseline was recorded on 2026-09-01 on Windows
x86_64 with an Intel Xeon w5-2445, 128 GiB memory, and Rust/Cargo 1.97.1. It
uses the same deterministic vectors, query count, rounds, and percentile
methodology and intentionally reports the L2 rows. The graph and ADC
implementations now also accept inner product, cosine, and MIPS-L2; their
metric-specific correctness and bounded-work evidence is in the public feature
matrix and ANN contract tests. Every positioned or mmap row closes and reopens the
collection read-only, asserts a validated cache hit and the requested backend,
performs one untimed warmup, and then traverses the A3S-native sidecar. The
benchmark also asserts identical mapped and positioned result IDs and exact
backend telemetry counts. PQ uses eight balanced chunks and up to 256 centroids
per chunk. Open-time full-file validation and the mmap backend's full-copy setup
are excluded; operating-system page-cache effects on positioned reads are not
controlled.

| Mode | Metric | Recall@10 | Median round (µs/query) | p50 (µs) | p95 (µs) | p99 (µs) | Estimated payload (bytes) | Sidecar (bytes) | 4 KiB sectors/query |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | L2 | 1.0000 | 128.28 | 126.10 | 139.80 | 257.30 | n/a | n/a | 0.00 |
| In-memory Vamana (`R=32`, build list 96, alpha 1.2, query list 64) | L2 | 1.0000 | 311.75 | 306.60 | 380.00 | 516.90 | 681,828 | n/a | 0.00 |
| Positioned Vamana after reopen (same parameters) | L2 | 1.0000 | 1,283.52 | 1,264.50 | 1,534.40 | 1,812.30 | 681,828 | 823,296 | 138.04 |
| Mmap-snapshot Vamana after reopen (same parameters) | L2 | 1.0000 | 846.68 | 827.50 | 1,071.30 | 1,325.40 | 681,828 | 823,296 | 138.04 |
| In-memory DiskANN PQ8 (same graph/query controls) | L2 | 1.0000 | 269.32 | 261.30 | 309.90 | 405.50 | 732,668 | n/a | 0.00 |
| Positioned DiskANN PQ8 after reopen | L2 | 1.0000 | 1,036.52 | 977.70 | 1,745.30 | 1,927.60 | 732,668 | 622,592 | 108.16 |
| Mmap-snapshot DiskANN PQ8 after reopen | L2 | 1.0000 | 702.38 | 708.70 | 766.40 | 825.80 | 732,668 | 622,592 | 108.16 |

This is correctness, compression, and I/O-volume evidence, not an exact-scan
speedup claim. PQ8 reduced sidecar bytes by 24.4 percent and staged sectors per
query by 21.6 percent while retaining recall@10 1.0000. Against its matching
positioned row, the mmap snapshot reduced the measured Vamana median by 34.0
percent and the PQ8 median by 32.2 percent. Both remained slower than the
128.28 µs exact L2 scan at this small corpus. The mapped backend is not a
memory-lazy file mapping: it copies the complete validated sidecar into a
read-only anonymous map during open and retains that sidecar-sized allocation,
which this query-only table excludes. The estimated in-memory payload is 7.5
percent larger for PQ because the derived cache deliberately retains
authoritative-equivalent full vectors for validation, fallback, and final
refinement in addition to PQ state. Each query owns a bounded extent/node cache,
so repeated graph edges within one query do not repeat backend reads. The
optional Tokio query methods move identical work to the blocking pool and
therefore make no latency claim. Native async file reads and direct file-backed
mmap remain open optimizations; RaBitQ evidence is reported in the cosine table
above.

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

The current format-10 implementation was re-run on 2026-09-02 with the same
100,000-document, 32-dimensional scalar+FTS-only fixture. Each value below is
the median of three independent process runs; each process used five warm
read-only opens per row. The host was Windows x86_64 with an Intel Xeon
w5-2445 and 128 GiB RAM. The documents-only row is the control for separating
authoritative snapshot recovery from derived-index work.

| Open mode | Median (ms) | Cache (bytes) | Snapshot (bytes) |
| --- | ---: | ---: | ---: |
| Documents only | 461.58 | 0 | 28,523,360 |
| Valid scalar + FTS cache | 745.66 | 17,107,426 | 28,523,501 |
| Forced scalar + FTS rebuild | 918.04 | 0 | 28,523,501 |

The valid cache is 18.8% faster than rebuilding the scalar and FTS indexes on
raw open time; the absolute difference is 172.38 ms. These numbers are warm
page-cache observations, not a portable
SLO, and they do not measure process RSS.

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
validation, cache decoding, and structural/content validation. Cache format 10
contains the shared ordinal table plus HNSW/IVF/RaBitQ/Vamana/DiskANN, PQ
state, scalar, FTS, parsed tokenizer, and ordered token-filter generations. A Vamana
or DiskANN hit also validates the native 4 KiB-sector graph/vector-or-code
sidecar. Both artifacts are
non-authoritative and bound to the format, schema, revision, and exact manifest
snapshot/checkpoint/committed-WAL identity. Exact format-2/3 fixtures, older
version bytes, and missing, stale, corrupt, oversized, or structurally invalid
files all fall back to rebuilding from documents; a read-only open never writes
or repairs them. This reopen fixture does not issue queries. Cache-restored
Vamana and DiskANN generations use the configured positioned or immutable
mmap-snapshot sidecar traversal; their separate recall, latency, capacity, and
sector-volume evidence appears above.

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
| Selective unique term + phrase | Indexed | 1 | 7.38 | 38,066.38 |
| Selective required + optional | Indexed | 1 | 4.62 | 38,372.50 |
| Selective wildcard | Indexed | 1 | 26,548.00 | 77,713.00 |
| Selective fuzzy distance 1 | Indexed | 36 | 33,811.00 | 98,363.38 |
| Selective exact range | Indexed | 1 | 429.62 | 50,439.12 |
| Selective ordered proximity | Indexed | 1 | 7.62 | 38,810.25 |
| Common phrase | Planner scan | 25,000 | 40,806.12 | 41,376.62 |
| Broad boolean + NOT | Planner scan | 25,000 | 43,191.25 | 42,977.25 |

The five one-candidate expressions reduced scored candidates 25,000 times and
were 117.4 to 8,305.7 times faster than their scan controls for term/phrase,
required/optional, range, and proximity execution. Wildcard expansion was 2.93
times faster than scan while still scoring one document. Fuzzy distance 1
expanded to 36 concrete terms and was 2.91 times faster than scoring the full
corpus. Dynamic wildcard/fuzzy expansion uses an index-local character-trigram
prefilter when a literal run (or fuzzy query) yields a non-zero shared-trigram
bound; otherwise it falls back to one analyzed-vocabulary scan. The concrete
term set is then shared by candidate construction and BM25.

The first run preserved the same plans and candidate counts. Its indexed/scan
pairs were 7.38/40,620.12, 4.62/38,979.88,
26,611.75/74,287.62, 31,729.75/93,866.88, 313.25/49,048.38, and
9.88/37,019.62 microseconds in the table's selective row order. Phrase
verification does not yet store token positions, so broad phrases can make
indexed retokenization more expensive than a sequential scan. The planner
therefore selects scan when a phrase estimate reaches half the corpus, or
another structured estimate reaches three quarters, unless a scalar bitmap
already narrows the eligible documents. Telemetry confirms that both broad rows
selected the same exact scan path as their controls.

A broader representative cross-project benchmark suite, long-duration/high-
scale mixed read/write workloads, and allocator-aware/process-level memory
accounting remain external qualification work. The refreshed same-host
cross-project scale table and the concurrent/mixed fixtures above provide
bounded comparison and contention evidence; they are not a replacement for
those larger-scale claims.
