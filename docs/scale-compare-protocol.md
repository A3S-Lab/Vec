# First-principles a3s-vec ↔ zvec comparison protocol

This protocol defines what a fair product comparison is allowed to claim.
It deliberately refuses overfitting (lower `ef`, drop exact re-ranking,
switch public scores to `f32`, enable zvec post-optimize only on one side,
or cherry-pick a single lucky process).

## Why re-do this

Prior `BENCHMARKS.md` / README rows mix hosts (Xeon vs Apple Silicon),
stacked same-day micro-optimizations ("after prefetch", "after ordinal
rerank"), and tables that are easy to read as a universal ranking. A
first-principles comparison starts from shared physical work, shared
controls, and **fresh** measurements on one declared host for the current
tree—not from the most flattering inherited cell.

## What is being compared

| Dimension | Shared control | Explicit asymmetry (disclose, do not hide) |
| --- | --- | --- |
| Corpus | SplitMix64 `f32` generator, same IDs / query schedule / batch size | — |
| Metric / top-k | Cosine, top-10, Recall@10 vs Flat exact ranking | — |
| HNSW knobs | `m=16`, `ef_construction=96`, `ef=64` | Graph construction algorithms differ |
| Build workers | a3s: `RAYON_NUM_THREADS=1`; zvec: `IndexOption(concurrency=1)` + `init(query_threads=1, optimize_threads=1)` | — |
| HNSW query refinement | a3s always exact-reranks authoritative vectors (`f64` public scores) | zvec harness sets `is_using_refiner=False` (refiner needs a separate flat index; not enabled) |
| Flat query threads | Fairness harness: one Rayon worker | Product-default harness: unset Rayon (host core count). Report both; do not conflate |
| ISA / language | Same host | a3s portable Rust crate vs zvec macOS arm64 C++ wheel via Python binding |
| Persistence | Manual durability / flush after insert on both | Snapshot / checkpoint semantics still differ |
| Flat "index build" | zvec Flat has no separate build (CSV `0`) | a3s may spend time in `rebuild_index` before Flat query; report raw CSV and treat Flat comparison as insert + query, not build |

## Fixtures (must run both)

1. **Smoke** (`A3S_VEC_BENCH_SCALE=smoke`): 96 × 8 — harness wiring only.
2. **Small**: 2,000 × 32 — recall should be 1.0 for HNSW at these knobs.
3. **Scale**: 100,000 × 128 — primary product comparison point.

Each fixture: Flat + HNSW, **three independent processes**, report **median**
per column. Keep process-level recall ranges for zvec HNSW (nondeterministic
graph).

## Harness commands

```bash
# a3s-vec (fairness: one worker)
RAYON_NUM_THREADS=1 \
A3S_VEC_SCALE_DOCUMENTS=100000 A3S_VEC_SCALE_DIMENSIONS=128 \
A3S_VEC_SCALE_MODE=both \
cargo bench --bench scale_compare

# zvec companion (venv with zvec==0.7.0)
.venv-zvec/bin/python scripts/scale_compare_zvec.py \
  --documents 100000 --dimensions 128 --mode both
```

Optional product-default Flat only (do **not** use for HNSW fairness claims):

```bash
# unset RAYON_NUM_THREADS
A3S_VEC_SCALE_MODE=flat A3S_VEC_SCALE_DOCUMENTS=100000 A3S_VEC_SCALE_DIMENSIONS=128 \
cargo bench --bench scale_compare
```

## Allowed claims

- Same-host directional evidence for the declared fixture and controls.
- Relative medians (build time, p50/p95/p99, QPS, Recall@10).
- Explicit statement when HNSW latency is **not** same-refiner.

## Forbidden claims

- Universal engine ranking / SLO.
- Wins obtained by lowering `ef`, dropping a3s exact re-ranking, or
  enabling zvec refiner/optimize on only one side.
- Treating Python binding overhead as absent, or treating portable Rust vs
  native-wheel SIMD as identical compilers.
- Replacing the release `ann_recall` gate with this smoke comparison.

## Completion evidence

Goal is done only when:

1. This protocol is checked in.
2. Fresh three-process CSVs for small + scale exist under
   `target/fp-compare-<date>/` for **current** HEAD on the declared host.
3. `BENCHMARKS.md` and README "Proof vs zvec" are rewritten from those CSVs
   (historical tables labeled historical, not presented as current proof).

### Recorded completion (2026-09-20)

| Requirement | Evidence |
| --- | --- |
| Protocol | this file + `scripts/run_fp_compare.sh` |
| Small 2k×32 | `target/fp-compare-20260920/small/medians.csv` (HEAD `8c11ad6`, M5 Max) |
| Scale 100k×128 | `target/fp-compare-20260920/scale/medians.csv` |
| Product-default Flat | `target/fp-compare-20260920/scale-flat-default/` (Rayon unset) |
| Docs | `BENCHMARKS.md` “First-principles…” section; README “Proof vs zvec” |
