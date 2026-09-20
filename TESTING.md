# a3s-vec Test Case Plan

First-principles catalog for comprehensive, deep testing of `a3s-vec`.
This document is the planning source of truth for new cases. It does not
replace executable gates in CI; every P0 case below must eventually land as
code with an explicit oracle or typed failure assertion.

Related: [ARCHITECTURE.md](ARCHITECTURE.md) (six invariants),
[ROADMAP.md](ROADMAP.md) (VEC-R2: change only when an invariant test fails),
[RELEASE.md](RELEASE.md) (hosted gates), [BENCHMARKS.md](BENCHMARKS.md)
(honest performance evidence, not correctness).

---

## 1. Objective

Plan tests that make the six architecture invariants *more true under
adversarial conditions*, not tests that merely exercise happy paths or
optimize for a single fixture.

**Deliverable of this plan:** a prioritized, evidence-backed case catalog
mapped to invariants, with gaps called out relative to the current suite
(~291 `#[test]` items across `src/` + `tests/`, plus smoke benches and
recovery fuzz).

**Out of scope for “correctness” cases:** lowering `ef`, dropping exact
re-rank, changing public `f64` scores, or adding indexes without a failing
invariant. Those are overfitting and are refused (VEC-R2).

---

## 2. First principles → test obligations

| # | Invariant | What every related test must prove |
| --- | --- | --- |
| I1 | Documents are authority | Derived indexes may be missing, stale, or wrong; DocumentMap exact scan still returns the authoritative ranking for that revision. |
| I2 | Schema owns meaning | Invalid type/dimension/metric/index combos fail *before* mutation with typed errors; no silent coercion. |
| I3 | One revision per query | Concurrent writers publishing newer revisions cannot tear a reader’s result set mid-query. |
| I4 | Durability is explicit | Ack’d mutations survive crash/reopen under the configured durability; torn WAL/snapshot/manifest fail closed. |
| I5 | Approximation never breaks correctness | ANN candidates are always exact-re-ranked; stale/missing ANN → Flat/exact fallback; scores match DocumentMap oracle bit-for-bit where claimed. |
| I6 | Portable path is default | SIMD / Rayon / mmap / async are accelerators; `RAYON_NUM_THREADS=1`, scalar kernels, and positioned I/O still match the same public results. |

**Cross-cutting score contract (Flat + ANN re-rank):**

- Public scores promote stored coordinates to `f64` (see `score_f64`).
- Cosine top-k may *rank* by `dot * inv_norm`, then **re-score** for the
  public value; ranking order must match the DocumentMap Cosine oracle.
- Bit-identical SIMD vs scalar for eligible kernels (`score_f64` unit tests).
- Dense Flat may use a packed rebuild-only cache; binary Flat stays on the
  DocumentMap Hamming path and must never enter `encode_vector`.

---

## 3. Current inventory (authoritative today)

### 3.1 Integration suites (`tests/`)

| Suite | Role |
| --- | --- |
| `contracts`, `public_api_contract`, `execution_contracts` | Schema/API allowlists, Send+Sync, unsupported controls |
| `ann_contracts/*` | HNSW/IVF/Vamana/DiskANN/RaBitQ exhaustive vs exact; lifecycle; recall bounds |
| `binary_queries`, `vector_codecs`, `document_api` | Typed vectors, Binary32/64, sparse, codecs |
| `differential_oracle`, `generated_differential`, `differential_fts` | Independent scan oracles on fixed/generated corpora |
| `filtered_ann`, `scalar_indexes`, `source_id_queries` | Filters composed with ANN / source-id |
| `fts_*`, `ngram_fts` | BM25, syntax, filters, n-grams |
| `durability`, `index_cache`, `mmap_diskann` | WAL/snapshot/cache/mmap |
| `concurrency`, `async_queries` | Readers vs writers; Tokio parity |
| `resource_limits`, `maintenance_health` | Limits, health states |
| `feature_matrix`, `workspace_retrieval_contract` | Advertised routes + Code shadow contract |

### 3.2 Unit / module tests (`src/`)

Notable: `score_f64` bit-identity, HNSW packed navigation, ordinals,
quantization, FTS postings, storage recovery fuzz + fault injection,
collection configuration.

### 3.3 Hosted gates (not unit tests)

Smoke benches + AWK validators: `feature_matrix`, `concurrent_queries`,
`mixed_workload`, `scale_compare`, `lifecycle_matrix`. Recovery fuzz smoke.
MSRV / fmt / clippy / audit / package jobs per `RELEASE.md`.

### 3.4 What exists for Flat specifically

| Case | Status |
| --- | --- |
| Unit: dense Flat Cosine top-k candidates | Present (`index/tests.rs`) |
| Integration: Flat is exact, not ANN telemetry | Present (`execution_contracts`) |
| Binary Flat L2/Hamming + reopen + stats ready | Present (`binary_queries`) |
| Feature matrix binary Flat smoke | Present |
| Packed dense Flat vs DocumentMap bit/order oracle | **Gap** |
| Flat after insert (cache absent) still exact | **Thin** — covered indirectly; needs explicit case |
| Flat `optimize`/`rebuild` refreshes pack; query matches | **Gap** |
| Cosine `inv_norm` ranking ≡ DocumentMap order | **Gap** as dedicated case |
| Rayon Flat (N>1) ≡ serial Flat results | **Gap** |
| Flat metrics matrix (L2/IP/Cosine/MipsL2) × dtypes | **Partial** via codecs/oracle; not Flat-index specific |
| FP64 Flat close-score primary-key ties | Mentioned in roadmap; **verify depth** |

---

## 4. Coverage model (how to place a case)

```text
L0  Pure kernel     score_f64 / quantization / tokenizer  — bit or math oracle
L1  Index unit      OrdinalMap, Flat pack, HNSW graph     — structural + candidate set
L2  Collection      Query + DML + schema                  — public API oracle
L3  Persistence     Close/reopen/crash/corruption         — I4
L4  Concurrency     Barriered readers/writers             — I3
L5  Differential    Generated corpus × filters × metrics  — I1+I5
L6  Gate            Smoke bench CSV validators            — regression SLO, not proof
```

Rules:

1. Prefer an **independent oracle** (DocumentMap scan, hand-computed Hamming,
   scalar `f64` loop) over “same engine, two paths.”
2. Prefer **deterministic seeds** and primary-key tie-breaks over approximate
   recall alone for correctness.
3. Recall@k gates prove *bounded ANN quality*; they do **not** prove score
   contract or Flat correctness.
4. Performance benches never substitute for L0–L5.

---

## 5. Gap analysis (deep)

### G1 — Packed dense Flat acceleration (post-0.1.1)

Recent Flat work (contiguous `f64` pack, lazy rebuild, Cosine `inv_norm`,
Rayon chunking) is under-tested relative to risk.

Required cases:

1. **Pack ≡ DocumentMap** for Cosine/L2/IP on FP32 (and FP16/FP64 if Flat
   accepts them): same IDs, same score bit patterns after public re-score.
2. **Insert without optimize:** Flat stats remain `ready`; query uses exact
   path; results match oracle.
3. **Optimize rebuilds pack:** second query may use packed path; results still
   match oracle bit-for-bit.
4. **Binary never packs:** `builds_packed_vector_index` false for Binary32/64;
   reopen never calls `encode_vector` on bits fields (regression for the
   2026-09-20 CI failure).
5. **Thread parity:** with `RAYON_NUM_THREADS=1` and default pool, identical
   ordered `(id, score_bits)` for the same corpus/query (I6).
6. **Tombstones / deletes / upserts** on Flat field between rebuilds.

### G2 — Score kernel adversarial inputs

1. Non-finite coordinates rejected at write or query boundary (typed error).
2. Zero-norm Cosine candidates: defined behavior vs DocumentMap (no NaN leak
   into public scores).
3. FP64 near-ties: stable primary-key order (roadmap claim).
4. Dimension tails not multiple of SIMD width: bit-identity vs scalar.
5. Empty collection / topk=0 / topk > n: typed or empty, never panic.

### G3 — ANN × exact re-rank matrix

Existing exhaustive tests are strong for L2/Cosine on core families. Deepen:

1. **Every metric × every ANN family** that schema allows, including
   MipsL2 where supported; refuse unsupported with typed error tests.
2. **`ef == n` HNSW ≡ Flat** ranking (roadmap exit gate language).
3. **Stale index revision:** mutate docs, *do not* rebuild ANN, assert query
   falls back and matches oracle (I5).
4. **Filter + ANN + exact re-rank:** selective and non-selective filters;
   empty eligible set.
5. **Quantized families** (RaBitQ, PQ/ADC): navigation may differ; **public
   scores** after re-rank must match unquantized DocumentMap on the returned
   IDs.

### G4 — Multi-query / group-by / fusion

1. RRF and weighted fusion: deterministic under fixed inputs; compare to
   hand-computed fusion on small corpora.
2. Group-by with projection / include_vector / filters: no field leakage
   across projections (binary suite pattern generalized).
3. Ambiguous dual routes fail at build time (already partial).

### G5 — FTS depth

Strong syntax/oracle coverage exists. Add:

1. Unicode edge: combining marks, CJK n-grams, empty tokens after filters.
2. Phrase slop at boundaries; prohibited clauses that eliminate all hits.
3. Incremental FTS after many upserts vs full rebuild BM25 equality.
4. Nullable text field missing vs empty string.

### G6 — Durability / fault

Storage fault + recovery fuzz exist. Add product-facing scenarios:

1. Kill between WAL append and manifest rename (if not already covered).
2. Index cache CRC mismatch → rebuild or fail closed; query still correct.
3. DiskANN sidecar corruption → positioned fallback / error contract.
4. Resource limit rejection mid-batch: no partial publish.

### G7 — Concurrency

1. One writer publishing generations; N readers each see a coherent revision
   (extend `concurrency.rs` to HNSW + Flat + FTS).
2. `optimize` concurrent with queries: no panic; results match some published
   revision’s oracle.
3. Async feature: every public query entry ≡ sync on same snapshot.

### G8 — Schema evolution

1. Add/rename/drop column with backfill worker counts 1 and N: identical
   published docs.
2. Alter vector dimension refused; alter metric refused when index live.
3. Create Flat on binary vs dense: allowlist assertions.

### G9 — Cross-project / workspace retrieval

`workspace_retrieval_contract` pins adapter behavior. Keep in sync when
Code/Vec commits move; treat score narrowing to `f32` as an explicit
contract, not Flat’s public `f64` contract.

### G10 — Performance gates vs correctness

Keep smoke CSV validators. Add **correctness harness flags** to
`scale_compare` / feature matrix that assert recall and (where cheap)
oracle agreement on a tiny subset—never replace L5 differential suites.

---

## 6. Prioritized case catalog

Priority: **P0** must land before claiming the related feature is release-safe;
**P1** deepens adversarial coverage; **P2** expands matrix breadth.

### 6.1 P0 — Correctness blockers

| ID | Case | Invariant | Suggested home | Evidence |
| --- | --- | --- | --- | --- |
| F-P0-1 | Dense Flat pack Cosine ≡ DocumentMap (IDs + score bits) after `optimize` | I1,I5 | `tests/flat_packed.rs` (new) | Independent `f64` scan oracle |
| F-P0-2 | Dense Flat without pack (post-insert, pre-optimize) ≡ oracle | I1,I5 | same | Stats `ready`; scores match |
| F-P0-3 | Binary Flat never builds packed index; reopen OK | I2,I5 | `tests/binary_queries.rs` (extend) | No `encode_vector` error; stats ready |
| F-P0-4 | Rayon Flat results ≡ `RAYON_NUM_THREADS=1` | I6 | `tests/flat_packed.rs` | Same ordered score bits |
| F-P0-5 | Stale HNSW after upsert → fallback ≡ oracle | I5 | `tests/ann_contracts/lifecycle.rs` | Force missing/stale generation |
| F-P0-6 | Public Cosine score finite for zero-ish norms | I5 | `src/score_f64.rs` + collection | No NaN in hit scores |
| F-P0-7 | `ef = n` HNSW ranking ≡ Flat on small corpus | I5 | `tests/ann_contracts/graph.rs` | Same ID order |
| F-P0-8 | Resource limit mid-batch leaves prior revision intact | I4 | `tests/resource_limits.rs` | Revision + doc set unchanged |

### 6.2 P1 — Deep adversarial

| ID | Case | Invariant | Suggested home |
| --- | --- | --- | --- |
| F-P1-1 | Flat L2/IP/Cosine/MipsL2 × FP16/FP32/FP64 | I2,I5 | `flat_packed` / codecs |
| F-P1-2 | Flat delete/upsert/tombstone then rebuild | I1 | `flat_packed` |
| F-P1-3 | Filtered Flat + scalar invert | I1 | `filtered_ann` |
| F-P1-4 | SIMD lane tails + odd dimensions | I6 | `score_f64` |
| F-P1-5 | FP64 near-tie PK order | I5 | `differential_oracle` |
| F-P1-6 | RaBitQ/PQ public scores ≡ unquantized oracle on hit set | I5 | `ann_contracts/rabitq` |
| F-P1-7 | MultiQuery RRF hand oracle (tiny) | I5 | new `tests/multi_query_oracle.rs` |
| F-P1-8 | FTS incremental ≡ rebuild BM25 | I1 | `differential_fts` |
| F-P1-9 | Cache CRC fail → correct exact query | I4,I5 | `index_cache` |
| F-P1-10 | Concurrent optimize + query | I3 | `concurrency` |
| F-P1-11 | Async ≡ sync matrix for Flat + HNSW + FTS | I6 | `async_queries` |
| F-P1-12 | Group-by projection field leakage | I2 | `binary_queries` pattern |

### 6.3 P2 — Breadth / soak

| ID | Case | Notes |
| --- | --- | --- |
| F-P2-1 | Generated differential corpus ≥1k docs × random metrics/filters | Extend `generated_differential` |
| F-P2-2 | Schema add-column worker 1 vs 256 identity | `execution_contracts` |
| F-P2-3 | Unicode FTS / n-gram adversarial corpus | `ngram_fts` |
| F-P2-4 | DiskANN mmap vs positioned identical hits | `mmap_diskann` |
| F-P2-5 | Long soak: 10k upserts + periodic optimize + oracle spot checks | Optional nightly |
| F-P2-6 | Cross-version snapshot reopen (legacy v3 already partial) | `storage` |
| F-P2-7 | Workspace retrieval contract pin when Code bumps Vec | `workspace_retrieval_contract` |

---

## 7. Oracle recipes (reuse, don’t reinvent)

1. **Dense DocumentMap oracle:** for each live doc with the field, compute
   `score_f64` (or the same promotion rules as the executor), sort by
   `(score ordered by metric, primary key)`.
2. **Binary Hamming oracle:** popcount of XOR on packed bytes; L2/Hamming
   metric as schema requires.
3. **Sparse oracle:** already patterned in `vector_codecs` /
   `differential_oracle`.
4. **BM25 oracle:** scan tokenizer + collection stats; keep in
   `differential_fts`.
5. **Fusion oracle:** run per-route oracles, then apply RRF/weighted formula
   in the test.

Shared helpers should live under `tests/support/` (new) once a second suite
needs them—avoid duplicating RNG/schema builders.

---

## 8. Landing rules

1. **VEC-R2:** no engine change without a failing case from this catalog (or
   a new case that cites an invariant).
2. New Flat/ANN performance work must add or extend **F-P0-1..4** before
   claiming a win in README/BENCHMARKS.
3. Prefer one focused integration file per theme (`flat_packed.rs`) over
   growing `feature_matrix` further.
4. Hosted CI must run P0 cases in `cargo test --locked --all-features`.
5. Do not mark a plan item ✅ until the test is merged and green on tip CI.

---

## 9. Implementation sequence (recommended)

1. **Wave A (P0 Flat):** `tests/flat_packed.rs` + binary pack refusal assertion.
2. **Wave B (P0 ANN fallback / ef=n / NaN):** lifecycle + graph + score_f64.
3. **Wave C (P1 matrix):** metrics × dtypes, quantized public scores, multi-query oracle.
4. **Wave D (P1 concurrency/async/FTS rebuild):** extend existing suites.
5. **Wave E (P2):** generated scale-up, soak, cross-pin.

After each wave: update this file’s checklist status, keep ROADMAP/RELEASE
gates honest, and refuse overfitting “wins” that weaken F-P0 evidence.

---

## 10. Checklist status

| Wave | Status |
| --- | --- |
| Plan published (`TESTING.md`) | ✅ |
| Wave A — Flat pack oracles | ✅ (`tests/flat_packed.rs`: pack/no-pack/L2/IP/Rayon/overlay/filter/binary) |
| Wave B — ANN fallback / ef / scores | ✅ (`tests/ann_fallback_contracts.rs`; score_f64 scalar≡SIMD) |
| Wave C — Metric/dtype/quantized matrix | 🔄 (`filtered_rabitq`, `multi_query_normalization`, schema contracts, `query_contract_rejects`) |
| Wave D — Concurrency / async / FTS rebuild | 🔄 (`mutation_validation`, FTS expression/lexer oracles, `schema_evolution`, storage fail-closed) |
| Wave E — Soak / generated scale | ⬜ |
| Line coverage (`cargo llvm-cov`) | ✅ **94.62%** lines (`cov28`: `cargo llvm-cov --locked --all-features --summary-only`); accepted as sufficient (≥94%) |

Measured line coverage is the authoritative gate for the coverage goal. The
accepted floor is ≥94% (`cov28` TOTAL). Do not claim a higher floor unless a new
measurement proves it.

---

## 11. Explicit refusals

Do not schedule tests whose only purpose is to:

- Justify lowering `ef` or skipping exact re-rank.
- Replace `f64` public scores with `f32` for speed.
- Treat zvec parity as “pass if faster,” without oracle equality.
- Count smoke bench CSV green as proof of I1–I6.

Those contradict first principles and VEC-R2.
