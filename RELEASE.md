# Release Qualification

`a3s-vec` `0.1.4` is published. Tag `0.1.4`, and crates.io SHA-256
`15c4220df078de9c350aea98e0f9187890cec066e4d3ee242170a2ab762ed80f` bind to
revision `9a07e9a33726dd187080b00a44505f9bbd31bd97`. Hosted CI run
`35510190796` closes the gate when green. The prior `0.1.3` package remains
on the registry as a historical artifact (tag `0.1.3` @
`88599126a4c179d8a0df24bd52963d372ea8eb67`, SHA-256
`c5c692f409c4870048a5f081f66f9072893af2047042487835f97b1d6aa3d9f2`). macOS 12
Monterey Intel is deliberately unsupported.

## 0.1.4 release notes

This patch release keeps the public storage and query contracts and turns
persistence DoS ceilings into first-class configuration:

- New public type `StorageCeilings` with product defaults **8 GiB** snapshot /
  index-cache / WAL-replay and **512 MiB** DiskANN sidecar (same magnitudes as
  `0.1.3`).
- Process default via `ConfigBuilder::storage_ceilings`; per-collection
  override via `CollectionOptions::set_storage_ceilings`. Zero is rejected;
  omitted fields keep the product default.
- Encode, open, restore, and DiskANN attach/validate paths honor the captured
  ceilings end-to-end (raised limits no longer break sidecar restore).
- Protocol/format guards stay hardcoded elsewhere (1 MiB manifest, 64 MiB
  single WAL frame, 4 KiB lock owner).
- The engine still never autodetermines ceilings from host RAM or free disk.

Do not lower `ef`, drop exact re-ranking, or switch public scores to `f32` to
manufacture a benchmark win.

Supported platforms: Linux x86_64/aarch64, Windows x86_64, and macOS
arm64/x86_64 on current hosted images (macOS deployment target 15.0).

## 0.1.3 release notes (historical)

This patch release kept the public storage and query contracts and raised the
finite storage DoS ceilings so million-document dense corpora can flush and
persist derived indexes on workstation hosts:

- Document snapshot write/recovery ceiling: **512 MiB → 8 GiB**.
- Derived index cache payload and on-disk index-cache file ceiling: **512 MiB →
  8 GiB** (aligned with snapshots so HNSW graphs can persist beside the
  authoritative corpus).
- Committed WAL replay ceiling: **512 MiB → 8 GiB**.
- DiskANN sidecar file ceiling remains **512 MiB** (unchanged).
- Honest same-host million-scale directional evidence under the existing
  fairness harness (`RAYON_NUM_THREADS=1`, protocol knobs unchanged): see
  [README.md](README.md) and [BENCHMARKS.md](BENCHMARKS.md). Protocol defaults
  are not tuned for high million-scale recall; do not market recall@10 from
  that table as an accuracy SLO.

## 0.1.2 release notes (historical)

- Character-trigram prefilter for FTS wildcard/fuzzy matcher expansion.
- Packed dense Flat acceleration; parallel Flat Cosine under the default Rayon
  pool; bit-identical `f64` public scores and exact re-ranking retained.
- First-principles a3s-vec ↔ zvec remeasure protocol and Apple Silicon medians.

## Public API review

The release-facing contract has the following boundaries:

- `zvec-core` remains a private algorithm dependency and cannot be named
  through the public crate surface.
- Collection and process configuration use typed Rust values. Unsupported
  controls and index/query combinations fail with typed errors before
  mutation.
- Persistence DoS ceilings use `StorageCeilings`; document/query resource
  budgets remain separate as `CollectionResourceLimits`.
- Schema backfills and candidate-schema validation accept a bounded typed
  worker count through `AddColumnOption`/`AlterColumnOption`; the effective pool
  is capped by work size, host parallelism, and 256 workers, while publication
  stays atomic and deterministic across worker counts.
- Public embedding and query-executor ports require `Send + Sync`. The
  `public_api_contract` integration test also enforces `Send + Sync` for the
  owned public handles, schemas, queries, values, statistics, errors, and
  `StorageCeilings`.
- `unsafe_code = "deny"` remains active. The mmap option is an immutable
  anonymous snapshot of a fully validated sidecar, not a mutable file-backed
  mapping.
- `version()`, the numeric version accessors, and `check_version()` are checked
  against the package's `0.1.4` identity.
- The public feature matrix checks every advertised query/lifecycle route,
  all six ANN families across their supported metrics (including metric-aware
  Vamana and DiskANN/PQ), cache/sidecar reopen, and the explicit binary-query
  boundary against deterministic fixtures. Binary32/Binary64 radius and
  projection/include-doc-id combinations also have asserted matrix rows. Its
  smoke-scale feature-matrix, concurrent-reader, mixed-workload,
  scale-comparison, and lifecycle-matrix performance CSVs are required hosted
  CI artifacts; same-host p50/p95/p99 baselines are
  recorded in [`BENCHMARKS.md`](BENCHMARKS.md).
- The locked dependency graph passes `cargo audit --deny unsound`: no known
  vulnerability or unsoundness advisory is present at the candidate revision.
  The audit still reports four upstream maintenance warnings (`bincode`,
  `bitmaps`, `fxhash`, and `paste`); they are non-blocking because no patched
  release or unsound finding is currently available for those paths, and the
  exact versions remain pinned in `Cargo.lock`.
- `SearchQueryBuilder` dense and pure-FTS routes are executed against the same
  collection oracles, and `include_doc_id` is checked for deterministic,
  generation-local ordinals across projections and reopen.

## Reproducible candidate artifact

After every required hosted CI job passes on `main`, the `Versioned release
candidate` job runs `cargo package --locked`. It uploads these files in one
revision-bound Actions artifact:

- `a3s-vec-0.1.4.crate`;
- `a3s-vec-0.1.4.crate.sha256`;
- `a3s-vec-0.1.4.release.json`, which records the package version, source
  revision, workflow run, and build runner.
- `feature-matrix.csv`, `concurrent-queries.csv`, `mixed-workload.csv`,
  `scale-compare.csv`, and `lifecycle-matrix.csv`, which record the
  smoke-scale correctness/performance gates, including management-plane
  lifecycle, resource, and maintenance operations.
- One `a3s-vec-platform-performance-<platform>-<revision>` directory for each
  hosted platform, containing the same five validated smoke CSVs. These
  artifacts show whether the metrics and recall gate hold across the supported
  OS/architecture matrix.

The same package can be reproduced locally without changing external state:

```text
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo audit --deny unsound
cargo test --locked --all-features
cargo test --locked --test feature_matrix
cargo bench --locked --bench feature_matrix --features async
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench feature_matrix --features async
cargo bench --locked --bench concurrent_queries
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench concurrent_queries
cargo bench --locked --bench mixed_workload
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench mixed_workload
cargo bench --locked --bench scale_compare
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench scale_compare
cargo bench --locked --bench lifecycle_matrix
A3S_VEC_BENCH_SCALE=smoke cargo bench --locked --bench lifecycle_matrix
cargo doc --locked --no-deps --all-features
cargo package --locked --allow-dirty
```

## Post-publish checklist

Record here when `0.1.4` is tagged and published:

1. Hosted CI run ID on the release revision (binding confirmed when green).
2. The published crate SHA-256.
3. Formal git tag `0.1.4` points at that revision, and `cargo publish`
   uploaded the matching crate.

## Deliberate non-support

- macOS 12 Monterey on Intel x86-64 is unsupported. The former
  `macOS 12 Intel Runtime Qualification` workflow and host-fenced script have
  been removed. A `MACOSX_DEPLOYMENT_TARGET=12.0` build is not a supported
  configuration.
- Native async file reads and direct file-backed mmap remain refused until an
  invariant test fails (see VEC-R2 in [`ROADMAP.md`](ROADMAP.md)).
