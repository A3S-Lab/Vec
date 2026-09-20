# Release Qualification

`a3s-vec` `0.1.1` is ready for formal tag and registry publication when every
release gate below is green for the same source revision. macOS 12 Monterey
Intel is deliberately unsupported: the project no longer requires, tests, or
advertises that runtime.

## 0.1.1 release notes

This patch release carries the qualified HNSW traversal improvements from the
previous candidate: runtime-dispatched `f32` SIMD scoring for unquantized
navigation, lazy primary-key resolution on exact ties, and a bounded visited
ordinal bitset. Graph construction, encoded-vector scoring, and final public
re-ranking retain their authoritative arithmetic and fallbacks. The change
also refreshes the reproducible performance and cross-project qualification
record without changing the public storage or query contracts.

Supported platforms for this release are Linux x86_64/aarch64, Windows
x86_64, and macOS arm64/x86_64 on current hosted images (macOS deployment
target 15.0). Older macOS 12 Intel hosts are out of scope.

## Public API review

The release-facing contract has the following boundaries:

- `zvec-core` remains a private algorithm dependency and cannot be named
  through the public crate surface.
- Collection and process configuration use typed Rust values. Unsupported
  controls and index/query combinations fail with typed errors before
  mutation.
- Schema backfills and candidate-schema validation accept a bounded typed
  worker count through `AddColumnOption`/`AlterColumnOption`; the effective pool
  is capped by work size, host parallelism, and 256 workers, while publication
  stays atomic and deterministic across worker counts.
- Public embedding and query-executor ports require `Send + Sync`. The
  `public_api_contract` integration test also enforces `Send + Sync` for the
  owned public handles, schemas, queries, values, statistics, and errors.
- `unsafe_code = "deny"` remains active. The mmap option is an immutable
  anonymous snapshot of a fully validated sidecar, not a mutable file-backed
  mapping.
- `version()`, the numeric version accessors, and `check_version()` are checked
  against the package's `0.1.1` identity.
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

- `a3s-vec-0.1.1.crate`;
- `a3s-vec-0.1.1.crate.sha256`;
- `a3s-vec-0.1.1.release.json`, which records the package version, source
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
cargo package --locked --offline
cargo publish --dry-run --locked
```

The candidate artifact is not itself a crates.io publication until the formal
tag and `cargo publish` step below.

The previous `0.1.0` candidate was produced by
[CI run 33772179017](https://github.com/A3S-Lab/Vec/actions/runs/33772179017)
for revision `13585ccd3f956f6cb7d669b2ee6acc7096fca03d`; its manifest and
checksum remain historical evidence.

The current `0.1.1` candidate was produced by
[CI run 35481932811](https://github.com/A3S-Lab/Vec/actions/runs/35481932811)
for revision `a08413a48b05f5457d3b164e98d6f85988269541`. Hosted matrix jobs
(Linux/Windows/macOS Intel & arm64), the public feature performance matrix,
and the versioned release-candidate packaging job are green for that
revision. Artifact
`a3s-vec-0.1.1-a08413a48b05f5457d3b164e98d6f85988269541` binds:

- package version `0.1.1`;
- source revision `a08413a48b05f5457d3b164e98d6f85988269541`;
- crate SHA-256
  `9688ce6ab8dac12f804b0ddc00f3d1c69db52ec23430b73e825ac4712766b68f`;
- runner `Linux/X64`.

Same-host HNSW directional evidence versus zvec 0.7.0 (three-process
medians, exact re-rank retained) is recorded in
[`README.md`](README.md) and [`BENCHMARKS.md`](BENCHMARKS.md).

## Registry status

The crates.io index currently contains `a3s-vec` `0.1.0`, published on
2026-09-02. That package predates the current qualification revision and must
not be treated as the `0.1.1` candidate. `cargo publish --dry-run --locked`
should now validate the new package metadata without the already-published
version collision. After the release gates below pass, publish `0.1.1` with an
artifact, checksum, and source-revision manifest that all bind to the same
qualified revision.

## Release gates

Enterprise GA for `0.1.1` is closed when all of the following bind to one
revision:

1. Hosted CI on `main` is green for that revision (quality, MSRV, recovery
   fuzz smoke, performance matrix, platform matrix including macOS 15 Intel
   and arm64, and the versioned release-candidate package job).
2. The published crate SHA-256 matches the release-candidate artifact for
   that revision.
3. The formal git tag `0.1.1` (or `v0.1.1` if the repository adopts a `v`
   prefix) points at that revision, and `cargo publish` uploads the matching
   crate.

Root submodule / Cloud lock bumps are separate consumers and must not invent
new engine work. Code commit `ff226ebe` removed the Vec shadow migration note
and made official `zvec-rust` 0.7 FTS the workspace lexical default; a Vec
shadow re-qualification is not a publish blocker.

## Deliberate non-support

- macOS 12 Monterey on Intel x86-64 is unsupported. The former
  `macOS 12 Intel Runtime Qualification` workflow and host-fenced script have
  been removed. A `MACOSX_DEPLOYMENT_TARGET=12.0` build is not a supported
  configuration.
- Native async file reads and direct file-backed mmap remain refused until an
  invariant test fails (see VEC-R2 in [`ROADMAP.md`](ROADMAP.md)).
