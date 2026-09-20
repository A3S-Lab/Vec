# Release Qualification

`a3s-vec` `0.1.2` is published. Tag `0.1.2`, hosted CI run
`35503763590`, and crates.io SHA-256
`2b2c5194e05cc8d17ac4f1ba5f3b609e2b5aba403bc25ab18ebf5f0af1ec6cc0` all bind to
revision `0c7894fedb62d3ea76057486cd4e4b5fad3978b2`. The prior `0.1.1` package
remains on the registry as a historical artifact (tag `0.1.1` @
`af22076eb32386b7272c2d2acfa2fef742cb2f73`, SHA-256
`94b28f42fbc14967cab8368aabd4187bddac11f706e3f8fd11befffc2110cfa3`). macOS 12
Monterey Intel is deliberately unsupported.

## 0.1.2 release notes

This patch release keeps the public storage and query contracts and adds:

- Character-trigram prefilter for FTS wildcard/fuzzy matcher expansion
  (recall-safe prune before the pattern matcher; full vocabulary fallback when
  no long enough literal run exists).
- Packed dense Flat acceleration on the ANN path while binary Flat stays on
  `DocumentMap`; bit-identical `f64` public scores and exact re-ranking retained.
- Parallel Flat Cosine scan under the default Rayon pool (product default),
  with the one-worker fairness harness unchanged for HNSW comparisons.
- First-principles a3s-vec ↔ zvec remeasure protocol and fresh Apple Silicon
  medians (`docs/scale-compare-protocol.md`, `scripts/run_fp_compare.sh`).
- Expanded invariant-driven integration coverage (`TESTING.md`).

Do not lower `ef`, drop exact re-ranking, or switch public scores to `f32` to
manufacture a benchmark win.

Supported platforms: Linux x86_64/aarch64, Windows x86_64, and macOS
arm64/x86_64 on current hosted images (macOS deployment target 15.0).

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
  against the package's `0.1.2` identity.
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

- `a3s-vec-0.1.2.crate`;
- `a3s-vec-0.1.2.crate.sha256`;
- `a3s-vec-0.1.2.release.json`, which records the package version, source
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

## Registry status

The crates.io index contains `a3s-vec` `0.1.2`, published from tag `0.1.2`
at revision `0c7894fedb62d3ea76057486cd4e4b5fad3978b2`. The published crate
SHA-256 is
`2b2c5194e05cc8d17ac4f1ba5f3b609e2b5aba403bc25ab18ebf5f0af1ec6cc0`, matching
the hosted release-candidate artifact from
[CI run 35503763590](https://github.com/A3S-Lab/Vec/actions/runs/35503763590).
The earlier `0.1.1` and `0.1.0` packages remain historical.

## Release gates

Enterprise GA for `0.1.2` is closed: all of the following bind to revision
`0c7894fedb62d3ea76057486cd4e4b5fad3978b2`:

1. Hosted CI on `main` is green (quality, MSRV, recovery fuzz smoke,
   performance matrix, platform matrix including macOS 15 Intel and arm64, and
   the versioned release-candidate package job) —
   [run 35503763590](https://github.com/A3S-Lab/Vec/actions/runs/35503763590).
2. The published crate SHA-256 matches the release-candidate artifact
   (`2b2c5194e05cc8d17ac4f1ba5f3b609e2b5aba403bc25ab18ebf5f0af1ec6cc0`).
3. Formal git tag `0.1.2` points at that revision, and `cargo publish`
   uploaded the matching crate.

## Deliberate non-support

- macOS 12 Monterey on Intel x86-64 is unsupported. The former
  `macOS 12 Intel Runtime Qualification` workflow and host-fenced script have
  been removed. A `MACOSX_DEPLOYMENT_TARGET=12.0` build is not a supported
  configuration.
- Native async file reads and direct file-backed mmap remain refused until an
  invariant test fails (see VEC-R2 in [`ROADMAP.md`](ROADMAP.md)).
