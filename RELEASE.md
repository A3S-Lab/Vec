# Release Qualification

`a3s-vec` is currently qualified as a `0.1.0` release candidate. A formal tag
or registry publication must not be created until every release gate below is
green for the same source revision.

## Public API review

The release-facing contract has the following boundaries:

- `zvec-core` remains a private algorithm dependency and cannot be named
  through the public crate surface.
- Collection and process configuration use typed Rust values. Unsupported
  controls and index/query combinations fail with typed errors before
  mutation.
- Public embedding and query-executor ports require `Send + Sync`. The
  `public_api_contract` integration test also enforces `Send + Sync` for the
  owned public handles, schemas, queries, values, statistics, and errors.
- `unsafe_code = "deny"` remains active. The mmap option is an immutable
  anonymous snapshot of a fully validated sidecar, not a mutable file-backed
  mapping.
- `version()`, the numeric version accessors, and `check_version()` are checked
  against the package's `0.1.0` identity.
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

- `a3s-vec-0.1.0.crate`;
- `a3s-vec-0.1.0.crate.sha256`;
- `a3s-vec-0.1.0.release.json`, which records the package version, source
  revision, workflow run, and build runner.
- `feature-matrix.csv`, `concurrent-queries.csv`, `mixed-workload.csv`,
  `scale-compare.csv`, and `lifecycle-matrix.csv`, which record the
  smoke-scale correctness/performance gates, including management-plane
  lifecycle, resource, and maintenance operations.
- One `a3s-vec-platform-performance-<platform>-<revision>` directory for each
  hosted platform, containing the same five validated smoke CSVs. These
  artifacts show whether the metrics and recall gate hold across the supported
  OS/architecture matrix; the hosted Intel image is not macOS 12.

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

The candidate artifact is not a crates.io publication and is not evidence of
an actual macOS 12 Intel runtime.

## Formal release blockers

- Register an actual Intel Mac running macOS 12 as a repository runner with the
  `a3s-macos-12` label, then manually dispatch
  `macOS 12 Intel Runtime Qualification` with the exact candidate commit. The
  workflow's host-fenced script runs the locked exact/FTS, recovery, async,
  DiskANN, example, rustdoc, package, and all five smoke-scale performance
  fixtures (with the same CSV validators as hosted CI) offline. A deployment
  target or newer hosted Intel image is insufficient.
- Attach that machine-readable result to the release record and verify the
  source revision matches the candidate artifact.
- Re-run the Code migration and root compatibility-lock checks against that
  same Vec revision before creating the formal tag and registry artifact.

Until those gates pass, A3S Code keeps A3S Memory as the serving authority and
uses Vec only as a failure-isolated differential shadow.
