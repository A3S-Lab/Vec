# Compatibility examples

These binaries are executable compatibility gates, not illustrative snippets
that are allowed to drift out of date.

`upstream/crud_operations.rs` is copied from
`zvec-ai/zvec-rust@0d40cb1aef081bae175061fef35c89269e6a80f4` with only the
crate namespace changed from `zvec_rust` to `a3s_vec`. The upstream fixture's
two upsert documents omit its required `id` field, so both the official zvec
core and `a3s-vec` correctly report per-document validation failures. The file
is intentionally not corrected here because its purpose is source-level
compatibility evidence. The top-level `crud_operations.rs` executable only
includes that source under narrowly scoped Clippy allowances for upstream
style that does not follow this repository's lint policy.

The copied upstream fixture remains covered by the Apache License 2.0
reproduced in `upstream/LICENSE`; that license notice applies to the copied
fixture, not to the rest of this repository.

The upstream revision does not ship examples for every advanced API exposed by
its README and C API. The remaining binaries therefore provide asserted,
project-owned gates for:

- `retrieval_workflows.rs`: vector, FTS, RRF, weighted hybrid reranking, and
  normalization;
- `group_by.rs`: grouped top-k search and output projection;
- `schema_iteration.rs`: isolated iteration plus add, rename, alter, drop,
  flush, and reopen.

Run all gates from the crate root:

```text
cargo run --locked --example crud_operations
cargo run --locked --example retrieval_workflows
cargo run --locked --example group_by
cargo run --locked --example schema_iteration
```
