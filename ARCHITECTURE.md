# A3S Vec Architecture

This file defines the engine-internal contract. Cross-project ownership,
`a3s-code`/`vgrep` integration, model-provider policy, and migration from the
current Memory/BM25 path are defined in the
[A3S local retrieval platform architecture](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md).
The corresponding cross-project delivery gates are in the
[A3S local retrieval platform roadmap](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-roadmap.md).

This document defines the architecture for `a3s-vec`, the native Rust vector
database embedded in A3S. The target is behavioural parity with the zvec Rust
surface (collection lifecycle, typed documents, vector and full-text queries,
indexes, persistence, and maintenance), without a C/C++ runtime or a language
binding layer.

## 1. First principles

The engine is built around six invariants:

1. **Documents are the authority.** A document snapshot plus its write-ahead
   log (WAL) is the only source of truth. Every ANN, scalar, and text index is
   derived state and can be rebuilt from documents.
2. **Schema owns meaning.** Field type, nullability, vector dimension, metric,
   and index parameters are validated at the write boundary. Query and index
   code never infer a different type or dimension.
3. **A query observes one revision.** A query runs against an immutable read
   snapshot. Background indexing, compaction, and writes may publish a newer
   revision, but cannot change the result set halfway through a query.
4. **Durability is explicit.** A mutation is acknowledged only after its WAL
   durability policy is satisfied. Snapshots and manifests are published with
   an atomic rename and checksum validation.
5. **Approximation never breaks correctness.** Every approximate index has an
   exact flat-scan fallback and an exact re-ranking stage. Invalid or stale
   index state fails closed to the fallback path.
6. **The portable path is the default.** The baseline uses stable Rust and
   portable scalar code. Runtime AVX2/AVX-512, ARM NEON, mmap, and platform
   asynchronous I/O are optional accelerators, never required for correctness.

## 2. Module layout

The checked-in implementation is intentionally smaller than the target index
layout:

```text
crates/vec/
├── src/
│   ├── lib.rs                 # public API and prelude
│   ├── error.rs               # typed errors and zvec status mapping
│   ├── config.rs              # process and collection configuration
│   ├── types.rs               # public type vocabulary
│   ├── schema.rs              # field/vector/collection schemas and builders
│   ├── doc.rs                 # typed document values and projection
│   ├── query.rs               # vector, FTS, group, and query parameters
│   ├── multi_query.rs         # routes and RRF/weighted fusion
│   ├── collection.rs          # lifecycle and write transaction coordinator
│   ├── collection/
│   │   ├── query_api.rs       # query/fetch/iterator collection API
│   │   ├── query_contract.rs  # schema-derived route/type/dimension checks
│   │   ├── query_engine.rs    # exact vector/filter/FTS oracle
│   │   └── validation.rs      # write validation and index metadata
│   ├── iterator.rs            # isolated document iterators
│   ├── stats.rs               # counters, index status, and health
│   ├── embedding.rs           # caller-owned dense/sparse embedding traits
│   └── storage/
│       ├── mod.rs             # recovery and commit coordination
│       ├── manifest.rs        # generation and checksum metadata
│       ├── wal.rs             # revisioned framed WAL and replay
│       ├── snapshot.rs        # immutable generation snapshots
│       ├── lock.rs            # single-writer/multi-reader lock
│       └── tests.rs           # storage-boundary fault simulations
└── tests/
    ├── contracts.rs           # typed query/write contract coverage
    └── durability.rs          # public lifecycle/restart coverage
```

Real `index/` and `planner/` modules are added only when their phase gates have
recall, fallback, corruption, and persistence evidence. Private exact wrappers
named HNSW/IVF/DiskANN are not kept as placeholders.

The public modules deliberately mirror the zvec Rust SDK names where that
improves migration (`Collection`, `Doc`, `CollectionSchema`, `IndexParams`,
`SearchQuery`, `MultiQuery`, and `DocIterator`). Internal modules remain
replaceable and do not leak storage implementation details.

## 3. Runtime ownership and concurrency

`Collection` is a cheap, cloneable handle around an `Arc<CollectionInner>`:

```text
Collection
  └── Arc<CollectionInner>
      ├── RwLock<LogicalState>       # schema + document snapshot
      ├── RwLock<IndexCatalog>       # generation-tagged derived indexes
      ├── AtomicU64 revision
      ├── WriterLock                  # process and file lock
      ├── CollectionOptions
      └── StatsRegistry
```

Readers acquire a snapshot guard and release it before expensive result
materialisation. Writers are serialized per collection and publish a new
logical revision atomically. Index builds run on a snapshot and are installed
only when their input revision still matches; otherwise the build is discarded
and retried. This gives multiple processes read access while preserving the
zvec single-process writer contract.

No async runtime is required by the core API. Optional async helpers use
`spawn_blocking` around the same synchronous transaction boundaries, so a
caller never blocks an async executor thread on disk or index work.

## 4. Data and storage model

### Logical data

`Doc` contains a primary key, scalar values, dense vectors, sparse vectors, and
an optional score. Values are typed at the API boundary and have a lossless
serde representation. `FieldValue::Json` is an adapter input only: collection
writes canonicalize compatible JSON scalars and arrays to the schema's concrete
`FieldValue` variant before validation, WAL append, and storage. JSON cannot
represent binary fields in this contract. Recovery applies the same
normalization so query code never treats untyped JSON as stored authority.
Vector codecs preserve the original dimension and quantizer metadata.

### On-disk generations

```text
<collection>/
├── .a3s-vec.lock
├── manifest.json                       # sole commit point
├── wal/wal-<sequence:020>.bin          # version + length + payload + CRC
└── segments/snapshot-<generation:020>.json
```

The format-2 manifest is the commit point. Each acknowledged mutation first
writes a WAL record containing a monotonic revision/operation identity, then
publishes the manifest with the committed byte boundary for the active WAL
segment. Recovery reads only that boundary. Bytes after it—including a partial
frame—are uncommitted and ignored; truncation or corruption inside the boundary
is an error.

A checkpoint writes a new immutable snapshot generation to a temporary file,
optionally fsyncs it and its directory according to the caller's durability
boundary, renames it, and finally publishes the manifest. Only after that
manifest commit may old WAL and snapshot generations be pruned. Recovery loads
exactly the snapshot named by the manifest and replays consecutive WAL
revisions from `checkpoint_revision + 1` through `revision`.

Schema and backfilled documents are carried in the schema WAL operation until
the immediately following checkpoint publishes their snapshot. This keeps a
single replay transaction authoritative while the schema-change encoding is
still a prototype; a more compact schema delta format may replace it only with
equivalent recovery tests.

Manifest reads are capped at 1 MiB, individual WAL payloads at 64 MiB, total
committed WAL replay at 512 MiB, and snapshots at 512 MiB before allocation and
deserialization. The format is versioned and checksummed. Compatibility with the Alibaba C++
binary files is provided through an explicit importer/exporter milestone; the
native format is not silently interpreted as a different schema.

## 5. Index contracts

Future index implementations satisfy the following contract:

```rust,ignore
trait VectorIndex: Send + Sync {
    fn kind(&self) -> IndexType;
    fn dimension(&self) -> usize;
    fn build(&mut self, input: &IndexInput<'_>) -> Result<()>;
    fn search(&self, query: &VectorQuery, filter: Option<&DocSet>)
        -> Result<Vec<Candidate>>;
    fn insert(&mut self, doc: &IndexedDoc) -> Result<()>;
    fn remove(&mut self, id: &str) -> Result<()>;
    fn save(&self, writer: &mut IndexWriter) -> Result<()>;
}
```

The target and fallback implementations are:

| Capability | Implementation | Correctness path |
| --- | --- | --- |
| Dense/sparse exact search | Flat scan | Always available |
| HNSW | Hierarchical graph with bounded `ef` | Flat re-rank |
| IVF | Lloyd centroids and postings | Flat re-rank |
| Vamana/DiskANN | Sector-aligned graph, optional mmap/pread | Delta + flat re-rank |
| PQ | Per-subspace codebooks and ADC | Full-vector re-rank |
| RaBitQ | Binary/scalar codes and refinement | Full-vector re-rank |
| Scalar filters | Hash/B-tree postings and bitsets | AST scan fallback |
| FTS | Standard, whitespace, n-gram, optional jieba tokenizers + BM25 | Token scan fallback |

Index metadata always includes the source data revision, schema digest,
dimension, metric, and a format version. A mismatch makes the planner ignore
the index rather than return unverifiable results.

At the current baseline, collection queries always execute the exact oracle.
Schema metadata may describe a requested future index, but it does not change
execution to an approximate algorithm or indexed FTS path.

## 6. Query pipeline

```text
Request
  → resolve the schema field and validate route/type/dimension/limits
  → parse filter and FTS expressions
  → acquire one data/index snapshot
  → build scalar candidate set (if available)
  → execute dense, sparse, and/or FTS routes
  → merge candidates and exact re-rank
  → apply radius, top-k, group-by, and fusion rules
  → project requested fields/vectors
  → deterministic sort (score, then primary key)
```

Malformed filters, unsupported parameter combinations, dimension mismatches,
and stale generations are explicit errors or safe fallback decisions. A query
never starts a network request, child process, or implicit model download.

## 7. Public compatibility surface

The Rust API provides the zvec concepts below without requiring zvec's C API:

- lifecycle: `initialize`, `shutdown`, `version`, `ConfigBuilder`;
- schema: scalar/array/vector data types, nullable fields, index builders,
  schema evolution;
- documents: typed setters/getters, sparse vectors, nulls, field projection;
- collection DML: insert, update, upsert, delete, delete-by-filter, fetch;
- DQL: vector, sparse, FTS, hybrid, multi-query, group-by, and iterators;
- index management: create/drop/optimize and index statistics;
- durability: create/open, flush, WAL recovery, read-only mode, and locking;
- reranking: weighted and reciprocal-rank fusion with score normalization;
- caller-owned embedding traits for applications that want text-to-vector
  execution in Rust.

The crate does not promise ABI compatibility with the C API or source
compatibility with Python-only extension classes. Those are separate adapters
and are intentionally outside this project.

## 8. Platform policy

The release matrix includes Linux x86_64/aarch64, Windows x86_64, and macOS
arm64/x86_64 with a macOS deployment target of 12.0. Intel Monterey uses the
portable scalar/AVX2 path, POSIX file locks, and `pread`/ordinary file reads;
Linux-only `io_uring` is never a required dependency. CI must compile with
default features and with all optional index/FTS features enabled.

The default Cargo feature set is empty and its normal/build dependency graph
does not contain Jieba, `zstd-sys`, or `cc`. The `jieba` feature is explicit
because its embedded dictionary compression currently introduces that native
build chain. A schema that requests Jieba without the feature receives
`NotSupported`; it is never executed with a substitute tokenizer.

## 9. Non-negotiable quality gates

- no production `unwrap`, `expect`, or unchecked user-controlled allocation;
- `Send + Sync` for public handles where the contract permits it;
- crash/replay tests for every WAL operation and checkpoint boundary;
- deterministic result ordering and schema validation tests;
- recall and latency benchmarks against the flat reference implementation;
- fuzz coverage for filter parsing, WAL frames, and index metadata;
- Intel macOS 12 compile, smoke, and runtime benchmark before release.
