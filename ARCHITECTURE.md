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

```text
crates/vec/
├── Cargo.toml
├── README.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── src/
│   ├── lib.rs                 # public API and prelude
│   ├── error.rs               # typed errors and zvec status mapping
│   ├── config.rs              # process and collection configuration
│   ├── types.rs               # DataType, MetricType, IndexType, quantizers
│   ├── schema.rs              # field/vector/collection schemas and builders
│   ├── doc.rs                 # typed document values and projection
│   ├── query.rs               # vector, FTS, group, and query parameters
│   ├── multi_query.rs         # routes and RRF/weighted fusion
│   ├── collection.rs          # thread-safe collection handle
│   ├── iterator.rs             # isolated document iterators
│   ├── stats.rs               # counters, index status, and health
│   ├── embedding.rs           # caller-owned dense/sparse embedding traits
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── manifest.rs        # generation and checksum metadata
│   │   ├── wal.rs             # framed WAL and replay
│   │   ├── snapshot.rs        # immutable segment snapshots
│   │   └── lock.rs            # single-writer/multi-reader lock
│   ├── index/
│   │   ├── mod.rs             # VectorIndex and ScalarIndex contracts
│   │   ├── flat.rs             # exact dense/sparse scan
│   │   ├── hnsw.rs             # in-memory HNSW
│   │   ├── ivf.rs              # IVF coarse quantizer and postings
│   │   ├── diskann.rs          # Vamana/DiskANN disk index
│   │   ├── pq.rs               # product quantization
│   │   ├── rabitq.rs           # RaBitQ encoding and refinement
│   │   ├── quantize.rs         # FP16/INT8/INT4 codecs
│   │   ├── scalar.rs           # equality/range inverted indexes
│   │   └── fts.rs              # tokenizers, postings, and BM25
│   └── planner/
│       ├── mod.rs              # query planning and validation
│       ├── filter.rs           # parsed filter AST and bitmap execution
│       ├── hybrid.rs           # dense/sparse/FTS route execution
│       ├── fusion.rs           # normalization and result fusion
│       └── projection.rs       # output-field/vector shaping
└── tests/
    ├── api_compat.rs
    ├── crud_and_query.rs
    ├── durability.rs
    ├── indexes.rs
    └── concurrency.rs
```

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
serde representation. Vector codecs preserve the original dimension and
quantizer metadata.

### On-disk generations

```text
<collection>/
├── manifest.json                 # active generation, revisions, checksums
├── schema.json                   # canonical schema
├── wal/wal-<sequence>.log       # length + payload + CRC framed records
├── segments/segment-<generation>-<n>.bin
└── indexes/<field>/<generation>.idx
```

The manifest is the commit point. A checkpoint writes new segment and index
files to temporary names, fsyncs according to the selected durability mode,
renames them, and finally publishes the manifest. Recovery validates the
manifest, loads the last complete generation, and replays WAL records after its
checkpoint sequence. A truncated final WAL frame is tolerated; corruption in
an earlier frame is reported as an error.

The format is versioned and checksummed. Compatibility with the Alibaba C++
binary files is provided through an explicit importer/exporter milestone; the
native format is not silently interpreted as a different schema.

## 5. Index contracts

All index implementations satisfy the same contract:

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

The initial and fallback implementations are:

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

## 6. Query pipeline

```text
Request
  → validate schema/dimension/limits
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

## 9. Non-negotiable quality gates

- no production `unwrap`, `expect`, or unchecked user-controlled allocation;
- `Send + Sync` for public handles where the contract permits it;
- crash/replay tests for every WAL operation and checkpoint boundary;
- deterministic result ordering and schema validation tests;
- recall and latency benchmarks against the flat reference implementation;
- fuzz coverage for filter parsing, WAL frames, and index metadata;
- Intel macOS 12 compile, smoke, and runtime benchmark before release.
