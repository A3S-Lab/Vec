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
6. **The portable path is the default.** The baseline uses stable Rust,
   portable scalar code, and positioned file reads. Runtime AVX2/AVX-512, ARM
   NEON, immutable mmap snapshots, and platform asynchronous I/O are optional
   accelerators, never required for correctness.

## 2. Cross-project adapter boundary

The engine remains independent of CLI, provider, and serving policy. A3S Code
owns the optional workspace-retrieval adapter that exercises this engine as a
migration shadow:

- A single admitted embedding batch is passed to the authoritative A3S Memory
  path and mirrored once into a session-local temporary Vec collection. Vec
  never calls an embedding provider and never becomes the public result source.
- The adapter compares stable document identifiers, partition/filter selection,
  narrowed `f32` scores, and search accounting under one publication gate.
  Only the Memory result can be published; a Vec error records bounded
  diagnostics and degrades the shadow without changing retrieval behavior.
- The temporary collection and its resource counters are session-owned and are
  closed with the session. No Vec state is shared between sessions or persisted
  as serving data.

The delivered adapter is pinned by the current Code candidate commit
[`17113af`](https://github.com/A3S-Lab/Code/commit/17113af42d34cb95f2fa018a1999dc2d29623bc8) to Vec commit
[`41283f631`](https://github.com/A3S-Lab/Vec/commit/41283f6315906a2737b5a8e8612ac876a8dc9c04).
Its cross-SDK wire mapping, promotion gates, and rollback procedure live in
[Code's migration note](https://github.com/A3S-Lab/Code/blob/main/manual/WORKSPACE_RETRIEVAL_VEC_MIGRATION.md).
The engine revision documented and tested by this repository is now
[`41283f631`](https://github.com/A3S-Lab/Vec/commit/41283f6315906a2737b5a8e8612ac876a8dc9c04),
with the complete hosted gate recorded in
[CI run 33705867979](https://github.com/A3S-Lab/Vec/actions/runs/33705867979).
The root Cloud compatibility lock remains a separate, older graph until its
Code dependency and lock entry are promoted together.

## 3. Module layout

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
│   ├── schema/
│   │   └── index_contract.rs  # executable index-configuration allowlist
│   ├── doc.rs                 # typed scalar/document values and projection
│   ├── doc/
│   │   ├── vector_api.rs      # native vector conversion and typed access
│   │   └── vector_codec.rs    # vector validation and IEEE FP16 codec
│   ├── query.rs               # vector, FTS, group, and query parameters
│   ├── text.rs                # tokenizer/query/BM25 shared primitives
│   ├── text/
│   │   ├── filters.rs         # ordered lowercase/folding/stemmer pipeline
│   │   ├── query_expression.rs # structured query AST, expansion, and evaluator
│   │   └── query_expression/  # lexer, parser, token matchers, and tests
│   ├── multi_query.rs         # routes and RRF/weighted fusion
│   ├── collection.rs          # lifecycle and write transaction coordinator
│   ├── collection/
│   │   ├── checkpoint.rs      # checkpoint + derived-cache maintenance
│   │   ├── configuration.rs   # process defaults + collection overrides
│   │   ├── async_api.rs       # optional Tokio blocking-pool query entry points
│   │   ├── index_api.rs       # create/drop/rebuild/optimize publication
│   │   ├── maintenance.rs     # explicitly owned background maintenance
│   │   ├── mutation.rs        # serialized DML generation publication
│   │   ├── query_api.rs       # query/fetch/iterator collection API
│   │   ├── query_contract.rs  # schema-derived route/type/dimension checks
│   │   ├── query_engine.rs    # exact vector/filter/FTS oracle
│   │   ├── resource.rs        # typed limits and logical accounting
│   │   ├── tests.rs           # generation-sharing unit contracts
│   │   └── validation.rs      # write normalization and validation
│   ├── index/
│   │   ├── mod.rs             # revision-tagged index registry and planner
│   │   ├── build.rs           # complete immutable ANN base construction
│   │   ├── cache.rs           # validated derived-index cache + restore gate
│   │   ├── diskann.rs         # native sector-aligned Vamana/DiskANN codec
│   │   ├── diskann/
│   │   │   ├── codec.rs       # bounded little-endian format primitives
│   │   │   ├── reader.rs      # typed positioned/mmap full-vector/PQ traversal
│   │   │   └── tests.rs       # packing, PQ, corruption, large-record fixtures
│   │   ├── diskann_index.rs   # Vamana graph plus optional PQ/ADC generation
│   │   ├── fts.rs             # term-frequency postings + BM25 statistics
│   │   ├── fts/
│   │   │   ├── document_lengths.rs # persistent direct-address lengths
│   │   │   ├── expression.rs       # structured-expression candidate planner
│   │   │   ├── posting_list.rs     # contiguous ordinal posting + delta
│   │   │   └── term_dictionary.rs  # contiguous term dictionary + delta
│   │   ├── hnsw.rs            # deterministic hierarchical graph
│   │   ├── hnsw/
│   │   │   └── search.rs      # deterministic frontier/result heaps
│   │   ├── ivf.rs             # deterministic centroids and postings
│   │   ├── ordinal_map.rs     # direct-address storage by document ordinal
│   │   ├── ordinals.rs        # shared persistent ID/ordinal generation
│   │   ├── product_quantization.rs # deterministic codebooks/codes/ADC tables
│   │   ├── quantization.rs    # index-only FP16/INT8/packed-INT4 codecs
│   │   ├── rabitq.rs           # multi-bit codes and unbiased estimator
│   │   ├── rabitq/
│   │   │   └── rotation.rs     # deterministic signed Hadamard rotation
│   │   ├── rabitq_index.rs     # HNSW/IVF RaBitQ execution adapters
│   │   ├── rebuild.rs         # targeted generation rebuild coordination
│   │   ├── scalar.rs          # persistent dictionaries + Roaring postings
│   │   └── vector_query.rs     # ANN base/overlay candidate traversal
│   ├── iterator.rs            # isolated document iterators
│   ├── stats.rs               # counters, index status, and health
│   ├── embedding.rs           # caller-owned dense/sparse embedding traits
│   └── storage/
│       ├── mod.rs             # recovery and commit coordination
│       ├── manifest.rs        # generation and checksum metadata
│       ├── wal.rs             # revisioned framed WAL and replay
│       ├── snapshot.rs        # binary snapshots + legacy JSON reader
│       ├── snapshot/
│       │   └── codec.rs       # compact document/value wire representation
│       ├── lock.rs            # single-writer/multi-reader lock
│       ├── derived_file.rs     # bounded positioned + immutable mmap artifact I/O
│       ├── diskann_file.rs     # Vamana/DiskANN sector-sidecar storage
│       ├── index_cache.rs      # bounded atomic derived-cache storage
│       └── tests.rs           # storage-boundary fault simulations
└── tests/
    ├── concurrency.rs         # snapshot and writer-serialization evidence
    ├── ann_contracts.rs       # exhaustive, recall, lifecycle, quantization
    ├── async_queries.rs       # Tokio/DiskANN parity and corruption fallback
    ├── contracts.rs           # typed query/write contract coverage
    ├── differential_fts.rs    # independent scan-BM25 reference
    ├── differential_oracle.rs # independent dense/sparse reference
    ├── durability.rs          # public lifecycle/restart coverage
    ├── execution_contracts.rs # executable/unsupported option matrix
    ├── filtered_ann.rs        # filtered completeness/lifecycle/concurrency
    ├── fts_filters.rs         # filter semantics, validation, cache lifecycle
    ├── fts_indexes.rs         # BM25 differential, lifecycle, concurrency
    ├── fts_advanced_query_syntax.rs # wildcard/fuzzy/range/proximity differential
    ├── fts_query_syntax.rs    # boolean/phrase index-versus-scan differential
    ├── index_cache.rs         # hit/stale/corrupt/read-only cache lifecycle
    ├── mmap_diskann.rs        # backend parity, isolation, reopen fallback
    ├── ngram_fts.rs           # Unicode n-gram config, mutation, cache reopen
    ├── resource_limits.rs     # admission, atomicity, accounting, telemetry
    ├── scalar_indexes.rs      # bitmap semantics, lifecycle, hybrid planning
    └── vector_codecs.rs       # native codec/metric/storage contracts
```

The checked-in `index/` module owns real HNSW/IVF/RaBitQ/Vamana/DiskANN candidate generation,
scalar posting dictionaries, indexed BM25 term-frequency postings, and their
revision-aware selection contract. It also owns the native sector-aligned
Vamana/DiskANN codec and typed positioned/mmap-snapshot query reader plus its
fallback, corruption, and persistence evidence. Exact wrappers with
approximate names are not kept as
placeholders.

The public modules deliberately mirror the zvec Rust SDK names where that
improves migration (`Collection`, `Doc`, `CollectionSchema`, `IndexParams`,
`SearchQuery`, `SearchQueryBuilder`, `MultiQuery`, and `DocIterator`). The
builder validates one route at a time: either a dense vector or one pure FTS
`query_string`/`match_string` expression. Query result materialisation keeps
the authoritative document generation separate from public projection;
`include_doc_id` resolves the shared generation ordinal only for that request
and never mutates the stored document. Internal modules remain replaceable and
do not leak storage implementation details. The `zvec-core` algorithm
dependency is also private: it may implement internal filtering, tokenization,
or document conversion, but its modules and types are not re-exported as part
of the A3S contract.

The module inventory above is extended by `collection/maintenance.rs`, which
owns the explicit scheduler lifecycle; `collection/mutation.rs`, which owns
serialized DML publication; and `collection/resource.rs`, which owns typed
admission and deterministic accounting. `tests/maintenance_health.rs` and
`tests/resource_limits.rs` cover their public ownership, atomicity, readiness,
and rejection contracts.

## 4. Runtime ownership and concurrency

`Collection` is a cheap, cloneable handle around an `Arc<CollectionInner>`:

```text
Collection
  └── Arc<CollectionInner>
      ├── RwLock<CollectionState>    # schema + Arc<docs/indexes> + revision
      ├── Mutex<StorageHandle>       # manifest + file lock
      ├── Mutex<()>                  # in-process writer serialization
      └── AtomicBool                 # closed lifecycle state
```

Readers briefly acquire a state guard, clone immutable document/index `Arc`
generations, and release the guard before search and result materialisation.
The document generation is a persistent ordered tree of `Arc<Doc>` values. A
generation clone shares its root in O(1); insert, remove, and patch operations
copy-on-write only an O(log N) tree path plus changed documents. Deterministic
key order is retained, while large unchanged vectors and unrelated tree nodes
are not duplicated.

Writers are serialized per collection. For ANN mutations they share the
immutable graph/posting base, encode changed vectors into a small ordinal-keyed
delta, and use a Roaring tombstone bitmap to shadow deleted or replaced base
entries. Candidate selection merges ordinal base and delta sets under the
requested budget before authoritative exact re-ranking. An overlay compacts
into a complete base at the larger of 64 entries or one eighth of the base,
capped at 2,048 entries. Compaction is built while
the previous generation remains readable. Scalar mutations copy only the
affected paths in persistent value dictionaries and copy-on-write Roaring
postings. Stable `u64` document ordinals avoid coupling public/internal IDs to
the bitmap layout. Dense vector generations and HNSW/Vamana/DiskANN graph layers store those
ordinals in direct-address `Vec<Option<T>>` slots, avoiding a tree lookup in the
ANN inner loop while preserving deterministic ordinal iteration. One
registry-level persistent ordinal generation is shared by scalar postings,
vector maps/membership, HNSW/Vamana/DiskANN nodes and edges, IVF centroid postings, candidate
selections, and FTS postings. The registry maps primary keys to ordinals with a
persistent ordered map and resolves the dense append-only ordinal domain through
a persistent indexed vector. Generation clones structurally share both lookup
directions, while reverse lookup avoids ordered-map key comparisons. Retired
ordinals trigger one complete derived-index rebuild so those domains cannot
drift. The FTS term dictionary, each term posting, and the direct-address
document-length table own sorted or ordinal-addressed contiguous immutable
bases plus persistent ordered change maps. Term lookup checks the small delta
before binary-searching the base; posting queries merge both layers in ordinal
order. Mutations copy only affected change-map paths and compact once after the
document batch. At one eighth of the base, bounded to 64..=2,048 changes, each
structure is merged into a new base while previous generations remain
readable. A parsed tokenizer and its ordered filter pipeline belong to the same
FTS generation. The default filter is Unicode lowercase; an explicit empty
pipeline preserves case for the native standard, whitespace, and n-gram
tokenizers. NFKD-based ASCII folding and Snowball stemming are applied in
declaration order to documents and queries. The standard tokenizer enforces a
configurable Unicode-character token limit. The Unicode n-gram implementation
emits one or two adjacent gram sizes over either every valid UTF-8 character or
the configured letter, decimal-digit, whitespace, punctuation, and symbol
classes. Corpus count, total token count, and term document frequencies stay at
the same revision. Posting entries carry
frequency and document length together, removing a separate document-length
tree lookup from each BM25 contribution without enlarging the aligned
`(ordinal, posting)` slot on supported 64-bit targets. Indexed BM25 score
generations also remain ordinal-keyed and carry the immutable ordinal
generation that produced them. Query execution resolves borrowed primary keys
lazily and does not construct a second primary-key map or
candidate bitmap. Single-term queries append one ordered posting directly into
a contiguous score slice. Multi-term queries use direct-address `f64` scratch
only after at least 4,096 estimated posting visits and only when its ordinal
span is at most eight times that estimate; smaller or sparse intersections use
an ordered map. Flat analyzed-term AND queries start with the shortest posting,
intersect the remaining unique terms, and preserve repeated-term BM25
weighting; OR remains the default. Structured `query_string` expressions are
parsed once into an AND-before-OR tree with required/prohibited modifiers,
boosts, exact or ordered-proximity phrases, and analyzer-aware wildcard, fuzzy,
and range leaves. Dynamic leaves expand once against the revision-matched index
dictionary or scan-corpus vocabulary, after which both paths score the same
concrete terms. The indexed planner chooses the smallest required subtree as
its driver, verifies remaining posting membership by ordinal, and retokenizes
only phrase candidates to prove ordered proximity. Without a scalar prefilter, an
estimated phrase candidate set covering at least half the corpus—or another
structured candidate set covering at least three quarters—uses the scan oracle
instead of paying broad bitmap/refinement overhead. Complete scalar and FTS
rebuilds use mutable ordered maps and
Roaring bitmaps as construction scratch, then freeze each result into one
persistent generation; they do not pay copy-on-write costs for every source
document. The writer then appends the WAL
under the established `state → storage` lock order and publishes documents,
indexes, and revision in one assignment. `StatsRegistry` is shared from
collection state, while Flat statistics are derived from the current schema and
document generation. The file lock allows multiple read-only processes while
preserving the single-writer contract.

Public concurrency fixtures exercise these boundaries through cloned
`Collection` handles. Concurrent disjoint patches retain both fields and
advance separate revisions; an iterator keeps its captured revision across a
later write; synchronized readers racing repeated two-document upserts see
only a complete previous or next batch; ANN readers racing an incremental
index update observe only complete old or new generations; and scalar readers
racing whole-corpus posting updates always receive one complete bitmap/document
revision. An equivalent FTS fixture verifies that term postings, BM25 corpus
statistics, and documents also publish as one generation.

Background maintenance is explicit ownership rather than a constructor side
effect. `Collection::start_maintenance` claims the collection's single
standard-thread scheduler; the returned runtime must be retained and closed or
dropped. A due pass holds the existing writer gate, builds a complete derived
registry while queries keep the old immutable generation, publishes it once,
then checkpoints that exact authoritative revision. Already maintained
revisions are skipped. A condition-variable predicate makes immediate triggers
and shutdown immune to early-notification races, and shutdown joins the worker
before releasing the ownership claim. Read-only handles cannot start the
runtime.

`Collection::health` is independent of scheduler ownership. It compares the
published and durable authoritative revisions, assesses each index's state,
completeness, and source revision, and exposes checkpoint lag without treating
valid manual/interval WAL accumulation as a readiness failure. The maintenance
runtime separately exposes bounded worker progress, failure, and lifecycle
diagnostics.

No async runtime is required by the core API. With the `async` feature,
`query_async`, `multi_query_async`, and `group_by_async` use `spawn_blocking`
around the same synchronous snapshot and query boundaries, so a caller does
not block a Tokio runtime worker on disk or index work. Polling these helpers
without an active Tokio runtime returns `FailedPrecondition` instead of
panicking.

Runtime configuration follows an executable-contract rule. `ConfigBuilder`
owns the process durability and `IoBackend` defaults plus WAL operation/byte
checkpoint thresholds. `CollectionOptions` owns read-only mode, typed resource
limits, and optional durability/I/O overrides; absence means inheritance, not a
second hard-coded default. `IoBackend::Positioned` is the default.
`IoBackend::Mmap` selects the implemented validated anonymous-map snapshot; it
is not a flag for direct file-backed mapping. Threading, logging, buffer, and
segment controls remain absent until they select real bounded implementations.
Resolved process defaults and collection resource policy are captured when a
collection is created or opened, so a later `initialize` call cannot change an
active handle's policy.

Resource admission has one engine-owned source of truth. Each candidate
document/index generation is measured before WAL append or publication, and
its admitted `ResourceUsage` is cached beside that immutable generation so
statistics remain O(1) with respect to document count. The logical byte total
combines deterministic authoritative-document serialization with derived-index
payload estimates; it intentionally excludes allocator overhead, transient
build memory, mapped-file residency, and process RSS. If incremental deletion
tombstones would exceed the byte budget, the engine rebuilds a compact derived
generation before deciding admission. Query snapshots capture the same policy:
single queries check their planned exact/refinement set, while multi-query
branches accumulate one shared candidate count before any branch executes.
Write-batch and state-limit rejection happens before durable or in-memory
publication and increments only a metadata counter.

Index and query configuration follows the same rule. The exact executor owns
Flat metrics and query `metric`/`radius`; packed binary fields accept only Flat
L2, whose squared bit-space distance is Hamming count. HNSW owns `m`,
`ef_construction`, and `ef`; IVF owns `n_list`, training iterations, optional SOAR dual assignment,
`nprobe`, and candidate scaling;
Vamana owns `max_degree`, build/search list sizes, alpha, and deterministic
two-pass RobustPrune construction. DiskANN owns the same graph controls plus
`pq_chunk_num`; positive chunk counts train generation-scoped product
codebooks and select ADC traversal, while zero keeps full-vector traversal.
HNSW and IVF own optional FP16/INT8/INT4 index quantization. Their dedicated
RaBitQ families own deterministic center training, one-to-nine-bit compact
codes, HNSW `ef`, IVF `nprobe`, and bounded exact refinement; every ANN path
uses exact re-ranking. FTS owns standard, whitespace, n-gram, and optional
Jieba tokenizers, persistent
term-frequency postings, document lengths, exact `f64` BM25 corpus statistics,
the OR/AND analyzed-term default operator, ordered lowercase/ASCII-folding/
Snowball-stemmer filters, and structured boolean, field-qualified, wildcard,
boosted, fuzzy, range, and phrase-proximity expressions. Scalar
inverted indexes own equality,
optional ordered range, and optional string wildcard/prefix/suffix postings.
Public DiskANN mmap/backend-selection controls and tokenizer extras without an
execution consumer remain absent or return `NotSupported` before mutation.
Unknown deserialized keys return `InvalidArgument`. Non-zero segment sizing and
add/alter concurrency return `NotSupported`. Ready ANN, scalar, and FTS
generations appear in index telemetry and increment their respective query
counters.
Ready ANN entries also report `estimated_payload_bytes`, a deterministic
encoded-payload lower bound covering quantized vectors, ordinal slot
membership, graph edges, centroids, and postings. Allocator/container
overhead and authoritative document storage are deliberately excluded, so the
field is not RSS.

For a revision-matched scalar filter, `ef` bounds the eligible HNSW candidates
returned for exact re-ranking; internal graph navigation may visit additional
rejected nodes because they remain necessary connectivity bridges. `nprobe` is
the initial IVF probe window. The filtered IVF path intersects each ranked
centroid's ordinal posting bitmap with scalar eligibility and vector
tombstones, then may extend that window until it can fill top-k. Candidate
scaling remains the final IVF re-rank bound. These internal
expansions never make an ineligible identifier visible to the exact executor.

## 5. Data and storage model

### Logical data

`Doc` contains a primary key, scalar values, dense vectors, sparse vectors, and
an optional score. Values are typed at the API boundary and have a lossless
serde representation. `FieldValue::Json` is an adapter input only: collection
writes canonicalize compatible JSON scalars and arrays to the schema's concrete
`FieldValue` variant before validation, WAL append, and storage. JSON cannot
represent binary fields in this contract. Recovery applies the same
normalization so query code never treats untyped JSON as stored authority.
Vector payloads preserve their physical type and original dimension. Dense and
sparse FP16 use raw IEEE 754 half-precision bits; the f32 adapters apply
round-to-nearest-even and reject non-finite or out-of-range input. INT4 stores
one authoritative signed coordinate per element and enforces `-8..=7`; it is
not a packed scalar quantizer. INT8 and INT16 likewise represent native integer
coordinates. Binary32/Binary64 store packed bytes, require complete 32-/64-bit
chunks, and express schema dimensions in bits. Writes never coerce one vector
variant into another schema type.

The exact oracle decodes native numeric vectors into `f64` intermediates for
L2, inner product, cosine, and MIPS-L2. This preserves FP64 coordinates through
accumulation and result ordering; only after exact ranking is the public
document score checked and narrowed to `f32`. Packed Binary32/Binary64 queries
use L2 over bit coordinates, so their negative squared-distance score is the
negative XOR Hamming count. L2 radius is a non-negative distance and is
compared with the negative squared-distance score for numeric and binary
routes. Binary execution is exact-only: Flat L2 is live, while non-L2 metrics
and binary ANN descriptors return `NotSupported`. Scale-bearing
FP16/INT8/packed-INT4 quantizers are derived HNSW/IVF state only; they never
replace document vectors, and every selected candidate receives full-vector
exact refinement.

### On-disk generations

```text
<collection>/
├── .a3s-vec.lock
├── manifest.json                       # sole commit point
├── wal/wal-<sequence:020>.bin          # version + length + payload + CRC
├── segments/snapshot-<generation:020>.bin # bounded MessagePack authority
├── indexes/index-cache.bin              # optional derived-index cache
└── indexes/diskann-graph.bin            # optional Vamana/DiskANN sector sidecar
```

The format-4 manifest is the commit point. Each acknowledged mutation first
writes a WAL record containing a monotonic revision/operation identity, then
publishes the manifest with the committed byte boundary for the active WAL
segment. Recovery reads only that boundary. Bytes after it—including a partial
frame—are uncommitted and ignored; truncation or corruption inside the boundary
is an error.

A checkpoint serializes schema and documents into a bounded MessagePack
generation, writes it to a temporary file, optionally fsyncs it and its
directory according to the caller's durability boundary, renames it, and
finally publishes the manifest with its CRC. Only after that manifest commit
may old WAL and snapshot generations be pruned. Recovery loads exactly the
snapshot named by the manifest, requires the decoder to consume every byte,
and replays consecutive WAL revisions from `checkpoint_revision + 1` through
`revision`.

Snapshot-only scalar/vector wire enums use compact binary variant identifiers.
They are converted at the storage boundary, leaving the public adjacently
tagged JSON representation of `FieldValue` and `VectorValue` unchanged. This
also gives unit variants such as `Null` an unambiguous binary representation.

The reader retains explicit format-3 compatibility for legacy JSON snapshot
generations. A writable checkpoint publishes a new format-4 binary generation
before switching the manifest, so interruption leaves either the complete old
format or the complete new format recoverable. Cleanup recognizes both file
extensions and prunes only generations not named by the committed manifest.

Schema changes that alter documents and backfill values are carried in the
schema WAL operation until the immediately following checkpoint publishes
their snapshot. Index lifecycle and other metadata-only revisions use the
schema-only WAL operation and retain the already-authoritative document map
during replay. This distinction keeps one replay transaction authoritative
without placing a large unchanged document set into a single WAL frame; the
individual frame remains bounded at 64 MiB. Any future schema-delta encoding
must preserve the same atomic recovery tests.

`indexes/index-cache.bin` and `indexes/diskann-graph.bin` are never referenced
by the manifest and are never commit points. Cache format 10 stores the shared
ordinal table plus HNSW/IVF/RaBitQ/Vamana/DiskANN, RaBitQ rotations, centers,
compact codes, PQ codebooks and codes, scalar, FTS, parsed tokenizer, and
ordered token-filter generations; the structurally different
format-2/3/4/5/6/7/8/9
payloads are ignored and rebuilt rather than reinterpreted. The
cache has its own magic, format version, payload length, and CRC, with a 512
MiB payload bound. Reuse
additionally requires an exact schema digest, source revision, and storage
identity derived from the manifest's collection, snapshot, checkpoint,
committed WAL boundary, and document checksum. Decoded ordinals, vectors,
overlays, graph layers/edges, IVF postings, scalar bitmap partitions, FTS term
dictionaries, document-length slots, and postings receive structural and live-
membership validation. Every active cached vector and scalar value must also
equal the deterministic encoding of its authoritative recovered document
before publication.

A cache containing Vamana or DiskANN additionally requires the A3S-native
sector sidecar.
The file starts with a versioned header and variable field metadata padded to a
4 KiB boundary. Each fixed-length node record contains its `u64` ordinal,
degree, either decoded full-vector coordinates or one PQ code byte per chunk,
and maximum-degree neighbor slots. PQ field metadata stores the complete
codebook; validation compares it and every code with the document-validated
cache generation.
Records no larger than a sector are packed without crossing sector boundaries;
larger records start at a sector boundary and occupy a whole-sector stride.
The header binds the source revision while metadata binds the schema digest and
manifest-derived storage identity. One CRC covers metadata, canonical zero
padding, and every field data sector. Recovery also compares every vector/code and
edge with the already document-validated cache generation. This is a native
A3S format, not the Microsoft DiskANN C++ format.

A missing, unreadable, oversized, stale, corrupt, or invalid cache/required
sidecar is a cache miss, not a collection-open failure. The registry is rebuilt
from recovered documents; a writable handle then refreshes the sidecar first
and the cache marker last, each best-effort through a temporary file and atomic
rename, while a read-only handle never writes either artifact.
Flush/close, interval checkpoints, schema or index lifecycle changes, and
explicit rebuilds refresh the cache only after the authoritative storage state
is safe to identify. Cache persistence errors cannot turn an already committed
document transaction into an apparent failure. Public telemetry reports the
per-handle open result through `index_cache_hit`.

After full sidecar validation, each restored Vamana or DiskANN field receives a
shared typed random-access reader. The default positioned backend uses
`FileExt::read_at` on Unix, `seek_read` on Windows, and a cloned-handle seek
fallback elsewhere. The mmap backend safely allocates an anonymous map, copies
the validated bytes, makes the map read-only, and releases its dependency on
the source file. It therefore preserves query-time immutability even if that
file is later replaced or truncated. The tradeoff is a full open-time copy,
temporary peak memory for both buffers, and one sidecar-sized mapping retained
by the handle; direct file-backed mapping is deliberately not implied.

With either backend, packed records stage one 4 KiB sector and oversized
records stage their whole-sector stride. A query caches loaded extents and
decoded nodes only for that request, so repeated graph edges do not repeat
backend reads. PQ queries build one asymmetric-distance table from the query
and stored codebook, then score every loaded node by summing its chunk entries.
Incremental delta/tombstone generations retain the base reader through `Arc`;
a rebuilt base deliberately has no reader until its newly persisted sidecar is
validated on reopen. A query-time short read or invalid record falls back to
the equivalent in-memory full-vector or ADC graph.
`diskann_query_count` reports every successful sidecar traversal,
`diskann_mmap_query_count` reports its mmap subset, and
`diskann_sector_read_count` reports request-local sectors staged by either
backend. `io_backend` exposes the handle's resolved selection even when a cache
miss leaves the rebuilt generation in memory. The optional Tokio query methods
move this entire selected-backend traversal and authoritative refinement path
to the runtime's blocking pool; they do not fork planner, scoring, fallback, or
telemetry semantics. Native async file reads and direct file-backed mmap remain
separate optimizations.

`rebuild_index(field)` clones the current registry, rebuilds only the named
HNSW, IVF, HNSW/IVF RaBitQ, Vamana, DiskANN, scalar, or FTS generation against the shared ordinal table,
and
publishes once. Unrelated immutable generations remain shared. `optimize()` is
the explicit whole-registry rebuild; neither path mutates authoritative
documents or their revision. A scalar/FTS-only rebuild preserves exact logical
content and therefore does not rewrite the equivalent cache generation;
HNSW/IVF/RaBitQ/Vamana/DiskANN rebuilds and `optimize()` refresh it after publication.

Manifest reads are capped at 1 MiB, individual WAL payloads at 64 MiB, total
committed WAL replay at 512 MiB, and snapshots at 512 MiB before allocation and
deserialization. WAL frame version 4 adds the schema-only operation while
retaining readability for version-3 frames. The format is versioned and
checksummed. Compatibility with
the Alibaba C++ binary files is provided through an explicit importer/exporter
milestone; the native format is not silently interpreted as a different
schema. Prototype formats 1 and 2 are rejected. Format 3 introduced raw
half-precision bits for sparse FP16 and remains readable through its JSON
snapshot codec. Format 4 keeps those document semantics and changes only the
snapshot container to MessagePack. Failure-closed versioning prevents old
readers from treating incompatible bytes as another schema or vector encoding.

## 6. Index contracts

Future persistent index implementations extend the current in-memory contract:

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
| HNSW | Hierarchical graph + eligible-result heap + bounded delta/tombstones | Flat re-rank/filter fallback |
| IVF | Lloyd centroids + optional SOAR primary/secondary ordinal postings + adaptive filtered probes | Flat re-rank/filter fallback |
| Metric-aware Vamana | Two-pass RobustPrune graph for L2, inner product, cosine, and MIPS-L2 + bounded delta/tombstones + validated sector sidecar | Flat re-rank/filter fallback |
| Metric-aware DiskANN/PQ | Vamana graph + deterministic per-chunk codebooks/codes + metric-aware ADC + bounded overlays | In-memory ADC + flat re-rank |
| DiskANN query reader | Request-local positioned or immutable mmap-snapshot full-vector/PQ-code traversal after cache reopen | Equivalent in-memory graph + flat re-rank |
| HNSW/IVF RaBitQ | Fixed-seed signed Hadamard rotation + compact 1-to-9-bit residual codes + unbiased traversal/refinement estimator | Full-vector re-rank/filter fallback |
| Scalar filters | Persistent ordered postings + Roaring bitmaps | AST scan verification/fallback |
| FTS | Contiguous term/posting/length bases + persistent deltas; Unicode tokenizers/filters, advanced structured AST, BM25 | Exact token scan fallback with shared expansion/proximity semantics |

The current optional on-disk derived-index cache includes the source data
revision, schema digest, index parameters, shared ordinals, and a format
version. Its manifest-bound source identity prevents a cache from being reused
against a different committed snapshot or WAL boundary even when a revision
number is repeated. Any mismatch makes open ignore the cache rather than
return unverifiable results.

At the current baseline, Flat executes the exact oracle directly. HNSW, IVF,
HNSW/IVF RaBitQ, metric-aware Vamana, and metric-aware DiskANN select candidates
only when the immutable registry entry matches the captured collection revision
and metric; otherwise planning falls back to Flat. Each
revision shares a complete base with older readers and owns a bounded overlay;
base tombstones are filtered with a bitmap, delta vectors are scored without
materializing a decoded vector, and merged candidates remain ordinals while
retaining the configured ANN candidate limit. An exhaustive `ef`, `nprobe`, or
`list_size` returns every live indexed identifier across both layers, and exact
re-ranking then matches Flat ordering and scores. A freshly built Vamana base
traverses in memory; a cache-restored base uses the native sidecar's configured
positioned or immutable mmap-snapshot reader for bounded traversal and fails
closed to that same memory graph.
Vamana RobustPrune uses squared L2 for L2, angular distance for cosine, and a
standard norm-augmentation distance for inner product and MIPS-L2. The
augmentation bound is computed once from the immutable base, so graph pruning
never compares raw signed similarities. Selected ordinals are removed by
identity before occlusion filtering; this keeps duplicate or collinear vectors
canonical despite floating-point ties.
DiskANN follows the same lifecycle, but a positive `pq_chunk_num` uses a
metric-aware ADC table for in-memory and either sidecar traversal. L2 sums
asymmetric squared distances, inner-product/MIPS-L2 sums centroid similarities,
and cosine normalizes the reconstructed centroid vector against the query.
RaBitQ uses compact codes only for
candidate traversal/refinement and retains the same authoritative exact-score
handoff. Scalar generations use
the same source-revision
check. Equality,
range, `IN`, null, wildcard, prefix, and suffix leaves produce exact bitmaps;
boolean planning may retain a conservative bitmap superset and the executor
always checks the full filter AST. `NOT` complements only an exact subtree, so
a partial index can reduce work but cannot remove an eligible document.
With `use_soar=true`, IVF first assigns each base vector to its nearest Lloyd
centroid, then selects one distinct secondary centroid by minimizing the
[SOAR residual objective](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0973524e02a712af33325d0688ae6f49-Abstract-Conference.html):
squared secondary-residual length plus its squared projection onto the primary
residual. The fixed lambda is one, matching the public boolean control. Stable
centroid-index ties make construction reproducible. Posting unions deduplicate
ordinals, filtered probe expansion measures the unique union, and cache
validation requires exactly two assignments per vector whenever at least two
centroids exist. With SOAR disabled, validation retains the disjoint
single-assignment contract.

The registry pairs each scalar bitmap with its immutable ordinal generation.
Vector base/delta map keys, HNSW/Vamana/DiskANN graph nodes and edges, IVF postings,
tombstones, and candidate selections use the same domain, making eligibility
and candidate composition bitmap operations rather than primary-key scans.

HNSW layer search uses a best-first `BinaryHeap` frontier and a bounded
worst-first result heap. Heap entries carry compact ordinals and borrow the
corresponding primary key from the immutable ordinal table only for equal-score
comparison. The visited hash set is used only for membership, so randomized
hash layout cannot change traversal order. Scores use `total_cmp`; equal scores
sort by ascending primary key even when mutation order assigned the opposite
ordinal order. The same kernel builds graph connections and serves queries,
replacing repeated whole-vector sorts and front removals without changing the
deterministic candidate contract.

RaBitQ first normalizes vectors for cosine and trains deterministic Lloyd
centers over either the full generation or the configured evenly spaced
sample, raised to at least the requested center count. Four fixed-seed
sign/Hadamard rounds form a portable orthogonal random
rotation with power-of-two zero padding. Each rotated residual stores its sign
bit plus zero to eight extended magnitude bits, packed without byte alignment;
an iterative least-squares rescale maximizes residual/code alignment. Query
rotation is computed once. HNSW feeds the unbiased residual estimator directly
into the shared graph heap, while IVF probes its ordinal postings, ranks the
bounded refiner set with the same estimator, and then hands ordinals to exact
document scoring. L2 uses center-relative distance estimation; inner product
and normalized cosine use center plus residual dot estimation. Delta vectors
remain full precision until normal compaction rebuilds and retrains the base.

Vamana starts at the vector nearest the dataset centroid. It initializes a
seeded R-regular directed graph and performs two deterministic build passes,
first with alpha 1 and then with the configured alpha. Each pass uses greedy
search, RobustPrune, backward-edge insertion, and re-pruning to enforce the
degree bound. L2 uses squared distance, cosine uses angular distance, and
inner-product/MIPS-L2 use a norm-augmentation transform with an immutable-base
bound. Selected ordinals are removed by identity before occlusion filtering so
floating-point ties cannot duplicate edges.

DiskANN partitions the vector into `pq_chunk_num` contiguous, balanced chunks.
Each non-empty generation trains `min(256, vector_count)` centroids per chunk
with deterministic farthest-first seeds, stable lower-index tie breaks, and
eight Lloyd iterations. Every base ordinal stores one `u8` centroid code per
chunk. A query computes squared-L2 distances to all centroids once, then graph
navigation uses a query-local metric-aware ADC table: squared distances for L2,
centroid similarities for inner product/MIPS-L2, and normalized reconstructed
centroid scores for cosine. The approximate score is never returned publicly:
authoritative document vectors perform final `f64` ranking. A complete rebuild
re-trains codebooks; bounded delta vectors remain full precision until
compaction.

Filtered vector planning resolves the scalar generation before invoking ANN.
Small eligible sets are scored directly. A conservative bitmap is first
refined with the authoritative filter AST; this prevents false-positive
members from consuming a bounded ANN result heap. For larger exact sets, HNSW
uses all graph nodes for navigation but only eligible nodes for its result heap,
with an inverse-selectivity traversal budget. Vamana uses the same navigation-
bridge rule and a proportionally expanded list budget. IVF intersects eligible
ordinals
with ranked centroid postings and extends the requested probe window until
top-k is available. Delta vectors are filtered before scoring in every path. If
graph navigation is estimated to cost
at least an exact eligible scan, or a bounded ANN search still underfills,
planning selects the exact eligible set instead of returning a short page.

FTS generations store term frequency and document length together by document
ordinal, plus a document-length map and exact corpus token totals for lifecycle
accounting. Query terms visit only their postings; an
optional scalar prefilter resolves through the same registry ordinal generation
before scoring. The resulting score generation retains ordinals until document
lookup, avoiding per-query primary-key ownership and a redundant candidate
bitmap. Its score payload is a validated contiguous ordinal-score slice. A
single posting writes that slice directly; high-cardinality multi-term queries
use bounded direct-address accumulation, and selective queries avoid allocating
scratch proportional to the ordinal domain. Safe unfiltered or exact-filter
plans retain only the best ordinal scores before document lookup; conservative
filters keep the complete score generation for final AST evaluation. The scan
fallback and index both compute corpus statistics only over
documents that contain the queried text field, including zero-token strings,
and produce the same `f64` BM25 score. Simple token expressions keep specialized
single, conjunction, sparse, and dense score paths. Structured `query_string`
supports AND/OR/NOT, parentheses, required/prohibited modifiers, escapes,
same-field qualifiers, finite boosts, `*`/`?` wildcards, transposition-aware
fuzzy distances 1 and 2, independently inclusive lexical term ranges, and
ordered phrase slop through 1,024. Dynamic leaves expand against one analyzed
vocabulary before document evaluation. Candidate construction is exact for
boolean semantics; phrase postings form a conservative intersection and only
those documents are retokenized for ordered proximity verification. Broad
structured expressions switch to the same scan evaluator when indexed
refinement is estimated to cost more. Symbolic `&&` and `||` aliases remain
`NotSupported`; selecting both `match_string` and `query_string` is invalid and
ambiguous.

## 7. Query pipeline

```text
Request
  → acquire one schema/document snapshot
  → resolve the schema field and validate route/type/dimension/limits/options
  → parse each filter and FTS expression once
  → derive a revision-matched scalar bitmap prefilter when safe
  → refine a conservative bitmap before bounded vector candidate selection
  → select matching HNSW/IVF/RaBitQ/Vamana/DiskANN/FTS generations or exact scan fallbacks
  → exact-score a selective set, or push a larger eligible set into HNSW/IVF/RaBitQ/Vamana/DiskANN
  → traverse indexed term postings or execute scan BM25 with identical stats
  → retain exact dense/sparse/FTS results in a deterministic bounded top-k heap
  → fuse exact branch results when requested
  → apply radius, exact-score ordering, top-k, and group-by limits
  → project requested fields/vectors
  → expose the checked `f32` score after deterministic exact ordering
```

For a source-ID vector route, the captured document snapshot supplies the
authoritative dense or sparse numeric payload before scoring. A missing source
document is `NotFound`; a source that lacks the selected sparse vector is a
`FailedPrecondition`. Resolution therefore observes the same immutable
generation as candidate planning, filtering, and exact refinement.

Malformed filters, unsupported parameter combinations, dimension mismatches,
and stale generations are explicit errors or safe fallback decisions. A query
never starts a network request, child process, or implicit model download.

## 8. Public compatibility surface

The Rust API provides the zvec concepts below without requiring zvec's C API:

- lifecycle: `initialize`, `shutdown`, `version`, `ConfigBuilder`;
- schema: scalar/array/vector data types, nullable fields, index builders,
  schema evolution;
- documents: typed setters/getters, sparse vectors, nulls, field projection;
- collection DML: insert, update, upsert, delete, delete-by-filter, fetch;
- DQL: vector, sparse, FTS, hybrid, multi-query, group-by, and iterators;
- index management: create/drop/optimize, index statistics, explicit
  background maintenance, and health assessment;
- durability: create/open, flush, WAL recovery, read-only mode, and locking;
- reranking: weighted and reciprocal-rank fusion with score normalization;
- caller-owned embedding traits for applications that want text-to-vector
  execution in Rust.

Executable compatibility evidence lives under `examples/`. The pinned
upstream CRUD, vector-search, and schema-builder sources are the corresponding
`zvec_rust` fixtures with only their crate namespace replaced; top-level
executable wrappers add narrowly scoped lint allowances. The schema fixture
also executes the IVF SOAR construction contract. Project-owned asserted
binaries cover the advanced routes that the pinned upstream revision does not
ship as examples: vector/FTS hybrid fusion, group-by, iterator isolation, and
schema evolution through reopen. CI runs these binaries; compiling examples
alone is not treated as behavioral evidence. The upstream CRUD fixture intentionally
retains its incomplete replacement-upsert inputs, which both engines reject
under their shared non-null schema contract.

The crate does not promise ABI compatibility with the C API or source
compatibility with Python-only extension classes. Those are separate adapters
and are intentionally outside this project. It likewise does not expose the
dependency kernel as a low-level escape hatch; a kernel replacement must not
force downstream callers to migrate unrelated types.

## 9. Platform policy

The release matrix includes Linux x86_64/aarch64, Windows x86_64, and macOS
arm64/x86_64 with a macOS deployment target of 12.0. The default derived-file
backend uses cursor-independent positioned reads (`FileExt::read_at` on Unix
and `FileExt::seek_read` on Windows) with a stable-Rust fallback elsewhere.
The optional backend uses safe anonymous-map allocation and read-only
transition after validation, without mapping a mutable external file. Intel
Monterey uses the portable scalar/AVX2 path and POSIX file locks; Linux-only
`io_uring` is never a required dependency. CI must compile with default
features and with all optional index/FTS features enabled.

The external Intel gate has a separate manual workflow. It accepts an exact
40-character revision and runs only on a self-hosted runner labeled
`a3s-macos-12`; its script rejects non-Darwin, non-x86_64, non-macOS-12, dirty,
or revision-mismatched hosts before testing. After a locked dependency fetch,
formatting, strict Clippy, default/all-feature tests, examples, rustdoc, and
package verification run offline, it executes and validates the feature-matrix,
concurrent-reader, mixed-workload, scale-comparison, and lifecycle-matrix
smoke benchmarks. The workflow uploads a checksummed crate, the five
performance CSVs (including management-plane lifecycle, resource, and
maintenance metrics), and a machine-readable
host/revision report, so a newer hosted Intel image cannot be mistaken for
Monterey evidence.

The ordinary CI platform matrix runs those same five smoke benchmarks on
Linux x86_64/arm64, Windows x86_64, and macOS arm64/Intel, with one validated,
revision-bound CSV artifact per platform. This is cross-platform regression
evidence; the hosted macOS Intel image still does not satisfy the macOS 12
runtime gate.

The default Cargo feature set is empty and its normal/build dependency graph
does not contain Jieba, `zstd-sys`, or `cc`. The `jieba` feature is explicit
because its embedded dictionary compression currently introduces that native
build chain. A schema that requests Jieba without the feature receives
`NotSupported`; it is never executed with a substitute tokenizer. The default
feature set and its tests honor the declared Rust 1.75 MSRV, including explicit
compatible pins for broad Rayon and `rmp` dependency ranges. Jieba's current
dependency chain contains Rust 2024-edition manifests and therefore requires a
newer Cargo; that feature-specific compiler floor remains explicit until the
dependency is replaced or its minimum version is raised for the whole crate.
The locked graph is checked with `cargo audit --deny unsound`; the current
audit has no vulnerability or unsoundness findings. Four transitive crates are
reported as unmaintained (`bincode`, `bitmaps`, `fxhash`, and `paste`) and are
kept at explicit lockfile versions until compatible maintained replacements are
available.

## 10. Non-negotiable quality gates

- no production `unwrap`, `expect`, or unchecked user-controlled allocation;
- `Send + Sync` for public handles where the contract permits it;
- crash/replay tests for every WAL operation and checkpoint boundary;
- deterministic result ordering and schema validation tests;
- recall and latency benchmarks against the flat reference implementation;
- fuzz coverage for filter parsing, WAL frames, and index metadata;
- Intel macOS 12 compile, smoke, and runtime benchmark before release.
