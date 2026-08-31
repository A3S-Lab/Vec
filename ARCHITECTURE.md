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
│   │   └── query_expression.rs # structured boolean/phrase parser + evaluator
│   ├── multi_query.rs         # routes and RRF/weighted fusion
│   ├── collection.rs          # lifecycle and write transaction coordinator
│   ├── collection/
│   │   ├── checkpoint.rs      # checkpoint + derived-cache maintenance
│   │   ├── configuration.rs   # process defaults + collection overrides
│   │   ├── index_api.rs       # create/drop/rebuild/optimize publication
│   │   ├── query_api.rs       # query/fetch/iterator collection API
│   │   ├── query_contract.rs  # schema-derived route/type/dimension checks
│   │   ├── query_engine.rs    # exact vector/filter/FTS oracle
│   │   ├── tests.rs           # generation-sharing unit contracts
│   │   └── validation.rs      # write normalization and validation
│   ├── index/
│   │   ├── mod.rs             # revision-tagged index registry and planner
│   │   ├── cache.rs           # validated derived-index cache + restore gate
│   │   ├── fts.rs             # term-frequency postings + BM25 statistics
│   │   ├── fts/
│   │   │   ├── document_lengths.rs # persistent direct-address lengths
│   │   │   ├── posting_list.rs     # contiguous ordinal posting + delta
│   │   │   └── term_dictionary.rs  # contiguous term dictionary + delta
│   │   ├── hnsw.rs            # deterministic hierarchical graph
│   │   ├── hnsw/
│   │   │   └── search.rs      # deterministic frontier/result heaps
│   │   ├── ivf.rs             # deterministic centroids and postings
│   │   ├── ordinal_map.rs     # direct-address storage by document ordinal
│   │   ├── ordinals.rs        # shared persistent ID/ordinal generation
│   │   ├── quantization.rs    # index-only FP16/INT8/packed-INT4 codecs
│   │   ├── rebuild.rs         # targeted generation rebuild coordination
│   │   └── scalar.rs          # persistent dictionaries + Roaring postings
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
│       ├── index_cache.rs      # bounded atomic derived-cache storage
│       └── tests.rs           # storage-boundary fault simulations
└── tests/
    ├── concurrency.rs         # snapshot and writer-serialization evidence
    ├── ann_contracts.rs       # exhaustive, recall, lifecycle, quantization
    ├── contracts.rs           # typed query/write contract coverage
    ├── differential_fts.rs    # independent scan-BM25 reference
    ├── differential_oracle.rs # independent dense/sparse reference
    ├── durability.rs          # public lifecycle/restart coverage
    ├── execution_contracts.rs # executable/unsupported option matrix
    ├── filtered_ann.rs        # filtered completeness/lifecycle/concurrency
    ├── fts_filters.rs         # filter semantics, validation, cache lifecycle
    ├── fts_indexes.rs         # BM25 differential, lifecycle, concurrency
    ├── fts_query_syntax.rs    # boolean/phrase index-versus-scan differential
    ├── index_cache.rs         # hit/stale/corrupt/read-only cache lifecycle
    ├── ngram_fts.rs           # Unicode n-gram config, mutation, cache reopen
    ├── scalar_indexes.rs      # bitmap semantics, lifecycle, hybrid planning
    └── vector_codecs.rs       # native codec/metric/storage contracts
```

The checked-in `index/` module owns real HNSW/IVF/Vamana candidate generation,
scalar posting dictionaries, indexed BM25 term-frequency postings, and their
revision-aware selection contract. DiskANN disk layouts are added only when
their phase gate has fallback, corruption, and persistence evidence; exact
wrappers with approximate names are not kept as placeholders.

The public modules deliberately mirror the zvec Rust SDK names where that
improves migration (`Collection`, `Doc`, `CollectionSchema`, `IndexParams`,
`SearchQuery`, `MultiQuery`, and `DocIterator`). Internal modules remain
replaceable and do not leak storage implementation details. The `zvec-core`
algorithm dependency is also private: it may implement internal filtering,
tokenization, or document conversion, but its modules and types are not
re-exported as part of the A3S contract.

## 3. Runtime ownership and concurrency

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
the bitmap layout. Dense vector generations and HNSW/Vamana graph layers store those
ordinals in direct-address `Vec<Option<T>>` slots, avoiding a tree lookup in the
ANN inner loop while preserving deterministic ordinal iteration. One
registry-level persistent ordinal generation is shared by scalar postings,
vector maps/membership, HNSW/Vamana nodes and edges, IVF centroid postings, candidate
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
parsed once into an AND-before-OR tree with required/prohibited modifiers and
exact phrase leaves. The indexed planner chooses the smallest required subtree
as its driver, verifies remaining posting membership by ordinal, and retokenizes
only phrase candidates to prove adjacency. Without a scalar prefilter, an
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

No async runtime is required by the core API. Optional async helpers use
`spawn_blocking` around the same synchronous transaction boundaries, so a
caller never blocks an async executor thread on disk or index work.

Runtime configuration follows an executable-contract rule. `ConfigBuilder`
owns only the process durability default and WAL operation/byte checkpoint
thresholds. `CollectionOptions` owns read-only mode plus an optional durability
override; absence means inheritance, not a second hard-coded default. Memory,
threading, logging, I/O backend, mmap, buffer, and segment controls are not
public until they select a real bounded implementation. The resolved process
defaults are captured when a collection is created or opened, so a later
`initialize` call cannot change an active collection's acknowledgement policy.

Index and query configuration follows the same rule. The exact executor owns
Flat metrics and query `metric`/`radius`; HNSW owns `m`, `ef_construction`, and
`ef`; IVF owns `n_list`, training iterations, `nprobe`, and candidate scaling;
Vamana owns `max_degree`, build/search list sizes, alpha, and deterministic
two-pass RobustPrune construction. HNSW and IVF own optional FP16/INT8/INT4
index quantization; all three ANN paths use exact re-ranking. FTS
owns standard, whitespace, n-gram, and optional Jieba tokenizers, persistent
term-frequency postings, document lengths, exact `f64` BM25 corpus statistics,
the OR/AND analyzed-term default operator, ordered lowercase/ASCII-folding/
Snowball-stemmer filters, and structured boolean/phrase expressions. Scalar
inverted indexes own equality,
optional ordered range, and optional string wildcard/prefix/suffix postings.
Sector-aligned DiskANN files, RaBitQ/PQ, wildcard/fielded/boosted/fuzzy/range
FTS syntax, and
tokenizer extras without an execution consumer return `NotSupported` before
mutation. Unknown
deserialized keys return `InvalidArgument`. Non-zero segment sizing and
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

## 4. Data and storage model

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
document score checked and narrowed to `f32`. L2 radius is a non-negative
distance and is compared with the negative squared-distance score. Binary
search has no exact consumer yet and returns `NotSupported`. Scale-bearing
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
└── indexes/index-cache.bin              # optional derived-index cache
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

Schema and backfilled documents are carried in the schema WAL operation until
the immediately following checkpoint publishes their snapshot. This keeps a
single replay transaction authoritative while the schema-change encoding is
still a prototype; a more compact schema delta format may replace it only with
equivalent recovery tests.

`indexes/index-cache.bin` is never referenced by the manifest and is never a
commit point. Cache format 7 stores the shared ordinal table plus HNSW/IVF/
Vamana, scalar, FTS, parsed tokenizer, and ordered token-filter generations; the
structurally different format-2/3/4/5/6 payloads are ignored and rebuilt rather
than reinterpreted. The
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

A missing, unreadable, oversized, stale, corrupt, or invalid cache is a cache
miss, not a collection-open failure. The registry is rebuilt from recovered
documents; a writable handle then refreshes the cache best-effort through a
temporary file and atomic rename, while a read-only handle never writes it.
Flush/close, interval checkpoints, schema or index lifecycle changes, and
explicit rebuilds refresh the cache only after the authoritative storage state
is safe to identify. Cache persistence errors cannot turn an already committed
document transaction into an apparent failure. Public telemetry reports the
per-handle open result through `index_cache_hit`.

`rebuild_index(field)` clones the current registry, rebuilds only the named
HNSW, IVF, Vamana, scalar, or FTS generation against the shared ordinal table,
and
publishes once. Unrelated immutable generations remain shared. `optimize()` is
the explicit whole-registry rebuild; neither path mutates authoritative
documents or their revision. A scalar/FTS-only rebuild preserves exact logical
content and therefore does not rewrite the equivalent cache generation;
HNSW/IVF/Vamana rebuilds and `optimize()` refresh it after publication.

Manifest reads are capped at 1 MiB, individual WAL payloads at 64 MiB, total
committed WAL replay at 512 MiB, and snapshots at 512 MiB before allocation and
deserialization. The format is versioned and checksummed. Compatibility with
the Alibaba C++ binary files is provided through an explicit importer/exporter
milestone; the native format is not silently interpreted as a different
schema. Prototype formats 1 and 2 are rejected. Format 3 introduced raw
half-precision bits for sparse FP16 and remains readable through its JSON
snapshot codec. Format 4 keeps those document semantics and changes only the
snapshot container to MessagePack. Failure-closed versioning prevents old
readers from treating incompatible bytes as another schema or vector encoding.

## 5. Index contracts

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
| IVF | Lloyd centroids + ordinal postings + adaptive filtered probes | Flat re-rank/filter fallback |
| In-memory L2 Vamana | Two-pass RobustPrune graph + bounded delta/tombstones | Flat re-rank/filter fallback |
| DiskANN disk layout | Sector-aligned graph, optional mmap/pread | Future delta + flat re-rank |
| PQ | Per-subspace codebooks and ADC | Full-vector re-rank |
| RaBitQ | Binary/scalar codes and refinement | Full-vector re-rank |
| Scalar filters | Persistent ordered postings + Roaring bitmaps | AST scan verification/fallback |
| FTS | Contiguous term/posting/length bases + persistent deltas; Unicode tokenizers/filters, structured boolean/phrase AST, BM25 | Exact token scan fallback |

The current optional on-disk derived-index cache includes the source data
revision, schema digest, index parameters, shared ordinals, and a format
version. Its manifest-bound source identity prevents a cache from being reused
against a different committed snapshot or WAL boundary even when a revision
number is repeated. Any mismatch makes open ignore the cache rather than
return unverifiable results.

At the current baseline, Flat executes the exact oracle directly. HNSW, IVF,
and in-memory L2 Vamana select candidates only when the immutable registry
entry matches the captured collection revision and metric; otherwise planning
falls back to Flat. Each
revision shares a complete base with older readers and owns a bounded overlay;
base tombstones are filtered with a bitmap, delta vectors are scored without
materializing a decoded vector, and merged candidates remain ordinals while
retaining the configured ANN candidate limit. An exhaustive `ef`, `nprobe`, or
`list_size` returns every live indexed identifier across both layers, and exact
re-ranking then matches Flat ordering and scores. Descriptors for DiskANN disk
layouts and compressed index families remain serializable for adapters but
cannot attach to a schema. Scalar generations use the same source-revision
check. Equality,
range, `IN`, null, wildcard, prefix, and suffix leaves produce exact bitmaps;
boolean planning may retain a conservative bitmap superset and the executor
always checks the full filter AST. `NOT` complements only an exact subtree, so
a partial index can reduce work but cannot remove an eligible document.
The registry pairs each scalar bitmap with its immutable ordinal generation.
Vector base/delta map keys, HNSW/Vamana graph nodes and edges, IVF postings,
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

Vamana starts at the vector nearest the dataset centroid. It initializes a
seeded R-regular directed graph and performs two deterministic build passes,
first with alpha 1 and then with the configured alpha. Each pass uses greedy
search, RobustPrune, backward-edge insertion, and re-pruning to enforce the
degree bound. The currently validated execution contract is L2-only; other
metrics fail before schema mutation instead of receiving an unproven transform.

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
supports AND/OR/NOT, parentheses, required/prohibited modifiers, escapes, and
exact phrases. Candidate construction is exact for boolean semantics; phrase
postings form a conservative intersection and only those documents are
retokenized for adjacency verification. Broad structured expressions switch to
the same scan evaluator when indexed refinement is estimated to cost more.
Wildcard, fielded, boosted, fuzzy/proximity, and range syntax fails with
`NotSupported`; selecting both `match_string` and `query_string` is invalid and
ambiguous.

## 6. Query pipeline

```text
Request
  → acquire one schema/document snapshot
  → resolve the schema field and validate route/type/dimension/limits/options
  → parse each filter and FTS expression once
  → derive a revision-matched scalar bitmap prefilter when safe
  → refine a conservative bitmap before bounded vector candidate selection
  → select matching HNSW/IVF/Vamana/FTS generations or exact scan fallbacks
  → exact-score a selective set, or push a larger eligible set into HNSW/IVF/Vamana
  → traverse indexed term postings or execute scan BM25 with identical stats
  → retain exact dense/sparse/FTS results in a deterministic bounded top-k heap
  → fuse exact branch results when requested
  → apply radius, exact-score ordering, top-k, and group-by limits
  → project requested fields/vectors
  → expose the checked `f32` score after deterministic exact ordering
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
and are intentionally outside this project. It likewise does not expose the
dependency kernel as a low-level escape hatch; a kernel replacement must not
force downstream callers to migrate unrelated types.

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
`NotSupported`; it is never executed with a substitute tokenizer. The default
feature set and its tests honor the declared Rust 1.75 MSRV, including explicit
compatible pins for broad Rayon and `rmp` dependency ranges. Jieba's current
dependency chain contains Rust 2024-edition manifests and therefore requires a
newer Cargo; that feature-specific compiler floor remains explicit until the
dependency is replaced or its minimum version is raised for the whole crate.

## 9. Non-negotiable quality gates

- no production `unwrap`, `expect`, or unchecked user-controlled allocation;
- `Send + Sync` for public handles where the contract permits it;
- crash/replay tests for every WAL operation and checkpoint boundary;
- deterministic result ordering and schema validation tests;
- recall and latency benchmarks against the flat reference implementation;
- fuzz coverage for filter parsing, WAL frames, and index metadata;
- Intel macOS 12 compile, smoke, and runtime benchmark before release.
