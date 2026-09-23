<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec：进程内向量与全文检索">
</p>

<p align="center">
  <a href="https://crates.io/crates/a3s-vec"><img alt="crates.io" src="https://img.shields.io/crates/v/a3s-vec.svg"></a>
  <a href="https://docs.rs/a3s-vec"><img alt="docs.rs" src="https://img.shields.io/docsrs/a3s-vec"></a>
  <a href="https://github.com/A3S-Lab/Vec/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Vec/ci.yml?branch=main"></a>
  <img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.75-informational">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

# a3s-vec

给 Coding Agent 工作区用的进程内向量和全文检索。一个集合就是一个目录。文档快照和 WAL 是记录本身。HNSW、IVF、RaBitQ、Vamana、DiskANN、标量倒排和 BM25 都从这份记录建出来，也可以再建一次。

[架构](ARCHITECTURE.md) · [路线图](ROADMAP.md) · [测试](TESTING.md) · [基准](BENCHMARKS.md) · [docs.rs](https://docs.rs/a3s-vec)

当前版本：[`0.1.8`](https://crates.io/crates/a3s-vec)，tag `0.1.8` 指向 `a26d59e`，crate SHA-256 `755d3bee…`。旧 tag 见 [RELEASE.md](RELEASE.md)。`0.1.7` 的库与此相同，但它的测试调用了 `f32::next_up`，在 Rust 1.75 上编不过。

## 安装

```toml
[dependencies]
a3s-vec = "0.1.8"
```

Tokio 上的查询走同一个规划器，落在 `spawn_blocking`：

```toml
a3s-vec = { version = "0.1.8", features = ["async"] }
```

在 A3S monorepo 里：`a3s-vec = { path = "crates/vec" }`。

## 示例

```rust
use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, IndexParams, MetricType, Result,
    SearchQuery,
};

fn main() -> Result<()> {
    let mut embedding = FieldSchema::new("embedding", DataType::VectorFp32, false, 4)?;
    embedding.set_index_params(&IndexParams::flat(MetricType::Cosine)?)?;
    let schema = CollectionSchema::builder("notes")
        .add_field(embedding)
        .build()?;
    let collection = Collection::create("./notes-index", &schema, None)?;

    let mut doc = Doc::with_pk("src/index.rs")?;
    doc.add_vector_f32("embedding", &[1.0, 0.0, 0.0, 0.0])?;
    collection.insert(&[&doc])?;

    let hits = collection.query(&SearchQuery::new("embedding", &[1.0, 0.0, 0.0, 0.0], 1)?)?;
    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

全文走同一个集合。`standard`、`whitespace`、`ngram` 编进默认构建。`jieba` 要打开 `jieba` feature。

```rust
use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams, Result,
    SearchQuery,
};

fn main() -> Result<()> {
    let mut body = FieldSchema::new("body", DataType::String, false, 0)?;
    body.set_index_params(&IndexParams::fts(Some("standard"), None, None)?)?;
    let schema = CollectionSchema::builder("workspace")
        .add_field(body)
        .build()?;
    let collection = Collection::create("./workspace-index", &schema, None)?;

    let mut doc = Doc::with_pk("src/index.rs")?;
    doc.add_string("body", "Rust vector database for workspace retrieval")?;
    collection.insert(&[&doc])?;

    let mut expression = Fts::new()?;
    expression.set_query_string("rust AND \"vector database\"")?;
    let hits = collection.query(&SearchQuery::fts("body", &expression, 10)?)?;
    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

```rust
async fn search(
    collection: &a3s_vec::Collection,
    query: &a3s_vec::SearchQuery,
) -> a3s_vec::Result<Vec<a3s_vec::Doc>> {
    collection.query_async(query).await
}
```

更多程序见 [`examples/README.md`](examples/README.md)。

## 分数

一次查询先冻住一份 schema、文档和索引修订，再检查路由、类型、维度和限额。索引可以返回候选。命中上的分数是存储向量的精确 `f64` 分数。索引缺失或过期时，扫描直接读文档。Flat 的召回是 1。分数相同则保留较小的主键，主键只对留下的命中解析。

进程默认的持久化级别是 `Always`。HNSW 的默认 `ef` 是 64。精确重排保持打开。IVF 没有默认 `scale_factor`。

过滤解析、分词，以及 FP16/INT8/INT4 量化都在这个 crate 里，不对外再导出。

## 索引与字段

| 索引 | 说明 |
| --- | --- |
| Flat | 对存储向量做精确扫描。 |
| HNSW | 上层度数 `m`，第 0 层度数 `2m`。 |
| IVF | 可选 SOAR。 |
| HNSW RaBitQ、IVF RaBitQ | 1 到 9 bit 的码用于遍历。公开分数仍是完整向量。 |
| Vamana | L2、内积、cosine、MIPS-L2。 |
| DiskANN | PQ/ADC，定位读，或一份校验过的匿名 mmap 快照。 |

BM25 支持布尔、短语、通配、模糊和范围。通配和模糊在匹配器展开前先用字符 trigram 剪枝。

标量过滤有相等、范围、`IN`、null、通配、前缀、后缀和布尔组合，与向量和全文共用一个规划器。

编码：FP16、FP32、FP64、INT4、INT8、INT16、Binary32、Binary64、稀疏 FP16、稀疏 FP32。度量：L2、内积、cosine、MIPS-L2。二进制检索是精确的 Flat L2 或 Hamming。

`StorageCeilings` 的默认值：快照、索引缓存、WAL 回放各 8 GiB，DiskANN 边车 512 MiB。0 会被拒绝。库不会按主机内存改这些值。

`Collection` 上有只读打开、flush、rebuild、optimize、health，以及一个由调用方持有的维护调度器。DiskANN I/O、RaBitQ、分析器、资源限额和恢复写在 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 实测

2026-09-23，Apple M5 Max，同机一次。a3s-vec 用 `0.1.7` 的排序代码（`0.1.8` 只改了一个 Rust 1.75 测试辅助函数），对照 zvec 0.7.0。cosine，top-10，32 条查询 × 3 轮，batch 512，HNSW `m=16`、`ef_construction=96`、`ef=64`，单 worker。a3s-vec 用 `f64` 重排。zvec 设置 `is_using_refiner=False`。写入时间含最后一次 flush。2,000×32 和 100,000×128 是三次进程中位数。1,000,000×128 是一次进程。

协议：[docs/scale-compare-protocol.md](docs/scale-compare-protocol.md)。完整表：[BENCHMARKS.md](BENCHMARKS.md)。

| 规模 | a3s 写入 | zvec 写入 | a3s Flat p50 | zvec Flat p50 | a3s HNSW 构建 | zvec HNSW 构建 | a3s HNSW p50 | zvec HNSW p50 | a3s Recall@10 | zvec Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000×32 | 28.2 ms | 28.8 ms | 11.5 µs | 53.4 µs | 66.2 ms | 67.8 ms | 29.3 µs | 57.6 µs | 1.0000 | 1.0000 |
| 100,000×128 | 346 ms | 964 ms | 770 µs | 1,694 µs | 12.4 s | 45.3 s | 98.3 µs | 136 µs | 0.6000 | 0.5813 |
| 1,000,000×128 | 3.33 s | 10.0 s | 7.54 ms | 23.6 ms | 202 s | 559 s | 136 µs | 180 µs | 0.3063 | 0.2594 |

`ef=64` 下的 Recall@10 是这次协议跑出来的数。2026-09-20 的百万文档写入 `77,339.081` ms 是更早的一次未拆分测量，留在 [BENCHMARKS.md](BENCHMARKS.md)。

## 限度

- 磁盘格式不是 Alibaba zvec 的 C++ 存储，这个 crate 也不实现那套 ABI。
- 没有 C++ 线格式导入或导出，也没有二进制 ANN。
- 异步文件读和文件映射要等一个会失败的测试（[VEC-R2](ROADMAP.md)）。
- 不支持 macOS 12 Monterey 的 Intel。

## 开发

```sh
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --all-features
cargo clippy --locked --all-targets -- -D warnings
cargo +1.75.0 test --locked
```

托管 CI 在 Linux x86_64/aarch64、Windows x86_64、macOS arm64/x86_64 上跑这些检查、恢复 fuzz 和 smoke 基准。macOS 部署目标是 15.0。默认构建不要求 `io_uring`，也不要求按架构打开的 SIMD。

仓库：[A3S-Lab/Vec](https://github.com/A3S-Lab/Vec)。monorepo 把它挂在 `crates/vec`。[MIT](LICENSE)。
