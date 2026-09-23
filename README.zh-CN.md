<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec：面向 Coding Agent 工作区的进程内向量与全文检索">
</p>

<p align="center">
  <a href="https://crates.io/crates/a3s-vec"><img alt="crates.io" src="https://img.shields.io/crates/v/a3s-vec.svg"></a>
  <a href="https://docs.rs/a3s-vec"><img alt="docs.rs" src="https://img.shields.io/docsrs/a3s-vec"></a>
  <a href="https://github.com/A3S-Lab/Vec/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Vec/ci.yml?branch=main"></a>
  <img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.75-informational">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

# a3s-vec

**面向 Coding Agent 工作区的进程内检索。**

集合是一份持久化的文档日志。文档快照和 WAL 是权威来源。HNSW、IVF、RaBitQ、
Vamana、DiskANN、标量倒排和 BM25 是派生索引：它们提出候选，公开分数是权威
向量的精确 `f64` 重排。索引缺失或过期时回退到这次精确扫描。分数相同则保留
升序主键。

稠密/稀疏向量、BM25 全文与类型化标量过滤共享同一个带修订的序号域。没有服务端
进程，也没有 C/C++ 运行时。`0.1.7` 的过滤、分词和量化内核由本 crate 自己实现。
`0.1.8` 保持这份实现，并把一个测试辅助函数换成 Rust 1.75 能编译的写法
（`f32::next_up` 比 MSRV 新）。

**[`0.1.8` 已发布](https://crates.io/crates/a3s-vec)** · 已发布
（tag `0.1.8` @ `a26d59e` · SHA-256 `755d3bee…` · [RELEASE.md](RELEASE.md)）。
`0.1.7` 已发布（tag `0.1.7` @ `57fc476` · SHA-256 `90254cfd…`）。
它的库代码与这份排序实现相同；它的单元测试在 Rust 1.75 上编不过。
`0.1.6` 仍是 tag `0.1.6` · SHA-256 `67c238a0…`。
`0.1.5` 仍是 tag `0.1.5` · SHA-256 `bc42798f…`。
`0.1.4` 仍是 tag `0.1.4` · SHA-256 `15c4220d…`。

[架构](ARCHITECTURE.md) · [路线图](ROADMAP.md) ·
[测试](TESTING.md) · [基准](BENCHMARKS.md) ·
[docs.rs](https://docs.rs/a3s-vec)

---

## 特性

| 特性 | 作用 |
| --- | --- |
| **单一集合** | 向量、FTS、标量索引共享带修订的 `u64` 序号域——组合过滤时无需构建查询规模的主键映射。 |
| **精确优先正确性** | ANN 只提候选；权威向量做 **exact re-rank**，公开 `f64` 分数。索引缺失或过期则精确扫描回退——不做静默近似。 |
| **ANN 深度** | HNSW、IVF（可选 SOAR）、HNSW/IVF RaBitQ、Vamana、PQ/ADC DiskANN（定位读或 mmap sidecar）。 |
| **工作区文本** | BM25，`standard` / `whitespace` / `ngram` / 可选 `jieba`；布尔、短语、通配、模糊、范围；matcher 展开前的字符 trigram 剪枝。 |
| **类型化过滤** | 相等、范围、`IN`、null、通配/前缀/后缀、布尔组合——与 ANN/FTS 同一规划器。 |
| **持久化** | WAL、校验和快照、文件锁、派生索引缓存、类型化资源限额；类型化 `StorageCeilings`（默认 8 GiB / 8 GiB / 8 GiB / DiskANN 512 MiB）——显式策略，从不按主机自动探测。 |
| **失败封闭 API** | 不支持的路由与错误维度在变更前以类型化错误失败。 |
| **嵌入外置** | 嵌入模型留给调用方；a3s-vec 负责存储、索引与规划。 |

原生编码：FP16/32/64、INT4/8/16、Binary32/64、稀疏 FP16/32。度量：L2、IP、
cosine、MIPS-L2。

---

## 为什么选它

1. **跑在 Agent 进程内** — 无需运维旁路数据库；打开路径、写入、查询即可。
2. **分数可辩护** — 公开排序对权威向量做精确 `f64` 重打分；Flat 召回按构造为 1.0。
3. **混合检索无需胶水** — 语义 + 词法 + 结构化谓词在同一规划器与同一持久化世代。
4. **一次修订，一个校验和** — 托管 CI、git tag 与 crates.io 产物绑定同一提交。`0.1.6` 在 DiskANN 边车仍是上一修订时保留索引缓存。`0.1.5` 与 `0.1.4` 仍是各自的已发布绑定。
5. **相对 zvec 0.7.0 的实测** — 同一语料、cosine、top-10、`m=16`、`ef_construction=96`、`ef=64`、单 worker，并保留 exact re-rank。证据见下。百万文档 flush 仍在 8 GiB 存储上限之内。

**不是什么：** 托管向量云、zvec C++ ABI 克隆，或「全面碾压」式引擎排名。

---

## 安装

```toml
[dependencies]
a3s-vec = "0.1.8"
```

面向 Tokio 的查询（同一规划器，跑在 `spawn_blocking`）：

```toml
a3s-vec = { version = "0.1.8", features = ["async"] }
```

Monorepo path：`a3s-vec = { path = "crates/vec" }`。

## 快速开始

```rust
use a3s_vec::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams,
    Result, SearchQuery,
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
async fn search(collection: &a3s_vec::Collection, query: &a3s_vec::SearchQuery)
    -> a3s_vec::Result<Vec<a3s_vec::Doc>>
{
    collection.query_async(query).await
}
```

CI 示例：`crud_operations`、`vector_search`、`retrieval_workflows` — 见
[`examples/README.md`](examples/README.md)。

---

## 查询如何保持精确

```text
请求
  → 冻结一份 schema / 文档 / 索引修订
  → 校验路由、类型、维度、限额
  → 在选择性足够时组合标量 + FTS 候选
  → ANN 或精确向量路径
  → 在权威文档上验证过滤 / 短语
  → 精确打分、确定性 top-k、投影
```

同分按升序主键打破平局；仅对最终 top-k 解析主键。

---

## 运维表面

只读打开、flush、定向 rebuild、optimize、health，以及显式拥有的维护调度器。
DiskANN I/O、RaBitQ、FTS 分析器、`CollectionResourceLimits`、`StorageCeilings`
与恢复契约见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 相对 zvec 的证据（诚实，不是王冠）

同一次协议跑出来的同机证据，不是容量 SLO。
协议：[docs/scale-compare-protocol.md](docs/scale-compare-protocol.md) ·
[BENCHMARKS.md](BENCHMARKS.md)。

控制：SplitMix64 语料、cosine、top-10、32×3 查询、batch 512、HNSW
`m=16` / `ef_construction=96` / `ef=64`、单 worker。a3s-vec 保留 exact
re-rank 和公开 `f64` 分数；zvec 使用 `is_using_refiner=False`。
2,000×32 与 100,000×128 是三次进程中位数，1,000,000×128 是单进程。
数字来自 `0.1.7` 的排序实现；`0.1.8` 只改了 Rust 1.75 的测试辅助函数。
Apple M5 Max，zvec `0.7.0`，2026-09-23。
写入时间包含最后一次 flush。

| 规模 | a3s 写入 | zvec 写入 | a3s Flat p50 | zvec Flat p50 | a3s HNSW 构建 | zvec HNSW 构建 | a3s HNSW p50 | zvec HNSW p50 | a3s Recall@10 | zvec Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000×32 | 28.2 ms | 28.8 ms | 11.5 µs | 53.4 µs | 66.2 ms | 67.8 ms | 29.3 µs | 57.6 µs | 1.0000 | 1.0000 |
| 100,000×128 | 346 ms | 964 ms | 770 µs | 1,694 µs | 12.4 s | 45.3 s | 98.3 µs | 136 µs | 0.6000 | 0.5813 |
| 1,000,000×128 | 3.33 s | 10.0 s | 7.54 ms | 23.6 ms | 202 s | 559 s | 136 µs | 180 µs | 0.3063 | 0.2594 |

2026-09-20 的百万文档写入 `77,339.081` ms 仍是未拆分的历史测量。
协议默认 Recall@10 不是精度承诺。不以降低 `ef`、关闭 re-ranking
或把公开分数改成 `f32` 制造胜负。

---

## 边界

- 不是 Alibaba zvec 存储或 C++ ABI 的二进制兼容克隆。
- 过滤解析、全文分词，以及 FP16/INT8/INT4 索引量化都是本 crate 拥有的 Rust 实现。公开 API 由 A3S 拥有。
- Binary ANN 与 C++ 线格式导入/导出是刻意非目标。
- 原生异步文件读与直接文件映射 mmap 在有失败不变量测试前保持拒绝
  （[VEC-R2](ROADMAP.md)）。
- macOS 12 Monterey Intel 不受支持。

## 质量闸门

```sh
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --all-features
cargo clippy --locked --all-targets -- -D warnings
cargo +1.75.0 test --locked
```

托管 CI：Linux/Windows/macOS（arm64 + Intel，部署目标 15.0）上的质量、MSRV、
恢复 fuzz 与 smoke 性能。

## 平台与归属

正确性目标：Linux x86_64/aarch64、Windows x86_64、macOS arm64/x86_64
（macOS 15.0+）。不要求 `io_uring`、C/C++ 运行时或强制架构 SIMD。

仓库：[`A3S-Lab/Vec`](https://github.com/A3S-Lab/Vec)。A3S monorepo 以
`crates/vec` 消费。跨项目边界：
[检索平台架构](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md)。

采用 [MIT](LICENSE) 许可。
