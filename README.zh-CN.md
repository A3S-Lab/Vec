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
稠密/稀疏向量、BM25 全文与类型化标量过滤落在同一持久化 Rust 集合——无需
服务端进程，也无需 C/C++ 运行时。

**[`0.1.4` 已发布](https://crates.io/crates/a3s-vec)** · 已发布
（tag `0.1.4` · SHA-256 `15c4220d…` · [RELEASE.md](RELEASE.md)）

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
4. **已发布 Enterprise GA** — 多平台托管 CI、版本化 RC 与 crates.io 校验和绑定同一修订（`0.1.4` 类型化 `StorageCeilings`）。
5. **诚实 harness 下 HNSW 有竞争力** — 相同旋钮、单 worker、保留 exact re-rank；证据见下（方向性，非 SLO）。百万文档 flush 在工作站主机上已放开（8 GiB 存储上限）。

**不是什么：** 托管向量云、zvec C++ ABI 克隆，或「全面碾压」式引擎排名。

---

## 安装

```toml
[dependencies]
a3s-vec = "0.1.4"
```

面向 Tokio 的查询（同一规划器，跑在 `spawn_blocking`）：

```toml
a3s-vec = { version = "0.1.4", features = ["async"] }
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

同主机方向性证据——不是容量 SLO，也不是「全面碾压」。
协议：[docs/scale-compare-protocol.md](docs/scale-compare-protocol.md) ·
[BENCHMARKS.md](BENCHMARKS.md)。

控制：SplitMix64 语料、cosine、top-10、32×3 查询、batch 512、HNSW
`m=16` / `ef_construction=96` / `ef=64`、单 worker。a3s-vec 保留 exact
re-rank + `f64`；zvec 使用 `is_using_refiner=False`。三次进程中位数 · 包
`0.1.3` · Apple M5 Max / macOS 26.6.2 arm64 · zvec 0.7.0（10 万表）。百万
文档表为 `0.1.3` 提高 8 GiB 存储上限后、同一控制下的单进程同机结果。

### HNSW · 100k × 128（公平 harness）

| 引擎 | 索引构建 | 查询 p50 | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.3** | **26.3 s** | **103 µs** | **0.6000** |
| zvec 0.7.0 | 46.2 s | 149 µs | 0.5813 |

构建约快 **1.75×**，查询 p50 约低 **1.44×**，recall 更高且稳定。

### Flat · 同一单 worker

| 引擎 | 查询 p50 | Recall@10 |
| --- | ---: | ---: |
| a3s-vec 0.1.3 | 3,550 µs | **1.0000** |
| zvec 0.7.0 | **1,841 µs** | **1.0000** |

公开 `f64` 精确 Flat 在此约慢 **1.93×**（契约使然）。主机默认 Rayon 池下
a3s-vec Flat p50 约 **651 µs**——单独报告，勿混入 HNSW 公平表。

### HNSW · 1M × 128（同一控制，单进程）

| 引擎 | 写入 | 索引构建 | 查询 p50 | QPS | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **a3s-vec 0.1.3** | 77.3 s | **422 s** | **159 µs** | **5954** | 0.3063 |
| zvec 0.7.0 | **13.5 s** | 628 s | 231 µs | 4244 | 0.2437 |

方向性结论：此规模下 a3s HNSW 建图与查询更快；zvec Flat 加载更快。协议
默认 recall **不是**精度承诺——引用百万级召回前请先提高 `ef` /
`ef_construction`。

不以降低 `ef`、关闭 re-ranking 或把公开分数改成 `f32` 制造胜负。

---

## 边界

- 不是 Alibaba zvec 存储或 C++ ABI 的二进制兼容克隆。
- `zvec-core` 是私有纯 Rust 算法内核；公开 API 由 A3S 拥有。
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
