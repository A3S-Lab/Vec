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

`a3s-vec` 是原生 Rust 的**进程内**检索引擎：稠密/稀疏向量、标量过滤与 BM25
落在同一持久化集合中——无需服务端进程，也无需 C/C++ 运行时。

**`0.1.2` 已发布到 [crates.io](https://crates.io/crates/a3s-vec)。** 标签
`0.1.2`、托管 CI run `35503763590` 与 crate SHA-256
`2b2c5194e05cc8d17ac4f1ba5f3b609e2b5aba403bc25ab18ebf5f0af1ec6cc0` 绑定修订
`0c7894f`（[RELEASE.md](RELEASE.md)）。索引缺失、过期或不够选择性时，精确执行
仍是正确性预言机。macOS 12 Monterey Intel 不受支持。

[架构](ARCHITECTURE.md) · [路线图](ROADMAP.md) ·
[测试](TESTING.md) · [基准](BENCHMARKS.md) · [发布](RELEASE.md) ·
[docs.rs](https://docs.rs/a3s-vec)

## 存在理由

Coding Agent 工作区需要本地、可持久且对分数诚实的检索。`a3s-vec` 负责集合、
WAL/快照、索引与查询规划；嵌入模型、工作区扫描与 UI 策略留给调用方。

| 需求 | 你得到的 |
| --- | --- |
| 语义检索 | 精确稠密/稀疏扫描、HNSW、IVF/SOAR、RaBitQ、Vamana、PQ/ADC DiskANN，再做**精确**全向量重排 |
| 工作区文本 | BM25、Unicode n-gram、结构化布尔/短语/通配/模糊/范围语法 |
| 结构化过滤 | 类型化标量索引，经共享 `u64` 序号域与 ANN/FTS 组合 |
| 持久化 | WAL、校验和快照、文件锁、派生索引缓存、类型化资源限额 |
| 可预期失败 | 校验错误与精确回退——不做静默近似 |

## 相对 zvec 的证据（第一性 harness）

协议：[docs/scale-compare-protocol.md](docs/scale-compare-protocol.md)。
证据：[BENCHMARKS.md](BENCHMARKS.md)。同一主机、共享 SplitMix64 语料、单
worker、相同 HNSW 控制（`m=16`、`ef_construction=96`、`ef=64`）。a3s-vec
保留 exact re-ranking 与 `f64` 公开分数；zvec harness 设置
`is_using_refiner=False`。三次独立进程中位数，包版本 `0.1.2`，Apple M5 Max /
macOS 26.6.2 arm64。

### Apple Silicon · 100k × 128（公平：单 worker）

| 引擎 | 索引构建 | 查询 p50 | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.2** | **26.3 s** | **103 µs** | **0.6000** |
| zvec 0.7.0 | 46.2 s | 149 µs | 0.5813 |

构建约快 **1.75×**，查询 p50 约低 **1.44×**，recall 更高且稳定。

### 同一单 worker 下的 Flat

| 引擎 | 查询 p50 | Recall@10 |
| --- | ---: | ---: |
| a3s-vec 0.1.2 | 3,550 µs | 1.0000 |
| zvec 0.7.0 | **1,841 µs** | 1.0000 |

公开 `f64` 精确 Flat 在 `RAYON_NUM_THREADS=1` 时约慢 **1.93×**。主机默认
Rayon 池下 a3s-vec Flat 中位 p50 约 **651 µs**（约快 **2.83×**）——单独报告，
勿混入 HNSW 公平表。

### Windows Xeon · 100k × 128（历史候选）

| 引擎 | 索引构建 | 查询 p50 | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.1** | **49.3 s** | 355 µs | **0.6000** |
| zvec 0.7.0 | 70.9 s | **349 µs** | 0.5875 |

构建约快 **1.44×**；查询 p50 落在噪声内（约 2%）。保留为 Xeon 快照。

以上是单主机、单参数点的方向性证据，不是 SLO。不以降低 `ef`、关闭
re-ranking 或把公开分数改成 `f32` 制造胜负。

## 安装

```toml
[dependencies]
a3s-vec = "0.1.2"
```

可选 Tokio 安全查询入口：

```toml
a3s-vec = { version = "0.1.2", features = ["async"] }
```

在 A3S monorepo 中仍可使用 path 依赖：

```toml
a3s-vec = { path = "crates/vec" }
```

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

异步应用在 Tokio blocking pool 上走同一规划器：

```rust
use a3s_vec::{Collection, Doc, Result, SearchQuery};

async fn search(collection: &Collection, query: &SearchQuery) -> Result<Vec<Doc>> {
    collection.query_async(query).await
}
```

`query_async` / `multi_query_async` / `group_by_async` 与同步路径结果和遥测
一致。丢弃 future 不会取消已在 `spawn_blocking` 上启动的工作。

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

全部索引共享带修订的序号域，规划器无需构建查询规模的主键映射，仅对最终
top-k 解析主键。同分按升序主键打破平局。

## 能力一览

| 族 | 表面 |
| --- | --- |
| 向量 | FP16/32/64、INT4/8/16、Binary32/64；稀疏 FP16/32；L2、IP、cosine、MIPS-L2 |
| ANN | HNSW、IVF（可选 SOAR）、HNSW/IVF RaBitQ、Vamana、PQ DiskANN（定位读或 mmap sidecar） |
| FTS | `standard` / `whitespace` / `ngram` / 可选 `jieba`；lowercase、ASCII fold、Snowball |
| 过滤 | 相等、范围、`IN`、null、通配/前缀/后缀、布尔组合 |
| 运维 | 只读打开、flush、定向 rebuild、optimize、health、显式拥有的维护调度器 |

可执行示例纳入 CI：

```sh
cargo run --locked --example crud_operations
cargo run --locked --example vector_search
cargo run --locked --example retrieval_workflows
```

详见 [`examples/README.md`](examples/README.md)。DiskANN I/O、RaBitQ 控制、FTS
分析器选项、资源限额与恢复的完整契约见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 边界

- 不是 Alibaba zvec 存储或 C++ ABI 的二进制兼容克隆。
- `zvec-core` 是私有纯 Rust 算法依赖；公开 API 由 A3S 拥有。
- Binary ANN 与 Alibaba C++ 线格式导入/导出是刻意非目标。
- 原生异步文件读与直接文件映射 mmap 在有失败不变量测试前保持拒绝
  （[VEC-R2](ROADMAP.md)）。

## 质量闸门

```sh
cargo fmt --all -- --check
cargo test --locked
cargo test --locked --all-features
cargo clippy --locked --all-targets -- -D warnings
cargo +1.75.0 test --locked
```

托管 CI 在 Linux/Windows/macOS（arm64 + Intel，部署目标 15.0）上重复质量、
MSRV、恢复 fuzz 与 smoke 性能矩阵。可重复基准：[BENCHMARKS.md](BENCHMARKS.md)。

## 平台与归属

正确性目标：Linux x86_64/aarch64、Windows x86_64、macOS arm64/x86_64
（macOS 15.0+）。不要求 `io_uring`、C/C++ 运行时或强制架构 SIMD。

本仓库为 [`A3S-Lab/Vec`](https://github.com/A3S-Lab/Vec)；A3S monorepo 以
`crates/vec` 消费。跨项目边界：
[检索平台架构](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md)。

采用 [MIT](LICENSE) 许可。
