<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec：面向 Coding Agent 工作区的快速进程内向量与全文检索">
</p>

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

`a3s-vec` 是面向 Coding Agent 工作区的原生 Rust、进程本地检索引擎。它在同一持久集合中结合稠密与稀疏向量、标量过滤与 BM25——无需服务器进程，也无需 C/C++ 运行时。

项目是 `0.1.1` 发布候选。HNSW、可选 SOAR 分配的 IVF、HNSW/IVF RaBitQ、度量感知 Vamana、带类型化定位或不可变 mmap 快照遍历的乘积量化 DiskANN、标量倒排索引与 FTS 均已可用；每当索引缺失、陈旧或选择性不足时，精确执行仍是正确性预言机。正式将 `0.1.1` 发布到 crates.io 仍需 [RELEASE.md](RELEASE.md) 中的托管发布闸门。macOS 12 Monterey Intel 不受支持。

[架构](ARCHITECTURE.md) · [路线图](ROADMAP.md) ·
[可复现基准](BENCHMARKS.md) ·
[发布资格](RELEASE.md)

## 它交付什么

| 需求 | 当前实现 |
| --- | --- |
| 本地语义检索 | 稠密与稀疏精确搜索、HNSW、IVF/SOAR、HNSW/IVF RaBitQ、度量感知 Vamana、PQ/ADC DiskANN，以及精确重排 |
| 工作区文本搜索 | BM25、Unicode n-gram、布尔组、通配符/模糊/范围词项、有序短语邻近、boost，以及 token 过滤器 |
| 结构化收窄 | 类型化标量索引、range/null/IN/wildcard 谓词，以及位图预过滤 |
| 持久嵌入 | WAL、带校验和快照、manifest 提交、文件锁、已验证的派生索引缓存，以及 Vamana/DiskANN 扇区 sidecar |
| 可预测失败 | 类型化校验错误与精确回退，而非静默近似 |

## A3S Code 集成

引擎现由 A3S Code 通过会话本地迁移影子消费。当前 Code 依赖 pin 为提交
[`708a85e3`](https://github.com/A3S-Lab/Code/commit/708a85e3ac070640ca5fb8173d0b06e6070152e7)，其 pin 的 Vec 提交为
[`13585ccd`](https://github.com/A3S-Lab/Vec/commit/13585ccd3f956f6cb7d669b2ee6acc7096fca03d)。
适配器将每个已准入嵌入批次镜像一次到临时集合，并把 Vec 结果与 A3S Memory 结果比较，同时 Memory 仍是唯一服务权威。影子失败被隔离并以有界诊断浮出；它们不能改变公开检索结果。完整所有权、映射、资源与回滚契约见
[Code 的迁移说明](https://github.com/A3S-Lab/Code/blob/main/manual/WORKSPACE_RETRIEVAL_VEC_MIGRATION.md)。

本仓库当前引擎与基准证据由修订
[`13585ccd`](https://github.com/A3S-Lab/Vec/commit/13585ccd3f956f6cb7d669b2ee6acc7096fca03d) 承载。
其修订绑定的托管闸门为
[CI run 33772179017](https://github.com/A3S-Lab/Vec/actions/runs/33772179017)；
此前实现与方法论闸门仍可在仓库历史中查看。
当前修订还记录了借用精确打分内核的实测改进。根兼容性锁可能保留较旧的 Code 子模块 pin，直到其 Cloud 晋升降级工作流作为一份确切组件图一并更新；Code 候选本身已对照此修订验证。

所有向量、标量与 FTS 索引共享一个带修订的 `u64` 序号域。这让规划器可组合位图与候选而无需构建查询规模的主键映射，再仅解析确切 top-k 文档。

## 同机 HNSW 与 zvec 对比

第一性原理对比：同一主机、同一夹具、单 worker、相同 HNSW 控制参数。不改变公开分数契约：a3s-vec 保留 exact re-ranking 与 `f64` 公开分数；zvec harness 关闭可选 refiner（`is_using_refiner=False`）。a3s-vec 为可移植 Rust 默认 target；zvec 0.7.0 为发行原生 wheel。每个单元格为三次独立进程的中位数。完整 CSV 方法见 [BENCHMARKS.md](BENCHMARKS.md)。

主机：Windows x86_64，Intel Xeon w5-2445，128 GiB RAM（2026-09-20）。
控制：cosine，`m=16`，`ef_construction=96`，`ef=64`，32 查询 × 3 轮，`RAYON_NUM_THREADS=1` / zvec `concurrency=1`。

### 100,000 文档 × 128 维

| 引擎 | 索引构建 (ms) | 查询 p50 (µs) | 查询 p95 (µs) | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| **a3s-vec 0.1.1 候选** | **49,317** | 355.0 | **415.7** | **0.6000** |
| zvec 0.7.0 | 70,920 | **348.5** | 461.3 | 0.5875 |

该夹具上 a3s-vec 构建约快 **1.44×**，Recall@10 更高且稳定。查询 p50 与 zvec 落在噪声内（约 2%）。a3s-vec 仍在候选生成后做 exact re-ranking。

### 2,000 文档 × 32 维

| 引擎 | 索引构建 (ms) | 查询 p50 (µs) | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.1 候选** | 145.9 | **62.8** | **1.0000** |
| zvec 0.7.0 | **123.7** | 206.8 | 1.0000 |

较小规模下 a3s-vec 查询 p50 约低 **3.3×**，recall 相同。构建时间接近。

### Apple Silicon（macOS arm64，相同控制）

在 aarch64 navigation prefetch 与按序号 exact re-rank 之后（仍为 `ef=64`，
仍保留权威 `f64` re-ranking），本机 Apple Silicon 上三次交错独立进程的
HNSW-only 中位数：

| 引擎 | 索引构建 (ms) | 查询 p50 (µs) | Recall@10 |
| --- | ---: | ---: | ---: |
| **a3s-vec 0.1.1 tip** | **22,375** | **99.5** | **0.6000** |
| zvec 0.7.0 | 50,761 | 146.0 | 0.5844 |

该主机上 a3s-vec 构建约快 **2.27×**，查询 p50 约低 **1.47×**，Recall@10 更高且稳定。

以上是单主机、单参数点的方向性证据，不是 SLO，也不是 Flat 扫描排名。公开 `f64` Flat 路径按契约仍慢于 zvec 原生路径；不以关闭 re-ranking 或降低 `ef` 制造胜负。

## 实测证明

`cargo bench --bench structured_fts` 构建 25,000 份工作区形态文档，并对照扫描执行检查每个索引结果与公开分数位。变更后在当前开发机上连续两次运行中的第二次产出：

| 查询 | 规划器路径 | 候选/查询 | 延迟/查询 |
| --- | --- | ---: | ---: |
| 选择性短语 | Indexed | 1 | 7.38 µs |
| 选择性必选 + 可选 | Indexed | 1 | 4.62 µs |
| 选择性通配符 | Indexed | 1 | 26.55 ms |
| 选择性模糊 | Indexed | 36 | 33.81 ms |
| 选择性精确范围 | Indexed | 1 | 429.62 µs |
| 选择性邻近 | Indexed | 1 | 7.62 µs |
| 显式扫描控制 | Scan | 25,000 | 38.07–98.36 ms |
| 常见短语 | 自动扫描回退 | 25,000 | 40.81 ms |
| 宽布尔 + NOT | 自动扫描回退 | 25,000 | 43.19 ms |

五个选择性案例将打分候选减少 25,000×；模糊扩展将其从 25,000 降到 36。通配符与模糊查询含词汇扩展遍，而宽结构化查询在候选集工作不太可能划算时故意切到精确扫描路径。这些是本地回归测量——不是跨项目 zvec 基准。完整方法与重复观察见 [BENCHMARKS.md](BENCHMARKS.md)。
公开 API 发布闸门是确定性 [功能矩阵](BENCHMARKS.md#public-feature-matrix-and-performance-gate)，检查每条查询路由并为 sync、ANN、sidecar、mutation 与 Tokio 路径报告 p50/p95/p99 延迟。配套
[并发读与混合负载 fixture](BENCHMARKS.md#mixed-readwrite-contention)
在同一修订上闸读争用、读/写争用、Recall@10、QPS 与逻辑记账。生命周期矩阵额外测量管理操作、资源准入与维护所有权。CI 平台矩阵在 Linux x86/ARM、Windows x86 与 macOS ARM/Intel 上重复全部五个 smoke bench；其托管 Intel 结果是当前 macOS 15 镜像（部署目标 15.0）上的可移植性证据。macOS 12 Monterey 不受支持。
更大同机引擎对比请用 [规模 harness](BENCHMARKS.md#larger-corpus-scale-comparison)，以同一确定性语料驱动 a3s-vec 与可选 zvec 对照，并报告构建时间、p50/p95/p99、QPS 与 Recall@10。

## 快速开始

A3S monorepo 将本仓库消费为 `crates/vec`。在 crate 发布上线前，从 monorepo 根使用 path 依赖：

```toml
[dependencies]
a3s-vec = { path = "crates/vec" }
```

下面完整示例创建持久 FTS 集合、插入一份工作区文档，并执行结构化查询：

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
    let query = SearchQuery::fts("body", &expression, 10)?;
    let hits = collection.query(&query)?;

    assert_eq!(hits[0].get_pk(), Some("src/index.rs"));
    Ok(())
}
```

Tokio 应用可选用调度器安全的查询方法，而无需让核心集合依赖运行时：

```toml
[dependencies]
a3s-vec = { path = "crates/vec", features = ["async"] }
```

```rust
use a3s_vec::{Collection, Doc, Result, SearchQuery};

async fn search(collection: &Collection, query: &SearchQuery) -> Result<Vec<Doc>> {
    collection.query_async(query).await
}
```

`query_async`、`multi_query_async` 与 `group_by_async` 需要活跃 Tokio 运行时，并在其 blocking 池上执行完整的同步快照、规划器、sidecar I/O、回退与精确细化路径。它们产生与同步对应方法相同的结果与遥测；该功能是执行器安全边界，而非延迟声明。Tokio 无法在 `spawn_blocking` 工作开始后取消它，因此丢弃这些 future 之一不会取消其底层查询。

## 可执行兼容性示例

[`examples`](examples/README.md) 目录是回归面的一部分。上游 CRUD、向量搜索与 schema-builder fixture 跟踪
`zvec-ai/zvec-rust@0d40cb1aef081bae175061fef35c89269e6a80f4`，仅更改 crate 命名空间；其可执行包装仅添加本地 lint 允许。断言的项目自有二进制覆盖向量/FTS 与混合检索、分组 top-k、隔离迭代、持久 schema 演进与维护健康。CI 运行每个二进制，而非仅检查能否编译：

```text
cargo run --locked --example crud_operations
cargo run --locked --example vector_search
cargo run --locked --example schema_builder
cargo run --locked --example retrieval_workflows
cargo run --locked --example group_by
cargo run --locked --example schema_iteration
cargo run --locked --example maintenance_health
```

已 pin 的上游 CRUD fixture 含两次不完整替换 upsert；官方 zvec 与 `a3s-vec` 均因缺少必需 `id` 字段而拒绝。此已知上游 fixture 缺陷被保留，使「仅命名空间」声明可审计。断言的 A3S 自有示例在任何错误结果上失败。

## 检索能力

### 向量与索引

- 稠密 FP16、FP32、FP64、INT4、INT8、INT16、Binary32 与 Binary64 载荷。
- 稀疏 FP16 与 FP32 载荷。
- 稠密、稀疏与打包二进制查询接受显式载荷或源文档 ID。源 ID 查询使用相同的精确打分、过滤、半径、投影、持久化与可选 Tokio 执行路径；Binary32 与 Binary64 经各路由独立覆盖。
- Binary32 与 Binary64 精确搜索对位坐标使用 L2：公开分数是负 XOR Hamming 计数。支持 Flat L2；其他二进制度量与二进制 ANN 索引返回 `NotSupported`。
- `SearchQuery::builder()` 支持稠密、打包二进制或纯 FTS 的 `query_string`/`match_string` 路由，并拒绝歧义组合；`include_doc_id` 为返回的查询文档暴露世代序号。
- 精确数值 L2、内积、余弦与 MIPS-L2 打分，以及二进制 L2/Hamming 打分，均以 `f64` 排序中间值。
- 原生 HNSW 与 IVF 候选生成加精确全向量重排；IVF 可选将每个基向量赋给主质心与一个正交感知 SOAR 次质心。
- 可移植 HNSW/IVF RaBitQ，带确定性随机旋转、紧凑 1–9 bit 码、有界细化与精确全向量重排。
- 确定性两遍度量感知 Vamana 构建（L2、内积、余弦与 MIPS-L2）、有界 `list_size` 搜索、增量 overlay，以及精确全向量重排。
- 确定性乘积量化器训练，每块最多 256 质心、单字节码、查询本地 ADC 表，以及精确全向量重排。
- 原生 4 KiB 扇区 Vamana/DiskANN 文件，固定全向量或 PQ 码记录、CRC 校验、有界定位读或不可变匿名 mmap 快照，以及失败封闭的内存回退。
- 仅索引的 FP16、对称 INT8 与对称 INT4 量化。
- 标量倒排索引，用于相等、范围、`IN`、null、通配符、前缀、后缀与布尔过滤组合。

Vamana 接受 L2、内积、余弦与 MIPS-L2 向量，可选 FP16、INT8 或 INT4 仅索引量化与精确权威重排。
`IndexParams::diskann` 使用相同度量感知确定性图，并在 `pq_chunk_num > 0` 时启用语料训练的 PQ；零选择全向量图打分。新构建或重建的世代在内存中遍历。经验证缓存重开后，有界查询默认使用可移植定位读，并保留请求本地扇区/节点缓存。`IoBackend::Mmap` 则在打开时将已验证 sidecar 复制到只读匿名内存映射，并从该不可变快照服务相同有界范围。PQ 查询构建一张度量感知 ADC 表，并在图遍历期间累加码相似度或距离。增量 overlay 共享 reader；完整重建重训码本并使 reader 失效直至下次验证重开。短读或畸形记录回退到等价内存全向量或 ADC 图，权威向量仍做最终重排。该文件是 A3S 原生格式，不是 Microsoft DiskANN C++ 格式。mmap 快照独立于源文件之后的替换或截断，但打开会完整复制 sidecar 并在 handle 生命周期内保留该额外内存。可选 Tokio 入口将任一后端保持在运行时 worker 之外；原生异步文件读与直接文件后备 mmap 仍是未来加速器。

用类型化选项为单个集合 handle 选择 mmap：

```rust
use a3s_vec::{Collection, CollectionOptions, IoBackend, Result};

fn open_with_mmap(path: &str) -> Result<Collection> {
    let mut options = CollectionOptions::new()?;
    options.set_io_backend(IoBackend::Mmap)?;
    Collection::open(path, Some(&options))
}
```

同一查询控制为两种索引类型选择有界 list size：

```rust
use a3s_vec::{DiskannQueryParams, IndexParams, MetricType, Result, SearchQuery};

fn configure_diskann_pq(query: &mut SearchQuery) -> Result<IndexParams> {
    query.set_diskann_params(DiskannQueryParams::new(64))?;
    IndexParams::diskann(MetricType::L2, 32, 96, 8)
}
```

RaBitQ 是独立的 HNSW/IVF 索引族。它训练确定性中心，应用四轮有符号 Hadamard 旋转，并仅将紧凑码用于候选遍历或细化。权威向量仍是公开分数来源。HNSW 默认七 bit 与 16 中心；类型化选项构造器暴露 bit 宽、中心数与采样数。IVF 用 `scale_factor * topk` 作为有界精确细化集：

```rust
use a3s_vec::{
    IndexParams, IvfRabitqQueryParams, MetricType, Result, SearchQuery,
};

fn configure_rabitq(query: &mut SearchQuery) -> Result<IndexParams> {
    let mut controls = IvfRabitqQueryParams::new(8, 0.0, false, true);
    controls.set_scale_factor(8.0)?;
    query.set_ivf_rabitq_params(controls)?;
    IndexParams::ivf_rabitq(MetricType::Cosine, 64, 7, 1_000)
}
```

Vamana 现执行 `max_occlusion` RobustPrune 候选上限、`saturate` 图填充控制，以及独立 FP16/INT8/INT4 索引量化与权威精确重排。二进制 ANN 与阿里巴巴 C++ 线格式仍是独立、已文档化的边界。精确二进制路由是基于 zvec 先前二进制平方欧氏/Hamming 语义的 A3S 扩展；阿里巴巴在 [zvec PR #365](https://github.com/alibaba/zvec/pull/365) 中移除了 Hamming 度量，因此本项目不声称与当前上游二进制查询兼容。

### 全文搜索

FTS 管道用相同有序分词器与过滤器配置分析文档与查询。

| 组件 | 支持值 |
| --- | --- |
| Tokenizer | `standard`、`whitespace`、Unicode `ngram`、可选 `jieba` / `jieba_accurate` |
| Token filter | `lowercase`、`ascii_folding`、`stemmer` |
| 查询语法 | `AND`、`OR`、`NOT`、括号、`+` 必选、`-` 禁止、转义、`*` / `?` 通配符、同字段限定符、`^` boost、模糊词项、有序短语 slop，以及词项范围 |
| 默认算子 | 兼容用 `OR`，或显式 `AND` |

省略 `filters` 选择 `lowercase`。传入显式空切片使 standard、whitespace 与 n-gram 分词器输出保持大小写敏感。过滤器按声明顺序同时作用于索引文本与查询文本。

```rust
use a3s_vec::{IndexParams, Result};

fn workspace_text_index() -> Result<IndexParams> {
    IndexParams::fts(
        Some("standard"),
        Some(&["lowercase", "ascii_folding", "stemmer"]),
        Some(r#"{"max_token_length":255,"stemmer_lang":"english"}"#),
    )
}
```

Snowball stemmer 支持 Arabic、Danish、Dutch、English、Finnish、French、German、Greek、Hungarian、Italian、Norwegian、Portuguese、Romanian、Russian、Spanish、Swedish、Tamil 与 Turkish。ASCII folding 使用 Unicode 分解加常见拉丁兼容映射；不宣称与每个 zvec folding 表逐字节等价。

n-gram 分词器默认 Unicode bigram。`ngram_min`、`ngram_max` 与 `token_chars` 配置其范围与接受的 Unicode 字符类：

```rust
use a3s_vec::{IndexParams, Result};

fn identifier_index() -> Result<IndexParams> {
    IndexParams::fts(
        Some("ngram"),
        None,
        Some(
            r#"{"ngram_min":2,"ngram_max":3,"token_chars":["letter","digit"]}"#,
        ),
    )
}
```

对选择性标识符/路径查询，`default_operator=AND` 从最短 posting 开始。结构化表达式构建精确布尔候选集；短语仅对候选验证有序邻近。规划器对宽表达式回退到扫描执行，并在有标量预过滤可用时保留索引细化。

通配符（`rust*`、`r?sty`）、模糊（`rust~1` 或 `rust~2`）与范围（`[alpha TO omega]`、`{alpha TO omega}`）叶子对已分析词项词汇扩展一次。`*` 是无界范围端点。模糊词项、范围边界与通配符字面片段必须各自分析为一个词项，范围比较按结果词典序。如 `body:rust` 的限定符必须命名 `SearchQuery::fts` 已选字段；跨字段执行被拒绝。Boost 为有限值，范围 `(0, 1_000_000]`。

带引号短语接受 0 到 1,024 的显式 slop，例如 `"vector engine"~2`。Slop 计介入 token 总数同时保留词项顺序；不启用换位。索引与扫描执行使用相同扩展、BM25 与邻近规则。符号 `&&` 与 `||` 别名仍明确不支持。

## 执行如何保持精确

```text
request
  → capture one immutable schema/document/index revision
  → validate route, type, dimension, limits, and syntax
  → derive scalar and FTS candidate ordinals when selective
  → run HNSW/IVF/RaBitQ/Vamana/DiskANN or the exact vector path
  → verify filters and phrases against authoritative documents
  → exact-score, deterministic top-k, projection, and optional fusion
```

- Flat 向量与扫描 BM25 执行始终作为参考路径可用。
- 每个派生索引世代不可变，并标记其源修订。
- HNSW/IVF/RaBitQ/Vamana/DiskANN 候选用权威向量重排。
- 索引与扫描 FTS 共享 `f64` 语料/打分原语，并在差分 fixture 中产生位一致公开分数。
- 相等分数用升序主键作为确定性平局打破。

## 持久化与恢复

文档、快照与 WAL 记录是权威的。当前存储格式为版本 4：带校验和的 MessagePack 快照加 manifest 提交的 WAL 边界。版本 3 JSON 快照仍可读，并在下次可写检查点升级。

ANN、标量、FTS 与共享序号表作为非权威派生缓存单独持久化。缓存格式 10 含 RaBitQ 旋转、中心、紧凑码、Vamana/DiskANN 图、PQ 码本/码、已解析分词器与有序过滤器状态。Vamana 或 DiskANN 世代额外需要 `indexes/diskann-graph.bin`：绑定同一修订、schema 摘要与 manifest 身份的 A3S 原生 4 KiB 扇区镜像。其头、元数据、填充、全向量或 PQ 码/码本、图边与 CRC 在缓存命中前验证。缺失、陈旧、损坏、结构无效或 pre-v10 的缓存/sidecar 对被忽略并从恢复文档重建；只读打开永不修复它。

公开 API 支持只读 handle、可配置持久性与 sidecar I/O、显式 `flush`、定向 `rebuild_index`、整注册表 `optimize`，以及每 handle 的缓存命中/查询/候选加 DiskANN 后端/扇区读遥测。

## 资源限制与记账

资源策略是类型化、集合本地选项，在 handle 创建或打开时捕获：

```rust,no_run
use a3s_vec::{CollectionOptions, CollectionResourceLimits, Result};

fn bounded_options() -> Result<CollectionOptions> {
    let limits = CollectionResourceLimits::new()
        .try_with_max_documents(100_000)?
        .try_with_max_accounted_bytes(512 * 1024 * 1024)?
        .try_with_max_query_candidates(50_000)?
        .try_with_max_write_batch_documents(1_000)?;
    let mut options = CollectionOptions::new()?;
    options.set_resource_limits(limits)?;
    Ok(options)
}
```

`max_documents` 与 `max_accounted_bytes` 在新集合世代发布或追加到 WAL 之前检查。记账字节是权威文档映射的确定性 bincode 大小加索引统计报告的派生索引载荷估计。它们不宣称测量分配器开销、临时构建峰值、映射文件或进程 RSS。会增长墓碑 overlay 的删除会先压缩派生世代，使删除仍是恢复容量的实用方式。

`max_query_candidates` 限制单次查询的计划精确/细化候选；多查询分支共享一个累计预算。它不代表墙钟截止，也不含每次规划器/索引查找。写批次限制适用于 insert、update、upsert、显式 delete 输入，以及过滤删除的匹配集。被拒绝的世代是原子的且不推进修订。`stats` 与 `stats_snapshot` 暴露活跃策略、文档与索引记账、总记账字节，以及仅元数据的拒绝计数器；被拒绝的查询文本与文档永不记录。

## 健康与后台维护

`Collection::health` 报告显式 `healthy`、`degraded`、`unhealthy` 或 `closed` 状态。它对照已提交存储修订检查内存修订，并要求每个已配置派生索引就绪、完整且源自该修订。等待检查点的 WAL 操作单独报告，因为在间隔或手动持久性下它们正常，不会使本可恢复的集合变为 unhealthy。

集合构造永不启动隐藏线程。可写集合可选用一个显式拥有的标准线程调度器：

```rust,no_run
use a3s_vec::{Collection, CollectionMaintenanceOptions};
use std::time::Duration;

fn start(collection: &Collection) -> a3s_vec::Result<()> {
    let options = CollectionMaintenanceOptions::new()
        .try_with_interval(Duration::from_secs(60))?;
    let maintenance = collection.start_maintenance(options)?;
    maintenance.trigger()?;
    let health = maintenance.health();
    assert!(health.worker_alive);
    maintenance.close()?;
    Ok(())
}
```

每个到期修订在持有写闸时重建完整派生注册表并检查点同一权威世代；读者在构建期间继续使用先前不可变索引。未变修订被跳过。仅一个运行时可拥有集合调度，只读 handle 拒绝它，且 `close` 或 `Drop` 在释放所有权声明前唤醒并 join worker。

## 当前边界

| 领域 | 状态 |
| --- | --- |
| Flat、HNSW、IVF/SOAR | 已实现；SOAR posting 使用确定性主+次分配与唯一候选探测 |
| HNSW/IVF RaBitQ | 已实现 L2、内积与余弦，1–9 bit 码与精确重排 |
| 度量感知 Vamana 遍历与增量 overlay | 已实现 L2、内积、余弦与 MIPS-L2，可选 FP16/INT8/INT4 索引量化，内存中及重开后经定位或不可变 mmap 快照 sidecar 读 |
| 度量感知 DiskANN PQ/ADC 与增量 overlay | 已实现 L2、内积、余弦与 MIPS-L2，内存中及重开后经定位或不可变 mmap 快照 PQ 码读 |
| 扇区对齐原生 Vamana/DiskANN 文件 | 已实现 |
| 标量倒排索引 | 已实现 |
| BM25 + 结构化布尔/短语 FTS | 已实现 |
| FTS 通配符/字段/boost/模糊/邻近/范围语法 | 已实现，语义有界且感知分析器 |
| 稠密/稀疏/二进制源 ID 查询 | 已实现；缺失源返回 `NotFound`，缺失源载荷返回 `FailedPrecondition` |
| 集合健康与后台维护 | 已实现，含显式所有权、有界调度、修订感知跳过、worker 诊断与 join 关停 |
| 集合资源准入 | 已实现保留文档/逻辑字节、累计查询候选、写批次，以及仅元数据拒绝遥测 |
| DiskANN 查询 reader | 可移植定位读或已验证不可变匿名 mmap 快照，加可选 Tokio blocking 池查询入口；原生异步文件读与直接文件后备 mmap 仍在路线图 |
| 乘积量化 / RaBitQ | PQ 已实现用于 DiskANN / RaBitQ 已实现用于 HNSW 与 IVF |
| 二进制向量查询执行 | Binary32/Binary64 精确 L2/Hamming 已实现于直接、源 ID、过滤、半径、投影/include-doc-id、多查询、group-by、持久化与可选 Tokio 路径；二进制 ANN 仍不支持 |
| 阿里巴巴 C++ 二进制格式兼容 | 需要显式未来导入/导出器 |

`a3s-vec` 在有用处跟随 zvec 的 Rust 词汇，但不是二进制兼容克隆。`zvec-core` 仍是私有纯 Rust 算法依赖；调用方仅使用 A3S 自有集合、schema、文档、查询与错误契约。

## 质量闸门

在本 crate 内运行检查：

```sh
cargo fmt --all -- --check
cargo audit --deny unsound
cargo test
cargo test --no-default-features
cargo test --all-features
cargo +1.75.0 test --locked
cargo +1.75.0 test --locked --features async
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo test --locked --test feature_matrix
```

可复现性能 fixture：

```sh
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
cargo bench --bench ann_recall
cargo bench --bench filtered_ann
cargo bench --bench scalar_filter
cargo bench --bench fts_index
cargo bench --bench ngram_fts
cargo bench --bench structured_fts
cargo bench --bench reopen_index
```

默认、无默认功能、全功能、严格 Clippy、rustdoc 与 Rust 1.75 闸门分别维护。可选 Jieba 依赖链目前需要较新 Cargo，因为一个传递包使用 Rust 2024 manifest。

## 平台与所有权

可移植正确性路径面向 Linux x86_64/aarch64、Windows x86_64 与 macOS arm64/x86_64，当前部署目标为 macOS 15.0。macOS 12 Monterey Intel 不受支持。它不要求 `io_uring`、C/C++ 运行时或架构特定 SIMD。

`a3s-vec` 拥有检索、持久化与索引执行。工作区扫描、嵌入模型运行时、Agent 会话与 UI 策略属于其调用方。跨项目边界见
[A3S 本地检索平台架构](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md)。

本仓库为 `A3S-Lab/Vec`；A3S monorepo 将其消费为 `crates/vec` 子模块。许可证为 [MIT](LICENSE)。
