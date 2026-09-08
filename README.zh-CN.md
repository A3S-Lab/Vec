<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="a3s-vec: fast process-local vector and full-text retrieval for Coding Agent workspaces">
</p>


<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

`a3s-vec` 是 Coding Agent 的原生 Rust、进程本地检索引擎
工作区。它结合了密集和稀疏向量、标量过滤和 BM25
在一个持久集合中——没有服务器进程或 C/C++ 运行时。

该项目是一个活跃的原型。 HNSW、IVF（可选择 SOAR 分配）、
HNSW/IVF RaBitQ，度量感知 Vamana，
具有类型化定位或不可变 mmap 快照遍历的乘积量化 DiskANN，
标量倒排索引和 FTS 已上线；
每当索引丢失时，精确执行仍然是正确性预言，
陈旧，或者选择性不够。

[Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) ·
[Reproducible benchmarks](BENCHMARKS.md) ·
[Release qualification](RELEASE.md)

## 它提供什么

|需要|当前实施 |
| ---| ---|
|局部语义检索|密集和稀疏精确搜索、HNSW、IVF/SOAR、HNSW/IVF RaBitQ、度量感知 Vamana、PQ/ADC DiskANN 和精确重新排名 |
|工作区文本搜索 | BM25、Unicode n-gram、布尔组、通配符/模糊/范围术语、有序短语邻近度、增强和标记过滤器 |
|结构化收窄|类型化标量索引、范围/空/IN/通配符谓词和位图预过滤 |
|耐用的嵌入 | WAL、校验和快照、清单提交、文件锁定、经过验证的派生索引缓存和 Vamana/DiskANN 扇区 sidecar |
|可预测的故障|输入验证错误和精确回退，而不是静默近似 |

## A3S Code 整合

该引擎现在由 A3S Code 通过会话本地迁移影子使用。
当前代码依赖 pin 已提交
[`708a85e3`](https://github.com/A3S-Lab/Code/commit/708a85e3ac070640ca5fb8173d0b06e6070152e7)，引脚 Vec
提交[`13585ccd`](https://github.com/A3S-Lab/Vec/commit/13585ccd3f956f6cb7d669b2ee6acc7096fca03d)。
适配器将每个已接纳的嵌入批次镜像一次到临时
收集并将Vec结果与A3S Memory结果进行比较，而Memory
仍然是唯一的服务机构。影子故障被隔离并浮出水面
作为有界诊断；他们无法更改公共检索结果。的
完整的所有权、映射、资源和回滚合同记录在
[Code's migration note](https://github.com/A3S-Lab/Code/blob/main/manual/WORKSPACE_RETRIEVAL_VEC_MIGRATION.md)。

该存储库中当前的引擎和基准证据由
修订[`13585ccd`](https://github.com/A3S-Lab/Vec/commit/13585ccd3f956f6cb7d669b2ee6acc7096fca03d)。
其修订版托管门是
[CI run 33772179017](https://github.com/A3S-Lab/Vec/actions/runs/33772179017)；
前面的实施和方法门仍然可用
存储库历史记录。
当前的修订版还记录了借用的精确分数内核的测量结果
改善。根兼容性锁可能会保留较旧的代码子模块
固定，直到其云升级工作流程更新为一个精确的组件图；
候选代码本身已针对此修订进行了验证。

所有向量、标量和 FTS 索引共享一个修订的 `u64` 序数域。
这使得规划器可以组合位图和候选项而无需构建
查询大小的主键映射，然后仅解析确切的前 k 个文档。

## 测量证明

`cargo bench --bench structured_fts` 构建 25,000 个工作区形状的文档
并根据扫描执行检查每个索引结果和公共分数位。
连续两次变更后的第二次运行在当前开发上
机器生产：

|查询 |规划路径 |候选人/查询 |延迟/查询 |
| ---| ---| ---: | ---: |
|选择性短语 |索引| 1 | 7.38 微秒 |
|选择性必需+可选|索引| 1 | 4.62 微秒 |
|选择性通配符 |索引| 1 | 26.55 毫秒 |
|选择性模糊|索引| 36 | 36 33.81 毫秒 |
|选择性精确范围|索引| 1 | 429.62 微秒 |
|选择性接近 |索引| 1 | 7.62 微秒 |
|显式扫描控制 |扫描| 25,000 | 25,000 38.07–98.36 毫秒 |
|常用短语|自动扫描回退 | 25,000 | 25,000 40.81 毫秒 |
|广泛布尔值 + NOT |自动扫描回退 | 25,000 | 25,000 43.19 毫秒 |

5 个选择性案例使考生得分减少 25,000 倍；模糊扩展
将它们从 25,000 个减少到 36 个。通配符和模糊查询包含词汇表
扩展过程，而广泛的结构化查询故意切换到确切的
当候选集工作不太可能收回成本时扫描路径。这些是
局部回归测量——不是跨项目 zvec 基准。满
方法论和重复观察存在于[BENCHMARKS.md](BENCHMARKS.md)。
公共 API 发布门是确定性的 [feature matrix](BENCHMARKS.md#public-feature-matrix-and-performance-gate)，
它检查每个查询路由并报告同步的 p50/p95/p99 延迟，
ANN、sidecar、mutation 和 Tokio 路径。同伴
[concurrent-reader and mixed-workload fixtures](BENCHMARKS.md#mixed-readwrite-contention)
门读争用、读/写争用、Recall@10、QPS 和逻辑
会计上同样的修订。生命周期矩阵还测量
管理操作、资源准入和维护所有权。 CI
平台矩阵在 Linux x86/ARM、Windows x86 和 macOS ARM/Intel 上重复所有五个烟雾台；它的托管
英特尔的结果是可移植性证据，而不是所需的 macOS 12 运行时门。
要进行更大的同主机引擎比较，请使用 [scale harness](BENCHMARKS.md#larger-corpus-scale-comparison)，
它以相同的确定性驱动 a3s-vec 和选择加入的 zvec 伴侣
语料库和报告构建时间、p50/p95/p99、QPS 和 Recall@10。

## 快速开始

A3S monorepo 使用此存储库作为 `crates/vec`。直到一个箱子
发布版本已发布，请使用 monorepo 根目录的路径依赖项：

```toml
[dependencies]
a3s-vec = { path = "crates/vec" }
```

这个完整的示例创建了一个持久的 FTS 集合，插入一个工作区
文档，并执行结构化查询：

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

Tokio 应用程序可以选择调度程序安全的查询方法，而无需执行
核心集合依赖于运行时：

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

`query_async`、`multi_query_async` 和 `group_by_async` 需要有效
Tokio运行时并执行完整的同步快照、规划器、
其阻塞池上的 sidecar-I/O、后备和精确细化路径。他们
产生与其同步对应物相同的结果和遥测；的
功能是执行者安全边界，而不是延迟声明。东京不能
开始后取消 `spawn_blocking` 工作，因此放弃其中一个 future
不取消其基础查询。

## 可执行兼容性示例

[`examples`](examples/README.md)目录是回归的一部分
表面。上游 CRUD、向量搜索和模式构建器固定装置跟踪
`zvec-ai/zvec-rust@0d40cb1aef081bae175061fef35c89269e6a80f4` 仅具有
crate命名空间已更改；他们的可执行包装器仅添加本地 lint
津贴。断言项目拥有的二进制文件涵盖矢量/FTS 和混合
检索、分组 top-k、隔离迭代、持久模式演化以及
维护健康。 CI 运行每个二进制文件，而不仅仅是检查它
编译：

```text
cargo run --locked --example crud_operations
cargo run --locked --example vector_search
cargo run --locked --example schema_builder
cargo run --locked --example retrieval_workflows
cargo run --locked --example group_by
cargo run --locked --example schema_iteration
cargo run --locked --example maintenance_health
```

固定的上游 CRUD 固定装置包含两个不完整的替换更新插入；
官方 zvec 和 `a3s-vec` 都拒绝它们，因为所需的 `id` 字段是
缺席。这个已知的上游装置缺陷被保留，因此仅命名空间
索赔仍可审核。所声称的 A3S 拥有的示例因任何错误而失败
结果。

## 检索能力

### 向量和索引

- 密集 FP16、FP32、FP64、INT4、INT8、INT16、Binary32 和 Binary64 有效负载。
- 稀疏 FP16 和 FP32 有效负载。
- 密集、稀疏和打包二进制查询接受显式有效负载
  或源文档 ID。源 ID 查询使用相同的精确评分，
  过滤、半径、投影、持久性和可选的 Tokio 执行
  路径； Binary32 和 Binary64 通过每条路线独立覆盖。
- Binary32 和 Binary64 精确搜索使用 L2 位坐标：公开的
  分数是负异或汉明计数。支持扁平L2；其他二进制文件
  指标和二进制 ANN 索引返回`NotSupported`。
- `SearchQuery::builder()` 支持密集、打包二进制或纯 FTS
  `query_string`/`match_string` 路由并拒绝不明确的组合；
  `include_doc_id` 公开返回的查询文档的生成序号。
- 精确数值 L2、内积、余弦和 MIPS-L2 评分以及二进制
  L2/Hamming评分，均为`f64`排名中级。
- 原生 HNSW 和 IVF 候选生成，具有精确的全向量重新排序；
  IVF 可选择将每个基向量分配给一个主质心和一个
  正交感知 SOAR 次要质心。
- 便携式 HNSW/IVF RaBitQ，具有确定性随机旋转，紧凑
  1 到 9 位代码、有界细化和精确的全向量重新排序。
- 确定性两遍度量感知 Vamana 构造（L2、内积、
  余弦和 MIPS-L2）、有界 `list_size` 搜索、增量覆盖和
  精确的全向量重新排序。
- 确定性乘积量化器训练，每个块最多 256 个质心，
  一字节代码、查询本地 ADC 表和精确的全向量重新排序。
- 具有固定全矢量或 PQ 代码的原生 4 KiB 扇区 Vamana/DiskANN 文件
  记录、CRC 验证、有界定位读取或不可变匿名 mmap
  快照和故障关闭内存回退。
- 仅索引 FP16、对称 INT8 和对称 INT4 量化。
- 标量倒排索引，用于相等、范围、`IN`、null、通配符、前缀、
  后缀和布尔过滤器组成。

Vamana 接受 L2、内积、余弦和 MIPS-L2 向量，可选
FP16、INT8 或 INT4 仅索引量化和精确的权威重新排名。
`IndexParams::diskann` 使用相同的度量感知确定性图并且
当 `pq_chunk_num > 0` 时启用语料库训练的 PQ；零选择全向量
图表评分。新建或重建的一代
在记忆中穿越。经过验证的缓存重新打开后，有界查询使用
默认情况下可移植定位读取并保留请求本地扇区/节点
缓存。 `IoBackend::Mmap` 相反，将已经验证的 sidecar 复制到
打开时只读匿名内存映射并提供相同的有界范围
来自那个不可变的快照。 PQ 查询构建一个可感知指标的 ADC 表，
在图遍历期间求和代码相似度或距离。增量叠加共享
读者；完全重建会重新训练密码本并使阅读器失效，直到
下一个验证的重新开放。短读或格式错误的记录会回退到
等效的内存中全向量或 ADC 图，并且权威向量仍然
执行最终重新排名。该文件是 A3S 原生格式，而不是 Microsoft
DiskANN C++ 格式。 mmap 快照独立于以后的替换或
截断源文件，但 open 执行完整的 sidecar 复制并保留
句柄生命周期的额外内存。可选的 Tokio 条目
点使任一后端远离运行时工作人员；本机异步文件读取和
直接文件支持的 mmap 仍然是未来的加速器。

使用类型化选项为一个集合句柄选择 mmap：

```rust
use a3s_vec::{Collection, CollectionOptions, IoBackend, Result};

fn open_with_mmap(path: &str) -> Result<Collection> {
    let mut options = CollectionOptions::new()?;
    options.set_io_backend(IoBackend::Mmap)?;
    Collection::open(path, Some(&options))
}
```

相同的查询控件为两种索引类型选择有界列表大小：

```rust
use a3s_vec::{DiskannQueryParams, IndexParams, MetricType, Result, SearchQuery};

fn configure_diskann_pq(query: &mut SearchQuery) -> Result<IndexParams> {
    query.set_diskann_params(DiskannQueryParams::new(64))?;
    IndexParams::diskann(MetricType::L2, 32, 96, 8)
}
```

RaBitQ 是一个独立的 HNSW/IVF 索引系列。它训练确定性中心，
应用四轮带符号哈达玛旋转，并仅使用紧凑代码
用于候选遍历或细化。权威向量仍然是
公共分数的来源。 HNSW默认为7位16中心；打字的
选项构造函数公开位宽、中心计数和样本计数。体外受精
使用 `scale_factor * topk` 作为有界精确细化器集：

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

Vamana 现在执行 `max_occlusion` RobustPrune 候选上限，
`saturate` 图形填充控制，以及独立的 FP16/INT8/INT4 索引
具有权威精确重新排名的量化。二元人工神经网络和阿里巴巴的
C++ 线格式保持独立、记录的边界。确切的二进制路由是基于A3S的扩展
zvec 的前二进制平方欧几里德/汉明语义；阿里巴巴删除了其
[zvec PR #365](https://github.com/alibaba/zvec/pull/365) 中的汉明度量，所以
该项目不声明当前上游二进制查询兼容性。

### 全文搜索

FTS 管道使用相同的有序标记器分析文档和查询
和过滤器配置。

|组件|支持的值 |
| ---| ---|
|分词器 | `standard`、`whitespace`、Unicode `ngram`、可选 `jieba` / `jieba_accurate` |
|令牌过滤器 | `lowercase`、`ascii_folding`、`stemmer` |
|查询语法 | `AND`、`OR`、`NOT`、括号、`+` 必需、`-` 禁止、转义符、`*` / `?` 通配符、同字段限定符、`^` 增强、模糊术语、有序短语斜率和术语范围|
|默认运算符 | `OR` 用于兼容性，或显式 `AND` |

省略 `filters` 选择 `lowercase`。传递显式空切片会保留
标准、空白和 n-gram 分词器输出区分大小写。过滤器
按声明顺序对索引文本和查询文本运行。

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

Snowball 词干分析器支持阿拉伯语、丹麦语、荷兰语、英语、芬兰语、法语、
德语、希腊语、匈牙利语、意大利语、挪威语、葡萄牙语、罗马尼亚语、俄语、
西班牙语、瑞典语、泰米尔语和土耳其语。 ASCII 折叠使用 Unicode 分解
加上常见的拉丁语兼容性映射；它不是按字节进行广告的
相当于每个 zvec 折叠桌。

n-gram 分词器默认为 Unicode 二元组。 `ngram_min`,
`ngram_max`和`token_chars`配置其范围和接受的Unicode
字符类：

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

对于选择性标识符/路径查询，`default_operator=AND`从
最短的发帖。结构化表达式构建精确的布尔候选集；
短语仅验证候选者的有序邻近度。计划者又回到了
扫描广泛表达式的执行，并在标量时保持索引细化
预过滤器可用。

通配符（`rust*`、`r?sty`）、模糊（`rust~1` 或 `rust~2`）和范围
(`[alpha TO omega]`, `{alpha TO omega}`) 叶子相对于分析的展开一次
术语词汇。 `*` 是无界范围端点。模糊项、范围界限、
通配符文字片段必须分别分析一个术语和范围
比较是根据最终的词典术语顺序进行的。限定符如
`body:rust`必须命名已被`SearchQuery::fts`选择的字段；
跨域执行被拒绝。提升是有限值
`(0, 1_000_000]`。

例如，引用的短语接受从 0 到 1,024 的显式斜率
`"vector engine"~2`。 Slop 计算干预令牌的总数，同时保留
术语顺序；它不启用转置。索引执行和扫描执行
使用这些相同的扩展、BM25 和邻近规则。象征性`&&`和`||`
别名仍然明确不受支持。

## 执行如何保持准确

```text
request
  → capture one immutable schema/document/index revision
  → validate route, type, dimension, limits, and syntax
  → derive scalar and FTS candidate ordinals when selective
  → run HNSW/IVF/RaBitQ/Vamana/DiskANN or the exact vector path
  → verify filters and phrases against authoritative documents
  → exact-score, deterministic top-k, projection, and optional fusion
```

- 平面矢量和扫描 BM25 执行始终可用作参考路径。
- 每个派生索引生成都是不可变的，并标有其来源
  修订。
- HNSW/IVF/RaBitQ/Vamana/DiskANN候选者重新排名，具有权威性
  向量。
- 索引和扫描 FTS 共享 `f64` 语料库/评分原语并生成
  不同赛程中的公共分数完全相同。
- 同等分数使用升序主键作为确定性抢七。

## 持久性和恢复

文档、快照、WAL记录都是权威的。当前存储
格式为版本 4：校验和 MessagePack 快照加上清单提交
WAL 边界。版本 3 JSON 快照保持可读并在下一个版本升级
可写检查点。

ANN、标量、FTS 和共享序数表分别作为
非权威派生缓存。缓存格式 10 包括 RaBitQ 旋转，
中心、紧凑代码、Vamana/DiskANN 图、PQ 码本/代码、已解析
分词器和有序过滤器状态。瓦玛纳或
另外还生成 DiskANN
需要 `indexes/diskann-graph.bin`：绑定到的 A3S 原生 4 KiB 扇区镜像
相同的修订版、模式摘要和清单标识。它的标题、元数据、
填充、完整向量或 PQ 代码/码本、图形边缘和 CRC 均经过验证
在缓存命中之前。丢失、陈旧、损坏、结构无效或 v10 之前的版本
缓存/边车对
被忽略并从恢复的文档中重建；只读打开永不修复
它。

公共 API 支持只读句柄、可配置的持久性和 sidecar
I/O、显式 `flush`、定向 `rebuild_index`、整个注册表 `optimize`，以及
每个句柄缓存命中/查询/候选加上 DiskANN 后端/扇区读取遥测。

## 资源限制和核算

资源策略是一个类型化的、集合本地的选项，当句柄被调用时捕获。
创建或打开：

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

在新集合之前检查`max_documents`和`max_accounted_bytes`
生成被发布或附加到 WAL。所计算的字节数是
权威文档图加上派生的确定性二进制代码大小
索引统计信息报告的索引负载估计。他们并不声称
测量分配器开销、临时构建峰值、映射文件或
处理 RSS。会增加墓碑覆盖层的删除首先会压缩
因此删除仍然是恢复容量的实用方法。

`max_query_candidates` 限制了一个计划的精确/细化候选者
查询；多个查询分支共享一个累积预算。它并不代表
挂钟截止日期或包括每个计划者/索引查找。写批处理
限制适用于插入、更新、更新插入、显式删除输入以及匹配的
一组已过滤的删除。被拒绝的一代是原子的并且不会前进
修订版。 `stats` 和 `stats_snapshot` 公开活动策略、文档
和索引统计、总字节数和仅元数据拒绝
柜台；被拒绝的查询文本和文档永远不会被记录。

## 健康与后台维护

`Collection::health` 报告显式 `healthy`、`degraded`、`unhealthy`、
或`closed`状态。它根据已提交的内容检查内存中的修订版本
存储修订并要求每个配置的派生索引准备就绪，
完整，并源自该修订版。 WAL 操作等待
检查点单独报告，因为它们在间隔或
手动持久性，不要进行其他可恢复的收藏
不健康。

集合构造永远不会启动隐藏线程。可写集合
可以选择一个显式拥有的标准线程调度程序：

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

每个到期的修订都会重建完整的派生注册表并检查点
作家门举行时，同样是权威一代；读者继续
在构建过程中使用以前的不可变索引。不变的修订版
被跳过。只有一个运行时可以拥有收集计划、只读句柄
拒绝它，`close`或`Drop`在释放之前唤醒并加入worker
该所有权主张。

## 当前边界

|面积 |状态 |
| ---| ---|
|公寓、HNSW、IVF/SOAR |实施的; SOAR 发布使用确定性主加辅助分配和唯一候选探测 |
| HNSW/IVF RaBitQ |针对 L2、内积和余弦实现，具有 1 至 9 位代码和精确的重新排序 |
|度量感知的 Vamana 遍历和增量覆盖 |针对 L2、内积、余弦和 MIPS-L2 实现，具有可选的 FP16/INT8/INT4 索引量化，在内存中并通过重新打开后定位或不可变的 mmap-snapshot sidecar 读取 |
|度量感知 DiskANN PQ/ADC 和增量叠加 |在内存中实现 L2、内积、余弦和 MIPS-L2，并通过重新打开后定位或不可变的 mmap 快照 PQ 代码读取 |
|扇区对齐的本机 Vamana/DiskANN 文件 |已实施 |
|标量倒排索引 |已实施 |
| BM25 + 结构化布尔/短语 FTS |已实施 |
| FTS 通配符/字段/提升/模糊/邻近/范围语法 |使用有界的、分析器感知的语义来实现 |
|密集/稀疏/二进制源ID查询|实施的;缺少源返回`NotFound`，缺少源有效负载返回`FailedPrecondition` |
|收藏健康及后台维护|通过显式所有权、有界时间表、修订感知跳过、工作诊断和联合关闭来实现 |
|馆藏资源入场|为保留文档/逻辑字节、累积查询候选、写入批次和仅元数据拒绝遥测而实施 |
| DiskANN 查询阅读器 |可移植的定位读取或经过验证的不可变匿名 mmap 快照，以及可选的 Tokio 阻塞池查询入口点；本机异步文件读取和直接文件支持的 mmap 仍然是路线图
|产品量化/ RaBitQ |为 DiskANN 实施 PQ /为 HNSW 和 IVF 实施 RaBitQ |
|二进制向量查询执行 | Binary32/Binary64 精确 L2/Hamming 跨直接、源 ID、过滤、半径、投影/包含 doc-id、多查询、分组、持久性和可选 Tokio 路径实现；二进制 ANN 仍然不受支持 |
|阿里巴巴C++二进制格式兼容性|需要明确的未来进口商/出口商 |

`a3s-vec` 遵循 zvec 的 Rust 词汇表，它是有用的，但它不是一个
二进制兼容的克隆。 `zvec-core` 仍然是私有的纯 Rust 算法
依赖性；调用者仅使用 A3S 拥有的集合、模式、文档、查询和
错误合同。

## 质量门

在此箱内运行检查：

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

可重现的性能夹具：

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

默认、无默认功能、全功能、严格 Clippy、rustdoc 和 Rust
1.75个门单独维护。可选的Jieba依赖链
目前需要更新的 Cargo，因为一个传递包使用 Rust
2024 年清单。

## 平台和所有权

可移植正确性路径针对 Linux x86_64/aarch64、Windows x86_64、
和 macOS arm64/x86_64，其中 macOS 12.0 作为 Intel 部署目标。确实如此
不需要`io_uring`、C/C++ 运行时或特定于体系结构的 SIMD。

`a3s-vec` 拥有检索、持久化和索引执行。工作空间
扫描、嵌入模型运行时、代理会话和 UI 策略属于
他们的来电者。跨项目边界记录在
[A3S local retrieval platform architecture](https://github.com/A3S-Lab/a3s/blob/main/docs/retrieval-platform-architecture.md)。

这个存储库是`A3S-Lab/Vec`； A3S monorepo 将其作为
`crates/vec`子模块。已获得[MIT](LICENSE)许可。
