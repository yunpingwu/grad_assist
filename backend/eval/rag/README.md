# RAG 四指标评测

用 [ragas](https://docs.ragas.io) 对「检索 + 生成」这条 RAG 链路做离线评测。包内自足，
不依赖 `eval/retrieval/`（那个包是旧的自研指标，待删）；检索实现直接复用生产代码
（`app/retrieval/`），测的就是线上真正跑的那条路径。

## 评什么

| 阶段 | 指标 | 回答的问题 | 输入 |
| --- | --- | --- | --- |
| 检索 | **Context Precision** | 相关的是不是排在前面、无关的是不是混进来了 | 问句 + 上下文 + 参考答案 |
| 检索 | **Context Recall** | 回答所需的信息，检索结果里是不是都有 | 同上 |
| 生成 | **Faithfulness** | 答案里每句话能不能在上下文里找到依据（= 幻觉程度） | 问句 + 回答 + 上下文 |
| 生成 | **Answer Relevancy** | 有没有正面回答问题，还是答非所问 | 问句 + 回答 |

指标实现取 `ragas.metrics.collections`（0.4 的新路径）。评分粒度：Context Precision 是
平均精度（AP，逐条 context 判相关性后按序累加），Context Recall 是参考答案**逐句**判
能否归因到检索结果，Faithfulness 是抽 statements 后逐条判能否被上下文支撑，
Answer Relevancy 是用回答反生成 3 个问题后与原问题算余弦相似度取均值。

## 两个配置

| 配置 | 含义 | 检索路径 |
| --- | --- | --- |
| `hyb_50` | 非深度 | 混合召回（dense/sparse 各 0.5 加权）取 top-5 |
| `rerank` | 深度（对齐 `tools/search.py` 的 deep 分支） | 混合召回 + HyDE 两路各扩到候选池 20 → RRF 融合 → 交叉编码精排（与 RRF 分数加权融合） |

`hyb_50` 与线上默认的 `hyb_80` 实测 **top-5 重合度 0.968**（50 题无一条完全不同），所以选它
不影响与线上行为的可比性；`hyb_50` 是 2026-09-18 报告推荐优先验证的候选。两个配置的 top-5
重合度是 **0.760**——有 24% 的内容不同，所以对照是有区分度的。

## 实测结论：深度路径值不值（2026-10-16）

300 题 × 2 配置跑完，配对检验（bootstrap 95% CI，300 组配对）：

| 指标 | hyb_50 | rerank | 差值 | 95% CI | 结论 |
| --- | --- | --- | --- | --- | --- |
| Context Precision | 0.8860 | 0.8899 | +0.0039 | [−0.0158, +0.0228] | 不显著 |
| Context Recall | 0.9092 | 0.9064 | −0.0028 | [−0.0200, +0.0150] | 不显著 |
| **Faithfulness** | 0.8687 | 0.8891 | **+0.0204** | **[+0.0019, +0.0406]** | **显著** |
| Answer Relevancy | 0.7765 | 0.7710 | −0.0054 | [−0.0162, +0.0047] | 不显著 |

**深度路径（rerank）只在一项上带来统计显著的提升——Faithfulness +0.020。检索侧两项与
Answer Relevancy 的差值都在 ±0.006 以内、置信区间跨 0。**

逐本方向很乱，不是「某类教材受益」那么简单：

| 数据集 | CP 差值 | CR 差值 | FA 差值 | AR 差值 |
| --- | --- | --- | --- | --- |
| C语言 | **+0.060** | +0.012 | +0.009 | −0.014 |
| 组成原理 | **−0.024** | +0.017 | +0.011 | −0.041 |
| 数据库 | +0.008 | 0.000 | +0.053 | −0.008 |
| 数据结构 | −0.008 | −0.017 | +0.016 | +0.013 |
| 计算机网络 | −0.004 | −0.018 | +0.025 | +0.012 |
| 操作系统 | −0.001 | −0.023 | +0.016 | +0.002 |

C语言精度涨 6 个点、组成原理反退 2.4 个点，其余基本不动。这与 2026-09-18 报告
「精排不是所有教材都受益」一致。

**读这个结论时的三条限制**（别把上面那张表当成定论）：

1. **成本侧没有数据**。这一轮没记延迟（`rag.py` 只存 chunk id 与回答）。「多花的检索代价」
   只能引用 2026-09-18 报告的 37×，**不能用这轮支撑**；要严谨得单独跑一轮串行单飞的延迟测量。
2. **300 题不是 300 个独立样本**：20 处是两题共用同一个 gold，CI 偏乐观，只能当信号看。
3. **Faithfulness 的 +0.02 要打折**：数据库贡献了最大的一块（+0.053），而它恰好是推导题最多、
   该指标最不适用的一本（见下面「读结果前必须知道的口径」）。

复现：`rag rerank` → `evals rerank`（`hyb_50` 同理），产物在 `results/<配置>/`。

## 怎么跑

```bash
# 1. 给某配置生成样本（检索 → 单轮生成；可断点续跑，--force 覆盖重跑）
uv run python -m eval.rag.rag hyb_50
uv run python -m eval.rag.rag rerank

# 2. 评测（指定配置）
uv run python -m eval.rag.evals hyb_50
uv run python -m eval.rag.evals rerank
uv run python -m eval.rag.evals hyb_50 os --limit 5          # 先小样本
uv run python -m eval.rag.evals rerank --judge-provider alibaba   # 换评审模型
```

⚠️ `--limit` 会**覆盖**该数据集的结果文件。小样本试跑前先把 `results/` 备份，或换一个配置名跑。

结果落在 `results/<配置>/<数据集>.json`（逐题四个分数 + 均值 + 用了哪个模型）和
`results/<配置>/summary.json`（总表），控制台也会打一张总表。

## 目录

```
eval/rag/
├── rag.py       # 被测系统：retrieve() + answer()，以及生成样本的 CLI
├── evals.py     # 四个指标：评审模型装配、并发控制、汇总报告
├── prompts/     # 冻结的生成提示词（进 git，改它等于换一套答案）
├── datasets/    # 题集 + gold 正文（config 无关）
├── samples/     # 样本，按配置分目录（答案 + 它那一轮的上下文）
└── results/     # 评测产物，按配置分目录
```

## 读结果前必须知道的口径

- **生成是单轮、无历史、冻结 prompt**，不跟线上 Agent 的提示词走（线上那套是给工具调用用的
  壳，直接把固定上下文塞进去会错配）。所以结论代表「固定检索结果 + 固定生成指令」的链路质量，
  **不代表线上 Agent 的端到端行为**。
- **Faithfulness 在推导/计算题上系统性偏低，那不等于幻觉**。实例：函数依赖推理题的答案是
  「由 A→B 和 B→C，根据传递律得 A→C」，教材里只有推理规则、没有这道题的推导过程，判据便全判
  不支持。实测 `computation` 类 0.61，而 `multi_hop` 0.95、`confusable` 0.92。**别把它当
  幻觉率读**，要逐题看 verdict。
- **评审模型默认与生成模型相同**（`deepseek-flash`），自己答、自己判会高估 Faithfulness。
  要认真出数就用 `--judge-provider` 换一家。
- **失败不静默**：单条指标失败计为 NaN 并打警告，总表下方会汇总 NaN 条数。
- 检索侧那两列是干净的（Context Precision / Recall 不受上述问题影响）。

## 三个踩过的坑（改代码时别踩回去）

1. **`AsyncOpenAI` 构造不收 `extra_body`**，得经 `llm_factory` 转发。DeepSeek 平台侧思考模式
   默认开启，不关的话输出预算被思考吃光，指标直接抛 `IncompleteOutputException`。同时 judging
   长上下文要显式给足 `max_tokens`。
2. **collections 指标不能塞进 `aevaluate()`**（没有 `single_turn_ascore`），而它们自带的
   `abatch_score` 是无上限全量并发。所以并发必须自己用信号量控（`evals.py` 的 `_score`）。
3. **吞吐上限在供应商侧**。实测 12 / 24 / 36 并发分别是 1.06 / 1.11 / 1.12 次/秒——加并发
   完全没用。总调用量 = 每样本 11 次（Context Precision 是按 context 数走的，k=5 就是 5 次），
   所以 300 题一轮约 50 分钟，想优化得换供应商而不是调并发。

## 依赖

`ragas==0.4.3` + **`langchain-community<0.4`**。后者是钉死的：ragas 在模块级硬 import
`langchain_community.chat_models.vertexai`，而该模块在 langchain-community 0.4 里已被删除——
不钉版的话 `import ragas` 直接失败（`uv lock` 不会报错，只有 import 才炸）。
