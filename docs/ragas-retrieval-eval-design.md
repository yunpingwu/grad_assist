# RAG 检索评测重构方案（ragas 版）

> 状态: 设计中(未实现) · 日期: 2026-10-15 · v0.2
> 范围: **只重构检索侧离线评测**（现 `app/eval/retrieval/run_ablation.py` 这条线）。
> 生成侧（faithfulness / 答案正确性）、Agent 端到端、多轮、意图识别**不在本版**，但第 5 节的数据产物为它们预留接缝。
>
> v0.2 变更：工程结构改为对齐 ragas 官方 `rag_eval/` 布局（第 5 节）；新增运行口径与攒批器实测（第 6 节）。

***

## 1. 结论速览

| 问题 | 结论 |
| --- | --- |
| 框架 | ragas 0.4.3（当前最新），只用 `ragas.metrics.collections` 的 4 个指标 + 它们自带的 `ascore` / `abatch_score`（**不用** `evaluate()`，见 3.6） |
| 数据集 | 沿用现有 6 教材 × 50 题 = 300 题，**不重建标注、不改题面**；格式先统一为 [eval-dataset-schema.md](./eval-dataset-schema.md) 的 v2（字段 8~10 → 4 必填 + 2 可选，体积 -79%） |
| 指标 | 按标准四件套重排：**检索侧 2 个**（Context Precision / Context Recall，各配一条确定性护栏口径）在 v1；**生成侧 2 个**（Faithfulness / Answer Relevancy）留 v2，前置条件是补 `response` |
| 工程结构 | 对齐官方布局：一个被测系统接缝 `rag.py` + 一个评测工作流 `evals.py`；**实验（一次运行）升为一等公民** |
| 落点 | 代码进 `app/eval/rag/`（不建第二个项目），数据留在 `data/`（不入库） |
| 要不要 LLM | 确定性两指标不要（离线、可单测、可进 CI）；语义两指标要，但只在代表配置上跑 |
| 攒批器 | 线上**保留**（实测并发 2.7× 收益）；离线串行是**纯税 22.8ms/次**，需加直通入口 |
| 主要收益 | ① 精度维度第一次可测（绕开"gold 未穷举"）② 口径标准化、外部可比 ③ 采样产物直接喂下一版生成评测 |

***

## 2. 现状体检：指标冗余是可证明的，不是感觉

先说结论：**旧指标 12 个数里只有 2 个自由维度**——「命中了几条 gold」和「首个命中排在第几位」。其余是同一变量的单调变换或恒等重复。逐条给证据。

### 2.1 三处硬冗余

**① `mrr` 与 `mrr@5` 是同一个数。**
`app/eval/retrieval/metrics.py:37` 的 `mean_reciprocal_rank` 遍历整个 `retrieved`；而 `app/eval/retrieval/channels.py` 返回的 `retrieved_ids` 长度不超过 `rerank_config.top_k`（=5，`dense/sparse/hybrid_search` 都按 `limit=top_k` 取），`mrr@5` 取的是 `retrieved[:5]`。长度 ≤ 5 时两者输入逐元素相同 → 逐题相等。
消融链路里这个重复被 `run_ablation._compute_at5()` 的 `endswith("@5")` 过滤顺手丢掉了，但 `run_agent_eval` / `run_multi_user_eval` 走的是 `compute_metrics` 原样输出，那里两个字段并存。

**② 单 gold 题上 `hit@k ≡ recall@k`。**
`|gold| = 1` 时 `|gold ∩ top-k| ∈ {0,1}`，于是 `recall@k = |gold∩topk| / 1`，恰等于 `hit@k = 1[命中]`。题集实测：**278/300 是单 gold**（C语言 / 数据结构 / 操作系统 / 组成原理 各 50 题全单 gold；数据库 41 单 + 9 双；计算机网络 37 单 + 13 双）→ 六本教材里有四本的 `hit@k` 整列是 `recall@k` 的逐字复制。

**③ 单 gold 题上 `mrr@k` 与 `ndcg@k` 对配置排序严格同序。**
两者都只依赖首个命中的排名 `r`：`mrr@k = 1/r`，`ndcg@k = 1/log2(r+1)`（`r > k` 时为 0）。`1/r` 与 `1/log2(r+1)` 在 `r` 上严格单调 → 同一题内，任意两个配置的大小关系两指标完全一致，差异只在跨题聚合的非线性形状。只有多 gold 题（22 题，7.3%）才有实质分歧。

**④ `recall@1/@3/@5` 三档只消费了 @5。**
`run_ablation._compute_at5()` 显式剔除其余档位。实测 2026-09-18 那份六教材报告：`Recall@5` 出现 22 次、`Hit@5` 17 次、`NDCG` 14 次、`MRR` 21 次，**@1/@3 档一次都没出现**。

> 合并结论：旧指标体系的信息量 = { `|gold ∩ top-k|` , `rank(first hit)` }。k 只取 5 一档，其余档位与其余四项指标都在重复表达这两个量。

### 2.2 题集口径决定的结构性盲区（旧指标补不上的那条轴）

`qa_set.json` 里写明：

> `gold_policy`: 直接证据集合；每题保留支持答案要点的主证据 chunk，**未做全库穷举相关性标注**。

含义：一个 chunk 可以「对回答这道题有用」却不在 `gold_chunk_ids` 里。于是任何「返回结果里有多少比例真相关」的指标都被系统性压低——分母对，分子漏了未标注的真相关。旧实现的处理是**干脆不算精度**，代价是报告只能说"gold 有没有被检索到"，说不出"返回的另外 4 条是不是噪声"。而后者用户体感更直接：噪声进上下文 → 干扰生成、白烧 token。

这一条**不能靠重新标注解决**（300 题全库穷举不现实），只能换判据：让 LLM 直接判「这条 chunk 对回答这个问题有没有用」，把「是否被标注过」和「是否相关」解耦。这是本版新增语义指标的主要理由。

***

## 3. ragas 事实核对（0.4.3）

以下口径来自官方文档 + 源码逐行核对（`evaluation.py` / `dataset_schema.py` / `metrics/_context_precision.py` / `metrics/_context_recall.py` / `executor.py` / `run_config.py`，已确认这些文件在 v0.4.3 tag 与 main 一致）。

### 3.1 版本与依赖

- 最新版 **0.4.3**（2026-01-13），`requires-python >= 3.9`，仓库已迁至 `vibrantlabsai/ragas`。
- 增量依赖（核心，非 `all` extra）：`numpy<3`、**`datasets>=4.0.0`（拖 pyarrow/pandas/fsspec/huggingface-hub）**、`tiktoken`、`pydantic>=2`、`nest-asyncio`、`appdirs`、`diskcache`、`typer`、`rich`、`openai>=1.0`、`tqdm`、**`instructor`**、`pillow`、`networkx`、`scikit-network`、`langchain`、`langchain-core`、`langchain-community`、`langchain_openai`。
- **⚠️ 依赖实测：直接装最新的 `import ragas` 会失败，必须钉 `langchain-community<0.4`。**
  ragas 在 `src/ragas/llms/base.py` 模块级硬 import `from langchain_community.chat_models.vertexai import ChatVertexAI`，而该模块在 `langchain-community 0.4.x` 里已被删除（0.4.2 的 `chat_models/` 下只剩 `google_palm.py`；`llms/vertexai.py` 倒还在）。
  实测三条：

  | 组合 | 结果 |
  | --- | --- |
  | ragas 0.4.3 + langchain-community 0.4.2（uv 默认解） | ❌ `ModuleNotFoundError: langchain_community.chat_models.vertexai` |
  | ragas 0.4.3 + **langchain-community 0.3.31**（钉 `<0.4`） | ✅ `import ragas` 成功 |
  | 钉版后其余依赖 | 不变：langchain 1.3.14 / langchain-core **1.5.2** / langchain-openai 1.4.1 |

  即：**不是版本冲突，是上游没跟上 langchain-community 的日落**。解析能过（`uv` 不会报错），但 import 才炸——所以第 8 节的验收里必须有一条 `import ragas` 冒烟，光看 `uv lock` 成功是发现不了的。ragas 对 langchain 系依赖**不写上下界**，这类问题会再发生（见第 10 节）。
- 需要在 `pyproject.toml` 显式补 `rapidfuzz`（仅当选 `NonLLMContextPrecisionWithReference`，见 4.4）。
- **`uv lock` 后必须提交 `uv.lock`**：CI 走 `uv sync --locked`，不同步直接红。

### 3.2 数据模型

**旧路径**才有样本类（`SingleTurnSample` / `EvaluationDataset`，字段全 Optional）。**本方案不用**——collections 指标直接吃 kwargs，样本就是我们自己的 dict/dataclass。旧路径的样本字段列在这里仅供对照：`user_input / retrieved_contexts / reference_contexts / retrieved_context_ids / reference_context_ids / response / reference …`，我们在 v2 需要 `response` 时用的就是这个名字。

### 3.3 可用指标的确切口径

> 下面用**旧路径**（`ragas.metrics`）的类名列表，因为它们的口径已被逐行核对过；本方案实际采用的是第 4.1 节四个 `ragas.metrics.collections` 类（**口径相同**，且不吃 deprecation 告警）。这里保留全表是为了说明"为什么选这四个、排除其余"。

| 指标 | 读取字段 | 判据（源码级） | 顺序敏感 | 需要 LLM |
| --- | --- | --- | --- | --- |
| `IDBasedContextRecall` | `retrieved_context_ids`, `reference_context_ids` | `\|ref ∩ ret\| / \|ref\|`，两侧都先 `str()` 归一化并用 **set** 去重 | **否** | 否 |
| `IDBasedContextPrecision` | 同上 | `\|ref ∩ ret\| / \|ret\|` | **否** | 否 |
| `NonLLMContextPrecisionWithReference` | `retrieved_contexts`, `reference_contexts` | 每条 retrieved 与任一 reference 的最大字符串相似度 `>= 0.5` 记相关，按序算 **AP**（需 `rapidfuzz`） | 是 | 否 |
| `NonLLMContextRecall` | `retrieved_contexts`, `reference_contexts` | 每条 reference 的最大相似度 `> 0.5` 记命中，命中数 / reference 条数 | 否 | 否 |
| `LLMContextPrecisionWithReference` | `user_input`, `retrieved_contexts`, `reference` | LLM 逐条判「该 context 是否有助于得出 reference」，按序算 **AP** | 是 | **是** |
| `LLMContextRecall` | `user_input`, `retrieved_contexts`, `reference` | LLM 把 reference 逐句判「能否归因到拼接后的 retrieved」，命中句数 / 总句数 | 否 | **是** |
| `LLMContextPrecisionWithoutReference` | `user_input`, `response`, `retrieved_contexts` | 同 AP，但用生成的 `response` 替代 reference | 是 | 是 |

### 3.4 四个必须知道的坑

1. **没有 `k` 参数，ID 类还完全不看顺序。**
   所有指标都在传入的整个列表上算。`IDBasedContextRecall` 走的是集合交并，**重排列表不改变它的值**。所以「@k」不是指标参数，而是**构造样本前自己做的切片**。这是本版最核心的一处口径差异。
2. **`IDBasedContextPrecision` 不是 AP，不能当排序指标。**
   它是 `|gold∩ret| / |ret|`（集合语义、不看顺序）。在本题集上它与 `IDBasedContextRecall` **严格成比例**：`precision = recall × |gold| / k`，`|gold|` 每题固定 → 逐题零信息增量；而 278/300 题的 `|gold| = 1`、`k = 5`，于是它在数值上恒等于 `recall/5`。**结论：不要用它。**（这条反直觉，若不核对源码很容易误以为它是"平均精度"。）
3. `evaluate()` 在 0.4 起带 deprecation 告警（指向新的 `@experiment` 装饰器）；`ragas.metrics` 下的旧名在 import 时发 `DeprecationWarning`，官方迁移目标是 `ragas.metrics.collections`——**但两个 `IDBased*` 类不在 collections 里**，目前只有旧路径。见第 10 节风险。
4. **指标全是非 LLM 时不会去初始化 OpenAI。** 源码里默认 LLM 的构造被 `isinstance(metric, MetricWithLLM)` 守卫，且 `from openai import OpenAI` 也在这个分支内 → 只用确定性指标时完全离线、不需要任何 API key。这条保证了 4.1 里那两个指标可以进单测和 CI。

### 3.5 组件怎么造：必须用 `llm_factory` + `AsyncOpenAI`，**不能**用 LangChain 包装

这一条推翻了本方案早先的一个假设，单列出来：

- collections 指标**只接受现代 `InstructorLLM`**。传别的（包括 `LangchainLLMWrapper`）构造时就 `ValueError: Collections metrics only support modern InstructorLLM. Found: … Use: llm_factory('gpt-4o-mini', client=openai_client)`。所以**不能**复用 `app/clients/llm.py` 的 `get_llm_client`，得另起一条构造路径。
- `llm_factory(model, client=...)` 的 **`client` 是必填**（`ValueError: llm_factory() requires a client instance. Text-only mode has been removed.`）。
- **必须是 async client**：四个指标的 `ascore` 都只调 `llm.agenerate`，而 Agenerate 在同步 client 下直接 `TypeError`。所以用 `AsyncOpenAI`。
- 供应商走 OpenAI 兼容端点即可：`AsyncOpenAI(base_url=llm_config.base_url, api_key=llm_config.api_key)` + `llm_factory(model, client=client)`（provider 默认走 `openai`）。百炼与 DeepSeek 都是兼容端点，**但这条组合只做到"机制上成立"，未实测**——接的时候先拿一条样本打通。
- ⚠️ **embeddings 的静默回退**：`embedding_factory` 在**不传 `client`** 时会悄悄构造 `langchain_openai.OpenAIEmbeddings`（默认模型 `text-embedding-ada-002`，读环境变量 `OPENAI_API_KEY`），只发一条 DeprecationWarning。**必须显式传 `client=`**，或者别用它——我们的 BGE-M3 是 FlagEmbedding 加载的，不是 sentence-transformers，多半要自己实现一个 `BaseRagasEmbedding` 子类包一层（v2 才需要，v1 用不到）。

### 3.6 两套 API：旧 `evaluate()` 与新 collections（本方案走新路）

| | 旧路径 | **新路径（本方案采用）** |
| --- | --- | --- |
| 指标 | `ragas.metrics.LLMContextRecall` 等 | `ragas.metrics.collections.ContextRecall` 等 |
| 样本 | `SingleTurnSample` / `EvaluationDataset` | 不需要样本类，直接传 kwargs |
| 调用 | `aevaluate(dataset, metrics)` | 逐个 `ascore(**kwargs)`，批量 `abatch_score(list[dict])` |
| 并发 | `RunConfig.max_workers`（默认 16，asyncio 信号量），超时 180s，失败转 `np.nan` | 见下方警告 |
| 结果 | `EvaluationResult`（`scores` / `to_pandas()` / `total_tokens()` / `total_cost()`） | `MetricResult`（逐条，取 `.value`） |
| 状态 | `evaluate()` 带 deprecation 告警 | 0.4 的迁移方向 |

**两套不互通**（实测）：collections 的 `BaseMetric` 有 `ascore` / `score` / `abatch_score` / `batch_score`，但**没有 `single_turn_ascore`** —— 而 `aevaluate()` 正是靠那个方法分发。所以 collections 指标**塞不进 `aevaluate()`**。

选新路径的理由：四个主口径都在 collections 里；省掉 `EvaluationDataset` + HF `Dataset` 两层转换（连带省掉 `datasets`/pyarrow 在运行期的实际使用）；避开弃用告警。

⚠️ **`abatch_score` 没有并发上限**。源码是把每个输入建成 `ascore` task 后一次性 `asyncio.gather`。300 题一把扔进去 = 300 路并发 LLM 调用。实施时必须确认清楚（必要时自己分批，比如每 16 条一批），别等撞限流才发现。

`@experiment` 装饰器（`ragas.experiment` 本身就是那个装饰器）是 0.4 推荐的**工作流**层封装（带结果落盘后端），本方案不用——我们的 experiment 目录结构（5.3）已经定义好了，多引一层反而不好对齐。

***

## 4. 指标重设计

### 4.0 框定：检索看「找得对不对」，生成看「说得对不对」

| 阶段 | 关注 | 两个指标 |
| --- | --- | --- |
| 检索 | 找到的文档**对不对、全不全** | Context Precision（相关的是否排前面、无关的是否混进来）· Context Recall（回答所需信息是否都在） |
| 生成 | 基于找到的文档，**话说的对不对** | Faithfulness（每句是否能在上下文里找到依据 → 直接反映幻觉）· Answer Relevancy（是否正面回答问题） |

这个四件套是标准框架，本版**照它重排**，不再自造指标名。

### 4.1 四个指标在 ragas 里的对应物（签名为本机实测）

| 阶段 | ragas 类（`ragas.metrics.collections`） | `ascore(...)` 参数 | 判据 | 每样本 LLM 调用 | 额外依赖 |
| --- | --- | --- | --- | --- | --- |
| 检索 | `ContextPrecisionWithReference(llm=…)` | `user_input, **reference**, retrieved_contexts` | LLM **逐条**判「该 context 是否有助于得出 reference」→ 按序累加 **AP** | **len(retrieved_contexts)**（k=5 即 5 次） | LLM |
| 检索 | `ContextRecall(llm=…)` | `user_input, retrieved_contexts, reference` | LLM 把 reference **逐句**判「能否归因到 retrieved」 | 1 | LLM |
| 生成 | `Faithfulness(llm=…)` | `user_input, response, retrieved_contexts` | 拆 statements（1 次调用）→ 一次调用对全部 statements 判「能否在 retrieved 里找到依据」 | 2 | LLM（**不需要 reference**） |
| 生成 | `AnswerRelevancy(llm=…, embeddings=…, strictness=3)` | `user_input, response` | 用 response 反生成 3 个问题 → 与原文算余弦相似度取均值；**全部**判定为 noncommittal 时归零 | 3（= strictness） | LLM + **embeddings** |

⚠️ **`reference` 的位置在两个指标里不一样**（ContextPrecision 是第 2 个、ContextRecall 是最后一个）。**一律用关键字调用**，别靠位置。

⚠️ **ContextPrecision 的调用数是按 context 数走的**，不是 1 次/样本。k=5 就是 5 次。这条直接决定预算（见 4.1.1）。

四个都在**新路径** `ragas.metrics.collections`（不吃 deprecation 告警），返回 `MetricResult`。

三个细节：

- **Context Precision 必须用 `WithReference` 变体**：另一版（`ContextUtilization`）用生成的 `response` 当基准，v1 没有 response。
- **Answer Relevancy 要 embeddings**，而 `embedding_factory` 不传 `client` 会**静默回退到 OpenAI**（见 3.5）。我们的 BGE-M3 是 FlagEmbedding 加载的，不是 sentence-transformers，所以大概率要自己实现一个 `BaseRagasEmbedding` 子类包一层。
- 新旧路径的**打分公式、提示词、阈值一致**（逐模块源码对照，旧版只多一层 ensembler 包装）；但「同输入数值完全一致」属**源码推断、未实测**。所以第 8 节的迁移锚点只能靠确定性护栏口径，不能靠 LLM 口径。

### 4.1.1 预算：按实测的每样本调用数重算

| 轮 | 每样本调用 | 300 题（1 教材） | 300 题 × 6 教材 |
| --- | --- | --- | --- |
| v1，只跑 2 个代表配置 | CP 5 + CR 1 = **6** | 1 800 | 10 800 |
| v1，若 8 配置全跑 | 同一个 6 | 1 800/配置 | 43 200 |
| v2 生成侧（FA + AR） | 2 + 3 = **5** | 1 500 | 9 000 |

这比早先按「300 题 × 2 指标 = 600 次」的估算**高了 3 倍**——错在把 `ContextPrecision` 当成 1 次调用，它其实是**按 context 数**走的。

结论：**8 配置全跑 LLM 口径不现实**（4.3 万次）。4.3 里「护栏口径全跑、LLM 口径只跑代表配置」因此从"省钱"升级为**硬约束**。

### 4.2 生成侧两个这一版跑不了——缺 `response`

Faithfulness 和 Answer Relevancy 的输入都是**系统真正生成的答案**，而评测集里只有 `reference_answer`（人写的参考答案）。跑检索评测不产生答案。

要补答案，两条路，取舍要提前定：

| 路径 | 真实度 | 代价 |
| --- | --- | --- |
| 走 `run_agent_eval` 的 SSE 链路让 Agent 真答 | 高（就是线上行为） | Agent 每轮自己决定检索几次，**它用到的上下文与被评的检索配置不是一回事**；样本必须记录"那一轮实际用到的上下文" |
| 固定检索配置 + 固定生成 prompt 生成一次 | 低（不是线上行为） | 可控、可复现，但结论只能代表那个 prompt |

仓库里**已经有带答案的历史产物**（C语言 `agent_results.json` 50 条、多轮 `multi_user_agent_results.json` 200 条，都有 `answer_content`），可以直接当 v2 的第一批样本。但它们只存了 `retrieved_chunk_ids`、**没有上下文正文**，而 Faithfulness 恰恰要拿「答案 vs 它那轮的上下文」比——所以 v2 还有一步「按 id 反查正文」。注意：**git 里的 `gold_contexts/` 只含 gold chunk**，反查任意 retrieved id 需要全量语料转储（`data/eval/chunks/`，不入库）或直接查 Milvus。

### 4.3 v1（检索侧）：对外 2 个指标，每个配一条确定性口径

| 对外指标 | 主口径（LLM，语义） | 护栏口径（确定性，免费） |
| --- | --- | --- |
| Context Precision | `ContextPrecisionWithReference`（AP） | `MRR@5` |
| Context Recall | `ContextRecall`（逐句归因） | `IDBasedContextRecall@5`（gold chunk 命中） |

护栏口径不是「又多两个指标」，而是同一根轴上的第二把尺子，理由是实的：

- **能进 CI**：无 LLM、无网络、无 Milvus，`tests/` 里可以真跑。
- **8 个配置全跑不花钱**：LLM 口径是 6 次调用/样本（4.1.1），8 配置 × 6 教材要 4.3 万次——现实上只够跑 2 个代表配置。护栏口径零成本，全跑。
- **跨版本可比**：LLM 判据会随评审模型/温度/抽样漂移。没有确定性锚点时，指标变了分不清是**系统变了**还是**尺子变了**。
- **迁移有硬锚点**：`IDBasedContextRecall@5` 逐题等于旧 `recall@5`、`MRR@5` 逐题等于旧 `mrr@5`（见第 8 节）。

**两条口径怎么一起读**（说清楚，免得当成可分解的等式）：它们**不构成不等式关系**，不能拿差值直接减出「未标注但相关」的占比。它们的用法是**定性对照**——若 LLM 精度整体远高于确定性精度，提示"未标注但相关"普遍存在；若两条口径**对配置的排序**不一致，说明有值得深挖的个案。真正的诊断在逐题明细（第 7 节）。

**Context Precision 这条轴为什么必须靠 LLM**：题集的 `gold_policy` 明说「未做全库穷举相关性标注」，所以任何基于 gold 的精度指标都会把"未标注但真相关"的内容当噪声惩罚——「无关的是否混进来」这半个问题在本题集上**没有可靠的确定性判据**（`IDBasedContextPrecision` 也不行，见 4.5）。LLM 逐条判「对回答有没有用」是唯一绕开标注完整性的口径。

### 4.4 Context Recall 的分辨率问题（实测）

`ContextRecall` 是拿 `reference` **逐句**判归因，所以得分档数 = 参考答案的句数。本题集实测：

| `reference_answer` 句数 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 题数 | 32 | 145 | 101 | 20 | 2 |

中位 **2 句**，均值 2.38 → 大多数题目的得分只能落在 2~3 档上，分辨率偏粗。

备选判据：`answer_points`（答案要点清单）每题 **3~4 条**（3 条 201 题 / 4 条 99 题，均值 3.33），比散文句更细、且每条是原子断言，更适合做归因判定。用法是把它拼接后当 `reference` 喂给同一个指标。

**建议先按 `reference_answer` 跑**（忠于 ragas 标准用法，外部可比），跑完看 `ContextRecall` 的分数分布：如果大量挤在 0 和 1，就切到 `answer_points`。这是一个有数据支撑的开关，不是现在拍脑袋。

### 4.5 与旧指标体系的关系

旧 12 个数（`recall@1/3/5`、`hit@1/3/5`、`ndcg@1/3/5`、`mrr@1/3/5`、`mrr`）的信息量只有两个自由维度（第 2 节有证明），在新框架下归位为：

| 旧指标 | 去向 |
| --- | --- |
| `recall@5` | → Context Recall 的护栏口径 `IDBasedContextRecall@5` |
| `mrr@5` | → Context Precision 的护栏口径 `MRR@5` |
| `hit@1/3/5` | 删：单 gold 题上与 recall 逐字相等（278/300 是单 gold） |
| `ndcg@1/3/5` | 删：单 gold 题上与 MRR 对配置同序；无分级相关性时无额外信息 |
| `recall@1` / `recall@3` | 删：从未被任何报告消费；k 现在是样本构造参数（4.6），要哪档切片即可 |
| `mrr`（无截断） | 删：输入已被 top_k 截断，与 `mrr@5` 逐题相等 |

### 4.6 k 从「指标参数」变成「样本构造参数」

ragas 的指标**没有 k 参数**：`IDBasedContextRecall` / `IDBasedContextPrecision` 是集合语义、完全不看顺序；LLM 版在传入的整个列表上算。所以「@k」是**构造样本前自己做的切片**。

- 采集阶段存**生产口径下的完整有序结果**（dense/sparse/hyb 是 `top_k=5` 条；深度路径 `candidate_pool=20 → top_k=5`，可存融合候选全量）。评估阶段按 k 切片，**不为换 k 重跑检索**。
- 默认 `k=5`（对齐 `rerank_config.top_k`，也保证与历史可比）。
- 可选 `k=20`：**只对深度配置出**，回答"候选池里还有多少没进前 5 的证据"。
- ⚠️ **不要为了让 k=20 有数据而把 `limit` 调大**：`hybrid_search` 内部单路召回数是 `max(limit, 10)`，limit 一变融合排序就可能变，那就不再是生产口径。

### 4.7 明文不选的

- `IDBasedContextPrecision`：它是 `|gold∩ret| / |ret|`（集合语义，**不是 AP**），在本题集上与 `IDBasedContextRecall` **严格成比例**（`precision = recall × |gold| / k`，`|gold|` 每题固定）→ 零信息增量。278/300 题的 `|gold|=1`、`k=5`，它在数值上恒等于 `recall/5`。
- `NonLLMContextPrecisionWithReference` / `NonLLMContextRecall`：Levenshtein 0.5 阈值在中文长块（700~1500 字）上的可靠性没验证，且与确定性护栏口径落在同一根轴。
- `ContextPrecisionWithoutReference`：需要生成侧 `response`，v2 才能考虑。
- `NoiseSensitivity` / `ContextEntityRecall`：前者需要 response，后者需要 LLM 抽实体且与精度轴重叠。

***

## 5. 工程结构

### 5.1 对齐官方布局：要搬的是信息架构，不是脚手架

ragas 官方给的 `rag_eval/` 模板长这样：

```
rag_eval/                     # 独立项目脚手架
├── pyproject.toml            # 自己的依赖与打包
├── rag.py                    # 被测 RAG 应用
├── evals.py                  # 评测工作流
└── evals/{datasets, experiments, logs}/
```

那是个**从零起一个项目的脚手架**——它假设 RAG 应用也写在 `rag.py` 里。我们不适用这份脚手架（应用早就存在），但它的**信息架构值得搬**：把「被测系统接缝」「评测工作流」「数据集/实验/日志」三类东西**物理分开**。逐项映射：

| 官方 | 我们的落点 | 理由 |
| --- | --- | --- |
| `rag.py`（被测应用） | `app/eval/rag/rag.py` | 应用是现成的 `app/study_agent/`，这里只能做**适配层**：把 8 个检索配置收敛成**一个函数**（`retrieve(question, textbook, config) → RetrievalResult`）。配置从"架构"降级为"参数" |
| `evals.py`（评测工作流） | `app/eval/rag/evals.py` | 装配指标 → 采集 → 评估 → 落 experiment。**唯一 CLI 入口** |
| `evals/datasets/` | `app/eval/rag/datasets/`（**已迁入，随代码入库**） | 旧位置 `data/eval/` 被 `.gitignore` 忽略，评测集看不到、改不了、review 不了。格式同步统一为 v2，见 [eval-dataset-schema.md](./eval-dataset-schema.md) |
| `evals/experiments/` | `data/eval_rag/<run_id>/`（新） | **本方案最重要的结构改进**，见 5.3 |
| `evals/logs/` | `data/eval_rag/<run_id>/run.log` | ⚠️ **绝不落顶层 `logs/`**：那是 `app/trace` 的落盘区（`logs/<日期>/metrics.log`），混进去会让 `backend/tools/` 的探针把评测日志当请求日志解析 |
| `pyproject.toml` | **不建** | 单一 uv 项目共享一份 `uv.lock`；另起项目要么上 workspace（改 CI 的 `uv sync --locked` 契约），要么维护第二份 lockfile。而评测代码本来就要 import `app.*`（channels / config / embedder），跨项目引用更别扭 |
| `README.md` | `app/eval/rag/README.md` | 面向使用者：怎么跑、产物在哪。设计文档留在 `docs/`（本文件） |
| `__init__.py` | `app/eval/rag/__init__.py` | 常规 |

### 5.2 模块划分

```
app/eval/rag/
├── __init__.py
├── README.md        # 怎么跑、产物在哪
├── rag.py           # 被测系统接缝（唯一）：retrieve() → RetrievalResult
├── evals.py         # 评测工作流入口（CLI）：装配 → 采集 → 评估 → 落 experiment
├── samples.py       # 题集加载 + gold 正文索引 + 样本构造（切 k / envelope）——纯函数，可单测
├── metrics.py       # 指标装配表 + 自定义 MRR@5
└── report.py        # 汇总表 / CSV / 控制台
```

`rag.py` 是**唯一接缝**，也是 v2 的扩展点：

```python
@dataclass(frozen=True)
class RetrievalResult:
    context_ids: list[str]      # 有序
    contexts: list[str]         # 与 ids 同序同长
    latency_ms: float

async def retrieve(question: str, textbook: str, config: str) -> RetrievalResult:
    """config ∈ {dense, sparse, hyb_50/70/80/90, rrf_hyde, rerank}（见 6.1）。
    本版只实现检索；生成侧将来在同文件加 answer()，ragas 的 faithfulness 直接接上。"""
```

它薄封装现成的 `app/eval/retrieval/channels.py`，**唯一必改**的是让底层带回 `retrieved_contexts`——Milvus hit 的 `entity["text"]` 里就有（`SEARCH_OUTPUT_FIELDS` 已含 `text`），不需要额外查询。

**复用（import，不复制）**：

- `app.eval.retrieval.channels.run_dense / run_sparse / run_hybrid / run_hyde_rrf / run_deep`：检索实现。
- `app.eval.retrieval.metrics.recall_at_k / mean_reciprocal_rank_at_k`：① 的对照实现与 ② 的实现。
- `app.ml.reranker` / 底层模型单例：预热。

**⛔ 硬约束**：`app/eval/rag/` 是唯一 import ragas 的地方，**绝不能让 ragas 出现在 `app/api` / `app/study_agent` 的导入路径上**。服务进程在 lifespan 里 `build_graph` + 连 PostgreSQL（见 CLAUDE.md），多一个 `datasets`/`tiktoken` 级别的重依赖会直接拖慢启动和 CI。

### 5.3 experiment 升为一等公民

旧结构把结果摊成一堆 `results.json`，看不出"这份报告是谁、什么时候、用哪版代码、什么口径跑的"。新结构把**一次运行 = 一个自描述的 experiment 目录**：

```
data/eval_rag/
└── 2026-10-15T1430_6books_8configs/
    ├── manifest.json          # 数据集哈希 / 配置集 / 指标集 / git rev / 评审模型 / 口径声明
    ├── samples/<config>.json  # 阶段 A 产物（可复用，见 6.2）
    ├── scores/<config>.json   # 逐题分数 + 均值
    ├── scores/<config>.csv
    ├── summary.json           # 配置 × 指标 总表（报告的主产物）
    └── run.log
```

好处：① 两次运行可比（换的是模型还是代码，看 manifest）；② **换指标重算 = 在同一个 experiment 的 samples 上开新 experiment**，不重跑检索；③ 报告里的数字永远能追到源头。

### 5.4 数据产物与口径（容易踩的三处）

**两阶段**：采集（async，需 Milvus + BGE）→ 落盘 samples → 评估（ragas，只在选中语义指标时才需要 LLM）。切开的理由见 6.2。

`samples/<config>.json` 用自造 envelope 包一层，元数据**不**进 ragas 样本：

```jsonc
{
  "schema_version": 1,
  "textbook": "...", "collection": "tb_xxxx", "config": "hyb_80", "k_default": 5,
  "captured_at": "2026-10-16T12:00:00+08:00",
  "dataset": "app/eval/rag/datasets/os.json", "revision": "2026-10-16.1",   // 可追溯到题集版本
  "retriever": {"top_k": 5, "candidate_pool": 20},
  "latency_ms": {"p50": …, "p90": …},     // ragas 不管延迟，见下
  "samples": [
    {
      "query_id": "OS-20260918-001",       // 非 ragas 字段：只在我们自己的 envelope 里
      "user_input": "为什么操作系统把进程看成…",
      "retrieved_context_ids": ["v1_tb_..."],
      "retrieved_contexts": ["# 第3章 进程描述和控制 > …"],
      "reference_context_ids": ["v1_tb_..."],
      "reference_contexts": ["# 第3章 进程描述和控制 > …"],
      "reference": "进程不是静态程序文件，而是程序在执行中的实例；…",
      "latency_ms": 127.3
    }
  ]
}
```

传给指标时只取该指标 `ascore` 声明的那几个 kwargs（collections 指标不吃多余的键）。`query_id` / `latency_ms` 这些非 ragas 字段留在我们自己的 envelope 里。

- **文本来源（关键，有实测依据）**：
  - `retrieved_contexts`：取自**实时检索返回的 hit**（`entity["text"]`），因为那正是被测系统实际看到的东西。
  - `reference_contexts`：取自 **`app/eval/rag/datasets/gold_contexts/<key>.json`**（随题集一起入库的 gold 正文），加载时逐条重算 `sha1(正文)[:12]` 与 chunk id 末段对账，**不需要 Milvus 也不需要全量语料转储**就能自证正文没被改过（见 [eval-dataset-schema.md](./eval-dataset-schema.md) 3.3 / 3.4）。题集里**不存** gold 正文副本——旧版的 `gold_evidence[].excerpt` 正是被删掉的那份：C语言与数据结构两本的 50/50 条曾被截断，而另四本没有，两份正文不一致且无人察觉。
- **正文口径**：一律用**裸 chunk 正文**。检索侧不做 `safety.wrap_untrusted`（那是给模型看的东西，不是评测对象），也不要去掉 markdown 标题路径前缀——自己清洗会引入不可比的差异。
- **边界**：**ragas 管质量，采集脚本管成本。** 延迟不进 ragas（它没有这个概念），由采集阶段用 `time.perf_counter()` 记录，落在 envelope 里单独报告。也不要为离线评测去接 `app/trace`：无请求上下文时 `mark_stage` 是静默 no-op，接了也没数据，反倒制造"这里有观测"的错觉。

***

## 6. 运行口径与攒批器

### 6.1 配置集合

保留原 8 配置以维持与 2026-09-18 报告的可比性，但改成注册表 + 命令行筛选：

| 配置 | 含义 | 延迟量级 | 语义指标默认跑 |
| --- | --- | --- | --- |
| `dense` | BGE-M3 稠密单路 | 快 | 否 |
| `sparse` | BGE-M3 稀疏单路 | 快 | 否 |
| `hyb_50/70/80/90` | Milvus WeightedRanker 加权融合 | 快 | 只 `hyb_80`（现线上快速路径） |
| `rrf_hyde` | 混合召回 + HyDE → RRF(k) 融合 | 中 | 否 |
| `rerank` | 融合候选 → 交叉编码精排（加权融合） | 慢（≈37× 快速路径） | 是（深度路径代表） |

- 默认跑全部 8 个配置的确定性指标（不花钱）；语义指标默认只跑 `hyb_80` + `rerank`，`--llm-configs` 可改。
- **顺带澄清一个已知不一致**：`merge_recalls.py` 的 `RRF_K = 10`，但同文件 docstring 与 `run_rewrite_ablation` 都写 k=60。重构时把常量与文档对齐，并在报告中注明实际用的 k（这个数字直接影响排名）。

### 6.2 攒批器：线上要，离线不要（实测）

`app/ml/` 的 `BatchEmbedder` / `BatchReranker` 是**动态攒批器**：把并发到达的小请求（1~6 条）攒一批，一次前向再按序拆分回填。drainer 拿到第一条后会**开一个 `_MAX_WAIT_S = 10ms` 的窗口**，用几毫秒延迟换 GPU 吞吐。

**这一段是本机实测**（RTX 5060 + BGE-M3 fp16，串行单飞 8 次取中位）：

| 场景 | 每条耗时 |
| --- | --- |
| A. 攒批器 + 串行单飞（现消融路径） | **42.9 ms** |
| B. 直通 + 串行单飞 | **20.2 ms** |
| C. 攒批器 + 8 路并发 | 总 60.9 ms，均摊 **7.6 ms** |

结论很干净：

- **并发时攒批是真收益**：7.6ms/条 对 20.2ms/条，约 **2.7×**。所以线上路径（`/study/chat` 并发、`run_multi_user_eval` 专门验证这个）**必须保留**它。
- **串行时是纯税**：没有第二个请求会进来，10ms 窗口必然空等，实测差 **22.8 ms/次**（窗口 10ms + 事件循环在队列/超时路径上的额外一轮调度）。
- **它对评测的污染是双重的**：旧消融每题 8 次 embedding + 1 次精排都走这条路，`latency_s` 里混进了 ~23ms 的窗口税。最直接的证据：旧报告的快速路径平均 **0.127 秒**，其中约 **18% 是攒批窗口**，不是检索耗时。整轮 6 教材消融还多花了约 **328 秒**纯等待。
- 这个税在**线上不算错**（用户确实要等这 10ms，换的是吞吐），但它**不是单飞延迟**，不该混进"检索耗时"这个说法里。

**改造（最小、线上零行为变化）**：给 `BatchEmbedder.encode` / `BatchReranker.encode` 加一个攒批窗口参数，drainer 用**首条请求**带来的窗口值；默认值仍是 10ms，`agenerate_embeddings` / `arerank_scores` 的线上语义一个字不改，另外导出 `agenerate_embeddings_nowait` / `arerank_scores_nowait`（窗口 0）给离线用。

两个实现细节值得写进注释，因为容易做错：

1. drainer 在应用窗口前应先 `get_nowait()` 把**已经在队列里**的请求吃掉——直通模式仍能合并同刻到达的请求，不会退化成"每次前向一条"。
2. 直通入口**仍必须走同一个 drainer**，不能直接 `asyncio.to_thread(generate_embeddings, ...)`。`app.ml.embedding.generate_embeddings` 的线程安全是靠"drainer 单任务串行调用"保证的（锁已按这个前提移除），绕开它去并发调用就是 GPU 并发前向。

**备选（若不想动共享代码）**：离线脚本自己 `await asyncio.to_thread(generate_embeddings, texts)` + 自加锁。更简单，但把"同一个东西两条路径"的问题留下来了，且要自己维护单飞保证。**推荐前者。**

### 6.3 质量与延迟必须分两轮跑

攒批的实测把一件旧事说清楚了：**同一轮运行里同时产出"质量"和"延迟"，两边都不讨好**。质量轮想并行（6 教材 × 8 配置 × 300 题，深度路径单题约 3s，串行要按小时计），并行就会共享 GPU、互相排队，延迟数字立刻失真。

所以拆成两轮，**报告里也分成两张表，不混**：

| 轮 | 目标 | 并发 | 攒批 | 规模 |
| --- | --- | --- | --- | --- |
| `--mode quality`（默认） | 4 个质量指标 | 有（信号量限流） | 走攒批器（此时是收益） | 全部 300 题 × 8 配置 |
| `--mode latency` | p50/p90/分阶段耗时 | 严格串行单飞 | **nowait 直通** | 30 题子集足够（延迟是分布不是逐题事实） |

延迟轮只跑 30 题是因为延迟说的是"一个用户等多久"，与题目数无关；这也让"想测延迟就得再跑一轮"的代价可接受。

> 一个必须记下的副作用：**GPU 攒批会改变浮点结果**（padding 与 batch 组成不同 → 前向数值有微小差异），因此并行质量轮与串行延迟轮的 top-5 理论上可能有一两条不一致。top-5 层面预计无影响，但不要拿两轮的检索 id 做逐条对比。

***

## 7. 报告形态

三层，都进 `summary.json` 与 `report.py` 的控制台输出：

1. **总表**：行 = 指标（4 个 × k 档），列 = 8 配置，格 = 300 题等权均值。
2. **配置结论**：确定性指标 + 延迟（**来自延迟轮**）并列，做质量-成本对照（如 `hyb_50` vs `hyb_80`、`rrf_hyde` vs `rerank`）。语义指标单独一张表并注明只在哪几个配置上跑过，**不要**和全量确定性指标混进同一张表，免得读者以为都跑全了。
3. **逐题明细**：`query_id` / 各指标分数 / 检索到的 id 与延迟；语义指标附 LLM 的逐条 verdict（能直接看出"哪几条被判噪声"），这是人工复盘最有用的一层。

报告中必须写死的口径声明（照抄旧报告的严谨度）：

- ① 是「**直接证据召回**」，不是「全部相关证据召回」，受 `gold_policy` 限制；
- ③④ 是 LLM 判据，标明评审模型、温度、日期与调用次数；
- 延迟来自**另一轮串行单飞 + nowait 直通**的测量，与质量轮不是同一次运行；
- 实际使用的 `RRF_K` 值。

***

## 8. 迁移与验收

每个阶段独立可 review、可回滚。

| 步 | 动作 | 验收 | 状态 |
| --- | --- | --- | --- |
| -1 | 评测集迁到 `app/eval/rag/datasets/` 并统一为 v2 格式 | 300 题 / 302 条 gold 正文全部自校验通过（见 [eval-dataset-schema.md](./eval-dataset-schema.md) 第 6 节） | **已完成 2026-10-16** |
| 0 | `uv add ragas "langchain-community<0.4"` + `uv lock`，提交 `uv.lock` | CI `uv sync --locked` 过；**`import ragas` 必须成功**（这是唯一能发现 3.1 那个上游 import bug 的检查——解析通过不等于能 import）；CI 加一条 import 冒烟 | |
| 1 | 攒批器改造：窗口参数化 + `*_nowait` 入口（第 6.2 节） | 线上入口行为零变化（默认窗口 10ms 不动）；`tests/ml/` 钉住"直通不等窗口、仍走同一 drainer" | |
| 2 | `samples.py`：题集加载 + gold 正文索引 + 样本构造（纯函数） | 单测：`schema != 2` 拒收；gold id 缺失、正文哈希对不上、`gold_contexts` 与题集脱节，三者都要报错 | |
| 3 | `channels.py` 补 `retrieved_contexts`（只加键，不改返回结构） | 现 `run_ablation` 与新接入方都不回归 | |
| 4 | `rag.py` + `evals.py` 阶段 A，先跑两条护栏口径（`IDBasedContextRecall@5` / `MRR@5`） | **数值回归锚点**：逐题等于旧 `recall@5` / `mrr@5`（对同一份旧结果文件比对，误差 0）。不等就是切片或 id 归一化有问题 | |
| 5 | 接两条 LLM 主口径（`ContextPrecisionWithReference` / `ContextRecall`），先在 `hyb_80` 单配置小样本（如 20 题）打通；**跑完看 `ContextRecall` 的分数分布**，若大量挤在 0/1 就切到 `answer_points` 当 reference（4.4） | LLM verdict 抽样人工核对 10 条；确认 `MetricResult.value` 的取值口径；**确认 `abatch_score` 的并发行为并定下我们的分批大小**（3.6 的警告） | |
| 6 | `--mode latency` 与 nowait 直通接上 | 与质量轮分表；快速路径延迟应比旧报告的 0.127 秒**明显低**（扣掉了 ~23ms 窗口税），差值与实测税同量级即为正确 | |
| 7 | 全量 6 教材 × 8 配置重跑；出报告 | 与 2026-09-18 报告对比，确定性指标差异只应来自检索侧随机性 / `RRF_K` 口径澄清 | |
| 8 | 旧入口退役 | `run_rewrite_ablation.py` 删除（依赖已下线链路，README 已承认新日志样本为 0）；`run_ablation.py` 标记废弃并在文件头指向新入口；`metrics.py` 降级为"对照实现" | |

**不要**在第 4 步之前删任何旧代码——数值回归锚点要靠旧实现。

***

## 9. 测试策略

遵循仓库既有约定（不用 `unittest.mock`、`tests/` 镜像 `app/`、每级 `__init__.py`、中文断言说明）。新增 `tests/eval/rag/`：

- `test_samples.py`：纯函数，**不依赖 ragas**。快照索引、切 k、id 归一化、envelope 字段、sha256 失败路径。
- `test_metrics.py`：**用真实 ragas 的确定性指标跑小样本**（无 LLM、无网络、无 Milvus）钉住口径——特别是钉住 `IDBasedContextRecall` 与 `recall_at_k` 的等价性，以及自定义 `MRR@5` 与 `mean_reciprocal_rank_at_k` 的等价性。这是本方案最重要的一条测试。
- `test_rag.py`：`rag.py` 的 8 配置注册表装配；检索函数用 `_Fake*` 桩掉（`monkeypatch` 打在 `app.eval.retrieval.channels` 里被引用的符号上）。
- 攒批器改造：`tests/ml/` 钉住窗口参数化的行为（nowait 不等窗口、并发仍合并、仍走同一 drainer）。
- 语义指标：只测**装配**（哪些配置选中、样本字段齐不齐），不测 ragas 内部；LLM 用 `GenericFakeChatModel` 走通链路即可。
- 需要真 Milvus/LLM/BGE 的端到端留一条 `@pytest.mark.integration`。

⚠️ 代价：CI 的 pytest job 从此要装 ragas（拖 pyarrow、tiktoken 等），安装时间与缓存体积上升。这是本方案对 CI 的主要副作用，实施前先估算一次。

***

## 10. 风险与取舍

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| **上游已断：ragas 硬 import 已在 langchain-community 0.4 删除的 `chat_models.vertexai`** | 装最新依赖时 `import ragas` 直接失败；而 `uv lock` 不报错，构建阶段看不出来 | 钉 `langchain-community<0.4`（实测 0.3.31 可用，其余依赖不变）；CI 加一条 `import ragas` 冒烟。上游修复后可以摘掉这个钉版 |
| ragas 对 langchain 系依赖不写上下界 | 上述这类断裂会再发生 | `uv.lock` 入库锁死 + 上面那条 import 冒烟；升级 ragas 时先跑冒烟 |
| `IDBasedContextRecall` 不在 `ragas.metrics.collections`，旧路径在 0.4 起带 deprecation 告警、v1.0 可能移除 | 确定性护栏口径可能失效 | 指标装配收敛在 `metrics.py` 一张表里，换实现只改一处；备选是自定义 metric 复制同样口径（15 行）或退到 `NonLLMContextRecall` |
| LLM 主口径按配置数线性花钱（`ContextPrecision` 是 **len(contexts) 次/样本**） | 8 配置全跑 6 教材 ≈ **4.3 万次**调用 | 硬约束：LLM 口径只跑 2 个代表配置（≈1.1 万次），其余配置靠确定性护栏；样本落盘使换模型/重算不重跑检索 |
| `abatch_score` / `@experiment` 都是**无上限全量并发**，且失败条目被静默吞掉 | 撞限流、结果缺行而不知 | 自写分批（Semaphore）而不是直接用 `abatch_score`；每轮结束**核对返回行数 == 输入行数**，缺行要显式列出 |
| collections 指标只能用 `llm_factory` + `AsyncOpenAI`，不能复用 `app/clients/llm.py` | 评审模型的构造与线上不一致，参数（如思考开关）要另配 | 单起一条构造路径，参数从 `llm_config` 读；先拿一条样本打通再批量 |
| LLM 评审与题集作者可能同族 → 相关性偏差 | 语义指标偏乐观 | 评审模型选与建集不同的模型/供应商；报告里标明；LLM 口径永远与确定性口径**并列**读，不单独下结论 |
| `AnswerRelevancy` 对短答案的偏差：**本方案早先说它"已知系统性偏低"，核源码后不成立**——公式是 `cosine_sim.mean() * int(not all_noncommittal)`，没有长度项，`strictness` 恒为 3。旧版 docstring 只有一句"信息不完整/冗余会被惩罚"，与长度无关 | 无（原判断撤回） | v2 实测分数分布后再下结论，别先按"短答案低分"解读 |
| 改攒批器触碰线上热路径 | 并发吞吐回退 | 默认窗口值不变、线上入口签名不变；`*_nowait` 是新名字不是改旧名；有单测钉行为 |
| 题集是 agent 起草、`review_status: ...pending_human_validation` | 确定性口径的分母本身可能不准 | 本版不解决；报告的结论措辞保持"相对比较"（配置间对比）而非"绝对质量" |
| 丢弃 NDCG | 将来若有分级相关性标注，无指标可承接 | `metrics.py` 里 `ndcg_at_k` 纯函数保留；需要时按 ragas 自定义 metric 接回 |
| `ContextPrecisionWithReference` 把"一个 chunk 只覆盖部分答案要点"判为不相关 | 系统性偏低 | 已知的判据粗糙处，靠逐题明细里的 LLM verdict 人工复盘暴露，不与确定性口径做跨表比较 |

***

## 附：本方案未采用的替代路线

- **保留自研指标、只借 ragas 的数据集 schema**：改动最小，但拿不到标准口径、外部可比与后续生成侧接缝，收益太薄。
- **照搬 `rag_eval/` 独立项目脚手架（含自己的 pyproject.toml）**：见 5.1——多一份 lockfile 或改 CI 契约，且评测代码要跨项目 import `app.*`。搬信息架构，不搬脚手架。
- **`LLMContextPrecisionWithoutReference`**（用生成的答案当基准）：需要先有 `response`，属于 v2 生成评测；届时与 faithfulness、answer correctness 一起上。
- **多轮 / 会话级评测走 ragas 的 `MultiTurnSample`**：`data/eval_multi_turn/qa_set.json` 有现成的多轮结构，但那是 agent 端到端评测的范畴（本轮要测的是会话理解与指代消解），留到 v2 再评估。
