# 评测集格式规范（v2）

> 状态: **已迁移（2026-10-16，revision `2026-10-16.1`）** · 适用范围: `app/eval/rag/datasets/`
> 关联: [ragas-retrieval-eval-design.md](./ragas-retrieval-eval-design.md)（检索评测重构，消费本格式）
>
> 数据集从 `data/eval/`（被 `.gitignore` 忽略）迁到仓库内，**随代码入库**。六个短键
> `c / co / db / ds / net / os` 沿用 `data/eval_multi_turn/qa_set.json` 的 `books` 键。

***

## 1. 为什么要统一

现在六本教材的 `qa_set.json` 是三套不同变体（`v2-rebuilt-20260916` / `2026-09-16.source-grounded.v1` / `2026-09-18.source-grounded.v1`），字段不一致、语义重叠、且混进了大量**作者期脚手架**：

| 病灶 | 实测 |
| --- | --- |
| 题目字段三套组合 | 8 字段（C语言、数据结构）/ 10 字段（操作系统、组成原理）/ 10 字段（数据库、网络），差异在 `topic` / `manual_validation_candidate` / `reasoning_note` |
| `difficulty` 是常量 | 300/300 题全是 `"medium-hard"`，顶层与题目级各存一份 |
| `gold_evidence[].excerpt` 是语料副本 | 占文件体积的 **79%**；且 C语言、数据结构两本的 50/50 条**被截断**（另四本没截断）——同一份正文有两个来源，其中一个还是坏的 |
| 两套 `category` 词表 | `{paraphrase, multihop, confusable, procedure}` vs `{reasoning, comparison, troubleshooting, multi_hop, calculation, sql}`；`multihop` 与 `multi_hop` 拼写还不一致 |
| 完整性锚只有 4/6 有 | C语言、数据结构两本没有 `source_snapshot_sha256`（也没有 `source_snapshot`） |
| 作者期流程记录混在数据里 | `review_status` / `manual_validation_candidates` / `manual_validation_note` / `usage` / `known_source_issues` |

**核心原则：字段由「消费者」定义，不由「作者」定义。** 每保留一个字段，都要说得出谁在读它。

***

## 2. 消费者清单

| 消费者 | 需要什么 |
| --- | --- |
| 采集阶段（`app/eval/rag/rag.py`） | `question` + `textbook` + `collection` |
| ragas 样本构造（`samples.py`） | `question`→`user_input`；`gold_chunk_ids`→`reference_context_ids`；`reference_answer`→`reference`；`reference_contexts` 由 **chunk 快照**按 id 查（不在数据集里） |
| 报告与逐题复盘 | `id` + `category`（分组）+ `question`（人眼读） |
| v2 生成侧评测 | `reference_answer` + `answer_points` |
| 完整性校验 | `source_snapshot_sha256` + `gold_chunk_ids` |
| 口径声明（报告必须引用） | `gold_policy` |

其余字段**没有消费者**，删除。

***

## 3. 目标格式

### 3.1 顶层

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema` | int | ✔ | 格式版本号，本次为 `2`。加载器据此拒绝旧格式，避免拿旧文件静默跑出新口径 |
| `textbook` | str | ✔ | 教材名，须与 Milvus 注册表一致 |
| `collection` | str | ✔ | Milvus 集合名（`tb_xxx`）；**同时决定 chunk 快照路径**（见 3.3） |
| `source_snapshot_sha256` | str | ✔ | 快照文件 `data/eval/chunks/<collection>.json` 的 sha256；加载时强制校验 |
| `gold_policy` | str | ✔ | 标注口径声明。必须随数据走，报告里要原样引用 |
| `revision` | str | ✔ | 本文件内容的修订号，格式 `<YYYY-MM-DD>.<n>`（同日多次修订用 n 区分） |
| `notes` | [str] | ✗ | 数据质量备注（原 `known_source_issues` 之类的去处） |
| `questions` | [obj] | ✔ | 题目列表 |

### 3.2 题目

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `id` | str | ✔ | 主键，形如 `OS-20260918-001`。报告与逐题明细都用它 |
| `question` | str | ✔ | **自包含**问句（评测输入不含对话历史）；→ ragas `user_input` |
| `gold_chunk_ids` | [str] | ✔ | 直接证据的 chunk 主键，1~2 条；→ ragas `reference_context_ids` |
| `reference_answer` | str | ✔ | 参考答案；→ ragas `reference`（`LLMContextRecall` / `LLMContextPrecisionWithReference` 的判据） |
| `category` | str | ✔ | **统一词表**（3.4），只用于报告分组，**不参与任何指标** |
| `answer_points` | [str] | ✗ | 答案要点清单。v1 不读；留给 v2 生成侧做要点覆盖 |

### 3.3 落盘布局：题集 + gold 正文，两份都入库

```
app/eval/rag/datasets/
├── README.md
├── os.json                 # 题集：schema/textbook/collection/sha256/gold_policy/revision/questions
├── c.json  co.json  db.json  ds.json  net.json
└── gold_contexts/
    └── os.json             # chunk_id → {chapter, section, block_type, text}，只含本题集用到的 gold
```

评测需要的 `reference_contexts` 从 **`gold_contexts/`** 取，**不写进题集文件**：

- 两者**生命周期不同**——改题只动 `os.json`，教材重灌只动 `gold_contexts/os.json`，git diff 各归各的；
- 题集里塞 gold 正文就是重蹈 `gold_evidence` 的覆辙（同一条正文两个来源）。

`source_snapshot_sha256` 保留为**出处戳**（记录 gold 是对着哪一版语料快照校验的），运行时不再是硬门禁——因为 8.7MB 的全量 chunk 转储属派生产物（可由 `python -m app.eval.retrieval.dump_chunks` 从 Milvus 再生），不入库。

### 3.4 gold 正文怎么自证没被改过

chunk id 的末段是 `sha1(归一化正文)[:12]`（`app/utils/chunk_id.py`，归一化＝换行统一、`[ \t]+`→空格、3+ 换行折成 2、strip）。所以**正文与 id 互为校验**：加载时对每条 gold 重算哈希与 id 对账，就能证明正文没漂移——**不需要 Milvus，也不需要那份全量转储**。

迁移时 302 条 gold 正文全部通过自校验；这也是允许把正文放进 git 的前提。

> 302 是**去重后**的 chunk 数（300 题共 322 次引用，20 处是两题共用同一个 chunk）。展开见 [`app/eval/rag/datasets/README.md`](../app/eval/rag/datasets/README.md)。

加载器行为（**全部是硬失败，不是警告**）：

1. 题集缺 `gold_contexts/<key>.json` → 报错；
2. 任何 `gold_chunk_ids` 在该文件里查不到 → 报错；
3. 正文重算哈希与 id 末段不符 → 报错（正文被改过或抽错了）；
4. `gold_contexts` 与题集引用的 gold 集合不吻合 → 报错（两边脱节）；
5. 本地若存在 `data/eval/chunks/<collection>.json`，再校验一次 `source_snapshot_sha256`；**文件不在就跳过**（不是错误）。

### 3.5 category 统一词表

两套词表按「检索会怎么失败」合并成 5 类（映射是确定性的，不依赖逐题人工判断）：

| 统一类 | 合并自 | 题数 | 检索上看什么 |
| --- | --- | --- | --- |
| `lexical` | `paraphrase` | 32 | 问法口语化、与教材用词不一致 → 词面/语义对齐的短板 |
| `confusable` | `confusable` | 36 | 近邻概念干扰 → 排序精度（MRR）的试金石 |
| `multi_hop` | `multihop`、`multi_hop`、`comparison` | 79 | 证据散在多处或需并列两处 → 召回与候选池深度的试金石 |
| `reasoning` | `reasoning`、`troubleshooting`、`procedure` | 107 | 单点证据但需理解，答「为什么/怎么办」→ 基础召回 |
| `computation` | `calculation`、`sql` | 46 | 需套用算法/写查询 → 能否检索到"方法/规则"原文 |

- 两处判断可以推翻：`comparison`（42 题）若想单列，从 `multi_hop` 拆出来即可；`sql`（3 题）若嫌并进 `computation` 不妥，可随数据库特有维度单列。
- 合并后最小类 32 题，分组分析仍有统计意义。
- 词表落成代码里一个常量（`app/eval/rag/samples.py`），加载时校验取值合法。

***

## 4. 示例（`app/eval/rag/datasets/os.json`，操作系统第 1 题，取自现网数据）

```json
{
  "schema": 2,
  "textbook": "操作系统：精髓与设计原理_第8版_(斯托林斯)_(z-library.sk,_1lib.sk,_.",
  "collection": "tb_a7874c71f752b140",
  "source_snapshot_sha256": "68e4c2b456c2953decbe35b7f5abac57c22c3cd4efcf09033c3398414c0738db",
  "gold_policy": "直接证据集合；每题保留支持答案要点的主证据 chunk，未做全库穷举相关性标注。",
  "revision": "2026-10-16.1",
  "notes": ["部分 chapter/section 标签跨章错位，以正文判断主题。"],
  "questions": [
    {
      "id": "OS-20260918-001",
      "question": "为什么操作系统把进程看成程序代码、数据集、当前状态和资源共同组成的活动单元？只把它理解成磁盘上的程序会漏掉什么？",
      "category": "reasoning",
      "gold_chunk_ids": [
        "v1_tb_36e0015b15_ch_d2fd4f71_sec_0b1f2c79_text_0000_447b79a9939a"
      ],
      "reference_answer": "进程不是静态程序文件，而是程序在执行中的实例；除了代码，还要有当前状态、数据集以及处理器和其他系统资源等信息。",
      "answer_points": [
        "进程是正在执行的程序实例或活动单元",
        "进程还包含状态、资源和相关数据",
        "磁盘上的程序代码本身不等于运行中的进程"
      ]
    }
  ]
}
```

一题从 8~10 个字段降到 4 个必填 + 2 个可选；整份文件从 1181KB 降到 **243KB（-79%）**，减的基本都是 `gold_evidence` 里的语料副本。

***

## 5. 字段迁移表

### 顶层

| 旧 | 处置 |
| --- | --- |
| `version`（"2026-09-18.source-grounded.v1" 等） | → `revision`（统一 `<日期>.<n>`）。原值里"生成方式"的信息并入 `gold_policy` 文字 |
| `source_snapshot`（路径） | **删**，按 `collection` 推导 |
| `source_snapshot_sha256` | 保留，**升为必填** |
| `gold_policy` | 保留 |
| `textbook` / `collection` | 保留 |
| `questions` | 保留 |
| `difficulty`（顶层） | **删**（300/300 恒为 "medium-hard"） |
| `usage` | **删**（"只向被测系统输入 question"是评测流程约定，属于文档） |
| `review_status` | **删**（作者期流程记录） |
| `manual_validation_candidates` / `manual_validation_note` | **删**（作者期流程记录） |
| `known_source_issues` | → `notes`（内容有价值，换个不绑定流程的名字留着） |
| —— | **新增** `schema` |

### 题目级

| 旧 | 处置 |
| --- | --- |
| `id` / `question` / `gold_chunk_ids` / `reference_answer` | 保留 |
| `category` | 保留，值按 3.4 重映射 |
| `answer_points` | 保留（可选） |
| `gold_evidence[]`（`chunk_id`/`chapter`/`section`/`block_type`/`anchor`/`excerpt`） | **删**。正文与章节信息都在 chunk 快照里；留存的是**截断且会漂移**的第二份副本 |
| `difficulty`（题目级） | **删**（常量） |
| `topic` | **删**（作者期笔记；分组用 `category` 已够） |
| `manual_validation_candidate` | **删**（作者期流程记录） |
| `reasoning_note` | **删**（作者的推理草稿，非评测输入） |

> 删 `gold_evidence` 的代价：数据集不再自带 gold 正文，人工复核要看快照。补偿：`app/eval/rag/` 提供一个按 `query_id` 打印 gold 正文的小工具（快照查表，十几行）。**这个代价是值得付的**——现在两处正文不一致（截断）而没人发现，正是"副本模型"的典型失败。

***

## 6. 迁移记录

**已完成（2026-10-16）**。由 `scripts/unify_eval_sets.py`（一次性脚本）从 `data/eval/<教材>/qa_set.json` 生成，产物落在 `app/eval/rag/datasets/`。旧文件**不备份、不删除**（用户要求不管历史），它们仍在 `data/eval/` 下被尚未迁移的脚本读取。

迁移校验（脚本内断言 + 独立复核，全部通过）：

| 检查 | 结果 |
| --- | --- |
| 六份题集 `schema == 2`、各 50 题、字段集合恰为 4 必填 + 2 可选 | ✔ 300 题 |
| `category` 全部落在 5 类词表内 | ✔ |
| 全局 `id` 唯一、题目内 `gold_chunk_ids` 无重复 | ✔ |
| gold 正文 302 条，逐条重算哈希与 chunk id 对账 | ✔ 全部通过 |
| `gold_contexts` 与题集引用的 gold 集合完全吻合 | ✔ |
| C语言、数据结构两本补上 `source_snapshot_sha256` 并与快照文件核对 | ✔ |

**尚未做**：`app/eval/retrieval/dataset.py` 仍读旧路径旧格式。现有脚本（`run_agent_eval` / `run_multi_user_eval` / `run_ablation` / `intent/*`）因此都还在读 `data/eval/` 下的旧文件，**没有受影响**。等 `app/eval/rag/` 的加载器写出来后再统一切换；兼容性已核对，被删字段的依赖只有两处且都不受影响：

- `run_agent_eval.py:225` 用 `question.get("answer_points", [])`（字段保留）；
- `run_multi_user_eval.py:109` 的 `topic` 取的是**多轮会话**的 topic（`data/eval_multi_turn/qa_set.json`，另一份数据），不是题目级 topic。

`scripts/build_*.py` 与 `scripts/rebuild_eval_sets.py` 是产生旧格式的一次性工具（不入库），不需要同步改；`unify_eval_sets.py` 是它们之后的新入口。

***

## 7. 不在本规范内

- **多轮集** `data/eval_multi_turn/qa_set.json`（`schema_version` / `books` / `conversations[].turns[]`）是另一套结构，服务于 agent 端到端评测。本轮不动；将来要统一时，`turns[]` 里的 `gold_chunk_ids` 应当复用本规范的 id 与快照校验约定。
- **意图集** `data/eval/intent/intent_extra.json`（`cases[]`）与检索无关，不适用。
