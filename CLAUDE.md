# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言与写作约定

- 注释、docstring、测试说明、日志、提交信息**一律中文**。docstring 讲「为什么这样设计/踩过什么坑」，不复述代码在做什么。
- 提交信息格式：标题 `主题：要点、要点`（可带实测数据与日期），正文分点写理由与证据。参考 `git log` 既有风格。
- 用户面向的内容（README、设计文档）也是中文。

## 常用命令

```bash
# 仓库根目录：中间件（milvus/etcd/minio/attu/postgresql）
docker compose -f docker/docker-compose.yml up -d

# 实际部署：compose/.env/data 在仓库外（默认 /home/ypw/docker/grad_assist，
# 该目录 .env 的 DATA_ROOT 指向仓库外的 data/）
docker compose --env-file /home/ypw/docker/grad_assist/.env \
  -f /home/ypw/docker/grad_assist/docker-compose.yml up -d

# backend/：后端安装、启动、测试（下文 app/... 均指 backend/app/...）
cd backend
uv sync                                        # 安装依赖（Python 3.13）
cp .env.example .env                           # 填入密钥；模型供应商见下
uv run python -m app.main                      # 启服务（启动时预热 BGE-M3 / Reranker）

uv run pytest                                  # 单测（默认 -m "not integration"）
uv run pytest tests/trace/test_metrics.py::test_xxx    # 单条
uv run pytest -m integration                   # 端到端（需真实 MinerU/Milvus/凭据，共 10 条，缺前置条件自动 skip）
uv run ruff check .                            # line-length 150，select E/F/W/I/UP/B

# frontend/：前端开发与构建
cd ../frontend
npm ci
npm run dev                                    # http://localhost:5173
npm run build                                  # vue-tsc 类型检查 + Vite 构建
```

- **依赖变更必须 `uv lock` 并提交 `uv.lock`**，CI 用 `uv sync --locked` 会直接失败。CI 的 ruff/uv 版本钉死（`ruff==0.16.3` / `uv==0.11.16`，见 `.github/workflows/ci.yml`），本地版本更旧时 CI 仍可能报错。
- CI 的 pytest job 起 `postgres:17-alpine` 并塞 `ALIBABA_API_KEY=dummy`：`tests/agent` 直接调用 `build_graph` 需要 dummy key；服务图的 `AsyncPostgresSaver` 改为在 FastAPI lifespan 内初始化，import 阶段不再连库。
- 评测/探针（评测在 `backend/eval/`，探针在 `backend/tools/`，在 `backend/` 下全部 `python -m` 运行）：
  - **ragas 四指标（现行，见 `backend/eval/rag/README.md`）**：`eval.rag.rag <配置>`（生成样本：检索 → 单轮生成）、`eval.rag.evals <配置>`（评测：Context Precision / Recall + Faithfulness / Answer Relevancy）。配置为 `hyb_50`（非深度）/ `rerank`（深度）。
  - 旧自研指标（待退役）：`eval.retrieval.run_ablation`（检索消融）、`run_multi_user_eval` / `run_agent_eval`（驱动**已启动服务**的 SSE 多轮评测）。
  - 意图识别：`eval.intent.run_eval` / `run_threshold_ablation`。
  - 日志探针：`tools.chat_probe`（按 `probe_questions.json` 跑多会话，读 `logs/<日期>/metrics.log` 汇总命中率曲线与压缩事件；想看压缩得用小窗口起服务，见其 docstring）。

## 架构总览

两条 LangGraph 流水线 + 一套存储（Milvus / PostgreSQL / MinIO / 本地盘），架构图与检索链路见 README。需要读多处代码才能拼出来的部分：

- **问答 Agent**（`backend/app/agent/`）：`create_agent` 单层 ReAct，动态提示词与前置理解全部走 middleware（`graph.py` 的列表就是装配全景图：意图识别 → 会话压缩 → 本轮指令 → 历史回收 → 静态壳 → 步数/工具限流/重试/HITL/成本预算护栏）。
- **教材摄入**（`backend/app/ingestion/`）：线性 6 节点 `load_textbook → split_contents → split → parse_to_md → enrich_md → split_text_and_store`，无分支；`split` 节点带人工确认**中断**（页码偏移）。
- **观测**（`backend/app/trace/`）：横切两条流水线，回调自动采集每次 LLM 调用与工具执行。
- **能力层**（`backend/app/retrieval/`、`backend/app/intent/`）：检索（dense / HyDE / RRF / 精排 / Milvus 存储）与意图识别被 Agent 与 eval 共用；能力层不得反向 import 流水线层（`tests/test_layering.py` 钉住依赖方向）。
- **开发探针**（`backend/tools/`）：`chat_probe` 等脚本不进入服务进程导入路径。
- **API**（`backend/app/main.py` + `backend/app/api/routes/`）：应用实例与生命周期在 `main.py`，`api/router.py` 只汇总路由，`api/routes/` 只做 HTTP 参数与响应；业务编排（SSE、HITL 续跑、会话/文件查询）在 `backend/app/services/`。对话与摄入两个 APIRouter 挂同一 FastAPI，各用独立 PostgreSQL schema。

## 关键不变量（违反会静默变贵/变错，不报错）

### 1. 前缀缓存纪律（整个仓库最贵的隐性约束）

提示词与消息的排布都是为「从第一条 token 起完全相同」的缓存命中服务的，改动前先想缓存：

- `dynamic_prompt` 的 system prompt **只装全局静态壳**；教材名 / 意图 / 任务目标一律由 `record_turn_instruction`（`graph.py`）作为**尾部 SystemMessage 每轮追加一次**（`lc_source="turn_instruction"` 标记，同内容不重复追加）。别把这些塞回 system prompt，也别每次模型调用重贴。
- **会话压缩只裁模型视图，不删 state**（`compaction.py`，自研件，不用官方 `SummarizationMiddleware`）：超「窗口 × 50%」（窗口默认 50w → 触发线 25w）时在 `before_agent` 生成一份回顾追加进消息流 + 推进水位线 `compacted_upto`，`awrap_model_call` 按水位线裁请求视图。三条纪律：①必须排在 `record_turn_instruction` **之前**（否则本轮指令会被折进回顾）；②摘要调用失败/产出为空就返回 None、state 一条不动（官方件会把报错写成回顾再删历史）；③触发读数取 provider 上报的 `usage_metadata.total_tokens`，字符估算（1.3 字符/token + 8k 固定开销）只作兜底且生效时告警——中文内容下 chars/4 会低估约 3 倍。
- 压缩注入的整篇回顾（`lc_source="summarization"`）**不是用户消息**：回显、会话计数、意图层历史一律用 `message_sources.is_summary_message` 排除（标记词表见 `app/core/message_sources.py`，别再抄字面量）。
- 预算降级的收尾说明是**转瞬即逝**的（不进 checkpoint、不动静态壳）。
- `safety.wrap_untrusted` 的边界随机串每次包裹新生成并**冻结进文本**；往轮检索原文回收（`tool_history.py`）是**一次性、永不回改**的。
- 判断缓存是否被打断看 `logs/<日期>/metrics.log` 里每次调用的 `cached` 与 `input_digest`（相邻调用 digest 相同即前缀未变）。

### 2. 观测单一来源：`app/trace`

- 调用方一律 `from app.trace import ...`，**不要**引 `app.core` 门面（会构成循环导入）。包内依赖方向见 `app/trace/metrics.py` 顶层 docstring。
- 每次 LLM 调用与工具执行自动进 `metrics.steps` 时间线（`LLMStep`/`ToolStep`，`tool_call_id`/`parent_id` 互相关联），覆盖 create_agent 内部、工具内部的调用，以及 `tool.ainvoke` 直调——**不要**在工具里手写新的请求级字段。
- 工具自定义指标走 `mark_tool(**fields)`（落在当前工具步的 `meta`）；阶段耗时走 `mark_stage`：工具内触发的进该步 `meta.phases`，工具外的进请求级 `stage_ms`。
- 回调 `run_inline=True`（ContextVar 要传进工具体）：回调里**只做 sha1/切片这类轻活，不能塞 I/O**。
- 请求级 `retrieved_chunk_ids` 由工具步 `meta.chunk_ids` 派生，别另存一份。
- 落盘 `logs/<日期>/metrics.log`（缩进 JSON + 空行分隔，探针按此格式解析）。`TRACE_PAYLOAD=digest|preview|full` 控制正文保真度，chars/sha1 恒定全量计算。

### 3. 工具失败处理分层（`app/agent/tools/`）

- **模型能自愈的**（文件不存在、没先读过、old_string 不匹配）→ 返回一句括号指引，不抛异常（不白耗一次重试）。
- **真故障**（路径越界、检索链路挂）→ 抛异常，交给图上的官方 `ToolRetryMiddleware` 兜成说明文本。
- 限流/重试等护栏只装在图侧（`graph.py`），工具文件里不写；**工具清单唯一来源**是 `tools/__init__.py` 的 `AGENT_TOOLS`。

### 4. 配置与 LLM 供应商

- `app/config/settings.py` 是唯一配置源：pydantic-settings，环境变量名写在 `validation_alias`；各域对象经 `app.config` 扁平导出（`llm_config` / `budget_config` / `trace_config`…），新增变量要同步补 `.env.example`。settings 在 import 时实例化一次，**运行时改值只能 monkeypatch 域对象属性**（测试即此法）。
- `app/infra/llm/llm.py` 是供应商门面：`get_llm_client(provider=...)`、`bind_structured_output(...)`。供应商差异（思考开关写法、结构化输出方式）各自关在 `llm_<供应商>.py`，契约是 `NAME / API_KEY_ENV / STRUCTURED_OUTPUT_METHOD / client_kwargs`。
- `enrich_md` 固定 `provider="alibaba"`（视觉模型只有百炼有）；agent 主体 `enable_thinking=False`。

### 5. 线程、checkpoint 与 HITL

- `thread_id = user_id:session_id`（对话）/ `user_id:task_id`（摄入）；`X-User-Id` 是**软隔离**（可伪造，非安全边界），同时充当恢复码。
- 两条流水线各用一个独立 PostgreSQL schema（避免 thread 撞车）：对话 `study`、摄入 `textbook`；表由 `AsyncPostgresSaver.setup()` 在 lifespan 里创建，连接通过 `search_path` 隔离。
- `AsyncPostgresSaver` 构造时要求当前线程有运行中的事件循环，所以图和 checkpointer 都在 `app/main.py` 的 lifespan 里 `init_graph`，不能在 import 阶段建。
- HITL 写盘确认：一轮内所有待确认调用打包进**同一个** interrupt，resume 的 decisions 数必须等于挂起调用数（`app/services/study.py` 的 `_hanging_write_count` 数 `action_requests`）；数错会抛 "Number of human decisions … does not match" 并卡死线程。
- SSE 事件协议（前端契约）：`tool`/`thought`/`token`/`ask_confirm`/`ask_clarify`/`done`/`error`；摄入侧另有 `message`/`toc`/`ask_offset`/`info`。

### 6. 提示词与行为块（`app/prompts/`）

- 提示词是 `.prompt` 文件，经 `load_prompt(name)` 读取，**带 lru_cache**——开发期不会热更新，改完要重启进程。
- 意图行为块在 `app/prompts/intent/*.prompt`，白名单在 `graph.py._INTENT_BLOCKS`；意图枚举与例句池在 `app/intent/detector.py`（三档漏斗：关键词+池核验 → embedding 例句池 → LLM 兜底）。

### 7. 外部内容边界与 artifact 分工（`app/core/safety.py`）

- 教材/网页/文件正文给模型前一律 `wrap_untrusted` 包边界；**artifact、落盘、前端回显、/history 一律用干净原文**（实时 SSE 与回显都走「优先 artifact、无 artifact 剥标记」）。
- 输出侧 `detect_prompt_leak` 只留痕告警（`output_leak_suspected`），不拦截。

### 8. 教材摄入的幂等与重灌

- 幂等判据是「目标产物**整组**存在 → 跳过」（目录/文件存在即完成，含 MinerU 下载的原子落盘），半成品会只补缺的那些。
- **已登记教材不会重新向量化**（`textbook_registry` 集合命中即跳过）；同教材换了内容要重灌时需先删注册表行（集合名 `tb_<sha1(教材名)[:16]>`）。
- chunk 主键含内容 hash（`app/ingestion/chunk_id.py`，`v1_tb_{...}_ch_{...}_sec_{...}_{block_type}_{idx}_{文本hash}`），改正文即新 ID——评测 gold 集因此能跨重灌保持稳定。
- 图片：二进制进 MinIO（`textbook/{教材}/{章}/images/`），正文内以 `【图: 简介】` 占位、`metadata_json.images` 带 URL，两者必须落在**同一个** chunk（检索侧 `search.py` 的 `_FIGURE_MARK_PATTERN` 负责回绑）。
- 产物目录布局（`textbooks/pdf-<时间戳>-<随机>/` 下）：`mineru_toc/{pdf_stem}/`、`pdf_split/`、`mineru_split/{教材名}/{章 stem}/`（`full.md` / `full_captioned.md`）。

### 9. 文件工具沙箱

`MATERIAL_ROOT = <backend>/study`，目标路径固定为 `study/{教材名}/{task_id}/`；拒绝对路径、`..`、symlink 逃逸；`edit_file` 要求本进程先 `read_file` 过（`_known_files`）。改落盘根只改 `tools/files.py` 顶部的常量。

## 测试约定（`tests/`）

- 目录镜像 `app/`，每级都有 `__init__.py`（测试间用 `tests.*` 包路径互相导入）。文件名 `test_<被测模块>.py`；模块 docstring 写清「桩了什么、钉的是什么」。
- **不用 `unittest.mock`**。主流写法：`monkeypatch` 打在**被测模块**的引用上（如 `search.search_by_query`，不是源模块）；手写 `_Fake*` 记录器；抓图装配用 `monkeypatch.setattr(graph_mod, "create_agent", lambda **kw: captured.update(kw) or "stub")`。
- 官方假件：`GenericFakeChatModel`（可造 usage/reasoning）、`InMemorySaver`、`httpx.MockTransport` 假 SSE；文件与日志类用 `tmp_path` + patch 常量（`LOGS_DIR` / `MATERIAL_ROOT`）。
- 带 lru_cache 的工厂（`get_llm_client`）要在用例前后 `cache_clear()`。
- 断言写「机制」而非精确耗时（用比较/比例），失败信息用中文说明断的是什么。
- 需要真实外部环境的一律 `@pytest.mark.integration` 并在缺前置条件时 `pytest.skip`；`tests/api`、`agent`、`trace`、`eval` **必须**能无外部服务跑过。

## 文档现状（部分陈旧，别照抄）

- `docs/image-caption-design.md`：承诺的「缺 `full_captioned.md` 时回退 `full.md`」**未实现**（缺富化副本的章节直接跳过）。
- `docs/image-chunk-offset-mismatch.md`：图片偏移部分**已修**（Plan A），「非首个子块的代码块丢失」仍是未修项。
- README 说的检索消融「显著性检验」在**旧的** `eval/retrieval/` 里不存在（只有均值对比）；`run_rewrite_ablation` 依赖已下线的重写链路，新日志样本数为 0。新的 `eval/rag/` 有配对 bootstrap 检验（见其 README 的实测结论）。
- `docs/java-gateway-design.md` 是**设计中**（未实现）的方案。
- `docs/ragas-retrieval-eval-design.md` 是评测重构的**设计过程稿**（含 ragas 0.4.3 的踩坑记录：`langchain-community` 必须钉 `<0.4`、collections 指标不能用 `aevaluate`、供应商吞吐顶棚等）；落地后的操作口径以 `eval/rag/README.md` 为准。
- `docs/eval-dataset-schema.md` 是评测集格式规范（已落地，`eval/rag/datasets/`）。

## 仓库边界

单仓形态：`backend/`（FastAPI + LangGraph 后端）与 `frontend/`（Vue3 + TypeScript，Vite 构建，走 `X-User-Id` + `/study/chat` SSE）**均入库**；`docker/`、`docs/`、`.github/` 服务于整个仓库。

以下本地运行数据**不入库**（见 `.gitignore`），改它们不会进 commit：`backend/data/`（评测集与结果）、`backend/textbooks/`、`backend/study/`（助手产物）、`backend/logs/`、`backend/scripts/`（一次性评测集构建工具）、`docker/data/`（容器数据；实际部署在仓库外 `/home/ypw/docker/grad_assist/data/`）。

`backend/mcp/milvus.py` 是给开发期用的只读 Milvus MCP 服务器（查集合元数据与数据），不参与运行时。

可选容器化：`backend/Dockerfile`（CPU 镜像，模型目录运行时挂载）+ `docker/` compose 的 `app` profile；默认 `docker compose up -d` 仍只起中间件。
