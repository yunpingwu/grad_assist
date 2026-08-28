# 教材复习资料生成 Agent（Study Material Agent）设计文档

> 状态: 设计中(未实现) · 日期: 2026-08-26（v0.2：主方案切换为 A）
> 目标: 基于已入库的教材内容，自动生成一份或多份复习资料（知识点总结 / 章节纲要 / 练习题 / 易错点清单等），并以 Markdown 文件落盘。主方案（A）为「**单层 ReAct 循环 + 护栏**」，对齐 OpenCode / DeepSeek Harness / SWE-agent 的开源实践；曾设计的「自适应路由 + Plan-Execute」（方案 B）降级为备选，见附录 A。

---

## 1. 背景与现状

当前项目已有两条 LangGraph 流水线（workflow），本 Agent 直接复用于它们之上：

- **教材摄入流水线** `app/textbook_flow/`：`load_textbook → split_contents → split → parse_to_md → enrich_md → split_text_and_store`，把教材切块向量化后写入 Milvus（每教材一张集合，`tb_<sha1>`），并经 `textbook_registry` 注册表维护「教材名 → 集合名」映射。
- **检索问答流水线** `app/query_flow/`：`rewrite_query → (embedding_search ∥ hyde_embedding_search) → merge_recalls(RRF) → rerank → [web_search] → generate_answer`，用于单轮/多轮问答。

**动机**：现有两条流水线是「固定拓扑」的确定性 workflow，适合摄入与问答这类流程固定的任务；但对于「根据教材内容，做出一份/多份复习资料」这类**开放式、多步骤、需要边检索边生成、还要写文件落盘**的任务，固定拓扑难以表达。因此引入 Agent 形态，让 LLM 自己决定步骤、按需调用工具、迭代产出并写盘。

**复用原则**：不重写检索/生成能力，而是把现有能力「封装成工具」交给 Agent 调用；文件读写则自建轻量工具（本项目无既有文件编辑能力），参考开源 agent 的成熟实现。

---

## 2. 需求分析

- **输入**：`textbook_name`（教材名，已在向量库中）+ 用户对复习资料的要求（可选：类型、范围、深度、章节、风格等）。
- **输出**：一份或多份 Markdown 复习资料，例如：
  - 全教材/单章「知识点总结」
  - 「章节纲要 / 思维导图（文本大纲）」
  - 「章节练习题 + 参考答案」
  - 「易错点 / 重点难点清单」
  - 「考前冲刺要点」
- **落盘**：资料以 Markdown 文件写入磁盘沙箱目录（`materials/{textbook_name}/{task_id}/`），元信息入 Mongo 供前端列表与下载；URL 方式可选上传 MinIO。
- **约束**：
  - 内容必须**忠于教材内容**（以向量检索到的片段为准），联网仅作补充并需注明来源；
  - 支持流式推送过程事件（工具调用、token、文件写入）与断点续跑（沿用 checkpointer）；
  - 用**护栏**而非固定拓扑控制长任务：步数上限、上下文压缩、文件沙箱、失败降级（见 §3.3）。

---

## 3. 总体方案（方案 A：单层 ReAct + 护栏）

### 3.1 为什么选单层循环（开源实践调研，context7 源码确认）

| 项目 | 循环结构 | 护栏机制 | 文件工具要点 |
|---|---|---|---|
| **OpenCode** (`anomalyco/opencode`) | 单层 turn 循环：模型出 tool-call 就继续，出纯文本就结束；**无独立 planner** | `agent.steps` 步数上限（到顶 `toolChoice:"none"` + 注入「请给最终答案」提示）、`compactAfterOverflow` 上下文压缩、工具权限授权 | `read` 拒绝绝对路径/`..`/symlink 逃逸、超长文本按行范围分页；`edit` 要求**先 Read 过**该文件；`apply_patch` 多文件补丁经 `writeWithDirs` 写入（自动建父目录） |
| **DeepSeek Harness** (`deepseek-ai/deepseek-harness`) | 统一 turn/step 生命周期（pre-step → request → stream → tool execute → next step）；编排/多 agent 是**可选工具**（`ralph`/`workflow`），不是顶层模式 | `maxParallelToolCalls`（默认 10）、沙箱模式与工具选集分离 | `edit` 用 `old_string` **精确唯一匹配**（`replace_all` 显式开启）；`writeFileAtomic` 原子写 + `withFileLock` 读改写锁 + 版本守卫防陈旧更新；沙箱 `read-only / workspace-write / danger-full-access` + `writableRoots` 白名单 |
| **SWE-agent** (`swe-agent/swe-agent`) | 纯 ReAct：`thought + action → 执行 → observation → 循环`，`exit` 结束；无 planner | shell blocklist、`HistoryProcessor` 历史压缩 | 专用文件查看器**每轮约 100 行 + 滚动/搜索**；`_state` 命令只回传「当前打开文件 + 工作目录」省上下文；编辑走 shell 命令 |

**结论**：主流开源 agent **不做「简单/复杂」二分类路由，也不前置规划**，而是让规划通过模型推理 + 可选工具**在 ReAct 循环内隐式表达**，靠「步数上限 + 上下文压缩 + 文件沙箱」三件套扛住长任务。本方案采纳同一路线：单层 `create_react_agent` + 四道护栏。

### 3.2 架构图

```text
┌──────────────────── 单层 StateGraph（薄封装）────────────────────┐
│                                                                  │
│  START ──▶ agent ──▶ finalize ──▶ END                            │
│            │  ▲                                                  │
│            ▼  │                                                  │
│   create_react_agent 预置循环：model ⇄ tools（自定步数、自定终止）│
│                                                                  │
│  tools 分两类：                                                  │
│   · 检索类（只读）: search_textbook / list_chapters / search_web │
│   · 文件类（读写）: write_material / read_file / edit_file /     │
│     append_file / list_materials                                 │
│   · 辅助（可选）: summarize / plan                               │
└──────────────────────────────────────────────────────────────────┘

finalize：解析最终消息 → 提取落盘文件清单 → materials 登记入 Mongo → done 事件
```

### 3.3 四道护栏（对齐开源三件套 + 本项目降级惯例）

| # | 护栏 | 机制 | 开源依据 |
|---|---|:--|---|
| 1 | **步数上限** | `create_react_agent` 的 `remaining_steps` 限步；system prompt 明示「接近步数上限时直接输出当前成果，勿再调工具」（等价 OpenCode 到顶后的 `MAX_STEPS_PROMPT` + `toolChoice:"none"`） | OpenCode `agent.steps` |
| 2 | **上下文压缩** | 检索工具返回强制 TOP-K 截断；提供 `summarize` 工具；prompt 纪律「**每检索完一章先提炼要点，只累积摘要不累积原文**」；后续可升级为 OpenCode 式溢出自动压缩（检测 token 阈值触发总结） | OpenCode `compactAfterOverflow`、SWE-agent `HistoryProcessor` |
| 3 | **文件沙箱** | 所有路径相对 `materials/{textbook_name}/{task_id}/` 解析；拒绝绝对路径、`..` 逃逸、symlink 逃逸；`edit_file` 要求先 `read_file`、`old_string` 唯一匹配；写入用「临时文件 + rename」原子落盘 | OpenCode `read` 路径拒绝 + `edit` 先读约束、DeepSeek `workspace-write` 沙箱 + 原子写 |
| 4 | **失败降级** | 检索/联网失败返回空结果 + 告警不中断（沿用 `query_flow` 的降级风格）；文件工具失败返回明确错误信息让模型换文件名/路径重试 | DeepSeek 工具错误「模型可见」的 `formatError` 思路 |

---

## 4. 状态设计（`StudyMaterialState`）

新增 `app/material_flow/state.py`，`TypedDict` 风格与现有 `QueryState` 保持一致：

```python
class StudyMaterialState(TypedDict):
    # 任务入口
    user_id: NotRequired[str]              # 匿名设备身份（多用户隔离）
    task_id: str                           # 任务 id（thread 后缀，断点续跑用）
    textbook_name: str                     # 教材名（检索过滤键 + 落盘目录名）
    requirement: str                       # 用户对复习资料的要求（缺省有默认指令）

    # 循环必需（create_react_agent 依赖；add_messages 每轮追加而非覆盖）
    messages: Annotated[list[AnyMessage], add_messages]

    # 收尾产物（finalize 节点写回）
    materials: NotRequired[list[Material]]  # 落盘资料清单（标题/类型/相对路径/大小）
    finished: NotRequired[bool]             # 是否完成
```

`Material` 用 `pydantic.BaseModel` 定义（finalize 从最终 AIMessage 解析 + 目录扫描写回）：

```python
class Material(BaseModel):
    title: str        # 资料标题（来自文件首个一级标题或文件名）
    filename: str     # 文件名，如 第3章-知识点总结.md
    rel_path: str     # 相对 materials/{textbook_name}/{task_id}/ 的路径
    size: int         # 字节数
```

> 说明：不再有 `mode` / `plan` / `current_step` 字段——单层方案没有 router/planner 节点；规划若发生，是工具调用历史的一部分，天然留在 `messages` 里，不单独建模。

---

## 5. 工具集设计（新增 `app/material_flow/tools/`）

原则：**检索工具是现有能力的薄封装；文件工具是自建的轻量封装**。全部用 LangChain `@tool` + Pydantic `args_schema` 定义；**docstring 即工具说明书**，直接决定 ReAct 是否正确调用。

### 5.1 检索类工具（只读，复用现有能力）

| 工具 | 说明 | 复用来源 | 状态 |
|---|---|---|---|
| `search_textbook(query, chapter=None)` | 向量混合检索（dense+sparse）教材内容，返回 TOP-K 片段（含 chapter/section/图片简介），已按「片段编号」截断合并 | `app/query_flow/nodes/embedding_search.py` 的 `rewrite_query_search`（hybrid search 逻辑），加 `chapter` 过滤参数 | 改造复用 |
| `list_chapters(textbook_name)` | 列出教材章节结构（chapter/section 聚合去重），供模型了解教材骨架、决定检索范围 | **新增**：基于 `milvus_util` 对集合 `chapter`/`section` 字段去重聚合 | 新增 |
| `search_web(query)` | 联网搜索补充外部资料（注明来源，仅补充） | `app/query_flow/nodes/web_search.py` 的 `_search_web` | 复用 |

### 5.2 文件类工具（读写落盘，自建 + 吸收开源实践）

| 工具 | 说明 | 关键约束 | 开源依据 |
|---|---|---|---|
| `write_material(filename, content)` | 在沙箱内**创建/整文件覆盖**一个 Markdown 资料文件（自动建父目录） | 路径白名单内解析；「临时文件 + rename」原子写；返回绝对路径 + 字节数 | OpenCode `apply_patch` 的 `writeWithDirs`、DeepSeek `writeFileAtomic` |
| `append_file(filename, content)` | **追加**内容到已有文件（长文档分段写，避免整文重写） | 文件必须已存在；锁定写入 | DeepSeek `withFileLock` 读改写 |
| `read_file(filename, offset=1, limit=100)` | 按**行范围分页**读取沙箱内文件（检查已写内容/续写定位） | UTF-8；超长自动按 limit 分页，返回「共 N 行」信息 | SWE-agent 每次约 100 行的文件查看器、OpenCode 按行范围分页 |
| `edit_file(filename, old_string, new_string)` | **精确串替换**修改已写文件 | **必须先 `read_file` 过该文件**；`old_string` 默认须唯一命中（`replace_all` 显式开启才全替换）；替换后返回上下文确认 | OpenCode「先 Read 再 Edit」纪律、DeepSeek `old_string` 唯一匹配 |
| `list_materials()` | 列出沙箱内已生成的文件（名称/大小/时间，按目录树） | 只列白名单内，缺省按名排序 | OpenCode `read` 目录列表 |

**沙箱设计（护栏 3 的落地）**：

- 根目录：`materials_root = materials/{textbook_name}/{task_id}/`（类比 `textbook_service` 的 `TEXTBOOK_ROOT` 模式，任务级隔离，天然防跨任务/跨用户串扰）；
- 所有相对路径先 `Path.resolve()`，再校验 `is_relative_to(root)`，**拒绝绝对路径、`..` 逃逸、symlink 逃逸**（不 resolve 的路径一律拒绝）；
- `write/append/edit` 走「临时文件 + `os.replace`」原子落盘；`append/edit` 加文件锁（本项目单进程内用 `threading.Lock` 即可，多 worker 时再上 `filelock`）。

示例（示意，实现时放 `app/material_flow/tools/files.py`）：

```python
from pathlib import Path
from langchain_core.tools import tool

@tool
async def write_material(filename: str, content: str) -> str:
    """在资料沙箱内创建或覆盖一个 Markdown 复习资料文件。

    Args:
        filename: 相对路径文件名（可含子目录，如 第3章/知识点总结.md）。
        content: 完整 Markdown 内容，用 UTF-8 编码。
    返回: 写入成功后的绝对路径与字节数；路径非法时返回错误原因。
    """
    root = get_materials_root()          # materials/{textbook_name}/{task_id}/
    path = safe_resolve(root, filename)  # 拒绝绝对路径/.. /symlink 逃逸
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)                # 原子替换
    return f"已写入 {path}（{path.stat().st_size} 字节）"


@tool
async def edit_file(filename: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """对已写文件做精确文本替换。调用前必须先 read_file 读取该文件。

    Args:
        filename: 相对路径文件名。
        old_string: 要被替换的原文（必须与文件内容完全一致，默认要求唯一出现）。
        new_string: 替换后的文本；传空串表示删除。
        replace_all: 是否替换所有匹配（默认 False，old_string 不唯一时报错提示）。
    返回: 替换结果确认（含替换处上下文）。
    """
    ...
```

### 5.3 辅助工具（可选，MVP 可后置）

| 工具 | 说明 | 来源 |
|---|---|---|
| `summarize(text, instruction)` | 压缩工具：把检察到的长片段/已写章节提炼为要点摘要（护栏 2 的显式开关，供模型主动调用） | 对齐 OpenCode 自动压缩的「手动版」 |
| `plan(steps)` | 可选规划工具：模型主动声明后续步骤并写进对话（DeepSeek 的「编排即工具」思路）；不设顶层 planner，不强制 | DeepSeek `ralph`/`workflow` 工具化 |

---

## 6. 图结构与节点职责（`app/material_flow/graph.py`）

### 6.1 节点

| 节点 | 类型 | 职责 | 关键实现 |
|---|---|---|---|
| `agent` | 预置 ReAct 子图 | 单层循环：以 `requirement` 为首条 HumanMessage，自主调用检索/文件工具，产出 Markdown 并以 `write_material` 落盘 | `create_react_agent(model, tools, prompt=load_prompt("material_agent"))`；内层 `stream_mode="messages"` 转 custom 事件 |
| `finalize` | 普通节点 | 解析最终 AIMessage + 扫描沙箱目录，写回 `materials`、置 `finished`、推 `done` 事件 | 目录扫描 + Mongo 登记（见 §8） |

### 6.2 图拓扑（示意）

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

def build_graph(checkpointer=None):
    builder = StateGraph(StudyMaterialState)

    builder.add_node("agent", agent_node)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
```

`agent` 节点内，单层 ReAct（context7 确认的 `create_react_agent`，配 system prompt + 步数护栏）：

```python
from langgraph.prebuilt import create_react_agent

react_agent = create_react_agent(
    model=get_llm_client(),                          # 支持 bind_tools 的 init_chat_model 实例
    tools=[search_textbook, list_chapters, search_web,
           write_material, append_file, read_file, edit_file, list_materials],
    prompt=load_prompt("study_agent"),            # system prompt：角色 + 四道护栏纪律
    checkpointer=checkpointer,                       # 沿用外层 checkpointer，断点续跑
    state_schema=StudyMaterialState,                 # 使用外层状态（含 messages）
)
```

> `remaining_steps` 由 `create_react_agent` 基于 `recursion_limit` 自动计算；到顶行为配合 prompt 纪律（护栏 1）实现「强制输出而非报错」。

### 6.3 流式输出协议（沿用 `stream_mode="custom"` + `StreamWriter`）

| 事件 | payload | 触发点 |
|---|---|---|
| `stage` | `{stage, message}` | agent 起止 / finalize（「正在生成资料」「正在整理产物」） |
| `tool` | `{name, status: start/done, detail?}` | 每次工具调用起止（检索与文件操作对前端可见） |
| `file` | `{action: write/append/edit, path}` | 文件类工具成功落盘时 |
| `token` | `{content}` | 模型逐 token 流式（agent 内层转出） |
| `error` | `{message}` | 工具调用失败但未中断循环时（护栏 4 的可见化） |
| `done` | `{task_id, materials}` | finalize 完成后 |

---

## 7. 目录与文件清单

**新增**（沿用 `*_flow` 命名，与 `textbook_flow` / `query_flow` 并列）：

```text
app/material_flow/
  __init__.py
  graph.py            # 薄封装图：agent(react 子图) + finalize + build_graph
  state.py            # StudyMaterialState + Material
  nodes/
    __init__.py
    agent.py          # 节点适配：调用 create_react_agent 子图，messages 流转 custom 事件
    finalize.py       # 扫描沙箱目录 → materials 清单 → Mongo 登记 → done 事件
  tools/
    __init__.py       # 汇总导出 tools 列表 + 沙箱材料根路径工具函数
    search.py         # search_textbook（复用 hybrid search + chapter 过滤）
    chapters.py       # list_chapters（新增：Milvus chapter/section 聚合）
    web.py            # search_web（复用 _search_web）
    files.py          # write_material / append_file / read_file / edit_file / list_materials（路径沙箱 + 原子写）
    summarize.py      # summarize（可选，护栏 2 显式工具）
  service.py          # 对外服务函数（启动任务、列出/读取资料、断点续跑）
  api.py              # APIRouter（可选，也可并入 material_service）
```

**提示词**（新增 `app/prompts/`）：

| 文件 | 用途 |
|---|---|
| `material_agent.prompt` | agent 的 system prompt：角色定位、**忠于教材（基于检索片段、不得编造）**、**压缩纪律（每章先摘要再继续）**、**步数纪律（接近上限直接交成果）**、文件工具使用规范（先 read 再 edit、分段写长文档） |

**复用/改造**：

| 文件 | 改动 |
|---|---|
| `app/query_flow/nodes/embedding_search.py` | 将 `rewrite_query_search` 抽为可传 `chapter` 过滤的公共函数（或直接在 `tools/search.py` 复用其 hybrid-search 逻辑，不改原文件） |
| `app/utils/milvus_util.py` | 新增 `list_chapters(textbook_name)`：按集合 `chapter`/`section` 字段去重聚合（若 Milvus 不支持 distinct 则 query 后本地去重） |
| `app/api/main.py` | `include_router(material_router)` |

---

## 8. API 接入（MVP 与查询端点同风格）

- `POST /material/generate`（SSE 流式，契约对齐 `query_service`）：入参 `textbook_name`、`requirement`、可选 `task_id`（断点续跑）、`user_id`（复用 `get_user_id`）。
- `thread_id = f"{user_id}:material:{task_id}"`，独立 `MongoDBSaver` checkpoint collection（如 `material_checkpoints`），避免与 query/textbook 图冲突。
- 落盘：`materials/{textbook_name}/{task_id}/`（沙箱根）；finalize 后元信息（task_id、textbook_name、材料清单）入 Mongo；可选上传 MinIO 供跨端下载。
- 读取接口：`GET /material/tasks`（任务列表）、`GET /material/files?task_id=`（资料列表）、`GET /material/file?task_id=&path=`（下载/预览，路径同样过沙箱校验）。

---

## 9. 复用映射总览

| 现有能力 | Agent 中的角色 |
|---|---|
| `get_llm_client()`（`init_chat_model`，支持 `bind_tools`） | 单层 ReAct 的模型底座 |
| `langgraph.prebuilt.create_react_agent` | 单层循环本体（含 tool-call 循环 / `remaining_steps` / `state_schema`） |
| `rewrite_query_search` / hybrid search（dense+sparse+Reranker） | `search_textbook` 工具 |
| `get_collection_by_name` / `list_textbooks` / Milvus collection | 定位教材集合、`list_chapters` 工具 |
| `_search_web`（百炼 WebSearch MCP） | `search_web` 工具 |
| `load_prompt` / `ChatPromptTemplate` | `material_agent.prompt` system prompt |
| `MongoDBSaver`（`langgraph-checkpoint-mongodb`） | 断点续跑 |
| `minio_client` / `mongo_client` | 资料可选上传、任务/资料元信息登记 |
| （无既有能力，新增） | 文件类工具 `tools/files.py`：路径沙箱 + 原子写（吸收 OpenCode / DeepSeek / SWE-agent 实践） |

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 长任务循环失控、上下文膨胀 | 护栏 1+2：`remaining_steps` 限步（到顶强制输出）+ TOP-K 截断 + `summarize` 工具 + 「每章摘要化」prompt 纪律 |
| 模型写文件越界/覆盖系统文件 | 护栏 3：任务级沙箱目录 + 路径 resolve 校验（拒绝绝对路径/`..`/symlink 逃逸）+ 原子写 |
| 大文件被整文重写造成 token 浪费 | 提供 `append_file` 分段追加；`edit_file` 精确替换替代整写；prompt 规范长文档分章落盘 |
| 「忠于教材」失控、胡编内容 | 检索工具 docstring + system prompt 双重约束「必须基于检索片段」；联网结果标注来源仅补充 |
| 工具失败拖垮整个生成 | 护栏 4：检索/联网失败返回空 + 告警（沿用 query_flow 降级风格）；文件工具错误信息给模型重试线索 |
| 步数到顶时输出残缺资料 | 到顶提示要求「输出当前已完成的成果并说明未覆盖部分」；`materials` 仍正常登记，前端可二次触发增量生成 |
| 多用户/多任务文件串扰 | 沙箱根含 `textbook_name` + `task_id`，目录级天然隔离 |

---

## 11. 待实现时的验证计划

1. **步数上限**：构造一个会持续调工具的任务，验证到顶「强制输出 + 正常收尾」而非超时报错。
2. **压缩纪律**：全书任务实测 messages 增长曲线，检索多章后确认模型按纪律先摘要再继续（可进一步做后续自动压缩）。
3. **沙箱安全**：`write_material` 传 `../../etc/x.md`、绝对路径、symlink 文件 → 均应拒绝；白名单内子目录写入 → 成功。
4. **文件工具闭环**：`write → read（分页验证）→ edit（未 read 先 edit 应报错；old_string 不唯一应报错；replace_all 生效）→ append`，确认落盘内容与原子性（无 `.tmp` 残留）。
5. **检索忠实性**：检查 `search_textbook`/`list_chapters` 返回结果与教材一致、chapter 过滤正确。
6. **端到端**：simple（单章知识点总结）与复杂（全书多资料）各跑一轮：资料落盘正确、`materials` 登记、可经 API 下载。
7. **流式**：SSE 按 `stage → tool → (file) → token → done` 顺序推送，前端可消费。
8. **断点续跑**：生成中途中断，用同一 `task_id` 重连，确认从断点继续而非重头执行。
9. **降级**：临时断开 Milvus / 联网，确认不拖垮整个 Agent（空结果 + 告警 + 继续）。

---

## 附录 A. 备选方案 B：自适应路由 + Plan-Execute（记录，暂不实施）

**曾设计**（v0.1）：图入口加 `router` 节点（`with_structured_output(RouteDecision)` 判 `simple/complex`）→ simple 直进单步 ReAct；complex 走 `planner + Plan-Execute 循环 + reporter`。

**降级原因**（基于 context7 对 OpenCode / DeepSeek Harness / SWE-agent 源码调研）：

1. 三家均不做复杂度二分类路由——分类器是额外 LLM 调用 + 误判点，主流用「步数上限 + 压缩」替代；
2. 规划可由 ReAct 内隐式完成（CoT + 可选 `plan` 工具），Plan-Execute 的冻结式计划反而失去「边做边改」的灵活性；
3. 官方生态趋势：`create_react_agent` 为第一公民，Plan-Execute 预置件已边缘化。

**何时重新启用**：若实测单循环在「全书多章多资料」任务上出现高频步数耗尽、上下文压缩后仍质量劣化，再把 router/planner 加回（状态、节点、事件契约已在历史版本中设计成形，改动成本可控）。

---

## 参考来源（开源实现，均经 context7 查询确认）

- [OpenCode 会话 runner 循环 (session/runner/llm.ts)](https://github.com/anomalyco/opencode/blob/dev/packages/core/src/session/runner/llm.ts)
- [OpenCode 文件工具 (src/tool/)](https://github.com/anomalyco/opencode/tree/dev/packages/opencode/src/tool)
- [DeepSeek Harness 架构 (docs/architecture.md)](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)
- [DeepSeek Harness 文件系统子系统 (docs/subsystems/filesystem.md)](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/filesystem.md)
- [DeepSeek Harness 原子写 (packages/util/atomic-write)](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/util/atomic-write/README.md)
- [SWE-agent ACI (docs/background/aci.md)](https://github.com/swe-agent/swe-agent/blob/main/docs/background/aci.md)
- [SWE-agent 工具配置 (docs/config/tools.md)](https://github.com/swe-agent/swe-agent/blob/main/docs/config/tools.md)