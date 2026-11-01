"""教材助手 Agent 的状态定义：create_agent 输入输出 + 任务信息与收尾产物字段。"""

from typing import Annotated, NotRequired, TypedDict

from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class Material(BaseModel):
    """一份已落盘的复习资料：标题、文件名、相对路径与字节数。"""

    title: str
    filename: str
    rel_path: str
    size: int


class StudyState(TypedDict):
    """复习资料生成任务各节点间传递的状态。"""

    # 匿名设备身份（前端 X-User-Id 请求头传入），用于多用户任务隔离
    user_id: NotRequired[str]
    # 任务 ID（thread 后缀，断点续跑用）
    task_id: str
    # 教材名（检索过滤键 + 落盘目录名）
    textbook_name: str
    # 用户的要求（缺省有默认指令）
    requirement: str
    # 本轮深度检索档（前端「深度检索」开关传入，per-turn 覆盖）；search_textbook 经 InjectedState 读取
    deep: NotRequired[bool]
    # 意图分类（前置 query 理解识别；explain/generate/quiz/plan/chat/unclear），system prompt 据此选块
    intent: NotRequired[str]
    # 意图判定置信度与来源档（keyword/embedding/llm/fallback），随意图一并透传给 system prompt
    # 的「意图预判」段——低置信提示模型自行斟酌 ask_clarification（降级路径可缺省）
    intent_confidence: NotRequired[float]
    intent_source: NotRequired[str]
    # 多轮对话（create_agent 依赖；add_messages 每轮追加而非覆盖）
    messages: Annotated[list[AnyMessage], add_messages]
    # 落盘清单（收尾后写回）
    materials: NotRequired[list[Material]]
    # 是否完成（收尾后置 True）
    finished: NotRequired[bool]

    # —— 会话成本预算（per-session；由 SessionBudgetMiddleware 读写，口径见 app/trace/llm_usage.weighted_cost）——
    # 已折叠的会话累计加权成本（PrivateStateAttr：不许 API 输入侧写入，只随 checkpoint 持久化）
    session_cost: NotRequired[Annotated[float, PrivateStateAttr]]
    # 上次折叠时当轮 trace 读数 / 所属请求 id（trace 计数器请求级，跨请求按 request_id 重新基线）
    cost_fold_mark: NotRequired[Annotated[float, PrivateStateAttr]]
    cost_fold_req: NotRequired[Annotated[str, PrivateStateAttr]]
    # 已触发成本降级（禁 deep / 停追问 / 收尾说明），会话内持续生效
    budget_exhausted: NotRequired[bool]

    # —— 会话压缩（视图层，见 agent/compaction.py）——
    # 水位线：该 id（含）之前的消息已被一份回顾取代，不再进模型视图。**state 原文一条不删**，
    # 回显/审计照旧读得到；只裁发给模型的请求视图（与往轮检索回收同一条设计线）
    compacted_upto: NotRequired[Annotated[str, PrivateStateAttr]]
