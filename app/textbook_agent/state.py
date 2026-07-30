from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class RetrievedChunk(TypedDict):
    """单条检索到的课本片段。"""

    content: str
    source: str
    score: Optional[float]
    metadata: dict[str, Any]


class TextBookState(TypedDict):
    """课本 RAG 流程中各节点之间传递的状态。

    字段按流程分组：
    入口路由 → 数据摄入(可选) → 检索 → 重排 → 生成 → 输出

    """

    # ===== 入口路由 =====
    # 是否已有教材在向量库中，决定走哪条入口
    textbook_exists: bool
    # 用户上传的教材文件路径（本地路径或 URL）
    textbook_path: NotRequired[str]
    # 教材名称（用于标识和检索过滤）
    textbook_name: NotRequired[str]

    # ===== 数据摄入（仅 textbook_exists=False 时使用） =====
    # 章节分割后的子 PDF 路径列表
    sub_pdf_paths: NotRequired[list[str]]
    # 目录页 mineru 解析后解压的目录路径列表
    extracted_contents_dirs: NotRequired[list[str]]
    # 章节 mineru 解析后解压的目录路径列表
    extracted_dirs: NotRequired[list[str]]
    # 切分后的文本块列表
    chunks: NotRequired[list[str]]
    # 向量化入库是否完成
    ingestion_done: NotRequired[bool]

    # ===== 输入 / 对话 =====
    # add_messages: 同一节点多次写入会追加而非覆盖，并自动按 id 去重
    messages: Annotated[list[AnyMessage], add_messages]
    # 原始用户问题（不经过任何改写，便于溯源/日志）
    original_question: str
    # 经过改写后用于检索的问题（多轮对话场景下常用）
    rewritten_question: NotRequired[str]

    # ===== 检索 =====
    # 检索召回的原始片段（未排序前）
    retrieved_chunks: NotRequired[list[RetrievedChunk]]
    # 重排后保留给 LLM 的片段
    reranked_chunks: NotRequired[list[RetrievedChunk]]
    # 本次检索是否足够支撑回答；不够时可触发兜底策略
    is_sufficient: NotRequired[bool]

    # ===== 生成 =====
    # 最终回答内容
    answer: NotRequired[str]
    # 引用来源列表（与 reranked_chunks 中的 source 对应）
    citations: NotRequired[list[str]]

    # ===== 控制 / 元信息 =====
    # 当前轮次（用于限制最大检索-生成循环次数）
    iteration: NotRequired[int]
    # 错误信息（任意节点失败时写入，便于路由到兜底节点）
    error: NotRequired[Optional[str]]
