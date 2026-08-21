from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class QueryState(TypedDict):
    # 会话 ID
    session_id: str
    # 匿名设备身份（前端 X-User-Id 请求头传入），用于多用户会话隔离
    user_id: NotRequired[str]
    # 教材名
    textbook_name: str
    # 原始问题
    original_query: str
    # 多轮对话历史（由 checkpointer 持久化；add_messages 每轮追加而非覆盖）
    messages: Annotated[list[AnyMessage], add_messages]
    # 重写问题
    rewritten_query: NotRequired[str]
    # 普通向量检索回来的切片
    embedding_chunks: NotRequired[list]
    # HyDE 检索回来的切片
    hyde_embedding_chunks: NotRequired[list]
    # RRF 融合后的最终召回（保留 distance）
    merged_chunks: NotRequired[list]
    # Rerank 精排后的 TOP-K 片段（未启用/精排失败时生成节点回退 merged_chunks）
    reranked_chunks: NotRequired[list]
    # 是否启用联网搜索（前端按钮控制，默认开启）
    is_web_search: NotRequired[bool]
    # 联网搜索结果 [{title, url, content}]，供生成节点合并进上下文
    web_chunks: NotRequired[list]
    # 最终答案（文本，可能含 markdown 图片引用）
    answer: NotRequired[str]
