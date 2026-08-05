from typing import NotRequired, TypedDict


class QueryState(TypedDict):

    # 会话 ID
    session_id: str
    # 教材名
    textbook_name: str
    # 原始问题
    original_query: str
    # 最近对话历史（纯文本，用于多轮消歧）
    chat_history: NotRequired[str]
    # 重写问题
    rewritten_query: NotRequired[str]
    # 普通向量检索回来的切片
    embedding_chunks: NotRequired[list]
    # HyDE 检索回来的切片
    hyde_embedding_chunks: NotRequired[list]
    # RRF 融合后的最终召回（保留 distance）
    merged_chunks: NotRequired[list]
    # 最终答案（文本，可能含 markdown 图片引用）
    answer: NotRequired[str]
