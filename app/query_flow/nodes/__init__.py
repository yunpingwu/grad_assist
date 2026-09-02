"""query_flow 检索函数库：供工具（search_textbook）复用。

query_flow 图与节点封装已随"统一 agent + 工具化检索"方案移除，本包仅保留
被 search_textbook 直接复用的纯检索函数。
"""

from app.query_flow.nodes.embedding_search import rewrite_query_search
from app.query_flow.nodes.hyde_embedding_search import hyde_doc_generate, hyde_doc_search
from app.query_flow.nodes.merge_recalls import RRF_K, rrf_merge
from app.query_flow.nodes.rerank import rerank_chunks
from app.query_flow.nodes.rewrite_query import format_questions, rewrite

__all__ = [
    "RRF_K",
    "format_questions",
    "rewrite",
    "rewrite_query_search",
    "hyde_doc_generate",
    "hyde_doc_search",
    "rrf_merge",
    "rerank_chunks",
]
