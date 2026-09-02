"""向量混合检索工具函数：供 search_textbook 做稠密+稀疏混合召回。

从 query_flow 收编而来：仅保留纯函数 rewrite_query_search（原节点封装已移除）。
"""

from __future__ import annotations

from pymilvus import WeightedRanker

from app.clients.milvus_client import get_client
from app.core import logger
from app.utils.embedding_util import generate_embeddings
from app.utils.milvus_util import create_hybrid_search_requests, get_collection_by_name


async def rewrite_query_search(textbook_name: str, rewrite_query: str, chapter: str | None = None) -> list[dict]:
    """根据重写后的问题进行向量混合搜索（可选按章节过滤）。

    Args:
        textbook_name: 教材名。
        rewrite_query: 重写后的问题。
        chapter: 限定章节名（可选，转义后作为 Milvus 过滤表达式），缺省全书检索。

    Returns:
        检索到的 TOP5 文本片段。
    """
    if not rewrite_query:
        raise ValueError("问题重写为空")
    # 生成问题向量
    query_embedding = generate_embeddings([rewrite_query])
    dense_vec = query_embedding.get("dense")[0]
    sparse_vec = query_embedding.get("sparse")[0]
    logger.info(f"提问向量生成结果: {query_embedding}")

    # 按教材名定位集合（注册表精确匹配，内部已处理名称转义）
    collection_name = get_collection_by_name(textbook_name)
    if not collection_name:
        raise ValueError(f"教材未登记: {textbook_name}")
    # 章节过滤表达式：转义双引号，防止破坏 filter 语法（与注册表查询同一策略）
    chapter_expr = None
    if chapter:
        safe_chapter = chapter.replace('"', '\\"')
        chapter_expr = f'chapter == "{safe_chapter}"'
    # 构造混合搜索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        limit=10,
        expr=chapter_expr,
    )
    # 执行混合搜索
    client = get_client()
    if not client:
        raise ValueError("Milvus 客户端无法连接")
    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=WeightedRanker(0.8, 0.2),
        limit=5,
        output_fields=["text", "chapter", "section", "metadata_json"],
    )
    logger.info(f"查询向量搜索结果: {res}")
    return res[0]


# 冒烟测试：仅验证空查询守卫（真实检索依赖 Milvus + embedding 模型，由 search_textbook 集成验证）
if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(rewrite_query_search("C语言程序设计", ""))
        raise AssertionError("空查询应当抛出 ValueError")
    except ValueError as exc:
        assert "问题重写为空" in str(exc)
        print("rewrite_query_search 空查询守卫通过")
