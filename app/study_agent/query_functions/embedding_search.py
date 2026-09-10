"""向量混合检索工具函数：供 search_textbook 做稠密+稀疏混合召回。

从 query_functions 收编而来：``search_by_vectors`` 为「向量 → 检索」纯函数，
``rewrite_query_search`` 在其之上补 embedding 生成，供快速路径使用；深度路径
直接用 ``search_by_vectors`` 复用一次批量 embedding 的产物。
"""

from __future__ import annotations

from app.core import logger, stage
from app.utils.embedding_util import agenerate_embeddings
from app.utils.milvus_util import get_collection_by_name, hybrid_search


async def search_by_vectors(
    textbook_name: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    chapter: str | None = None,
) -> list[dict]:
    """用预生成的 dense/sparse 向量在教材集合执行混合检索（可选章节过滤）。

    不生成 embedding，接收调用方已算好的向量，供深度路径对 query/hyde 两路
    共享一次批量 embedding 的结果。

    Args:
        textbook_name: 教材名（须已登记）。
        dense_vec: 查询稠密向量（单条）。
        sparse_vec: 查询稀疏向量（单条）。
        chapter: 限定章节名（可选，转义后作为 Milvus 过滤表达式），缺省全书检索。

    Returns:
        检索到的 TOP 文本片段列表。
    """
    collection_name = get_collection_by_name(textbook_name)
    if not collection_name:
        raise ValueError(f"教材未登记: {textbook_name}")
    # 章节过滤表达式：转义双引号，防止破坏 filter 语法（与注册表查询同一策略）
    chapter_expr = None
    if chapter:
        safe_chapter = chapter.replace('"', '\\"')
        chapter_expr = f'chapter == "{safe_chapter}"'
    with stage("milvus_search_ms"):
        return hybrid_search(dense_vec, sparse_vec, collection_name, expr=chapter_expr)


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
    # 生成问题向量（异步派发到线程池，避免阻塞事件循环）
    query_embedding = await agenerate_embeddings([rewrite_query])
    logger.info(f"提问向量生成结果: {query_embedding}")
    return await search_by_vectors(
        textbook_name,
        query_embedding.get("dense")[0],
        query_embedding.get("sparse")[0],
        chapter,
    )


# 冒烟测试：仅验证空查询守卫（真实检索依赖 Milvus + embedding 模型，由 search_textbook 集成验证）
if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(rewrite_query_search("C语言程序设计", ""))
        raise AssertionError("空查询应当抛出 ValueError")
    except ValueError as exc:
        assert "问题重写为空" in str(exc)
        print("rewrite_query_search 空查询守卫通过")