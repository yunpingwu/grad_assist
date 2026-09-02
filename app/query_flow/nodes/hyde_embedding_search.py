"""HyDE 假设文档检索工具函数：供 search_textbook 深度路径做第二路召回。

从 query_flow 收编而来：仅保留纯函数 hyde_doc_generate / hyde_doc_search
（原节点封装已移除）。
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import WeightedRanker

from app.clients.llm import get_llm_client
from app.clients.milvus_client import get_client
from app.core import load_prompt, logger
from app.utils.embedding_util import generate_embeddings
from app.utils.milvus_util import create_hybrid_search_requests, get_collection_by_name


async def hyde_doc_generate(rewritten_query: str) -> str:
    """借助 LLM 生成假设性回答文档（HyDE）。

    Args:
        rewritten_query: 重写后的问题。

    Returns:
        假设性文档（纯文本）。
    """
    if not rewritten_query:
        raise ValueError("问题重写为空")
    template = load_prompt("hyde_doc_generate")
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm_client()
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke(
        {
            "rewritten_query": rewritten_query,
        }
    )
    hyde_doc = output.strip()
    logger.info(f"假设性文档预览：{hyde_doc[:100]}")

    return hyde_doc


async def hyde_doc_search(hyde_doc: str, rewritten_query: str, textbook_name: str):
    """将假设性文档向量化后进行查询。

    Args:
        hyde_doc: 生成的假设性文档。
        rewritten_query: 重写后的问题（与 hyde_doc 拼接后向量化）。
        textbook_name: 教材名，用于定位检索集合。

    Returns:
        检索到的 TOP5 文本片段。
    """
    if not hyde_doc:
        raise ValueError("假设性文档为空")
    if not rewritten_query:
        raise ValueError("问题重写为空")
    hyde_doc = hyde_doc + rewritten_query
    hyde_doc_embedding = generate_embeddings([hyde_doc])
    dense_vec = hyde_doc_embedding.get("dense")[0]
    sparse_vec = hyde_doc_embedding.get("sparse")[0]
    logger.info(f"假设性文档向量生成结果: {hyde_doc_embedding}")
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        limit=10,
    )
    milvus_client = get_client()
    if not milvus_client:
        raise ValueError("Milvus无法连接")
    # 按教材名定位集合（注册表精确匹配，内部已处理名称转义）
    collection_name = get_collection_by_name(textbook_name)
    if not collection_name:
        raise ValueError(f"教材未登记: {textbook_name}")
    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=WeightedRanker(0.8, 0.2),
        limit=5,
        output_fields=["text", "chapter", "section", "metadata_json"],
    )
    logger.info(f"查询向量搜索结果: {res}")
    return res[0]


# 冒烟测试：仅验证空输入守卫（真实验证依赖 LLM/Milvus，由 search_textbook 集成覆盖）
if __name__ == "__main__":
    import asyncio

    async def _run() -> None:
        for fn, args in [
            (hyde_doc_generate, ("",)),
            (hyde_doc_search, ("", "问题", "教材")),
            (hyde_doc_search, ("文档", "", "教材")),
        ]:
            try:
                await fn(*args)
                raise AssertionError("空输入应当抛出 ValueError")
            except ValueError as exc:
                assert "为空" in str(exc)
        print("hyde 空输入守卫通过")

    asyncio.run(_run())
