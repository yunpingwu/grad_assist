"""HyDE 假设文档检索工具函数：供 search_textbook 深度路径做第二路召回。

从 query_functions 收编而来：``hyde_doc_generate`` 生成假设文档，``hyde_doc_search``
将其向量化后检索——检索部分复用 ``embedding_search.search_by_vectors``，与
query 路共享同一次批量 embedding 的产物。
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.clients.llm import get_llm_client
from app.core import load_prompt, logger
from app.study_agent.query_functions.embedding_search import search_by_vectors
from app.utils.batch_manager.embedder import agenerate_embeddings


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
    llm = get_llm_client(enable_thinking=False)
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke(
        {
            "rewritten_query": rewritten_query,
        }
    )
    hyde_doc = output.strip()
    logger.info(f"假设性文档预览：{hyde_doc[:100]}")

    return hyde_doc


async def hyde_doc_search(
    hyde_doc: str,
    rewritten_query: str,
    textbook_name: str,
    limit: int = 5,
) -> list[dict]:
    """将假设性文档向量化后进行查询。

    Args:
        hyde_doc: 生成的假设性文档。
        rewritten_query: 重写后的问题（与 hyde_doc 拼接后向量化）。
        textbook_name: 教材名，用于定位检索集合。
        limit: 融合后返回的最大命中数，缺省 5；深度路径评测扩候选池时传更大值。

    Returns:
        检索到的 TOP 文本片段。
    """
    if not hyde_doc:
        raise ValueError("假设性文档为空")
    if not rewritten_query:
        raise ValueError("问题重写为空")
    hyde_doc = hyde_doc + rewritten_query
    hyde_doc_embedding = await agenerate_embeddings([hyde_doc])
    logger.info(f"假设性文档向量生成结果: {hyde_doc_embedding}")
    return await search_by_vectors(
        textbook_name,
        hyde_doc_embedding.get("dense")[0],
        hyde_doc_embedding.get("sparse")[0],
        limit=limit,
    )


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