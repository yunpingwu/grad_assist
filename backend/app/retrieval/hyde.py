"""HyDE 假设文档检索：生成假设文档并复用 ``search_by_vectors`` 做第二路召回。"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.clients.llm import get_llm_client
from app.core import load_prompt, logger
from app.ml.embedding import agenerate_embeddings
from app.retrieval.dense import search_by_vectors


async def hyde_doc_generate(search_query: str) -> str:
    """用 LLM 生成假设性回答文档（HyDE）。"""
    if not search_query:
        raise ValueError("检索问句为空")
    template = load_prompt("hyde_doc_generate")
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm_client(max_retries=3, enable_thinking=False)
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke(
        {
            "search_query": search_query,
        },
        config={"tags": ["trace_role:hyde"]},
    )
    hyde_doc = output.strip()
    logger.info(f"假设性文档预览：{hyde_doc[:100]}")

    return hyde_doc


async def hyde_doc_search(
    hyde_doc: str,
    search_query: str,
    textbook_name: str,
    limit: int = 5,
) -> list[dict]:
    """把假设文档与检索问句拼接后向量化，再做第二路召回。"""
    if not hyde_doc:
        raise ValueError("假设性文档为空")
    if not search_query:
        raise ValueError("检索问句为空")
    hyde_doc = hyde_doc + search_query
    hyde_doc_embedding = await agenerate_embeddings([hyde_doc])
    logger.info(f"假设性文档向量生成结果: {hyde_doc_embedding}")
    return await search_by_vectors(
        textbook_name,
        hyde_doc_embedding.get("dense")[0],
        hyde_doc_embedding.get("sparse")[0],
        limit=limit,
    )


