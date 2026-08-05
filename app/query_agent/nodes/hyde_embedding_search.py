from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import WeightedRanker

from app.clients.llm import get_llm_client
from app.clients.milvus_client import get_client
from app.core import log_node, load_prompt,logger
from app.query_agent.state import QueryState
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
    output = await chain.ainvoke({
        "rewritten_query": rewritten_query,
    })
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
        collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
        reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
        ranker=WeightedRanker(0.8, 0.2),  # 稠/稀疏向量评分权重配比
        limit=5,  # 获取的TOP5相似度最高结果
        output_fields=["text", "chapter", "section", "metadata_json"],  # 输出的字段
    )
    logger.info(f"查询向量搜索结果: {res}")
    return res[0]


@log_node
async def hyde_embedding_search(state: QueryState) -> dict:
    """根据重写的问题生成假设性文档，向量化之后查询"""
    rewritten_query = state.get("rewritten_query")
    textbook_name = state.get("textbook_name")
    hyde_doc = await hyde_doc_generate(rewritten_query)

    hyde_embedding_chunks = await hyde_doc_search(hyde_doc,rewritten_query,textbook_name)
    # 只返回本节点写入的字段（避免并行分支携带整个 state 导致 key 冲突）
    return {"hyde_embedding_chunks": hyde_embedding_chunks}

# 单元测试
if __name__ == '__main__':
    import asyncio
    state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "C语言如何使用指针?",
        "rewritten_query": "C语言如何使用指针?",
        "chat_history": "",
    }
    asyncio.run(hyde_embedding_search(state))
