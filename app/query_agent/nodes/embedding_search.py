from langgraph.types import StreamWriter
from pymilvus import WeightedRanker

from app.clients.milvus_client import get_client
from app.core import log_node, logger
from app.query_agent.state import QueryState
from app.utils.embedding_util import generate_embeddings
from app.utils.milvus_util import create_hybrid_search_requests, get_collection_by_name


async def rewrite_query_search(textbook_name: str, rewrite_query: str) -> list[dict]:
    """根据重写后的问题进行向量搜索。

    Args:
        textbook_name: 教材名。
        rewrite_query: 重写后的问题。

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
    # 构造混合搜索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,  # 取用户问题的稠密向量（单条，故取索引0）
        sparse_vector=sparse_vec,  # 取用户问题的稀疏向量（单条，故取索引0）
        limit=10,  # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
    )
    # 执行混合搜索
    client = get_client()
    if not client:
        raise ValueError("Milvus 客户端无法连接")
    res = client.hybrid_search(
        collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
        reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
        ranker=WeightedRanker(0.8, 0.2),  # 稠/稀疏向量评分权重配比（pymilvus 3.0 用位置参数，可按业务调优）
        limit=5,  # 最终返回的TOP5相似度最高结果
        output_fields=["text", "chapter", "section", "metadata_json"],  # 指定返回的业务字段
    )
    logger.info(f"查询向量搜索结果: {res}")
    return res[0]


@log_node
async def embedding_search(state: QueryState, *, writer: StreamWriter) -> dict:
    """根据重写问题进行向量检索"""
    writer({"type": "stage", "stage": "search", "message": "正在检索教材内容…"})
    textbook_name = state.get("textbook_name")
    rewritten_query = state.get("rewritten_query")
    result = await rewrite_query_search(textbook_name, rewritten_query)
    # 只返回本节点写入的字段（避免并行分支携带整个 state 导致 key 冲突）
    return {"embedding_chunks": result}


# 单元测试
if __name__ == "__main__":
    import asyncio

    state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "C语言如何使用指针?",
        "rewritten_query": "C语言如何使用指针?",
        "chat_history": "",
    }
    asyncio.run(embedding_search(state))
