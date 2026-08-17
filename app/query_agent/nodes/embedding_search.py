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
        limit=10,  # 底层检索返回数量
    )
    # 执行混合搜索
    client = get_client()
    if not client:
        raise ValueError("Milvus 客户端无法连接")
    res = client.hybrid_search(
        collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
        reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
        ranker=WeightedRanker(0.8, 0.2),  # 稠/稀疏向量评分权重配比（pymilvus 3.0 用位置参数）
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
    # 向量检索
    result = await rewrite_query_search(textbook_name, rewritten_query)
    return {"embedding_chunks": result}


# 冒烟测试：桩掉真实检索（依赖 Milvus + embedding 模型），只验证节点编排
if __name__ == "__main__":
    import asyncio

    async def _fake_search(textbook_name: str, rewrite_query: str) -> list[dict]:
        return [
            {"id": "c1", "distance": 0.8, "entity": {"text": "指针是C语言的核心概念。"}},
            {"id": "c2", "distance": 0.7, "entity": {"text": "数组是相同类型元素的集合。"}},
        ]

    rewrite_query_search = _fake_search  # 覆盖真实检索，避免依赖外部服务

    def writer(chunk):
        print(f"  [writer] {chunk}")

    state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "C语言如何使用指针?",
        "rewritten_query": "C语言如何使用指针?",
    }
    result = asyncio.run(embedding_search(state, writer=writer))
    chunks = result["embedding_chunks"]
    assert len(chunks) == 2 and chunks[0]["id"] == "c1", f"embedding_chunks 不正确: {chunks}"
    print(f"embedding_search 测试通过，召回 {len(chunks)} 条")
