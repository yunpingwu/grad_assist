from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.query_agent.nodes import (
    embedding_search,
    generate_answer,
    hyde_embedding_search,
    merge_recalls,
    rewrite_query,
)
from app.query_agent.state import QueryState


def build_graph() -> StateGraph:
    """构建教材问答流水线 Graph。

    流程: rewrite_query → (embedding_search ∥ hyde_embedding_search) → merge_recalls → generate_answer
    """
    builder = StateGraph(QueryState)

    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("embedding_search", embedding_search)
    builder.add_node("hyde_embedding_search", hyde_embedding_search)
    builder.add_node("merge_recalls", merge_recalls)
    builder.add_node("generate_answer", generate_answer)

    # 入口：先重写问题
    builder.add_edge(START, "rewrite_query")
    # 重写后并行发起两路召回（普通向量 + HyDE）
    builder.add_edge("rewrite_query", "embedding_search")
    builder.add_edge("rewrite_query", "hyde_embedding_search")
    # 两路召回汇合后做 RRF 融合，最后生成答案
    builder.add_edge("embedding_search", "merge_recalls")
    builder.add_edge("hyde_embedding_search", "merge_recalls")
    builder.add_edge("merge_recalls", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# 单元测试
if __name__ == '__main__':
    import asyncio

    state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "C语言如何使用指针?",
        "chat_history": "",
    }

    graph = build_graph()
    result = asyncio.run(graph.ainvoke(state))
    print(f"答案: {result.get('answer', '')}")
