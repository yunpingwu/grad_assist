from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.textbook_flow.nodes import (
    enrich_md,
    load_textbook,
    parse_to_md,
    split,
    split_contents,
    split_text_and_store,
)
from app.textbook_flow.state import TextBookState


def build_graph(checkpointer=None) -> StateGraph:
    """构建教材处理流水线 Graph。

    流程: load_textbook → split_contents → split → parse_to_md → enrich_md → split_text_and_store
    """
    builder = StateGraph(TextBookState)

    builder.add_node("load_textbook", load_textbook)
    builder.add_node("split_contents", split_contents)
    builder.add_node("split", split)
    builder.add_node("parse_to_md", parse_to_md)
    builder.add_node("enrich_md", enrich_md)
    builder.add_node("split_text_and_store", split_text_and_store)

    builder.add_edge(START, "load_textbook")
    builder.add_edge("load_textbook", "split_contents")
    builder.add_edge("split_contents", "split")
    builder.add_edge("split", "parse_to_md")
    builder.add_edge("parse_to_md", "enrich_md")
    builder.add_edge("enrich_md", "split_text_and_store")
    builder.add_edge("split_text_and_store", END)

    return builder.compile(checkpointer=checkpointer)


# 集成测试已迁移至 tests/textbook_flow/test_graph.py
