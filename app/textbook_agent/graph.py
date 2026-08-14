from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.textbook_agent.nodes import (
    enrich_md,
    load_textbook,
    parse_to_md,
    split,
    split_contents,
    split_text_and_store,
)
from app.textbook_agent.state import TextBookState


def build_graph() -> StateGraph:
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

    return builder.compile()


# 单元测试
if __name__ == "__main__":
    from IPython.display import Image, display

    state: TextBookState = {
        "textbook_exists": False,
    }

    graph = build_graph()

    # 绘制图
    image = graph.get_graph().draw_mermaid_png()
    display(Image(image))

    graph.invoke(state)
    print("\nGraph 流水线执行完成")
