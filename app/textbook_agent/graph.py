from langgraph.constants import START, END
from langgraph.graph import StateGraph
from IPython.display import Image, display

from app.textbook_agent.nodes import parse_to_md, load_textbook, spilt_contents, split, split_text_and_store
from app.textbook_agent.state import TextBookState


def build_graph() -> StateGraph:
    """构建教材处理流水线 Graph。

    流程: load_textbook → spilt_contents → split → parse_to_md → split_text_and_store
    """
    builder = StateGraph(TextBookState)

    builder.add_node("load_textbook", load_textbook)
    builder.add_node("spilt_contents", spilt_contents)
    builder.add_node("split", split)
    builder.add_node("parse_to_md", parse_to_md)
    builder.add_node("split_text_and_store", split_text_and_store)

    builder.add_edge(START, "load_textbook")
    builder.add_edge("load_textbook", "spilt_contents")
    builder.add_edge("spilt_contents", "split")
    builder.add_edge("split", "parse_to_md")
    builder.add_edge("parse_to_md", "split_text_and_store")
    builder.add_edge("split_text_and_store", END)

    return builder.compile()




# 单元测试
if __name__ == '__main__':
    state: TextBookState = {
        "textbook_exists": False,
        "messages": [],
        "original_question": "",
    }

    graph = build_graph()

    # 绘制图
    image = graph.get_graph().draw_mermaid_png()
    display(Image(image))

    graph.invoke(state)
    print("\nGraph 流水线执行完成")
