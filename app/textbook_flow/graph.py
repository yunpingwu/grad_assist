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


# 单元测试
if __name__ == "__main__":
    from IPython.display import Image, display
    import asyncio

    state: TextBookState = {
        "textbook_exists": False,
        # 此处图入口需要指定偏移量
        "offsets": [
            {"textbook_name": "C语言程序设计（第五版）_(谭浩强)_(z-library.sk,_1lib.sk,_z-lib.", "offset": 25},
            {"textbook_name": "操作系统：精髓与设计原理_第8版_(斯托林斯)_(z-library.sk,_1lib.sk,_.", "offset": 21},
            {"textbook_name": "数据库系统概论(第6版)_(王珊,杜小勇,陈红)_(z-library.sk,_1lib.sk,.", "offset": 27},
            {"textbook_name": "数据结构 (陈越、何钦铭、徐镜春、魏宝刚、杨枨编) (z-library.sk, 1lib.sk, z-lib.sk)", "offset": 10},
            {"textbook_name": "机器学习_Machine_Learning_(Chinese_Edition)_(Zhou_Zh.", "offset": 17},
            {"textbook_name": "计算机组成原理_第6版_(白中英,_戴志涛)_(z-library.sk,_1lib.sk,_z.", "offset": 11},
            {"textbook_name": "计算机网络（第8版）_(谢希仁)_(z-library.sk,_1lib.sk,_z-lib.s.", "offset": 12},
        ],
    }

    graph = build_graph()

    # 绘制图
    image = graph.get_graph().draw_mermaid_png()
    display(Image(image))

    asyncio.run(graph.ainvoke(state))
    print("\nGraph 流水线执行完成")
