"""test_history_echo.py — /history 回显读 artifact：state 里降级成一行行为记录的检索结果，前端仍展示原文。

工具消息在轮首被降级时只换 ``content``，原文留在 ``artifact``（随 checkpoint 落库、不进模型请求）。
回显若直接读 ``content``，用户刷新后看到的就是那行内部记录——故 ``/history`` 优先取 artifact。
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api import study_service as svc
from app.study_agent.tools import search

# artifact 由生产方的构造器产出：键名改了，回显会静默回退成 state 里那行内部记录
_HITS = [
    {
        "id": "c1",
        "entity": {
            "id": "c1",
            "chapter": "第3章",
            "section": "3.1",
            "block_type": "text",
            "text": "指针变量存放的是地址……",
        },
    }
]
_ORIGINAL = search._format_hits(_HITS)
_STUBBED = "[历史记录] 上一轮已按「指针的用途」在《C语言程序设计》取回 1 段依据：第3章 > 3.1"


def test_history_shows_original_text_from_artifact() -> None:
    """降级过的那条：时间线里给前端的仍是原文，不是那行内部记录。"""
    messages = [
        HumanMessage(content="指针的用途"),
        AIMessage(content="", tool_calls=[{"name": "search_textbook", "args": {}, "id": "call_1"}]),
        ToolMessage(
            content=_STUBBED,
            tool_call_id="call_1",
            name="search_textbook",
            artifact=search._build_artifact("C语言程序设计", "指针的用途", False, _HITS, _ORIGINAL),
        ),
        AIMessage(content="指针用于间接访问……"),
    ]
    items = svc.build_history(messages)

    assert [i["role"] for i in items] == ["user", "assistant"]
    assert items[1]["steps"][-1] == {"type": "tool", "name": "search_textbook", "content": _ORIGINAL}


def test_history_falls_back_to_content_without_artifact() -> None:
    """没有 artifact 的工具（小体积输出、失败路径）照旧显示 content。"""
    messages = [
        HumanMessage(content="有哪些章"),
        AIMessage(content="", tool_calls=[{"name": "list_chapters", "args": {}, "id": "call_2"}]),
        ToolMessage(content="第1章 概述\n第3章 指针", tool_call_id="call_2", name="list_chapters"),
        AIMessage(content="共 2 章"),
    ]
    items = svc.build_history(messages)

    assert items[-1]["steps"][-1]["content"] == "第1章 概述\n第3章 指针"
