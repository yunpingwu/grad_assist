"""test_history_echo.py — /history 回显读 artifact：state 里被回收成占位的检索结果，前端仍展示原文。

工具消息在轮首被回收时只换 ``content``，原文留在 ``artifact``（随 checkpoint 落库、不进模型请求）。
回显若直接读 ``content``，用户刷新后看到的就是那句回收占位——故 ``/history`` 优先取 artifact。
"""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services import study as svc
from app.services.study import read_models
from app.study_agent.tools import search

# artifact 由生产方的构造器产出：键名改了，回显会静默回退成 state 里那句回收占位
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
_STUBBED = "（上一轮检索原文已回收：内容已用于当时的回答，需要依据时请重新检索）"


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


def _summary_recap() -> HumanMessage:
    """会话压缩注入的整篇回顾（打标成 HumanMessage + lc_source=summarization）。"""
    return HumanMessage(
        content="## 会话目标\n复习 Cache 的三种映射方式",
        additional_kwargs={"lc_source": "summarization"},
    )


def test_history_hides_summarization_recap() -> None:
    """压缩注入的回顾不进回显：它是 HumanMessage，不挡就会冒充成用户说过的话。

    回显侧唯一挡它的地方就是这里的 lc_source 判据；被压掉的历史照旧回显（压缩只裁模型视图）。
    """
    items = svc.build_history([_summary_recap(), HumanMessage(content="再讲讲组相联"), AIMessage(content="组相联…")])

    assert [i["role"] for i in items] == ["user", "assistant"], "摘要那条被当成用户消息展示出来了"
    assert items[0]["content"] == "再讲讲组相联"


def test_sessions_skip_summarization_recap(monkeypatch) -> None:
    """会话列表的计数与「最后问题」同样排除摘要回顾：否则计数虚高、列表摘要串味。"""

    async def _list_thread_ids(_prefix: str) -> list[str]:
        return ["u1:s1"]

    class _Graph:
        async def aget_state(self, _config: dict) -> SimpleNamespace:
            messages = [
                _summary_recap(),
                HumanMessage(content="第一问"),
                AIMessage(content="答一"),
                HumanMessage(content="第二问"),
                AIMessage(content="答二"),
            ]
            return SimpleNamespace(values={"textbook_name": "T", "messages": messages}, created_at="2026-10-16T00:00:00")

    # 会话列表只读 checkpoint 与 PostgreSQL 的 thread 名单，两处都换成假件（不起服务、不连库）
    monkeypatch.setattr(read_models, "list_thread_ids", _list_thread_ids)
    monkeypatch.setattr(svc.lifecycle, "study_graph", _Graph())

    items = asyncio.run(svc.sessions(textbook_name="T", user_id="u1"))

    assert items[0]["message_count"] == 4, "摘要那条被算进条目数了"
    assert items[0]["last_question"] == "第二问"
