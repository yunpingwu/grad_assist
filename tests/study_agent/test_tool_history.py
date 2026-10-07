"""历史检索结果降级测试：上一轮用过的工具原文在 state 里换成一行行为记录。

两条实测前提支撑这套设计：ChatOpenAI 序列化请求只带 ``ToolMessage.content``（artifact 不上线，
但随消息进 checkpoint），所以原文留在 artifact 里供回显与审计；``add_messages`` 按消息 id
原位替换，所以降级不必删消息——留着壳，``AIMessage.tool_calls`` 与结果的配对才不破。
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.study_agent.query_functions import tool_history as th
from app.study_agent.tools import search

# fixture 由生产方自己的构造器产出（不手写键名）：_build_artifact 改了键，这边的降级会静默失效
# ——`demote_messages` 对没有 chunks 的 artifact 是直接跳过的，只有共用真源才抓得到这种断链。
_HITS = [
    {"id": "c1", "entity": {"id": "c1", "chapter": "第3章", "section": "3.1", "block_type": "text",
                            "text": "指针是核心概念。" * 200}},
    {"id": "c2", "entity": {"id": "c2", "chapter": "第3章", "section": "3.2", "block_type": "text",
                            "text": "数组是相同类型元素的集合。"}},
]
_ART = search._build_artifact("C语言程序设计", "指针的用途", True, _HITS, search._format_hits(_HITS))


def _tm(msg_id: str, tool_call_id: str, artifact: dict | None, name: str = "search_textbook") -> ToolMessage:
    """造一条检索工具消息：content 为原文，artifact 为随消息落库的副本。"""
    return ToolMessage(
        id=msg_id,
        tool_call_id=tool_call_id,
        name=name,
        content=(artifact or {}).get("content", "网页摘要内容"),
        artifact=artifact,
    )


def _turn(prefix: str) -> list:
    """造一整轮：提问 → 工具调用 → 工具结果 → 回答。"""
    return [
        HumanMessage(id=f"{prefix}h", content=f"{prefix} 问题"),
        AIMessage(
            id=f"{prefix}a",
            content="",
            tool_calls=[{"name": "search_textbook", "args": {"query": "x"}, "id": f"{prefix}tc", "type": "tool_call"}],
        ),
        _tm(f"{prefix}t", f"{prefix}tc", dict(_ART)),
        AIMessage(id=f"{prefix}r", content=f"{prefix} 回答"),
    ]


def test_demote_replaces_used_search_result_with_one_line() -> None:
    """上一轮那条检索结果只剩一行行为记录，且仍是一行。"""
    messages = [*_turn("t1"), HumanMessage(id="t2h", content="再讲讲第二个例子")]

    updates = th.demote_messages(messages)

    assert len(updates) == 1
    demoted = updates[0]
    assert demoted.content != _ART["content"], "原文没被换掉"
    assert "\n" not in demoted.content, f"降级后应只有一行：{demoted.content!r}"
    assert len(demoted.content) < 200, f"一行记录不该又长出一段：{len(demoted.content)} 字符"
    # 行为记录要说清「查了哪本书、按什么问句、拿回几段」，否则模型不知道自己依据过什么
    assert "C语言程序设计" in demoted.content and "指针的用途" in demoted.content
    assert "2" in demoted.content, demoted.content


def test_demote_keeps_current_turn_results_intact() -> None:
    """本轮的检索结果必须原样全文：模型正要靠它写回答，此刻降级等于让它无据可答。

    本轮 = 最后一条 HumanMessage 之后的消息。降级只在轮首跑，所以这个边界真正防的是
    「resume 回中断点时本轮已查出结果、又被当成往轮抹掉」。
    """
    messages = [*_turn("t1"), HumanMessage(id="t2h", content="新问题"), *_turn("t2")[1:]]

    updates = th.demote_messages(messages)

    assert [m.id for m in updates] == ["t1t"], f"本轮结果不该动: {[m.id for m in updates]}"


def test_demote_preserves_pairing_and_position() -> None:
    """降级只换 content：同 id、同 tool_call_id、同 name，位置不前移——配对破了接口直接报错。"""
    messages = [*_turn("t1"), *_turn("t2"), HumanMessage(id="t3h", content="新问题")]

    demoted = th.demote_messages(messages)[0]

    assert demoted.id == "t1t" and demoted.tool_call_id == "t1tc" and demoted.name == "search_textbook"
    assert demoted.type == "tool"
    # 原文没丢：回显与审计从 artifact 取
    assert demoted.artifact["content"] == _ART["content"]
    assert [c["id"] for c in demoted.artifact["chunks"]] == ["c1", "c2"]


def test_demote_is_idempotent() -> None:
    """写回 state 后再跑一次应零更新：每条消息一生只改一次，resume 重入也不二改。"""
    messages = [*_turn("t1"), *_turn("t2"), HumanMessage(id="t3h", content="新问题")]

    updates = th.demote_messages(messages)
    by_id = {u.id: u for u in updates}
    merged = [by_id.get(m.id, m) for m in messages]

    assert th.demote_messages(merged) == []
    assert by_id["t1t"].artifact["content"] == _ART["content"], "降级后原文仍随 artifact 留在 state"


def test_demote_skips_tool_without_retrieval_artifact() -> None:
    """没有检索 artifact 的工具消息（web、list_chapters、失败降级）一律不动：无原文可留，也无从渲染记录。"""
    messages = [
        HumanMessage(id="t1h", content="问题"),
        AIMessage(id="t1a", content="", tool_calls=[{"name": "list_chapters", "args": {}, "id": "tc0", "type": "tool_call"}]),
        _tm("t0", "tc0", None, name="list_chapters"),
        *_turn("t2"),
        HumanMessage(id="t3h", content="新问题"),
    ]

    updates = th.demote_messages(messages)

    assert [m.id for m in updates] == ["t2t"], f"无检索 artifact 的工具消息不应被降级: {[m.id for m in updates]}"


def test_hook_writes_demoted_messages_into_state() -> None:
    """挂点（轮首）把该轮之前所有往轮结果交回 state：同 id 由 ``add_messages`` 原位替换。"""
    messages = [*_turn("t1"), *_turn("t2"), HumanMessage(id="t3h", content="新问题")]

    update = asyncio.run(th.demote_used_tool_results.abefore_agent({"messages": messages}, None))

    assert [m.id for m in update["messages"]] == ["t1t", "t2t"]
    assert update["messages"][0].tool_call_id == "t1tc"


def test_first_turn_has_nothing_to_demote() -> None:
    """首轮没有历史轮：一律放行全文，不该产生任何 state 更新。"""
    messages = [*_turn("t1"), HumanMessage(id="t1h2", content="追问")]
    # 首条 Human 之前没有任何消息，本轮检索结果全部在「最后一条 Human 之后」

    assert th.demote_messages([*messages[:-1]]) == []
