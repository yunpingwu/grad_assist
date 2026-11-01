"""历史工具原文降级测试：往轮工具原文在 state 里换成一句回收占位。

钉住：只换 content 不删消息；原文留在 artifact 供回显；判据是体积 + artifact，不是工具名单。
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import tool_history as th
from app.agent.tools import search

# fixture 由生产方自己的构造器产出（不手写键名）：_build_artifact 改了键，这边的降级会静默失效
# ——`demote_messages` 对取不到干净原文的 artifact 是直接跳过的，只有共用真源才抓得到这种断链。
# 正文体量要盖过 _MIN_DEMOTE_CHARS：回收判据按体积算，小结果本来就不该被回收（另有用例钉）
_HITS = [
    {"id": "c1", "entity": {"id": "c1", "chapter": "第3章", "section": "3.1", "block_type": "text",
                            "text": "指针是核心概念。" * 400}},
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


def test_demote_replaces_used_search_result_with_placeholder() -> None:
    """上一轮那条检索结果只剩一句回收占位：说明失忆与重新取用的出口，不复述原文。"""
    messages = [*_turn("t1"), HumanMessage(id="t2h", content="再讲讲第二个例子")]

    updates = th.demote_messages(messages)

    assert len(updates) == 1
    demoted = updates[0]
    assert demoted.content != _ART["content"], "原文没被换掉"
    assert "\n" not in demoted.content, f"占位应只有一行：{demoted.content!r}"
    assert len(demoted.content) < 200
    assert "已回收" in demoted.content and "重新取用" in demoted.content
    assert "指针是核心概念" not in demoted.content, "正文不该混进占位"


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


def test_demote_does_not_key_on_tool_name() -> None:
    """判据是「体积 + 有干净原文」，不认工具名单：未来任何把原文放进 artifact 的工具都自动纳管。

    旧版按名单只列了 search_textbook，新工具（大结果 + content_and_artifact）会悄悄漏网，
    体积照旧每轮进请求。
    """
    big_artifact = {"content": "网页正文" * 600, "results": [{"title": "t", "url": "u"}]}
    messages = [
        HumanMessage(id="t1h", content="问题"),
        AIMessage(id="t1a", content="", tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "tc0", "type": "tool_call"}]),
        _tm("t0", "tc0", big_artifact, name="search_web"),
        HumanMessage(id="t2h", content="新问题"),
    ]

    updates = th.demote_messages(messages)

    assert [m.id for m in updates] == ["t0"], "带大 artifact 的非检索工具同样该回收"


def test_demote_skips_small_or_unrecoverable_results() -> None:
    """两类不回收：体积没到线（重取一次的工具往返比它占的上下文更贵）、也没有干净原文可回显。

    ``read_file`` 分页上限 100 行、``list_chapters`` 就是份目录——它们留在上下文里更有用；
    没有 artifact 的消息一旦被换掉占位，回显就永远拿不回原文（``/history`` 只剩占位）。
    """
    messages = [
        HumanMessage(id="t1h", content="问题"),
        AIMessage(id="t1a", content="", tool_calls=[{"name": "read_file", "args": {}, "id": "tc0", "type": "tool_call"}]),
        _tm("small", "tc0", {"content": "短页面" * 100}, name="read_file"),  # 500 字符，未到线
        AIMessage(id="t1b", content="", tool_calls=[{"name": "list_chapters", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(id="noart", tool_call_id="tc1", name="list_chapters", content="第1章 概述\n" * 400),  # 大但无 artifact
        HumanMessage(id="t2h", content="新问题"),
    ]

    assert th.demote_messages(messages) == []


def test_demote_skips_messages_below_compaction_bookmark() -> None:
    """水位线之下的消息不再回收：它们早就不进模型视图（压缩已把它们移出请求），
    再换一次 content 只是白写一次 checkpoint。"""
    messages = [*_turn("t1"), *_turn("t2"), *_turn("t3"), HumanMessage(id="t4h", content="新问题")]

    # 水位线落在 t2 的工具结果上 → t1/t2 的工具消息都在线之下，只剩 t3 该回收
    updates = th.demote_messages(messages, bookmark="t2t")

    assert [m.id for m in updates] == ["t3t"]


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
