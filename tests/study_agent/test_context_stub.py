"""跨轮检索结果存根化：空档触发判定、按 mark 定盘、以及只改请求视图的 middleware。

设计要点（对齐 Claude Code 的「once seen, fate is frozen」）：
- **只在空档触发时定盘**：距上轮结束不足 ``IDLE_RECLAIM_SECONDS`` 时，前缀缓存大概率还活着，
  此时回溯改写已发内容会把整段历史划出公共前缀（探针实见跨轮命中塌回静态壳 3072），故不动；
- **定盘范围由 ``stub_mark_id`` 圈定**：mark 一旦写入 state 就只前进不回退，
  mark 之后的消息永不改——避免「这轮存根、下轮又放行全文」的翻转破缓存；
- 本轮（最后一条 HumanMessage 之后）的工具消息永不进 mark，模型要靠它生成最终回答；
- 替换只改 content（留壳）：type/name/tool_call_id 不变，tool_call↔tool_result 配对不破；
- 已是存根的消息不重复处理；``stub_search_messages`` 对同一入参恒等输出，可幂等重放。
"""

import asyncio
import dataclasses

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.study_agent.query_functions.context_stub import (
    IDLE_RECLAIM_SECONDS,
    plan_reclaim_mark,
    stamp_turn_activity,
    stub_old_search_results,
    stub_search_messages,
)

_OLD_SEARCH_CONTENT = (
    "[片段1｜c1｜第3章 > 3.1] 指针是C语言的核心概念。指针变量存储的是变量的内存地址，"
    + "通过解引用操作符可以访问该地址处的值，这是间接访问数据的基石机制。" * 8
    + "尾段标记T：这段位于摘录窗口之外的长篇背景不应出现在存根中。"
    + "\n\n[片段2｜c2｜第3章 > 3.2] 数组是相同类型元素的集合，在内存中连续存放。" * 8
    + "\n\n[代码｜c3_code｜第3章 > 3.1] ```c\nint main() { return 0; }\n```"
)

# mark 定在本轮提问之前的最后一条消息（a2）：t1 属历史轮该存根，t2 属本轮不动
_MARK = "a2"


def _messages(with_history: bool = True) -> list:
    msgs = [
        HumanMessage(content="指针是什么？", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"name": "search_textbook", "args": {}, "id": "tc1", "type": "tool_call"}],
        ),
        ToolMessage(content=_OLD_SEARCH_CONTENT, tool_call_id="tc1", name="search_textbook", id="t1"),
        AIMessage(content="指针是核心概念……（第一轮回答）", id="a2"),
        HumanMessage(content="它和数组什么关系？", id="h2"),
    ]
    if with_history:
        msgs += [
            AIMessage(
                content="",
                id="a3",
                tool_calls=[{"name": "search_textbook", "args": {}, "id": "tc2", "type": "tool_call"}],
            ),
            ToolMessage(
                content="[片段1｜c9｜第3章 > 3.2] 本轮新检索的完整原文。" + "长" * 700,
                tool_call_id="tc2",
                name="search_textbook",
                id="t2",
            ),
        ]
    return msgs


def test_history_search_message_is_stubbed() -> None:
    out = stub_search_messages(_messages(), _MARK)
    old = out[2]
    assert old.content.startswith("[存根]")
    assert len(out[2].content) < len(_OLD_SEARCH_CONTENT)
    # 定位信息与 id 保留，供重取
    for cid in ("c1", "c2", "c3_code"):
        assert cid in old.content
    assert "第3章 > 3.1" in old.content
    assert "read_chunk" in old.content
    # 摘录保留（首片段正文前若干字）
    assert "指针是C语言的核心概念" in old.content
    # 超长正文被丢弃（摘录窗口外的尾段不保留）
    assert "尾段标记T" not in old.content


def test_mark_before_first_search_returns_same_list() -> None:
    """没有 mark（本轮不回收）时一律放行全文：一个字节都不改，跨轮前缀才保得住。"""
    msgs = _messages()
    assert stub_search_messages(msgs, None) is msgs
    assert stub_search_messages(msgs, "h1") is msgs  # mark 早于任何检索消息 → 无可存根


def test_current_turn_message_untouched() -> None:
    """mark 只圈到本轮提问之前：本轮检索结果必须完整可见。"""
    out = stub_search_messages(_messages(), _MARK)
    assert out[-1].content == _messages()[-1].content


def test_message_identity_fields_preserved() -> None:
    out = stub_search_messages(_messages(), _MARK)
    src, dst = _messages()[2], out[2]
    assert dst.tool_call_id == src.tool_call_id
    assert dst.name == src.name
    assert dst.type == src.type == "tool"


def test_idempotent() -> None:
    once = stub_search_messages(_messages(), _MARK)
    twice = stub_search_messages(once, _MARK)
    assert [m.content for m in twice] == [m.content for m in once]


def test_mark_is_stable_across_new_turns() -> None:
    """定盘只前进：同一 mark 下，历史轮的存根文本跨轮逐字不变（新轮消息追加不影响旧存根）。"""
    turn2 = stub_search_messages(_messages(), _MARK)
    turn3 = [*turn2, AIMessage(content="第二轮回答", id="a4"), HumanMessage(content="第三轮问题", id="h3")]
    # mark 未推进时，第三轮视图里的 t1 与第二轮视图逐字相同
    assert stub_search_messages(turn3, _MARK)[2].content == turn2[2].content


def test_other_tool_messages_untouched() -> None:
    msgs = [
        HumanMessage(content="列一下章节", id="h1"),
        ToolMessage(content="第1章\n第2章\n" + "x" * 5000, tool_call_id="t1", name="list_chapters", id="m1"),
        AIMessage(content="共两章", id="m2"),
        HumanMessage(content="再看下", id="h2"),
    ]
    out = stub_search_messages(msgs, "m2")
    assert out[1].content == msgs[1].content


def test_legacy_format_without_id_still_stubbed() -> None:
    """旧格式 checkpoint（片段头无 id）：存根保留摘录与位置，不带 id 行不报错。"""
    legacy = "[片段1｜第3章 > 3.1] 旧格式正文，" + "长" * 700
    msgs = [
        HumanMessage(content="q1", id="h1"),
        ToolMessage(content=legacy, tool_call_id="t1", name="search_textbook", id="m1"),
        AIMessage(content="a1", id="m2"),
        HumanMessage(content="q2", id="h2"),
    ]
    out = stub_search_messages(msgs, "m2")
    assert out[1].content.startswith("[存根]")
    assert "第3章 > 3.1" in out[1].content


# --- 空档触发：回收时机与定盘范围 ---


def test_reclaim_skipped_while_cache_likely_alive(monkeypatch) -> None:
    """门槛调大后，活跃连问期间不推进 mark：回溯改写字节会把整段历史划出前缀。"""
    import app.study_agent.query_functions.context_stub as cs

    monkeypatch.setattr(cs, "IDLE_RECLAIM_SECONDS", 600.0)
    state = {
        "last_activity_at": 1_000.0 - cs.IDLE_RECLAIM_SECONDS / 2,
        "messages": _messages(),
    }
    assert plan_reclaim_mark(state, now=1_000.0) is None


def test_reclaim_advances_mark_to_last_history_message() -> None:
    """空档超阈值：缓存已按冷处理，mark 推进到本轮提问之前的最后一条消息。"""
    state = {"last_activity_at": 1_000.0 - IDLE_RECLAIM_SECONDS - 1, "messages": _messages()}
    assert plan_reclaim_mark(state, now=1_000.0) == "a2"


def test_reclaim_requires_existing_stamp() -> None:
    """旧线程没有 last_activity_at：本轮不定盘，交给本轮收尾写入的戳（宁少存根不多破前缀）。"""
    state = {"messages": _messages()}
    assert plan_reclaim_mark(state, now=1_000.0) is None


def test_mark_only_moves_forward() -> None:
    """已有更靠后的 mark 时不回退：定盘范围单调，避免把已存根的消息放回全文。"""
    state = {
        "last_activity_at": 1_000.0 - IDLE_RECLAIM_SECONDS - 1,
        "stub_mark_id": "a2",
        "messages": _messages()[:4],  # 上轮只跑到 a2
    }
    assert plan_reclaim_mark(state, now=1_000.0) is None


def test_after_agent_stamps_activity() -> None:
    """收尾写入本轮结束时刻，下一轮据此算空档。"""
    update = asyncio.run(stamp_turn_activity.aafter_agent({}, None))
    assert set(update) == {"last_activity_at"}
    assert update["last_activity_at"] > 1_600_000_000  # 墙钟秒，跨进程重启仍可比


def test_middleware_views_follow_state_mark() -> None:
    """挂点只按 state 里的 mark 改视图，不写回 state。"""
    from langchain.agents.middleware.types import ModelRequest, ModelResponse

    msgs = _messages()
    captured: dict = {}

    async def _handler(req: ModelRequest) -> ModelResponse:
        captured["req"] = req
        return ModelResponse(result=[])

    kwargs = {f.name: None for f in dataclasses.fields(ModelRequest)}
    kwargs.update(messages=list(msgs), state={"stub_mark_id": _MARK})
    asyncio.run(stub_old_search_results.awrap_model_call(ModelRequest(**kwargs), _handler))
    sent = captured["req"].messages
    assert sent[2].content.startswith("[存根]")
    assert sent[-1].content == msgs[-1].content, "本轮检索结果必须完整可见"
    assert msgs[2].content == _OLD_SEARCH_CONTENT, "state 里的消息本体不动，只改请求视图"
