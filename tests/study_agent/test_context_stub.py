"""跨轮检索结果存根化：stub_search_messages 纯函数与 wrap_model_call middleware。

设计要点：
- 只处理"最后一个 HumanMessage 之前"的 search_textbook ToolMessage（历史轮），
  本轮工具消息原样保留（模型要靠它生成最终回答）；
- 替换只改 content（留壳）：type/name/tool_call_id 不变，tool_call↔tool_result 配对不破；
- 已是存根的消息不重复处理（幂等）；
- 无历史检索消息时返回原列表（恒等，zero-copy 便于上层判断）。
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.study_agent.query_functions.context_stub import stub_search_messages

_OLD_SEARCH_CONTENT = (
    "[片段1｜c1｜第3章 > 3.1] 指针是C语言的核心概念。指针变量存储的是变量的内存地址，"
    + "通过解引用操作符可以访问该地址处的值，这是间接访问数据的基石机制。" * 8
    + "尾段标记T：这段位于摘录窗口之外的长篇背景不应出现在存根中。"
    + "\n\n[片段2｜c2｜第3章 > 3.2] 数组是相同类型元素的集合，在内存中连续存放。" * 8
    + "\n\n[代码｜c3_code｜第3章 > 3.1] ```c\nint main() { return 0; }\n```"
)


def _messages(with_history: bool = True) -> list:
    msgs = [
        HumanMessage(content="指针是什么？"),
        AIMessage(content="", tool_calls=[{"name": "search_textbook", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content=_OLD_SEARCH_CONTENT, tool_call_id="tc1", name="search_textbook"),
        AIMessage(content="指针是核心概念……（第一轮回答）"),
    ]
    msgs.append(HumanMessage(content="它和数组什么关系？"))
    if with_history:
        msgs += [
            AIMessage(content="", tool_calls=[{"name": "search_textbook", "args": {}, "id": "tc2", "type": "tool_call"}]),
            ToolMessage(content="[片段1｜c9｜第3章 > 3.2] 本轮新检索的完整原文。", tool_call_id="tc2", name="search_textbook"),
        ]
    return msgs


def test_history_search_message_is_stubbed() -> None:
    out = stub_search_messages(_messages())
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


def test_current_turn_message_untouched() -> None:
    out = stub_search_messages(_messages())
    assert out[-1].content == "[片段1｜c9｜第3章 > 3.2] 本轮新检索的完整原文。"


def test_message_identity_fields_preserved() -> None:
    out = stub_search_messages(_messages())
    src, dst = _messages()[2], out[2]
    assert dst.tool_call_id == src.tool_call_id
    assert dst.name == src.name
    assert dst.type == src.type == "tool"


def test_idempotent() -> None:
    once = stub_search_messages(_messages())
    twice = stub_search_messages(once)
    assert [m.content for m in twice] == [m.content for m in once]


def test_no_search_history_returns_same_list() -> None:
    msgs = [HumanMessage(content="你好"), AIMessage(content="你好！")]
    assert stub_search_messages(msgs) is msgs


def test_other_tool_messages_untouched() -> None:
    msgs = [
        HumanMessage(content="列一下章节"),
        ToolMessage(content="第1章\n第2章\n" + "x" * 5000, tool_call_id="t1", name="list_chapters"),
        AIMessage(content="共两章"),
        HumanMessage(content="再看下"),
    ]
    out = stub_search_messages(msgs)
    assert out[1].content == msgs[1].content


def test_legacy_format_without_id_still_stubbed() -> None:
    """旧格式 checkpoint（片段头无 id）：存根保留摘录与位置，不带 id 行不报错。"""
    legacy = "[片段1｜第3章 > 3.1] 旧格式正文，" + "长" * 700
    msgs = [
        HumanMessage(content="q1"),
        ToolMessage(content=legacy, tool_call_id="t1", name="search_textbook"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
    ]
    out = stub_search_messages(msgs)
    assert out[1].content.startswith("[存根]")
    assert "第3章 > 3.1" in out[1].content
