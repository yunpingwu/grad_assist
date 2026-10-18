"""消息来源标记测试：压缩注入的回顾必须能被三处消费方认出来。

回显、会话计数、意图层历史全靠 ``is_summary_message`` 挡它；标记对不上时三处会一起静默失效
（回顾以「用户」身份冒出来）。所以这里跑真压缩、拿它实际写进消息的标记断言，而不是只比常量。
"""

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.study_agent.compaction import HistoryCompactionMiddleware
from app.study_agent.message_sources import SUMMARY_SOURCE, TURN_INSTRUCTION_SOURCE, is_summary_message


def test_compaction_output_is_recognized_as_summary() -> None:
    """压缩产出的那条消息要能被认出来，且必须是 HumanMessage（回显侧正是靠这两点挡它）。"""
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="## 会话目标\n复习 Cache")] * 4))
    middleware = HistoryCompactionMiddleware(llm=fake, context_window=40_000)  # 触发线 2w
    messages = [
        HumanMessage(id="h1", content="问题一"),
        AIMessage(id="a1", content="细" * 20_000),
        HumanMessage(id="h2", content="问题二"),
        AIMessage(id="a2", content="回答二"),
        HumanMessage(id="h3", content="问题三"),
    ]

    update = asyncio.run(middleware.abefore_agent({"messages": messages}, None))

    assert update is not None, "样本没触发压缩，用例本身失效"
    recaps = [m for m in update["messages"] if is_summary_message(m)]
    assert recaps, f"压缩写的标记不再是 {SUMMARY_SOURCE!r}，回显/计数/意图三处会一起静默失效"
    assert recaps[0].type == "human", "回顾是 HumanMessage——正因为如此才必须在回显侧挡掉"


def test_plain_messages_are_not_summaries() -> None:
    """普通对话消息（含缺 additional_kwargs 的）一律不误伤。"""
    assert not is_summary_message(HumanMessage(content="用户的真实问题"))
    assert not is_summary_message(AIMessage(content="回答"))


def test_turn_instruction_marker_is_distinct() -> None:
    """两个标记不能撞车：撞了的话「本轮指令」会被当成回顾从回显里删掉。"""
    assert SUMMARY_SOURCE != TURN_INSTRUCTION_SOURCE
