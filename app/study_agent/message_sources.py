"""消息来源标记：``state.messages`` 里那些「不是对话本身」的消息靠什么认领。

消息流里混着几类非对话消息，统一用 ``additional_kwargs["lc_source"]`` 打标：
``summarization``（压缩注入的整篇回顾，值沿用 LangChain 官方件的写法）与
``turn_instruction``（本轮指令 SystemMessage）。写方在中间件、读方在 API 回显与意图层，
值写错一处就静默漏出去（不报错），判据与常量因此收在这一处，各消费点别再抄字面量。

``budget_exhausted``（预算收尾说明）不在此列：它转瞬即逝、没有消费方，且那个值同时是 prompt 文件名。
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage

# 会话压缩注入的回顾（由 tests/study_agent/test_message_sources.py 钉住）
SUMMARY_SOURCE = "summarization"
# 本轮指令（教材名 + 意图行为块 + 任务目标），见 graph.record_turn_instruction
TURN_INSTRUCTION_SOURCE = "turn_instruction"


def is_summary_message(message: AnyMessage) -> bool:
    """该消息是否为会话压缩注入的整篇回顾（回显、会话计数、意图层历史一律排除它）。"""
    return (message.additional_kwargs or {}).get("lc_source") == SUMMARY_SOURCE
