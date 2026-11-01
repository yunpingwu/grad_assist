"""消息来源标记：用 ``additional_kwargs["lc_source"]`` 区分回顾、本轮指令等非对话消息。

常量与判据只在本文件定义，回显、会话计数与意图层一律复用，别再抄字面量。
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
