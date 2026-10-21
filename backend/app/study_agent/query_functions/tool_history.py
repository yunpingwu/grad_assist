"""历史工具原文降级：把往轮大体积工具结果的正文换成一句回收占位。

只回收「≥ ``_MIN_DEMOTE_CHARS`` 且 ``artifact`` 里有干净原文」的结果，只换 content、
不删消息；每条消息只改写一次，原文仍可按 artifact 回显。
"""

from __future__ import annotations

from langchain.agents.middleware import before_agent
from langchain_core.messages import AnyMessage, ToolMessage

# 回收的体量门槛（字符）：低于它的结果留在上下文里更划算——重取一次的工具往返，
# 成本高于它每轮占的那点 token（检索实测 p50 9.1k 字符，这条只挡住零散小结果）
_MIN_DEMOTE_CHARS = 2000
# 降级标记写在消息自身：同一线程重入（resume 回中断点）据此跳过，不产生二次改写
_DEMOTED_KEY = "tool_history_demoted"
# 降级后的占位正文：不复述「查了什么、在哪」——检索问句在工具调用参数里本就可见、
# 教材名每轮指令都带，占位只为说明「原文已回收、要依据请重新取用」。内容由代码写死，
# 不含任何外部字段，无需截断/清洗。
_DEMOTED_CONTENT = "（上一轮工具原文已回收：内容已用于当时的回答，需要依据时请重新取用）"


def recoverable_text(message: AnyMessage) -> str:
    """工具消息里可回显的干净原文（``artifact["content"]``）；没有就返回空串。

    这就是「回收后回显还拿得回原文」的依据，也是回收判据的一半——artifact 里没有干净原文
    的工具结果一律不动（宁可多占点上下文，也不能让回显丢掉原文）。
    """
    artifact = getattr(message, "artifact", None) or {}
    text = artifact.get("content") if isinstance(artifact, dict) else None
    return text if isinstance(text, str) else ""


def demote_messages(messages: list[AnyMessage], bookmark: str | None = None) -> list[ToolMessage]:
    """把最后一条用户提问之前的工具原文降级，返回写回 state 的替换消息。

    本轮消息与压缩水位线之下的消息都跳过：前者正被使用，后者已不在模型视图。
    """
    last_human = -1
    for index, msg in enumerate(messages):
        if msg.type == "human":
            last_human = index
    if last_human <= 0:
        return []
    floor = next((i for i, m in enumerate(messages) if m.id == bookmark), -1) if bookmark else -1

    updates: list[ToolMessage] = []
    for index, msg in enumerate(messages[:last_human]):
        if index <= floor:
            continue  # 已在水位线之下：不在视图里，回收没有意义
        if msg.type != "tool":
            continue
        if (msg.additional_kwargs or {}).get(_DEMOTED_KEY):
            continue
        if len(recoverable_text(msg)) < _MIN_DEMOTE_CHARS:
            continue  # 无原文可回显 / 体积本来就有界：留着比回收划算
        updates.append(
            msg.model_copy(
                update={
                    "content": _DEMOTED_CONTENT,
                    "additional_kwargs": {**(msg.additional_kwargs or {}), _DEMOTED_KEY: True},
                }
            )
        )
    return updates


@before_agent
async def demote_used_tool_results(state: dict, runtime: object) -> dict:
    """挂点（轮首一次）：把往轮的检索原文回收成一句占位，写回 state。"""
    updates = demote_messages(state.get("messages") or [], state.get("compacted_upto"))
    return {"messages": updates} if updates else {}
