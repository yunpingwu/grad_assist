"""历史工具原文降级：上一轮用过的工具原文在 state 里换成一句回收占位。

回收谁由「**体积大 + 原文另有去处**」两条判据决定，不按工具名单：结果过线
（≥ ``_MIN_DEMOTE_CHARS``）且 ``artifact`` 里存着干净原文的工具消息，跨轮留在上下文里
就只是每轮重新计费。``search_textbook`` 单轮返回实测 p50 9.1k 字符（最大 12.6k），
而跨轮真正被引用的只有历史用户问题与回答本身——原文一旦被那一轮的回答消化完，
留在上下文里就只是每轮重复发送。

体积有界的工具结果**不回收**：``read_file`` 分页上限 100 行、``search_web`` 固定条数，
它们留在上下文里比让模型重读更划算（回收反而换来一次多余的工具往返）。这条判据同时是
给未来工具的契约：结果可能很大时把原文放进 ``artifact``（``response_format=
"content_and_artifact"``），回收机制自动认得，不必回来改名单——名单式判据的坏处正是
新工具悄悄漏网（旧版只列了 search_textbook）。

三条机制前提决定了现在的做法：

1. **只换内容、不删消息**：``AIMessage.tool_calls`` 必须有同 ``tool_call_id`` 的结果兜着，
   删整条会被接口直接拒；留着壳换 ``content`` 即可，``id``/``name``/配对都不动。
2. **原文另有去处**：ChatOpenAI 序列化请求只带 ``content``，``artifact`` 不上线但随消息进
   checkpoint，所以降级后历史回显与审计仍从 ``artifact`` 取原文（见 ``/history``）。
3. **改写一次到位**：降级写进 state（``add_messages`` 按 id 原位替换），每条消息一生只改一次、
   永不回改，断裂点只随轮首前移，不会像阈值/滑窗那样来回翻转字节。代价实测（10-07 两轮线程，
   :8001 真跑）：第二轮首次调用 input 5411 / cached 3072——命中的只剩静态壳，往轮那行记录及其后的
   回答都排在断裂点之后按全价重发；换回来的是那 17.6k 字符原文（约 9k token）从此不再进请求。
   按命中 20% 计费折算，第二轮三次调用 ≈18.9k token 等价，「原文留着且全命中」的对照推算 ≈23.7k，
   省约两成，线程越长省得越多。

代价要认：模型对上一轮引过的原文彻底失忆，追问细节只能重新检索——重取工具 ``read_chunk``
已随这套设计一起下线，降级处只留一句「已回收」占位：不复述查过什么、在哪（检索问句在工具
调用参数里本就可见、教材名每轮指令都带），也不提供捞回原文的入口。
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
    """把最后一条用户提问之前的工具原文降级，返回用于写回 state 的替换消息。

    "最后一条 HumanMessage 之后"属于本轮：模型正要靠它生成回答，此刻降级等于让它无据可答。
    已在压缩水位线之下的消息同样跳过——它们早就不进模型视图（见 ``compaction.py``），
    再回收只是白写一次 checkpoint。

    Args:
        messages: 本轮开始时的完整消息列表。
        bookmark: 会话压缩水位线（``compacted_upto``）；缺省表示还没压缩过。

    Returns:
        需要降级的消息副本（同 id、同 ``tool_call_id``，仅 ``content`` 与标记变化）；无需降级时为空列表。
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
