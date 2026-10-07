"""历史检索结果降级：上一轮用过的工具原文在 state 里换成一行行为记录。

``search_textbook`` 单轮返回中位 2.3k token，实测占线程上下文 76%–78%（同线程里模型自己的
回答只有 6.5k），而跨轮真正被引用的只有历史用户问题与回答本身——原文一旦被那一轮的回答
消化完，留在上下文里就只是每轮重复发送。

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
已随这套设计一起下线，历史里那行记录只说明"当时依据过什么"，不再提供捞回原文的入口。
"""

from __future__ import annotations

from langchain.agents.middleware import before_agent
from langchain_core.messages import AnyMessage, ToolMessage

# 只降级这几类"带检索 artifact、体积大、内容已被回答消化"的工具结果
_DEMOTE_TOOLS = frozenset({"search_textbook"})
# 降级标记写在消息自身：同一线程重入（resume 回中断点）据此跳过，不产生二次改写
_DEMOTED_KEY = "tool_history_demoted"
# 行为记录里最多列几个小节位置，多了只是把一行撑成一段
_MAX_LOCATIONS = 3


def _one_line_record(artifact: dict) -> str:
    """把一次检索压成一行：哪本书、按什么问句、拿回几段、在哪些小节。"""
    chunks = artifact.get("chunks") or []
    textbook = artifact.get("textbook_name") or "教材"
    query = (artifact.get("query") or "").strip()
    locations: list[str] = []
    for chunk in chunks:
        loc = chunk.get("location") or ""
        if loc and loc not in locations:
            locations.append(loc)
    shown = "、".join(locations[:_MAX_LOCATIONS])
    if len(locations) > _MAX_LOCATIONS:
        shown += f" 等 {len(locations)} 处"
    deep_note = "（深度召回）" if artifact.get("deep") else ""
    tail = f"：{shown}" if shown else ""
    return (
        f"[历史记录] 上一轮已按「{query}」在《{textbook}》取回 {len(chunks)} 段依据{deep_note}{tail}，"
        "其内容已用于当时的回答，原文不在当前上下文中。"
    )


def demote_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    """把最后一条用户提问之前的检索结果降级，返回用于写回 state 的替换消息。

    "最后一条 HumanMessage 之后"属于本轮：模型正要靠它生成回答，此刻降级等于让它无据可答。

    Args:
        messages: 本轮开始时的完整消息列表。

    Returns:
        需要降级的消息副本（同 id、同 ``tool_call_id``，仅 ``content`` 与标记变化）；无需降级时为空列表。
    """
    last_human = -1
    for index, msg in enumerate(messages):
        if msg.type == "human":
            last_human = index
    if last_human <= 0:
        return []

    updates: list[ToolMessage] = []
    for msg in messages[:last_human]:
        if msg.type != "tool" or getattr(msg, "name", None) not in _DEMOTE_TOOLS:
            continue
        if (msg.additional_kwargs or {}).get(_DEMOTED_KEY):
            continue
        artifact = getattr(msg, "artifact", None) or {}
        if not artifact.get("chunks"):
            continue  # 失败或零命中的检索没有原文可留，保持原样
        updates.append(
            msg.model_copy(
                update={
                    "content": _one_line_record(artifact),
                    "additional_kwargs": {**(msg.additional_kwargs or {}), _DEMOTED_KEY: True},
                }
            )
        )
    return updates


@before_agent
async def demote_used_tool_results(state: dict, runtime: object) -> dict:
    """挂点（轮首一次）：把往轮的检索原文降级为一行行为记录，写回 state。"""
    updates = demote_messages(state.get("messages") or [])
    return {"messages": updates} if updates else {}
