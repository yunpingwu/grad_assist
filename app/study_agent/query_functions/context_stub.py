"""跨轮检索结果存根化：按定盘位回收范围 + 只改请求视图，控制发给模型的历史体积。

``search_textbook`` 单次返回可达 8k+ 字符，实测占线程上下文近八成；而跨轮消歧只依赖
历史用户问题（意图兜底层可见）与最终回答，检索原文属重复信息——模型要看原文时，
按存根里的 chunk id 调 ``read_chunk`` 重取即可。

但存根化是拿「前缀切断」换「删冗余」：百炼隐式缓存按「从第一条 token 起完全相同」匹配，
上一轮已经**全额发过**的检索原文，这一轮换成存根就等于在那个位置把公共前缀切断，
其后整段历史（含本轮真正需要复用的部分）一起掉命中——探针实见跨轮首轮命中塌回
静态壳 3072。两条规矩把这笔账管住：

1. **回收时机收进一个门槛**（``IDLE_RECLAIM_SECONDS``）：默认 0＝每轮回收，因为实测删冗余省下的
   比破前缀亏掉的多（对照数见常量注释）；门槛调到缓存 TTL 以上就变成「只在缓存已冷时改字节」，
   改写不花钱、活跃会话满命中，代价是上下文与计费基数一起涨；
2. **改过的范围定盘冻结**：范围写进 ``state.stub_mark_id``，只前进不回退——同一 mark 下
   视图逐字稳定，不会出现「这轮存根、下轮又放行全文」的翻转。

state 里的消息本体不动，前端历史回显与审计链路仍拿原文。
"""

from __future__ import annotations

import re
import time

from langchain.agents.middleware import ModelRequest, after_agent, before_agent, wrap_model_call
from langchain_core.messages import AnyMessage

# 只存根这几类"可重取"的工具结果；list_chapters 等小体积输出不处理
_STUBBED_TOOLS = frozenset({"search_textbook"})
# 片段头：[片段N｜id｜章 > 节] 正文…（兼容旧格式 [片段N｜章 > 节]）
_HEADER = re.compile(r"^\[(片段\d+|代码)｜(.+?)\]\s?(.*)$", re.S)
# 正文短于该长度的消息存根收益为负，原样保留
_MIN_STUB_CHARS = 600
_EXCERPT_CHARS = 80
# 调大该值只在「命中率优先、且检索量小」的线程上才划算。
IDLE_RECLAIM_SECONDS = 0.0


def _parse_fragments(content: str) -> list[dict[str, str]]:
    """把 search_textbook 输出解析为片段列表（兼容片段内多行与代码拼接）。

    Args:
        content: 工具消息原文。

    Returns:
        [{"label", "id", "location", "body"}, ...]；无片段头时返回空列表。
    """
    fragments: list[dict[str, str]] = []
    for segment in content.split("\n\n"):
        match = _HEADER.match(segment.strip())
        if match:
            label, inner, body = match.groups()
            parts = inner.split("｜", 1)
            if len(parts) == 2:
                chunk_id, location = parts
            else:
                chunk_id, location = "", parts[0]
            fragments.append(
                {"label": label, "id": chunk_id, "location": location, "body": body}
            )
        elif fragments:
            # 代码块拼回等无头延续段：属于上一片段正文
            fragments[-1]["body"] += "\n\n" + segment
    return fragments


def _make_stub(content: str) -> str:
    """把一条历史检索输出压缩为存根；不适合存根时返回空串。"""
    fragments = _parse_fragments(content)
    if not fragments:
        return ""
    lines = [
        f"[存根] 上一轮 search_textbook 返回 {len(fragments)} 个片段"
        f"（原文 {len(content)} 字符，已移出当前上下文）："
    ]
    for frag in fragments:
        prefix = "代码 " if frag["label"] == "代码" else ""
        loc = f"{prefix}{frag['location'] or '未标注位置'}"
        id_part = f" | id={frag['id']}" if frag["id"] else ""
        excerpt = " ".join(frag["body"].split())[:_EXCERPT_CHARS]
        lines.append(f"- {loc}{id_part} | 「{excerpt}」")
    lines.append("需要某片段完整原文时，从上方挑 id 调用 read_chunk(ids=[...]) 重取，勿凭摘录臆造。")
    return "\n".join(lines)


def _index_of(messages: list[AnyMessage], msg_id: str) -> int | None:
    """按消息 id 定位（id 缺失的旧 checkpoint 返回 None，交调用方放弃本次改写）。"""
    for index, msg in enumerate(messages):
        if getattr(msg, "id", None) == msg_id:
            return index
    return None


def stub_search_messages(messages: list[AnyMessage], stub_upto_id: str | None) -> list[AnyMessage]:
    """把定盘位及其之前的检索消息替换为存根，返回新列表；无改动时返回原列表对象。

    范围由 ``stub_upto_id`` 单独决定：它是某轮提问前最后一条消息的 id，之后的消息
    （本轮检索结果）必须完整可见——模型依赖它生成最终回答，写盘确认 resume 也依赖消息配对。
    范围内不留豁免：刚查完的东西换成 80 字摘录，模型按 id 调 ``read_chunk`` 重取即可。

    Args:
        messages: 发给模型前的完整消息列表。
        stub_upto_id: 定盘位消息 id；None 表示本轮不回收，一律放行全文。

    Returns:
        处理后的消息列表（无改动时为入参本身）。
    """
    if not stub_upto_id:
        return messages
    upto = _index_of(messages, stub_upto_id)
    if upto is None:
        return messages

    out = list(messages)
    changed = False
    for index in range(upto + 1):
        msg = out[index]
        if msg.type != "tool" or getattr(msg, "name", None) not in _STUBBED_TOOLS:
            continue
        content = str(msg.content)
        if len(content) < _MIN_STUB_CHARS or content.startswith("[存根]"):
            continue
        stub = _make_stub(content)
        if not stub or len(stub) >= len(content):
            continue
        # 替换仅改 content（留壳）：type/name/tool_call_id 不动，配对不破、可幂等重放
        out[index] = msg.model_copy(update={"content": stub})
        changed = True
    return out if changed else messages


def plan_reclaim_mark(state: dict, now: float | None = None) -> str | None:
    """本轮是否该把定盘位往前推进；返回新的 mark 消息 id，不该推进时返回 None。

    只有一道门：距上轮结束不足 ``IDLE_RECLAIM_SECONDS`` 时前缀缓存按热处理，改字节要付
    命中损失（当前取 0＝每轮都过，理由见常量注释）。没有 ``last_activity_at``（首轮或改造
    前的旧线程）也不动：宁少存根，不多破一次前缀。

    Args:
        state: 当前图状态（读 last_activity_at / stub_mark_id / messages）。
        now: 判定用的时刻，缺省取当前墙钟；测试传入以固定空档。

    Returns:
        新的定盘位消息 id（本轮提问前最后一条消息），不推进时为 None。
    """
    stamp = state.get("last_activity_at")
    if not isinstance(stamp, (int, float)):
        return None
    if (time.time() if now is None else now) - stamp < IDLE_RECLAIM_SECONDS:
        return None

    messages = state.get("messages") or []
    last_human = -1
    for index, msg in enumerate(messages):
        if msg.type == "human":
            last_human = index
    if last_human <= 0:
        return None
    candidate_id = getattr(messages[last_human - 1], "id", None)
    if not candidate_id:
        return None

    existing = state.get("stub_mark_id")
    if existing:
        existing_index = _index_of(messages, existing)
        # 只前进不回退：mark 已覆盖到候选位或更靠后时，本轮视图字节不变
        if existing_index is not None and existing_index >= last_human - 1:
            return None
    return candidate_id


@before_agent
async def reclaim_stale_tool_results(state: dict, runtime: object) -> dict:
    """挂点（轮首一次）：按空档决定是否推进定盘位，结果写进 state 供本轮各次调用共用。"""
    mark = plan_reclaim_mark(state)
    return {"stub_mark_id": mark} if mark else {}


@after_agent
async def stamp_turn_activity(state: dict, runtime: object) -> dict:
    """挂点（轮尾一次）：记下本轮结束时刻，下一轮据此算空档。"""
    return {"last_activity_at": time.time()}


@wrap_model_call
async def stub_old_search_results(request: ModelRequest, handler):
    """挂点：只按 state 里的定盘位改本次请求的消息视图，不写回 state。"""
    messages = stub_search_messages(request.messages, request.state.get("stub_mark_id"))
    if messages is request.messages:
        return await handler(request)
    return await handler(request.override(messages=messages))
