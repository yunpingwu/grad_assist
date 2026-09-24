"""跨轮检索结果存根化：控制发给模型的历史上下文体积。

``search_textbook`` 单次返回可达 8k+ 字符，实测占线程上下文近八成；
而跨轮消歧只依赖历史用户问题（意图兜底层可见）与最终回答，检索原文属重复信息。
本模块在每次 model 调用前把"历史轮"的检索 ToolMessage 替换为存根
（保留片段定位、chunk id 与短摘录），本轮消息原样保留；
state/checkpoint 不动，前端历史回显与审计链路不受影响。
模型需要上一轮片段原文时，按存根中的 id 调用 ``read_chunk`` 重取。
"""

from __future__ import annotations

import re

from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain_core.messages import AnyMessage

# 只存根这几类"可重取"的工具结果；list_chapters 等小体积输出不处理
_STUBBED_TOOLS = frozenset({"search_textbook"})
# 片段头：[片段N｜id｜章 > 节] 正文…（兼容旧格式 [片段N｜章 > 节]）
_HEADER = re.compile(r"^\[(片段\d+|代码)｜(.+?)\]\s?(.*)$", re.S)
# 正文短于该长度的消息存根收益为负，原样保留
_MIN_STUB_CHARS = 600
_EXCERPT_CHARS = 80


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


def stub_search_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """把历史轮的检索工具消息替换为存根，返回新列表；无改动时返回原列表对象。

    "历史轮"指最后一条 HumanMessage 之前的消息：本轮检索结果必须完整可见，
    模型依赖它生成最终回答，且写盘确认 resume 依赖消息配对。
    替换仅改 content（留壳）：type/name/tool_call_id 不动，配对不破、可幂等重放。

    Args:
        messages: 发给模型前的完整消息列表。

    Returns:
        处理后的消息列表（无改动时为入参本身）。
    """
    last_human = -1
    for index, msg in enumerate(messages):
        if msg.type == "human":
            last_human = index
    if last_human <= 0:
        return messages

    out = list(messages)
    changed = False
    for index in range(last_human):
        msg = out[index]
        if msg.type != "tool" or getattr(msg, "name", None) not in _STUBBED_TOOLS:
            continue
        content = str(msg.content)
        if len(content) < _MIN_STUB_CHARS or content.startswith("[存根]"):
            continue
        stub = _make_stub(content)
        if not stub or len(stub) >= len(content):
            continue
        out[index] = msg.model_copy(update={"content": stub})
        changed = True
    return out if changed else messages


@wrap_model_call
async def stub_old_search_results(request: ModelRequest, handler):
    """middleware 挂点：仅改本次请求的消息视图，不写回 state。"""
    messages = stub_search_messages(request.messages)
    if messages is not request.messages:
        request = request.override(messages=messages)
    return await handler(request)
