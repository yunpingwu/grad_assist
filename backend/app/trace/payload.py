"""steps 时间线的正文视图构造：按 digest/preview/full 三档保真度压缩落盘内容。

三档只作用于输入侧 message 与工具结果；模型输出恒定全量落。
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.config import trace_config
from app.trace.entities import MessageView

_OMIT = "…"
# 整串输入指纹里分隔各条消息的分隔符：取不可能出现在 type/sha1 里的控制字符，防相邻消息拼接歧义
_SEP = "\x00"


def digest(text: str) -> str:
    """内容指纹（sha1 前 12 位）——判「两次调用是否同一前缀」只看它。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def preview(text: str) -> tuple[str, bool]:
    """按保真度档位取正文。

    Returns:
        ``(正文, 是否被截断)``：digest 档只留空正文；preview 档首尾各留一段；
        full 档全文。截断标记独立成字段，读的人不必靠 ``len(preview)`` 反推。
    """
    mode = trace_config.payload
    if mode == "full":
        return text, False
    if mode == "digest":
        return "", bool(text)
    head, tail = trace_config.preview_head, trace_config.preview_tail
    if len(text) <= head + tail:
        return text, False
    return f"{text[:head]}{_OMIT}{text[-tail:] if tail > 0 else ''}", True


def _content_text(content: Any) -> str:
    """把 message.content 归一成文本：str 原样；多模态块列表只拼文本块、其余以 ``[类型]`` 占位。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(f"[{block.get('type', 'block')}]")
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def build_message_view(message: Any, index: int) -> MessageView:
    """把一条输入消息压成视图；工具调用/回填关联字段按消息类型带出。"""
    text = _content_text(getattr(message, "content", ""))
    body, truncated = preview(text)
    view = MessageView(
        i=index,
        type=getattr(message, "type", "unknown"),
        chars=len(text),
        sha1=digest(text),
        preview=body,
        truncated=truncated,
    )
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        # args 已在「发起它的那次 LLM 步 output」里，这里只留 id/name 供 join
        view.tool_calls = [{"id": tc.get("id"), "name": tc.get("name")} for tc in tool_calls]
    if view.type == "tool":
        view.tool_call_id = getattr(message, "tool_call_id", None)
        view.name = getattr(message, "name", None)
    return view


def build_llm_input(messages: list[Any]) -> tuple[list[MessageView], str, int]:
    """整批输入消息 → ``(视图列表, 整串指纹, 总字符数)``。

    整串指纹由「逐条 type + 内容指纹」拼接后再取 hash：内容相同即指纹相同，
    跨调用直接判前缀是否未变（前缀缓存判读的直读口径），不必比对正文。
    """
    views = [build_message_view(message, i) for i, message in enumerate(messages)]
    joined = _SEP.join(f"{v.type}{_SEP}{v.sha1}" for v in views)
    return views, digest(joined), sum(v.chars for v in views)


def build_llm_output(response: Any) -> dict[str, Any] | None:
    """从 on_llm_end 的 LLMResult 里取输出：``{content, reasoning, tool_calls}``。

    流式在结束时触发一次，聚合消息带 tool_calls；非 chat 的纯文本补全只有 ``text``。
    取不到 generation 时返回 None（不编造空输出）。
    """
    try:
        generation = response.generations[0][0]
    except (IndexError, AttributeError, TypeError):
        return None
    message = getattr(generation, "message", None)
    if message is None:
        return {"content": _content_text(getattr(generation, "text", "")), "reasoning": None, "tool_calls": []}
    reasoning = getattr(message, "reasoning_content", None) or (
        getattr(message, "additional_kwargs", {}) or {}
    ).get("reasoning_content")
    tool_calls = [
        {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
        for tc in (getattr(message, "tool_calls", None) or [])
    ]
    return {
        "content": _content_text(getattr(message, "content", "")),
        "reasoning": str(reasoning) if reasoning else None,
        "tool_calls": tool_calls,
    }


def extract_usage(response: Any) -> Any:
    """从 on_llm_end 的 LLMResult 取 ``usage_metadata``；取不到或服务端没报时返回 None。"""
    try:
        generation = response.generations[0][0]
    except (IndexError, AttributeError, TypeError):
        return None
    message = getattr(generation, "message", None)
    return getattr(message, "usage_metadata", None) if message is not None else None


def build_tokens(usage: Any) -> dict[str, int | None] | None:
    """``usage_metadata`` → ``{"in", "cached", "out"}``；缺字段取 None（服务端没报不编 0）。

    cached 来自 ``input_token_details.cache_read``（langchain-openai 自
    ``usage.prompt_tokens_details.cached_tokens`` 映射而来；流式在流结束时才有）。
    """
    if not usage:
        return None
    details = usage.get("input_token_details") or {}
    return {
        "in": usage.get("input_tokens"),
        "cached": details.get("cache_read"),
        "out": usage.get("output_tokens"),
    }


def build_tool_result(output: Any) -> dict[str, Any]:
    """工具返回值 → ``{chars, sha1, preview, truncated}``。

    正文优先取 ``artifact["content"]``（干净原文）：模型侧的 content 带外部资料边界与
    随机串（见 app/study_agent/safety.py），落盘要的是原文。裸 str（直调工具）与
    ToolMessage 的差异在这里抹平。
    """
    artifact = getattr(output, "artifact", None)
    text = artifact.get("content") if isinstance(artifact, dict) else None
    if not text:
        raw = getattr(output, "content", output)
        text = raw if isinstance(raw, str) else _content_text(raw)
    body, truncated = preview(text)
    return {"chars": len(text), "sha1": digest(text), "preview": body, "truncated": truncated}
