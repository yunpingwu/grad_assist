"""会话压缩：把模型视图收成「最新回顾 + 最近若干轮」，state 原文一条不动。

只追加回顾并推进 ``compacted_upto`` 水位线；摘要失败或产出为空时 state 原样保留。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string
from langgraph.runtime import Runtime

from app.clients.llm import get_llm_client
from app.config import current_context_window
from app.core import load_prompt, logger
from app.core.message_sources import SUMMARY_SOURCE, is_summary_message
from app.study_agent.state import StudyState
from app.trace import get_metrics, mark_stage

# 中文正文的 token 折算口径：logs 里 192 次真实调用实测的中位数（p5 0.76 / p95 1.57 字符）
CHARS_PER_TOKEN = 1.3
# 触发线 = 窗口 × 该比例：留一半窗口给本轮增量（实测单轮可新增 ~3w token）与模型输出。
# 它同时是摘要请求的上限——被压区域 ≤ 触发线，序列化后放大 ~1.67 倍，比例过 0.6 就装不下窗口（构造时告警）
TRIGGER_RATIO = 0.5
_SERIALIZED_OVERHEAD = 1.67
# 压缩后视图里保留的最近轮数（按 HumanMessage 数）：留给指代消解与追问足够的上下文
KEEP_TURNS = 2
# 可压部分至少占触发线这么多才值得压：不然只是把上一份回顾回炉，越摘越糊
MIN_DROPPED_RATIO = 0.1
# 请求里不随消息走的固定开销（静态壳 + 工具 schema ≈ 8k token，实测），兜底估算时补上
FIXED_OVERHEAD_TOKENS = 8_000
# 摘要调用在观测里的角色标签（口径见 app/trace/llm_usage.py 的 _ROLE_TAG）
_SUMMARY_ROLE_TAG = "trace_role:summary"


# ── 纯函数：视图、触发、切点（都不依赖 state，单独可测）──────────────


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """字符口径的 token 估算（兜底用）：正文 + 工具调用实参的字符数按实测比例折算。"""
    chars = 0
    for message in messages:
        chars += len(str(message.content))
        for call in getattr(message, "tool_calls", None) or ():
            chars += len(str(call.get("args")))
    return int(chars / CHARS_PER_TOKEN)


def reported_tokens(messages: list[AnyMessage]) -> int | None:
    """取最后一条 AI 消息里 provider 上报的 ``total_tokens``（含系统提示词与工具 schema，比字符估算可信）；没上报返回 None。"""
    for message in reversed(messages):
        if message.type != "ai":
            continue
        usage = getattr(message, "usage_metadata", None) or {}
        total = usage.get("total_tokens")
        return int(total) if isinstance(total, int) else None
    return None


def view_messages(messages: list[AnyMessage], bookmark: str | None) -> list[AnyMessage]:
    """按水位线裁出模型视图：最新回顾置顶 + 水位线之后的消息。

    没有水位线、或水位线指的条目已不在列表里（历史被外部改动过）时原样返回——
    宁多给模型看几条，也不要在这种边上悄悄吞掉历史。
    """
    if not bookmark:
        return list(messages)
    index = next((i for i, m in enumerate(messages) if m.id == bookmark), None)
    if index is None:
        return list(messages)
    # 回顾本身不占位置：旧回顾是台账，模型只看最新那一份（置顶注入）
    kept = [m for m in messages[index + 1 :] if not is_summary_message(m)]
    latest = latest_summary(messages)
    return [latest, *kept] if latest is not None else kept


def latest_summary(messages: list[AnyMessage]) -> AnyMessage | None:
    """取最后一条回顾消息（压缩只保留最新一份进视图）。"""
    return next((m for m in reversed(messages) if is_summary_message(m)), None)


def summarize_cutoff(view: list[AnyMessage], keep_turns: int) -> int:
    """保留区起点：倒数第 ``keep_turns`` 条**用户提问**的下标；不够则 0（无可压）。

    回顾本身是 HumanMessage，数轮次时要排除，否则它会被当成一轮用户提问。
    """
    humans = [i for i, m in enumerate(view) if m.type == "human" and not is_summary_message(m)]
    return humans[-keep_turns] if len(humans) >= keep_turns else 0


# ── 中间件 ──────────────────────────────────────────────


class HistoryCompactionMiddleware(AgentMiddleware[StudyState]):
    """轮边界压缩：超线才动、压完只追加、失败不写。

    Args:
        llm: 摘要模型；缺省取主模型并关思考（带 ``trace_role:summary`` 标签）。
        context_window: 模型窗口（token）；缺省按当前供应商配置取。
    """

    state_schema = StudyState

    def __init__(
        self,
        *,
        llm: Any = None,
        context_window: int | None = None,
        trigger_ratio: float = TRIGGER_RATIO,
        keep_turns: int = KEEP_TURNS,
    ) -> None:
        super().__init__()
        self.llm = llm if llm is not None else get_llm_client(enable_thinking=False).with_config(tags=[_SUMMARY_ROLE_TAG])
        window = context_window if context_window is not None else current_context_window()
        self.threshold = max(int(window * trigger_ratio), 1)
        self.keep_turns = keep_turns
        if trigger_ratio * _SERIALIZED_OVERHEAD >= 1.0:
            logger.warning(
                f"压缩比例 {trigger_ratio} × 序列化放大 {_SERIALIZED_OVERHEAD} ≥ 1：被压区域整段喂给摘要模型时"
                f"可能超过窗口（{window}），摘要会失败——比例调回 0.5 附近或把窗口配准"
            )

    # ── 轮首：要不要压 ────────────────────────────────────

    async def abefore_agent(self, state: StudyState, runtime: Runtime) -> dict[str, Any] | None:
        """超线且确有可压内容时生成回顾，返回 state 增量；否则原样返回 None（不动 state）。"""
        messages = state.get("messages") or []
        bookmark = state.get("compacted_upto")
        view = view_messages(messages, bookmark)
        trigger = self._trigger(view)
        if trigger is None:
            return None

        cutoff = summarize_cutoff(view, self.keep_turns)
        if cutoff <= 0:
            return None
        dropped, kept = view[:cutoff], view[cutoff:]
        dropped_tokens = estimate_tokens(dropped)
        if dropped_tokens < self.threshold * MIN_DROPPED_RATIO:
            # 可压的只剩上一份回顾：硬压等于把回顾反复回炉，等信息攒够再说
            logger.info(f"会话已超触发线但可压内容太薄（≈{dropped_tokens} token），本轮不压缩")
            return None

        start = time.perf_counter()
        summary = await self._summarize(dropped, state.get("textbook_name") or "")
        mark_stage("compaction_ms", (time.perf_counter() - start) * 1000)
        if not summary:
            return None

        # 新水位线 = 被压掉消息里最后一条**真实消息**的 id（回顾不占水位线）
        new_bookmark = next(
            (str(message.id) for message in reversed(dropped) if not is_summary_message(message) and message.id),
            None,
        ) or bookmark
        logger.info(
            f"会话压缩（{trigger['source']} {trigger['tokens']} ≥ {self.threshold}）：压掉 {len(dropped)} 条"
            f"（≈{dropped_tokens} token），保留最近 {len(kept)} 条；原文留在 checkpoint，只裁模型视图"
        )
        metrics = get_metrics()
        if metrics is not None:
            metrics.compacted = {
                "trigger_tokens": trigger["tokens"],
                "trigger_source": trigger["source"],
                "trigger_reported": trigger["reported"],
                "trigger_estimated": trigger["estimated"],
                "dropped_msgs": len(dropped),
                "dropped_tokens": dropped_tokens,
                "kept_msgs": len(kept),
                "bookmark": new_bookmark,
            }
        # 只追加：回顾进消息流（打标，回显与计数会过滤掉），水位线写 state 供视图裁剪
        return {
            "messages": [HumanMessage(content=summary, additional_kwargs={"lc_source": SUMMARY_SOURCE})],
            "compacted_upto": new_bookmark,
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """按水位线裁请求视图：回顾置顶 + 水位线之后的消息（未压缩过时原样下发）。"""
        bookmark = (request.state or {}).get("compacted_upto")
        if not bookmark:
            return await handler(request)
        return await handler(request.override(messages=view_messages(list(request.messages), bookmark)))

    # ── 内部 ────────────────────────────────────────────

    def _trigger(self, view: list[AnyMessage]) -> dict[str, Any] | None:
        """到线返回触发读数，否则 None。

        口径取「上报值」与「字符估算」的较大者：两者量的不是同一刻——上报值是**上一次调用**
        的用量，估算量的是**当前视图**（多一条新提问、可能多几段工具结果），所以估算常常略大，
        这不是异常。只有**压根没有上报读数**才说明服务端没回 usage、口径真的退化了，那时才告警。
        """
        reported = reported_tokens(view)
        estimated = estimate_tokens(view) + FIXED_OVERHEAD_TOKENS
        used = max(reported or 0, estimated)
        if used < self.threshold:
            return None
        if reported is None:
            logger.warning(f"会话压缩触发口径没有 provider 上报读数（服务端没回 usage？），退化为字符估算 ≈{estimated}")
        return {
            "tokens": used,
            "source": "reported" if reported is not None else "estimated",
            "reported": reported,
            "estimated": estimated,
        }

    async def _summarize(self, dropped: list[AnyMessage], textbook_name: str) -> str:
        """生成回顾正文；失败或空结果返回空串（调用方据此保持 state 不动）。"""
        prompt = load_prompt("study_summary").format(
            textbook_name=textbook_name or "（未指定）",
            messages=get_buffer_string(dropped, format="xml"),
        )
        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as exc:
            # 压缩是省流优化，不该把本轮对话拖下水：保持完整历史，下一轮再试
            logger.warning(f"会话压缩调用失败，本轮保留完整历史（下轮再试）: {exc!r}")
            return ""
        return str(getattr(response, "text", "") or "").strip()

__all__ = [
    "CHARS_PER_TOKEN",
    "HistoryCompactionMiddleware",
    "estimate_tokens",
    "latest_summary",
    "reported_tokens",
    "summarize_cutoff",
    "view_messages",
]
