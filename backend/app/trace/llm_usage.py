"""LLM 与工具调用的自动采集：回调处理器、工具观测上下文与成本折算。

回调以 ``run_inline=True`` 执行（ContextVar 要传进工具体），只能做轻活、不能塞 I/O。
"""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from app.config import budget_config
from app.trace.entities import LLMStep, ToolStep
from app.trace.payload import build_llm_input, build_llm_output, build_tokens, build_tool_result, extract_usage

# 调用角色标签前缀：调用点用 ``config={"tags": [f"{_ROLE_TAG}intent"]}`` 声明自己是谁，
# 未打标的就是 agent 主体调用（落库 role 为空串）。探针靠它挑出「每轮的首次 agent 调用」，
# 否则会把意图兜底那次 267 token 的小调用当成首轮去读命中。
_ROLE_TAG = "trace_role:"

# 认领工具调用时的缺省哨兵：入参里没有这个键，就与模型实参对不上
_MISSING = object()

# 当前正在执行的工具步：on_tool_start 挂上、on_tool_end/on_tool_error 摘下。
# 并行工具各在自己的 task 上下文里（run_inline 下 set 发生在触发回调的 task 内），互不串台；
# 工具内部的 LLM 调用与 mark_tool/mark_stage 都靠它认领归属。
_current_tool: ContextVar[ToolStep | None] = ContextVar("grad_assist_current_tool", default=None)


class LLMMetricsCallbackHandler(BaseCallbackHandler):
    """统一采集 LLM 调用与工具执行，逐次写入共享的 ``steps`` 时间线。

    每次调用开一条 LLMStep / ToolStep；工具重试会产生新的 run，失败与成功各留一条步。
    """

    # 同步回调改在触发它的 task 内执行（理由见模块 docstring：ContextVar 传递 + 免线程池开销）
    run_inline = True

    def __init__(self, steps: list[LLMStep | ToolStep] | None = None) -> None:
        super().__init__()
        self.llm_calls = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        # 逐次留痕：与请求级 metrics 共享同一个 list（attach_handler 传入），未挂观测时自建
        self.steps: list[LLMStep | ToolStep] = steps if steps is not None else []
        # 请求锚点：handler 随 start_request 建立，故 start_ms 相对它就是「距本轮开始」
        self._t0 = time.perf_counter()
        self._llm_steps: dict[str, LLMStep] = {}
        self._tool_steps: dict[str, ToolStep] = {}
        self._starts: dict[str, float] = {}
        self._firsts: dict[str, float] = {}
        self._tool_tokens: dict[str, Token[ToolStep | None]] = {}
        # tool_call_id → 发起它的 LLM 步；未认领清单 (tool_call_id, 模型侧 tool_call, 已认领?) 按序
        self._tool_call_owner: dict[str, LLMStep] = {}
        self._pending_tool_calls: list[list[Any]] = []

    # ── LLM 调用 ─────────────────────────────────────────

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        """chat 模型入口：messages 即本次调用真正发出的完整消息序列。"""
        self._begin_llm(kwargs, messages=messages)

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        """非 chat 模型（BaseLLM）入口；chat 模型只触发 on_chat_model_start，两边按 run_id 去重。"""
        self._begin_llm(kwargs, messages=None)

    def _begin_llm(self, kwargs: dict[str, Any], messages: Any) -> None:
        run_id = kwargs.get("run_id")
        if run_id is None:
            return
        key = str(run_id)
        if key in self._llm_steps:
            return
        self.llm_calls += 1
        start = time.perf_counter()
        self._starts[key] = start
        owner = _current_tool.get()
        step = LLMStep(
            call_id=key,
            seq=len(self.steps) + 1,
            role=_role_from_tags(kwargs.get("tags")),
            # 工具内部发起的调用（hyde/detect 等）挂在所属工具步下。
            # 不能靠 parent_run_id：实测它的直接父级是 ``prompt | llm | parser`` 的链 run，
            # 工具 run 还在更上一层，单跳拿不到。
            parent_id=owner.call_id if owner is not None else None,
            start_ms=round((start - self._t0) * 1000, 1),
        )
        if messages:
            # 批量调用时 messages 是 list[list]，当前没有并发批量调用，如实只记第一组
            batch = messages[0] if isinstance(messages[0], list) else messages
            step.input, step.input_digest, step.input_chars = build_llm_input(list(batch))
        self.steps.append(step)
        self._llm_steps[key] = step

    def on_llm_new_token(self, token: Any, *, chunk: Any = None, run_id: Any = None, **kwargs: Any) -> None:
        """按 run_id 记下首个 chunk 时刻；已记过就直接返回，避免逐 token 重复取时。"""
        if run_id is None:
            return
        key = str(run_id)
        if key not in self._firsts:
            self._firsts[key] = time.perf_counter()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        step = self._llm_steps.pop(str(kwargs.get("run_id")), None)
        if step is None:
            return
        step.output = build_llm_output(response)
        step.tokens = build_tokens(extract_usage(response))
        _close_llm(step, self._starts, self._firsts)
        step.status = "ok"
        if step.tokens:
            self.input_tokens += step.tokens["in"] or 0
            self.cached_tokens += step.tokens["cached"] or 0
            self.output_tokens += step.tokens["out"] or 0
        # 登记本次发出的工具调用：工具步在 start 时按「同名 + 实参子集」认领，
        # 在 end 时用 ToolMessage 精确覆盖；跨请求（HITL 续跑）靠 tool_call_id 认亲
        for call in (step.output or {}).get("tool_calls") or []:
            if call.get("id"):
                self._tool_call_owner[call["id"]] = step
                self._pending_tool_calls.append([call["id"], call, False])

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        step = self._llm_steps.pop(str(kwargs.get("run_id")), None)
        if step is None:
            return
        _close_llm(step, self._starts, self._firsts)
        step.status = "error"
        step.error = type(error).__name__

    # ── 工具执行 ─────────────────────────────────────────

    def on_tool_start(self, serialized: Any, input_str: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        if run_id is None:
            return
        key = str(run_id)
        inputs = kwargs.get("inputs")
        start = time.perf_counter()
        self._starts[key] = start
        step = ToolStep(
            call_id=key,
            seq=len(self.steps) + 1,
            # 名字取 serialized：直调工具时 on_tool_end 的输出是裸值，拿不到名字；start 恒有
            name=(serialized or {}).get("name") or "",
            args=inputs if isinstance(inputs, dict) else None,
            start_ms=round((start - self._t0) * 1000, 1),
        )
        self._claim_tool_call(step)
        self.steps.append(step)
        self._tool_steps[key] = step
        self._tool_tokens[key] = _current_tool.set(step)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        step = self._close_tool(kwargs)
        if step is None:
            return
        step.result = build_tool_result(output)
        tool_call_id = getattr(output, "tool_call_id", None)
        if tool_call_id:
            # ToolMessage 是精确来源，覆盖 start 时的认领猜测
            step.tool_call_id = tool_call_id
            owner = self._tool_call_owner.get(tool_call_id)
            if owner is not None:
                step.parent_id = owner.call_id
        if not step.name and getattr(output, "name", None):
            step.name = str(output.name)
        step.status = "ok"

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        step = self._close_tool(kwargs)
        if step is None:
            return
        step.status = "error"
        step.error = type(error).__name__

    def _claim_tool_call(self, step: ToolStep) -> None:
        """按「同名 + 模型实参是本次入参的子集」在未认领的工具调用里找**唯一**候选。

        只为直调工具（拿不到 ToolMessage）也能留下关联；候选不唯一就不认——宁缺勿猜，
        ToolNode 路径的 ``on_tool_end`` 会用 ToolMessage 给出精确值并覆盖。
        """
        args = step.args or {}
        candidates = [
            entry
            for entry in self._pending_tool_calls
            if not entry[2]
            and entry[1].get("name") == step.name
            and isinstance(entry[1].get("args"), dict)
            and all(args.get(k, _MISSING) == v for k, v in entry[1]["args"].items())
        ]
        if len(candidates) != 1:
            return
        entry = candidates[0]
        entry[2] = True
        step.tool_call_id = entry[1].get("id")
        owner = self._tool_call_owner.get(step.tool_call_id)
        if owner is not None:
            step.parent_id = owner.call_id

    def _close_tool(self, kwargs: dict[str, Any]) -> ToolStep | None:
        """摘掉当前工具步上下文并回填时长；错误路径同样走这里（超时/取消也要有时间）。"""
        key = str(kwargs.get("run_id"))
        step = self._tool_steps.pop(key, None)
        token = self._tool_tokens.pop(key, None)
        start = self._starts.pop(key, None)
        if step is not None:
            step.ms = round((time.perf_counter() - start) * 1000, 1) if start is not None else None
        if token is not None:
            try:
                _current_tool.reset(token)
            except ValueError:
                # 结束事件跑在另一个上下文（框架内部换 task）时 reset 会报“不同 Context”，退回清空兜底
                _current_tool.set(None)
        return step


def _close_llm(step: LLMStep, starts: dict[str, float], firsts: dict[str, float]) -> None:
    """按 run_id 回填一次 LLM 调用的时长：``ms`` 总时长 + ``first_ms`` 首 chunk 等待。

    on_llm_start 没落到位就整组记 None，不用别的时间凑一个看起来对的数。
    """
    key = step.call_id
    start = starts.pop(key, None)
    first = firsts.pop(key, None)
    step.ms = round((time.perf_counter() - start) * 1000, 1) if start is not None else None
    step.first_ms = round((first - start) * 1000, 1) if (start is not None and first is not None) else None


def _role_from_tags(tags: Any) -> str:
    """从回调收到的 tags 里取 ``trace_role:`` 前缀标签；没有则空串（agent 主体调用）。"""
    for tag in tags or ():
        if isinstance(tag, str) and tag.startswith(_ROLE_TAG):
            return tag[len(_ROLE_TAG) :]
    return ""


# ── 工具观测的对外入口（工具函数按需补自定义指标）──────────────


def current_tool_step() -> ToolStep | None:
    """返回当前正在执行的工具步；不在工具里（或未挂观测）时为 None。"""
    return _current_tool.get()


def mark_tool(**fields: Any) -> None:
    """给当前工具步补挂自定义观测（工具自定义指标的唯一入口）。

    落在谁身上由上下文变量定：on_tool_start 把当前工具步挂进 ``_current_tool``，
    并行/嵌套的工具各在自己的 task 上下文里，互不串台；没有工具步（直调工具、未挂观测）时
    静默跳过——观测永远不该让业务路径失败。
    """
    step = _current_tool.get()
    if step is not None:
        step.meta.update(fields)


# 声明并注册 configure hook：模块 import 一次即全局生效；_cb_var 为空时不注入
_cb_var: ContextVar[LLMMetricsCallbackHandler | None] = ContextVar("grad_assist_llm_metrics", default=None)
register_configure_hook(_cb_var, inheritable=True)


def attach_handler(metrics: Any = None) -> None:
    """为当前请求上下文挂上一个新的采集 handler。

    Args:
        metrics: 请求级 ``RequestMetrics``；传入时 handler 与它**共享同一个 steps 时间线**
            （LLM 步与工具步同列一条线），不传则自建一个列表（测试/无观测场景）。
    """
    steps = metrics.steps if metrics is not None else None
    _cb_var.set(LLMMetricsCallbackHandler(steps=steps))


def detach_handler() -> None:
    """清空当前请求上下文的计数器（幂等）。"""
    _cb_var.set(None)


def get_handler() -> LLMMetricsCallbackHandler | None:
    """返回当前上下文的计数器；未挂载时返回 None。"""
    return _cb_var.get()


# ── 成本口径（预算护栏与落盘共用同一折算，单一来源）────────────────


def weighted_cost(input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """把一次/一段调用折算成加权 token 成本：未命中输入×1 + 命中输入×折扣 + 输出×权重。

    ``cached_tokens`` 是 ``input_tokens`` 的子集（命中部分仍计费、只是打折），
    折扣与输出权重取自 ``budget_config``（BUDGET_CACHE_WEIGHT / BUDGET_OUTPUT_WEIGHT）。
    与服务端 total_tokens 的口径不同：后者不含缓存折扣，直接拿它估成本会偏高。
    """
    uncached = max(input_tokens - cached_tokens, 0)
    return (
        uncached
        + cached_tokens * budget_config.cache_weight
        + output_tokens * budget_config.output_weight
    )


def current_weighted_cost() -> float:
    """当前请求已发生的加权成本（读上下文计数器的累计聚合）；未挂观测时返回 0.0。"""
    handler = get_handler()
    if handler is None:
        return 0.0
    return weighted_cost(handler.input_tokens, handler.cached_tokens, handler.output_tokens)
