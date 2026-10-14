"""会话成本预算护栏：把 app/trace 采到的 token/缓存数据变成 per-session 约束。

现有护栏都是「次数/体积」维度（步数 20 / 64k 摘要 / 写盘 HITL），本中间件补上成本维度：
用 trace 的加权口径（``app/trace/llm_usage.weighted_cost``）累计整个会话的用量，超限即降级
——禁 deep 检索、停澄清反问、提前收尾并说明，直到会话结束（标记随 checkpoint 持久）。

四个钩子各自的位置与理由：

- ``aafter_model``：把当轮 trace 读数折叠进 ``session_cost``。选 after_model 而不是轮末，
  是因为写盘确认/澄清的 interrupt 会终止本轮、轮末钩子不保证跑到（本中间件装配在 HITL 之后，
  after_model 链逆注册序执行，最后注册的先跑——被掐断的那次调用也能先折叠）；按 request_id
  重新基线，是因为 trace 计数器是请求级对象，跨请求从 0 重启；
- ``abefore_model``：越线时写 ``deep=False``（search_textbook 经 InjectedState 读取）与
  ``budget_exhausted`` 标记；
- ``awrap_model_call``：越线时把 ``ask_clarification`` 从模型可见工具表摘掉——这只是模型侧
  建议（实证：模型仍发出调用时工具体照样执行），并在消息末尾追加一条转瞬即逝的收尾说明
  （不落历史、不动 system 静态壳，前缀缓存不破）；
- ``awrap_tool_call``：真闸门——模型若仍发出澄清调用（工具表变更前的在途调用等），
  短路返回一行说明文本，不进入 interrupt。

钩子只实现 async 版本：本应用全程 astream 驱动，且成本读数来自 ContextVar，在事件循环内最稳
（与 graph.py 既有中间件同一约定）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from app.config import budget_config
from app.core import load_prompt, logger
from app.study_agent.state import StudyState
from app.trace import get_metrics
from app.trace.llm_usage import current_weighted_cost

# 澄清工具名与说明文本的 prompt 文件/来源标记（lc_source 与 turn_instruction 同款机制）
_CLARIFY_TOOL = "ask_clarification"
_BUDGET_PROMPT = "budget_exhausted"


def _tool_name(tool: Any) -> str:
    """取工具名：``request.tools`` 的元素可能是 BaseTool，也可能是 OpenAI 函数 schema 字典。"""
    if isinstance(tool, dict):
        return tool.get("name") or (tool.get("function") or {}).get("name") or ""
    return getattr(tool, "name", "")


class SessionBudgetMiddleware(AgentMiddleware[StudyState]):
    """per-session 加权 token 预算：越线后降级（禁 deep / 停追问 / 收尾说明）。

    Args:
        cost_fn: 当前请求已发生成本的读取函数（测试注入用）；缺省读 trace 计数器。
    """

    state_schema = StudyState

    def __init__(self, cost_fn: Callable[[], float] | None = None) -> None:
        super().__init__()
        self._cost_fn = cost_fn or current_weighted_cost

    # ── 折叠与实时值 ──────────────────────────────────────

    def _live_cost(self, state: StudyState) -> float:
        """实时会话成本 = 已折叠累计 + 当轮尚未折叠的增量（未挂观测的请求视为 0）。"""
        h = self._cost_fn()
        base = state.get("session_cost") or 0.0
        metrics = get_metrics()
        if metrics is None:
            return base
        if state.get("cost_fold_req") == metrics.request_id:
            return base + max(h - (state.get("cost_fold_mark") or 0.0), 0.0)
        return base + h

    async def aafter_model(self, state: StudyState, runtime: Any) -> dict[str, Any] | None:
        """把当轮 trace 读数按 request_id 重新基线后折叠进 ``session_cost``。

        新请求（request_id 变化）以当前读数为整段增量——trace 计数器从 0 重启，
        只做 ``h - mark`` 会把新一轮的花费一直吞到 h 追平上轮末值；未挂观测时
        跳过折叠且不推进 mark（如流式前的 resume 调用）。
        """
        metrics = get_metrics()
        if metrics is None:
            return None
        h = self._cost_fn()
        req = metrics.request_id
        if state.get("cost_fold_req") != req:
            delta = h
        else:
            delta = max(h - (state.get("cost_fold_mark") or 0.0), 0.0)
        if delta == 0.0 and state.get("cost_fold_req") == req and (state.get("cost_fold_mark") or 0.0) == h:
            return None  # 读数没动，不产生无谓的 checkpoint 写入
        return {
            "session_cost": (state.get("session_cost") or 0.0) + delta,
            "cost_fold_mark": h,
            "cost_fold_req": req,
        }

    # ── 降级（超限即持续生效）────────────────────────────

    async def abefore_model(self, state: StudyState, runtime: Any) -> dict[str, Any] | None:
        """越线时写 ``deep=False`` 与 ``budget_exhausted``（仅状态变化时返回，避免无谓写盘）。"""
        live = self._live_cost(state)
        if live < budget_config.limit:
            return None
        metrics = get_metrics()
        if metrics is not None:
            metrics.budget_exhausted = True  # 落盘留痕：本轮触发了降级
        updates: dict[str, Any] = {}
        if not state.get("budget_exhausted"):
            updates["budget_exhausted"] = True
            logger.warning(
                f"会话成本已用完（≈{live:.0f} >= {budget_config.limit:.0f}），"
                "降级：禁 deep / 停追问 / 收尾说明"
            )
        if state.get("deep"):
            updates["deep"] = False
        return updates or None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """越线时摘掉澄清工具 + 追加收尾说明（转瞬即逝，不进 checkpoint、不动静态壳）。"""
        over = request.state.get("budget_exhausted") or self._live_cost(request.state) >= budget_config.limit
        if not over:
            return await handler(request)
        tools = [t for t in request.tools if _tool_name(t) != _CLARIFY_TOOL]
        addendum = SystemMessage(
            content=load_prompt(_BUDGET_PROMPT),
            additional_kwargs={"lc_source": _BUDGET_PROMPT},
        )
        return await handler(request.override(tools=tools, messages=[*request.messages, addendum]))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """真闸门：越线后模型若仍发起澄清调用，短路返回说明文本，不进入 interrupt。"""
        state = request.state or {}
        if request.tool_call["name"] == _CLARIFY_TOOL and (
            state.get("budget_exhausted") or self._live_cost(state) >= budget_config.limit
        ):
            logger.info("会话成本已用完：拦截 ask_clarification 调用，改为直接作答提示")
            return ToolMessage(
                content="（会话额度已用完，不再发起澄清反问：请基于现有信息直接回答，或用一两句话说明缺口）",
                tool_call_id=request.tool_call["id"],
                name=_CLARIFY_TOOL,
                status="error",
            )
        return await handler(request)
