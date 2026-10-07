"""LLM 调用侧采集：回调处理器与它的请求级上下文。

经 ``register_configure_hook(inheritable=True)`` 全局注入，每次 LLM 调用自动挂上该 handler，
因此覆盖 create_agent 内部调用与检索工具内部的 hyde 调用，无需改 ``llm.py``。
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

# 调用角色标签前缀：调用点用 ``config={"tags": [f"{_ROLE_TAG}intent"]}`` 声明自己是谁，
# 未打标的就是 agent 主体调用（落库 role 为空串）。探针靠它挑出「每轮的首次 agent 调用」，
# 否则会把意图兜底那次 267 token 的小调用当成首轮去读命中。
_ROLE_TAG = "trace_role:"


class LLMMetricsCallbackHandler(BaseCallbackHandler):
    """统一统计 LLM 调用次数、token 用量、前缀缓存命中量与逐调用耗时（同步回调，AsyncCallbackManager 会自动包装）。

    - ``on_llm_start``：调用次数 +1，按 ``run_id`` 记下 tags 里的角色（同一请求内多次调用会并发到达，
      角色必须挂在 run_id 上而不是「最近一次」这种共享变量上），同时记下发起时刻；
    - ``on_llm_new_token``：只在流式调用上触发，记下该调用收到首个 chunk 的时刻——
      这是把「prefill 慢」与「decode 慢」分开的唯一依据。agent 主体经 ``astream`` 驱动会有该事件，
      intent/hyde/summary 走 ``ainvoke`` 收不到，其 ``first_ms`` 落 null；
    - ``on_llm_end``：从 ``AIMessage.usage_metadata`` 累加 input/output tokens，
      并读 ``input_token_details.cache_read`` 得缓存命中 token（langchain-openai 自
      ``usage.prompt_tokens_details.cached_tokens`` 映射而来；流式场景在流结束时触发一次，
      届时末块本身不带 usage，但回调收到的整段消息带）。
      总量不单独存：``total_tokens`` 恒等于 input+output，成本口径要靠 cached 折扣现算。
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_calls = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.calls_detail: list[dict[str, Any]] = []
        self._roles: dict[str, str] = {}
        # 请求锚点：handler 随 start_request 建立，故 start_ms 相对它就是「距本轮开始」
        self._t0 = time.perf_counter()
        self._starts: dict[str, float] = {}
        self._firsts: dict[str, float] = {}

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.llm_calls += 1
        run_id = kwargs.get("run_id")
        if run_id is not None:
            key = str(run_id)
            self._roles[key] = _role_from_tags(kwargs.get("tags"))
            self._starts[key] = time.perf_counter()

    def on_llm_new_token(
        self,
        token: Any,
        *,
        chunk: Any = None,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """按 run_id 记下首个 chunk 时刻；已记过就直接返回，避免逐 token 重复取时。"""
        if run_id is None:
            return
        key = str(run_id)
        if key not in self._firsts:
            self._firsts[key] = time.perf_counter()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except (IndexError, AttributeError):
            return
        usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
        if not usage:
            return
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        details = usage.get("input_token_details") or {}
        cached_tokens = int(details.get("cache_read", 0) or 0)
        self.input_tokens += input_tokens
        self.cached_tokens += cached_tokens
        self.output_tokens += output_tokens
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else ""
        end = time.perf_counter()
        start = self._starts.pop(key, None)
        first = self._firsts.pop(key, None)
        # on_llm_start 没落到位就整组记 null，不用别的时间凑一个看起来对的数
        start_ms = round((start - self._t0) * 1000, 1) if start is not None else None
        self.calls_detail.append(
            {
                "role": self._roles.pop(key, "") if run_id is not None else "",
                "input": input_tokens,
                "output": output_tokens,
                "cached": cached_tokens,
                "start_ms": start_ms,
                "ms": round((end - start) * 1000, 1) if start is not None else None,
                "first_ms": (
                    round((first - start) * 1000, 1) if (start is not None and first is not None) else None
                ),
            }
        )



def _role_from_tags(tags: Any) -> str:
    """从回调收到的 tags 里取 ``trace_role:`` 前缀标签；没有则空串（agent 主体调用）。"""
    for tag in tags or ():
        if isinstance(tag, str) and tag.startswith(_ROLE_TAG):
            return tag[len(_ROLE_TAG) :]
    return ""


# 声明并注册 configure hook：模块 import 一次即全局生效；_cb_var 为空时不注入
_cb_var: ContextVar[LLMMetricsCallbackHandler | None] = ContextVar("grad_assist_llm_metrics", default=None)
register_configure_hook(_cb_var, inheritable=True)


def attach_handler() -> None:
    """为当前请求上下文挂上一个新的计数器实例。"""
    _cb_var.set(LLMMetricsCallbackHandler())


def detach_handler() -> None:
    """清空当前请求上下文的计数器（幂等）。"""
    _cb_var.set(None)


def get_handler() -> LLMMetricsCallbackHandler | None:
    """返回当前上下文的计数器；未挂载时返回 None。"""
    return _cb_var.get()
