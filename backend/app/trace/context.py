"""请求级观测上下文：用 ContextVar 存 RequestMetrics 的引用，跨 task 共享同一实例。"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

from app.trace.entities import RequestMetrics
from app.trace.llm_usage import attach_handler, current_tool_step, detach_handler

# 请求级指标上下文：存 RequestMetrics 的引用，跨 task 共享累加
_current: ContextVar[RequestMetrics | None] = ContextVar("grad_assist_req_metrics", default=None)


def start_request(session_id: str, query: str) -> RequestMetrics:
    """开启一次请求观测：创建指标对象并挂载 LLM 采集上下文。"""
    metrics = RequestMetrics(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        query=query,
    )
    _current.set(metrics)
    # handler 与 metrics 共享同一条 steps 时间线：LLM 调用步与工具执行步同列一条线
    attach_handler(metrics)
    return metrics


def end_request() -> None:
    """结束一次请求观测，清空上下文（幂等）。"""
    _current.set(None)
    detach_handler()


def get_metrics() -> RequestMetrics | None:
    """返回当前上下文的指标对象；未开启观测时返回 None（无侵入降级）。"""
    return _current.get()


def mark_stage(name: str, cost_ms: float) -> None:
    """累加阶段耗时：工具步内触发的记进该步 ``meta.phases``，否则落请求级 ``stage_ms``。"""
    step = current_tool_step()
    if step is not None:
        phases = step.meta.setdefault("phases", {})
        phases[name] = round(phases.get(name, 0.0) + cost_ms, 2)
        return
    metrics = get_metrics()
    if metrics is not None:
        metrics.stage_ms[name] = metrics.stage_ms.get(name, 0.0) + cost_ms


@contextmanager
def stage(name: str) -> Iterator[None]:
    """同步阶段计时上下文管理器：``with stage("embedding_ms"):``。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        mark_stage(name, (time.perf_counter() - t0) * 1000)


@asynccontextmanager
async def astage(name: str) -> AsyncIterator[None]:
    """异步阶段计时上下文管理器：``async with astage("hyde_ms"):``。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        mark_stage(name, (time.perf_counter() - t0) * 1000)
