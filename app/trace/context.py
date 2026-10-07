"""请求级观测上下文：一次请求的指标对象与阶段计时埋点。

``RequestMetrics`` 通过 ``ContextVar`` 存其**引用**（而非值）贯穿一次请求：LangGraph 用
``asyncio.create_task`` 驱动节点时 context 被复制但引用不变，故各任务写入的是同一个实例，
跨任务可累加——这是 OpenTelemetry 同类方案的标准做法。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

from app.trace.entities import RequestMetrics
from app.trace.llm_usage import attach_handler, detach_handler

# 请求级指标上下文：存 RequestMetrics 的引用，跨 task 共享累加
_current: ContextVar[RequestMetrics | None] = ContextVar("grad_assist_req_metrics", default=None)


def start_request(session_id: str, query: str) -> RequestMetrics:
    """开启一次请求观测：生成 request_id，创建并挂载指标与 LLM 计数上下文。

    Args:
        session_id: 会话 ID（裸 ID）。
        query: 用户原始问题。

    Returns:
        新建的指标对象（调用方持有引用，收尾读取聚合结果）。
    """
    metrics = RequestMetrics(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        query=query,
    )
    _current.set(metrics)
    attach_handler()
    return metrics


def end_request() -> None:
    """结束一次请求观测，清空上下文（幂等）。"""
    _current.set(None)
    detach_handler()


def get_metrics() -> RequestMetrics | None:
    """返回当前上下文的指标对象；未开启观测时返回 None（无侵入降级）。"""
    return _current.get()


def mark_stage(name: str, cost_ms: float) -> None:
    """累加某阶段耗时到当前请求指标（无上下文时静默跳过）。

    Args:
        name: 阶段名，如 ``hyde_ms`` / ``embedding_ms``。
        cost_ms: 本次耗时（毫秒）。
    """
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
