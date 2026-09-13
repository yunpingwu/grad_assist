"""请求级可观测性指标：分阶段耗时埋点 + LLM 调用/token 统计 + JSONL 落库。

设计要点：
- ``RequestMetrics`` 是一个**可变对象**，通过 ``ContextVar`` 存其引用（而非值）贯穿一次请求。
  LangGraph 用 ``asyncio.create_task`` 驱动节点时，context 被复制但引用不变，故各任务写入的是
  同一个实例，跨任务可累加——这是 OpenTelemetry 同类方案的标准做法。
- LLM 调用次数与 token 用量由 ``LLMMetricsCallbackHandler`` 统一统计，经
  ``register_configure_hook(inheritable=True)`` 全局注入（每次 LLM 调用时自动挂上该 handler），
  因此能同时覆盖 create_agent 内部调用与检索工具内部的 rewrite/hyde 调用，无需改 ``llm.py``。
- 落库为 ``logs/metrics.log`` 逐行 JSON，失败仅告警不阻断主链路。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from app.core.logger import logger

# 指标落盘目录（项目根 logs/，与 study_service 的 messages.log 同目录）
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass
class RequestMetrics:
    """单次请求的观测指标聚合对象（可变，跨异步 task 共享同一引用）。"""

    request_id: str
    session_id: str = ""
    query: str = ""
    rewrite_query: str = ""
    intent: str = ""
    intent_source: str = ""
    deep: bool = False
    # 各阶段耗时(ms)，同名阶段多次触发时累加求和
    stage_ms: dict[str, float] = field(default_factory=dict)
    recall_count: int = 0
    rerank_count: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error_type: str | None = None


# 请求级指标上下文：存 RequestMetrics 的引用，跨 task 共享累加
_current: ContextVar[RequestMetrics | None] = ContextVar(
    "grad_assist_req_metrics", default=None
)

class LLMMetricsCallbackHandler(BaseCallbackHandler):
    """统一统计 LLM 调用次数与 token 用量（同步回调，AsyncCallbackManager 会自动包装）。

    通过 ``register_configure_hook(inheritable=True)`` 全局注入，覆盖所有模型调用：
    - ``on_llm_start``：调用次数 +1；
    - ``on_llm_end``：从 ``AIMessage.usage_metadata`` 累加 input/output/total tokens
      （langchain-core 标准字段，流式场景在流结束时触发一次）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.llm_calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except (IndexError, AttributeError):
            return
        usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
        if not usage:
            return
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)


# 声明并注册 configure hook：模块 import 一次即全局生效；_cb_var 为空时不注入
_cb_var: ContextVar[LLMMetricsCallbackHandler | None] = ContextVar(
    "grad_assist_llm_metrics", default=None
)
register_configure_hook(_cb_var, inheritable=True)


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
    _cb_var.set(LLMMetricsCallbackHandler())
    return metrics


def end_request() -> None:
    """结束一次请求观测，清空上下文（幂等）。"""
    _current.set(None)
    _cb_var.set(None)


def get_metrics() -> RequestMetrics | None:
    """返回当前上下文的指标对象；未开启观测时返回 None（无侵入降级）。"""
    return _current.get()


def mark_stage(name: str, cost_ms: float) -> None:
    """累加某阶段耗时到当前请求指标（无上下文时静默跳过）。

    Args:
        name: 阶段名，如 ``query_rewrite_ms`` / ``embedding_ms``。
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
    """异步阶段计时上下文管理器：``async with astage("query_rewrite_ms"):``。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        mark_stage(name, (time.perf_counter() - t0) * 1000)


def flush_metrics() -> None:
    """把当前请求指标合并 LLM 计数后序列化为一行 JSON 追加到 logs/metrics.log。

    失败仅告警不阻断；无观测上下文时静默跳过。
    """
    metrics = get_metrics()
    if metrics is None:
        return

    cb = _cb_var.get()
    if cb is not None:
        metrics.llm_calls = cb.llm_calls
        metrics.prompt_tokens = cb.input_tokens
        metrics.completion_tokens = cb.output_tokens
        metrics.total_tokens = cb.total_tokens

    _write_metrics_line(metrics)


def _write_metrics_line(metrics: RequestMetrics) -> None:
    """把一条指标写入 metrics.log（毫秒保留 2 位小数）。"""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        record = asdict(metrics)
        record["ts"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        record["stage_ms"] = {
            k: round(v, 2) for k, v in metrics.stage_ms.items()
        }
        line = json.dumps(record, ensure_ascii=False)
        with (LOGS_DIR / "metrics.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.warning(f"写入 metrics 失败(request_id={metrics.request_id}): {exc}")