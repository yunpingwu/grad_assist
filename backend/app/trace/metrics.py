"""观测聚合与落库：补齐请求级读数后追加写入 ``logs/<日期>/metrics.log``。"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logger import logger
from app.trace.context import get_metrics
from app.trace.entities import RequestMetrics
from app.trace.llm_usage import get_handler, weighted_cost

# 指标落盘目录（项目根 logs/，与后端运行日志同目录）
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_WRITE_LOCK = threading.Lock()


def flush_metrics() -> None:
    """把当轮聚合读数与派生标记合并进指标后追加到当天日期的 metrics.log。

    写失败仅告警不阻断；无观测上下文时静默跳过。
    """
    metrics = get_metrics()
    if metrics is None:
        return

    handler = get_handler()
    if handler is not None:
        metrics.llm_calls = handler.llm_calls
        metrics.prompt_tokens = handler.input_tokens
        metrics.cached_tokens = handler.cached_tokens
        metrics.completion_tokens = handler.output_tokens
        # 成本护栏的折算与预算侧共用同一函数（口径单一来源），落盘供离线重标定 BUDGET_LIMIT
        metrics.turn_cost = weighted_cost(
            handler.input_tokens, handler.cached_tokens, handler.output_tokens
        )

    _mark_resolved(metrics)
    _write_record(metrics.to_record())


def _mark_resolved(metrics: RequestMetrics) -> None:
    """给 agent 主体调用的 tool_calls 标 ``resolved``：本请求内是否有对应工具步执行过。

    被 HITL / 限流 / 中断拦下的调用会留下 false，续跑的执行步在另一条请求记录里。
    """
    executed = {
        step.tool_call_id for step in metrics.steps if getattr(step, "tool_call_id", None)
    }
    for step in metrics.steps:
        if getattr(step, "role", ""):
            continue
        output = getattr(step, "output", None)
        for call in (output or {}).get("tool_calls") or []:
            call["resolved"] = call.get("id") in executed


def _write_record(record: dict[str, Any]) -> None:
    """把记录序列化为缩进 JSON 追加到 ``logs/<日期>/metrics.log``，记录之间用空行分隔，方便人工查看长回答。

    使用进程内写锁，避免并发请求的多行记录交错写入。
    """
    try:
        # 目录取记录自身的日期，避免跨零点时记录时间戳与实际落盘目录分家
        day = str(record.get("ts", ""))[:10] or datetime.now().strftime("%Y-%m-%d")
        path = LOGS_DIR / day / "metrics.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        formatted = json.dumps(record, ensure_ascii=False, indent=2)
        with _LOG_WRITE_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(formatted + "\n\n")
    except Exception as exc:
        logger.warning(f"写入 metrics 失败(request_id={record.get('request_id')}): {exc}")
