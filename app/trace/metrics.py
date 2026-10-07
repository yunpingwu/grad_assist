"""观测聚合与落库：合并请求上下文指标与 LLM 采集结果，追加写入按天隔离的 ``logs/<日期>/metrics.log``。

本包分层（依赖单向，禁止反向引 app.core 门面）：
``entities``（数据形状与落库视图）← ``context``（请求上下文/阶段计时）/ ``llm_usage``（调用侧采集），
``metrics`` 是同时读这两者的编排与落盘点，对外入口在 ``app/trace/__init__.py``。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logger import logger
from app.trace.context import get_metrics
from app.trace.llm_usage import get_handler

# 指标落盘目录（项目根 logs/，与 study_service 的 messages.log 同目录）
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_WRITE_LOCK = threading.Lock()


def flush_metrics() -> None:
    """把当前请求指标合并 LLM 调用计数与 token/缓存明细后追加到当天日期的 metrics.log。

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
        metrics.llm_calls_detail = list(handler.calls_detail)

    _write_record(metrics.to_record())


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
