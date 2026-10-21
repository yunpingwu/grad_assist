"""请求级观测入口：start/end、阶段计时、工具标记与落盘。

调用方一律从 ``app.trace`` 导入，不要经 ``app.core`` 门面（会形成循环导入）。
"""

from app.trace.context import astage, end_request, get_metrics, mark_stage, stage, start_request
from app.trace.entities import LLMStep, MessageView, RequestMetrics, ToolStep
from app.trace.llm_usage import current_weighted_cost, mark_tool, weighted_cost
from app.trace.metrics import flush_metrics

__all__ = [
    "RequestMetrics",
    "LLMStep",
    "ToolStep",
    "MessageView",
    "start_request",
    "end_request",
    "get_metrics",
    "mark_stage",
    "stage",
    "astage",
    "mark_tool",
    "flush_metrics",
    "weighted_cost",
    "current_weighted_cost",
]
