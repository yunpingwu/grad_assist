"""请求级可观测性（trace）包对外入口。

一次请求的观测链路：``start_request`` 挂上下文 → 每次 LLM 调用与工具执行由
``app.trace.llm_usage`` 的回调自动落进同一条 ``steps`` 时间线（输入输出、token、耗时、
关联关系），各阶段用 ``stage`` / ``astage`` / ``mark_stage`` 记耗时（工具内部阶段自动归到
对应工具步的 ``meta.phases``），工具可用 ``mark_tool`` 补挂自家指标 →
``flush_metrics`` 格式化 JSON 追加到按天隔离的 ``logs/<日期>/metrics.log``，``end_request`` 收口。

分层与依赖方向见 ``app/trace/metrics.py`` 模块 docstring。调用方一律从 ``app.trace`` 导入，
不要引 ``app.core`` 门面（那会形成 app.core ↔ app.trace 的循环导入）。
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
