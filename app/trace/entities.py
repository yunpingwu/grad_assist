"""请求级观测指标实体：字段定义与落库视图。

只放数据形状，不引包内其它模块，供 context / llm_usage / metrics 单向依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RequestMetrics:
    """单次请求的观测指标聚合对象（可变，跨异步 task 共享同一引用）。"""

    # —— 身份 ——
    request_id: str
    session_id: str = ""
    query: str = ""
    # —— 意图路由（understand_query 前置节点写入）——
    intent: str = ""
    intent_source: str = ""
    # None=意图识别没跑成（走异常兜底），0.0=判得出但置信度极低
    intent_confidence: float | None = None
    # —— 检索链路：每次 search_textbook 一条，逐条留痕以免互相覆盖 ——
    # 单条形状 {"query", "deep", "recall"(召回候选数), "kept"(截断后送入模型的条数),
    # "error"(降级时的异常类名，正常为 None)}；补拉的代码块不计入 kept，体现在 chunk ID 里
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    # 跨多次检索去重累加，保序
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    # —— 回答与异常 ——
    answer_content: str = ""
    error_type: str | None = None
    # —— 阶段耗时：同名多次触发累加，故是整请求合计而非逐次明细 ——
    stage_ms: dict[str, float] = field(default_factory=dict)
    # —— LLM 用量：总 token 由 prompt + completion 现算，不单独落字段（服务端 total 就是二者之和，
    # 且它不含缓存折扣，拿它估成本会偏高）——
    llm_calls: int = 0
    prompt_tokens: int = 0
    # cached_tokens 是 prompt_tokens 的子集（命中部分按折扣计费），不是额外增量
    cached_tokens: int = 0
    completion_tokens: int = 0
    # 逐次 LLM 调用明细，每条 {"role", "input", "output", "cached",
    # "start_ms"(距本轮开始), "ms"(该调用总时长), "first_ms"(该调用到首 chunk 的等待)}：
    # role 由调用点 tags 声明（intent/hyde/summary），空串即 agent 主体调用。
    # 按调用序号逐条留痕是因为轮内后续调用本就能命中前缀，只看聚合值会把
    # 「每轮首次 agent 调用」的跨轮差异淹掉；first_ms 只在流式调用上有值（非流式为 None），
    # 它把 ms 拆成 prefill 等待与生成两段，是首屏归因的唯一落点，相邻调用的 start_ms/ms
    # 之差即工具执行与图开销
    llm_calls_detail: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """转成落库记录（毫秒保留 2 位小数）。

        显式排列字段：时间和请求标识靠前，回答/检索证据紧随其后，耗时与 token 指标置于末尾。
        """
        return {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "request_id": self.request_id,
            "session_id": self.session_id,
            "query": self.query,
            "answer_content": self.answer_content,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrievals": self.retrievals,
            "intent": self.intent,
            "intent_source": self.intent_source,
            "intent_confidence": self.intent_confidence,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls_detail": self.llm_calls_detail,
            "error_type": self.error_type,
        }
