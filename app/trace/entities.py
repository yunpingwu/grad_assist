"""请求级观测指标实体：字段定义与落库视图。

只放数据形状，不引包内其它模块，供 context / llm_usage / metrics 单向依赖。

一次请求的逐次留痕收敛为**一条按发起顺序的 ``steps`` 时间线**：每次 LLM 调用一条
``LLMStep``（输入 messages 视图 + 输出 + token/耗时），每次工具执行一条 ``ToolStep``
（入参 + 结果摘要 + 状态 + 工具自定义 meta）。两类步骤用 ``tool_call_id``（与
``AIMessage.tool_calls[].id`` 对齐）和 ``parent_id``（工具内部发起的调用挂在工具步下，
如 hyde → search_textbook）互相关联，一条记录即可复现 agent 这一轮的内部过程。

消息与工具结果的正文保真度由 ``trace_config.payload`` 三档控制（digest/preview/full），
见 app/trace/payload.py——实体本身只承接构造好的视图，不做文本处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar


@dataclass
class MessageView:
    """一次 LLM 调用输入里的一条消息视图（正文按保真度档位留 digest/preview/full）。

    ``chars``/``sha1`` 恒定全量计算：判「两次调用的输入是否同一前缀」只认指纹，
    预览截断与否不影响该判断，故截断档下仍能对齐前缀缓存判读。
    """

    i: int
    type: str
    chars: int
    sha1: str
    preview: str
    truncated: bool
    # ai 中间回合：本次让模型看到的工具调用（args 已在对应 LLM 步的 output 里，不重复）
    tool_calls: list[dict[str, Any]] | None = None
    # tool 消息：回填的是哪次调用、哪个工具
    tool_call_id: str | None = None
    name: str | None = None

    def to_record(self) -> dict[str, Any]:
        """落库视图：必含字段在前，可选字段按需出现。"""
        record: dict[str, Any] = {
            "i": self.i,
            "type": self.type,
            "chars": self.chars,
            "sha1": self.sha1,
            "preview": self.preview,
            "truncated": self.truncated,
        }
        if self.tool_calls:
            record["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            record["tool_call_id"] = self.tool_call_id
        if self.name:
            record["name"] = self.name
        return record


@dataclass
class LLMStep:
    """一次完整 LLM 调用的留痕：start 时建、end 时回填（时间线按发起顺序 append）。

    ``status="started"`` 表示发出了但没等到结束事件（请求被中断/异常）——如实描述，
    不用别的时间凑一个看起来完成的数。``first_ms`` 只在流式调用上有值（非流式为 None），
    它把 ``ms`` 拆成 prefill 等待与生成两段，是首屏归因的唯一落点。
    """

    kind: ClassVar[str] = "llm"
    call_id: str
    seq: int = 0
    # 调用角色：intent/hyde/summary 由调用点 tags 声明（trace_role: 前缀），空串即 agent 主体调用
    role: str = ""
    # 所属工具步的 call_id（工具内部发起的调用）；顶层调用为 None——不是框架 run 的父 id
    parent_id: str | None = None
    # 距本轮开始的发起时刻 / 该调用总时长 / 流式首 chunk 等待
    start_ms: float | None = None
    ms: float | None = None
    first_ms: float | None = None
    # {"in", "cached", "out"}；cached 是 in 的子集（命中部分仍计费、只是打折）
    tokens: dict[str, int | None] | None = None
    input: list[MessageView] = field(default_factory=list)
    # 整串输入的指纹与总字符数：相邻调用 digest 相同即前缀未变（前缀缓存判读的直读口径）
    input_digest: str = ""
    input_chars: int = 0
    # {"content", "reasoning", "tool_calls": [{"id","name","args","resolved"}]}；resolved 在 flush 时派生
    output: dict[str, Any] | None = None
    status: str = "started"
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """落库视图：固定键序、值为 None 的可选键整键丢掉（保持记录干净可读）。"""
        record: dict[str, Any] = {
            "kind": self.kind,
            "seq": self.seq,
            "call_id": self.call_id,
            "role": self.role,
        }
        _put(record, "parent_id", self.parent_id)
        _put(record, "start_ms", _round(self.start_ms))
        _put(record, "ms", _round(self.ms))
        _put(record, "first_ms", _round(self.first_ms))
        _put(record, "tokens", self.tokens)
        if self.input:
            record["input"] = [m.to_record() for m in self.input]
        if self.input_digest:
            record["input_digest"] = self.input_digest
            record["input_chars"] = self.input_chars
        _put(record, "output", self.output)
        record["status"] = self.status
        _put(record, "error", self.error)
        return record


@dataclass
class ToolStep:
    """一次工具执行的留痕；重试是新的 tool run，会另起一条（同一 ``tool_call_id``）。

    ``name`` 取 on_tool_start 的 serialized：直调工具的 on_tool_end 拿不到名字，而 start 恒有。
    ``args`` 是模型传入并按 schema 解析后的实参（含 InjectedState 注入项，如实记录）。
    """

    kind: ClassVar[str] = "tool"
    call_id: str
    seq: int = 0
    name: str = ""
    # 与发起它的 AIMessage.tool_calls[].id 对齐；跨请求也是认亲键（HITL 续跑那轮同 id）
    tool_call_id: str | None = None
    # 发起它的 LLM 步 call_id（按 tool_call_id 反查）
    parent_id: str | None = None
    start_ms: float | None = None
    ms: float | None = None
    args: dict[str, Any] | None = None
    # {"chars", "sha1", "preview", "truncated"}；成功/失败只看步级 status，不重复记
    result: dict[str, Any] | None = None
    # 工具自定义观测的唯一入口（mark_tool 写入）：检索的 query/deep/recall/kept/chunk_ids、
    # 内部阶段 meta["phases"] 等
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "started"
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """落库视图：固定键序、值为 None/空的可选键整键丢掉。"""
        record: dict[str, Any] = {
            "kind": self.kind,
            "seq": self.seq,
            "call_id": self.call_id,
            "name": self.name,
        }
        _put(record, "tool_call_id", self.tool_call_id)
        _put(record, "parent_id", self.parent_id)
        _put(record, "start_ms", _round(self.start_ms))
        _put(record, "ms", _round(self.ms))
        _put(record, "args", self.args)
        _put(record, "result", self.result)
        if self.meta:
            record["meta"] = self.meta
        record["status"] = self.status
        _put(record, "error", self.error)
        return record


@dataclass
class RequestMetrics:
    """单次请求的观测指标聚合对象（可变，跨异步 task 共享同一引用）。

    请求级字段只留聚合口径；逐次明细一律进 ``steps``（唯一事实来源，不做两处维护）。
    """

    # —— 身份 ——
    request_id: str
    session_id: str = ""
    query: str = ""
    # —— 意图路由（understand_query 前置节点写入）——
    intent: str = ""
    intent_source: str = ""
    # None=意图识别没跑成（走异常兜底），0.0=判得出但置信度极低
    intent_confidence: float | None = None
    # —— 回答与异常 ——
    answer_content: str = ""
    error_type: str | None = None
    # —— 逐次留痕：LLM 调用与工具执行按发起顺序同列一条时间线（LLM 侧由回调 append，
    # 工具侧由 mark_tool 的宿主工具步承载），完成事件乱序不影响阅读（seq 是发起序）——
    steps: list[LLMStep | ToolStep] = field(default_factory=list)
    # —— 阶段耗时：请求级阶段（总耗时/首事件/首 token/intent/embedding/模型加载）；
    # 工具内部阶段（hyde/rerank 等）由 mark_stage 路由进对应工具步的 meta.phases，不在此处 ——
    stage_ms: dict[str, float] = field(default_factory=dict)
    # —— LLM 用量聚合：总 token 由 prompt + completion 现算，不单独落字段（服务端 total 就是二者之和，
    # 且它不含缓存折扣，拿它估成本会偏高）——
    llm_calls: int = 0
    prompt_tokens: int = 0
    # cached_tokens 是 prompt_tokens 的子集（命中部分按折扣计费），不是额外增量
    cached_tokens: int = 0
    completion_tokens: int = 0
    # —— 成本护栏：本轮加权成本（口径见 llm_usage.weighted_cost）与是否触发了降级 ——
    # budget_exhausted 由护栏中间件在越线时写入；turn_cost 在 flush 时统一折算，供离线重标定 BUDGET_LIMIT
    turn_cost: float = 0.0
    budget_exhausted: bool = False
    # 输出侧留痕：答案疑似复述系统提示词/漏出配置形态（由 study_service 在 flush 前置位）
    output_leak_suspected: bool = False

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        """本轮检索最终返回的 chunk ID（跨多次检索去重保序）。

        由各工具步 ``meta["chunk_ids"]`` 派生而非单独维护一份——证据的落点只有一个，
        不会出现「steps 里记了、请求级忘了记」的分家。SSE done 事件与评测脚本读它。
        """
        seen: dict[str, None] = {}
        for step in self.steps:
            for cid in getattr(step, "meta", {}).get("chunk_ids") or ():
                seen.setdefault(str(cid), None)
        return list(seen)

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
            "intent": self.intent,
            "intent_source": self.intent_source,
            "intent_confidence": self.intent_confidence,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "turn_cost": round(self.turn_cost, 1),
            "budget_exhausted": self.budget_exhausted,
            "output_leak_suspected": self.output_leak_suspected,
            "steps": [step.to_record() for step in self.steps],
            "error_type": self.error_type,
        }


def _round(value: float | None, digits: int = 1) -> float | None:
    """毫秒统一保留 1 位小数；None 原样透传（不存在的读数不编造）。"""
    return round(value, digits) if isinstance(value, (int, float)) else value


def _put(record: dict[str, Any], key: str, value: Any) -> None:
    """可选键仅在非 None 时写入（0/False/空串照常保留——它们是有意义的读数）。"""
    if value is not None:
        record[key] = value
