"""会话压缩测试：触发口径、切点、视图裁剪，以及「失败不动 state」这条底线。

桩掉摘要调用本身（假模型），其余走生产代码——这里钉住的每条一旦破了都是静默劣化：
模型看不到本轮指令、回显塌掉、或历史被一条报错的摘要顶掉。
"""

import asyncio
import dataclasses

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.study_agent.compaction import (
    FIXED_OVERHEAD_TOKENS,
    HistoryCompactionMiddleware,
    estimate_tokens,
    reported_tokens,
    summarize_cutoff,
    view_messages,
)
from app.study_agent.message_sources import SUMMARY_SOURCE

# 测试用的窗口与内容规模：触发线 = 40_000 × 0.5 = 20_000，而字符估算含 8000 的固定开销
# （静态壳 + 工具 schema，实测口径），所以「过线」必须靠实打实的长回答，不能靠小消息堆轮数
_THRESHOLD = 20_000
_LONG_ANSWER_CHARS = 9_000


class _ScriptedSummarizer(GenericFakeChatModel):
    """照单吐预置回顾的假摘要器；``boom=True`` 时调用即抛（验失败路径）。

    ``boom`` 要声明成字段：GenericFakeChatModel 是 pydantic 模型，未声明的属性赋不上去。
    """

    boom: bool = False

    def __init__(self, reply: str = "## 会话目标\n复习 Cache", boom: bool = False):
        super().__init__(messages=iter([AIMessage(content=reply)] * 8), boom=boom)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.boom:
            raise RuntimeError("摘要服务不可用")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _turn(index: int, answer_chars: int = _LONG_ANSWER_CHARS) -> list:
    """造一轮：用户提问 → 回答。回答的体积用来把视图撑到触发线以上。"""
    return [
        HumanMessage(id=f"h{index}", content=f"第{index}轮问题：讲讲 Cache 的映射方式"),
        AIMessage(id=f"a{index}", content="回答：" + "细" * (answer_chars - 3)),
    ]


def _middleware(**kwargs) -> HistoryCompactionMiddleware:
    """压缩中间件（假摘要器 + 低窗口）：只换被注入的模型与窗口，逻辑全走生产代码。"""
    kwargs.setdefault("llm", _ScriptedSummarizer())
    kwargs.setdefault("context_window", _THRESHOLD * 2)  # trigger_ratio 默认 0.5 → 阈值 = _THRESHOLD
    return HistoryCompactionMiddleware(**kwargs)


# ── 触发口径 ──────────────────────────────────────────────


def test_reported_tokens_win_over_char_estimate() -> None:
    """触发读数取 provider 上报的 total_tokens：它是「上一次请求真实发了几 token」。

    中文实测 ≈1.3 字符/token（logs 里 192 次调用：p5 0.76 / p95 1.57），字符估算必须按这个
    口径折算并按固定开销补足，否则会低估到够不着触发线；两者取大者，上报值更大时以它为准。
    """
    messages = [
        HumanMessage(content="问题"),
        AIMessage(
            content="回答",
            usage_metadata={"input_tokens": 25_000, "output_tokens": 8, "total_tokens": 25_008},
        ),
    ]
    mw = _middleware()

    assert reported_tokens(messages) == 25_008
    assert estimate_tokens(messages) + FIXED_OVERHEAD_TOKENS < 25_008, "这批短消息的估算本就该低于上报值"
    assert mw._trigger(messages) == {"tokens": 25_008, "source": "reported"}


def test_trigger_falls_back_to_estimate_without_reported_usage() -> None:
    """没有 usage 读数时用字符估算顶上（含固定开销），并按估算口径写进读数与日志。"""
    messages = [HumanMessage(content="问题"), AIMessage(content="细" * 20_000)]
    mw = _middleware()

    trigger = mw._trigger(messages)

    assert trigger is not None and trigger["source"] == "estimated", trigger
    assert trigger["tokens"] == estimate_tokens(messages) + FIXED_OVERHEAD_TOKENS


def test_under_threshold_stays_untouched() -> None:
    """没到线就什么都不做：压缩一触发就是一次摘要调用 + 一次跨轮前缀作废，不能常态化。"""
    mw = _middleware()
    assert mw._trigger([HumanMessage(content="短的"), AIMessage(content="也短")]) is None


# ── 切点与视图 ────────────────────────────────────────────


def test_cutoff_keeps_recent_turns_by_human_messages() -> None:
    """保留区从倒数第 N 条用户提问起算；回顾本身是 HumanMessage，不能算作一轮。"""
    view = [HumanMessage(id=f"h{i}", content="q") for i in range(4)]

    assert summarize_cutoff(view, keep_turns=2) == 2
    assert summarize_cutoff(view, keep_turns=1) == 3
    assert summarize_cutoff(view, keep_turns=5) == 0, "轮数不够时无从下手，返回 0"

    summary = HumanMessage(id="s0", content="回顾", additional_kwargs={"lc_source": SUMMARY_SOURCE})
    assert summarize_cutoff([summary, *view], keep_turns=2) == 3, "回顾不该被当成一轮提问"


def test_view_hoists_latest_summary_and_hides_compacted() -> None:
    """视图 = 最新回顾置顶 + 水位线之后的消息；旧回顾与已压消息都不出现。"""
    old_summary = HumanMessage(id="s0", content="上一份回顾", additional_kwargs={"lc_source": SUMMARY_SOURCE})
    new_summary = HumanMessage(id="s1", content="最新回顾", additional_kwargs={"lc_source": SUMMARY_SOURCE})
    messages = [
        HumanMessage(id="h1", content="被压掉的问题"),
        AIMessage(id="a1", content="被压掉的回答"),
        old_summary,
        HumanMessage(id="h2", content="保留的问题"),
        new_summary,
    ]

    view = view_messages(messages, "a1")

    assert [m.id for m in view] == ["s1", "h2"], [m.id for m in view]
    assert view_messages(messages, None) == messages, "没有水位线时原样下发"
    assert view_messages(messages, "missing-id") == messages, "水位线指向的消息不在时宁多不少"


# ── 中间件行为 ────────────────────────────────────────────


def test_compaction_appends_summary_and_advances_bookmark() -> None:
    """超线时：只追加一条回顾 + 推进水位线，**不返回任何删除指令**（state 原文一条不删）。"""
    mw = _middleware()
    messages = [*_turn(1), *_turn(2), *_turn(3), HumanMessage(id="h4", content="第四轮问题")]

    update = asyncio.run(mw.abefore_agent({"messages": messages, "textbook_name": "计算机组成原理"}, None))

    assert update is not None
    assert [m.type for m in update["messages"]] == ["human"], "压缩只追加回顾，不删消息"
    assert update["messages"][0].additional_kwargs["lc_source"] == SUMMARY_SOURCE
    assert update["compacted_upto"] == "a2", "水位线 = 被压掉的最后一条真实消息（回顾不占位）"


def test_compaction_failure_leaves_state_untouched() -> None:
    """摘要调用失败 → 返回 None（state 一条不动），下一轮再试；历史绝不能因一次故障丢掉。"""
    mw = _middleware(llm=_ScriptedSummarizer(boom=True))
    messages = [*_turn(1), *_turn(2), *_turn(3), HumanMessage(id="h4", content="第四轮问题")]

    assert asyncio.run(mw.abefore_agent({"messages": messages}, None)) is None


def test_empty_summary_leaves_state_untouched() -> None:
    """回顾正文为空（模型抽风）同样不写：空回顾顶掉历史比不压缩更糟。"""
    mw = _middleware(llm=_ScriptedSummarizer(reply=""))
    messages = [*_turn(1), *_turn(2), *_turn(3), HumanMessage(id="h4", content="第四轮问题")]

    assert asyncio.run(mw.abefore_agent({"messages": messages}, None)) is None


def test_thin_droppable_region_is_left_alone() -> None:
    """可压的只剩上一份回顾时不动手：反复回炉会把回顾越摘越糊，等新内容攒够再说。"""
    mw = _middleware()
    summary = HumanMessage(id="s0", content="上一份回顾", additional_kwargs={"lc_source": SUMMARY_SOURCE})
    messages = [
        summary,
        HumanMessage(id="h1", content="问题一"),
        AIMessage(id="a1", content="细" * 20_000),
        HumanMessage(id="h2", content="问题二"),
    ]

    assert asyncio.run(mw.abefore_agent({"messages": messages}, None)) is None


def test_view_is_applied_to_model_request() -> None:
    """请求视图按水位线裁剪：模型看得见回顾与保留区，看不见已压消息与旧回顾。"""
    mw = _middleware()
    messages = [
        HumanMessage(id="h1", content="被压掉的"),
        AIMessage(id="a2", content="被压掉的回答"),
        HumanMessage(id="h3", content="保留的"),
        HumanMessage(id="s1", content="回顾正文", additional_kwargs={"lc_source": SUMMARY_SOURCE}),
    ]
    state = {"compacted_upto": "a2", "messages": messages}
    kwargs = {field.name: None for field in dataclasses.fields(ModelRequest)}
    kwargs.update(messages=list(messages), state=state, system_message=SystemMessage(content="壳"))
    request = ModelRequest(**kwargs)
    captured: dict = {}

    async def _handler(req: ModelRequest):
        captured["messages"] = req.messages
        return "model-response"

    result = asyncio.run(mw.awrap_model_call(request, _handler))

    assert result == "model-response"
    assert [m.id for m in captured["messages"]] == ["s1", "h3"]


# ── 端到端：真图里跑跨轮 ────────────────────────────────────

# 记录每次模型调用实际看到的消息（跨用例共享，用前 clear）：断言的是「视图」而非 state
_SEEN: list[list[str]] = []


class _RecordingModel(GenericFakeChatModel):
    """脚本化模型：照单吐预置回复，并记下每次调用看到的消息正文。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003 - 与 test_tool_guard 同款对齐
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003
        _SEEN.append([str(getattr(m, "content", "")) for m in messages])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_agent_view_is_compacted_between_turns() -> None:
    """端到端：真 create_agent 跨三轮跑，第三轮的请求里已压消息不见、回顾置顶、保留区原样。

    单测覆盖钩子行为，这条覆盖接线——中间件真的挂上了、state 字段真的写得进去、
    视图真的裁到了模型请求上（三处任一断了，压缩都只是白调一次摘要）。
    """
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    model = _RecordingModel(
        messages=iter(
            [AIMessage(id="a1", content="细" * 20_000), AIMessage(id="a2", content="回答二"), AIMessage(id="a3", content="回答三")]
        )
    )
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="静态壳",
        middleware=[_middleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-compaction"}}
    _SEEN.clear()
    for index in (1, 2, 3):
        asyncio.run(agent.ainvoke({"messages": [HumanMessage(id=f"h{index}", content=f"第{index}轮问题")]}, config=config))

    third = _SEEN[-1]
    assert any("## 会话目标" in content for content in third), "压缩回顾没有置顶进第三轮的请求"
    assert not any("第1轮问题" in content for content in third), "已压掉的历史不该再出现在请求里"
    assert any("第2轮问题" in content for content in third), "保留区内的消息必须原样下发"
    assert any("第3轮问题" in content for content in third), "本轮提问必须在场"


def test_summary_call_is_tagged_and_thinking_off(monkeypatch) -> None:
    """摘要那次调用带 ``trace_role:summary``（观测里与 agent 调用分开）且关思考。

    摘要是机械的信息抽取，思考只加时延与成本——与 intent/hyde 同一口径；不带标签时
    它在观测里与 agent 调用混在一起，前后对比会把摘要的账记到别处。
    """
    from app.study_agent import compaction as compaction_mod

    seen: dict = {}

    class _Client:
        def with_config(self, **kwargs):
            seen.update(kwargs)
            return self

    # 桩打在 compaction 模块自己的引用上（不是 app.clients.llm 的门面）
    monkeypatch.setattr(compaction_mod, "get_llm_client", lambda **kw: seen.update(kw) or _Client())

    compaction_mod.HistoryCompactionMiddleware(context_window=1000)

    assert seen.get("tags") == ["trace_role:summary"], seen
    assert seen.get("enable_thinking") is False, "摘要调用要关思考（与 intent/hyde 同口径）"
