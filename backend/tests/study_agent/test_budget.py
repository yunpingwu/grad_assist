"""会话成本预算护栏测试：折算数学、跨请求折叠、降级行为与装配位置。

护栏的输入是 trace 的请求级计数器（app/trace/llm_usage），折叠进会话级 state；
跨请求必须按 request_id 重新基线，否则新一轮花费会被上一轮的 mark 吞掉——这条最易写错，重点覆盖。
"""

import asyncio
import dataclasses

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest

import app.study_agent.graph as graph_mod
import app.study_agent.tools.search as search_mod
from app.config import budget_config
from app.study_agent.middleware import SessionBudgetMiddleware
from app.study_agent.tools.clarify import ask_clarification
from app.study_agent.tools.search import search_textbook
from app.trace import end_request, start_request
from app.trace.llm_usage import (
    attach_handler,
    current_weighted_cost,
    detach_handler,
    get_handler,
    weighted_cost,
)

# ── 折算数学 ──────────────────────────────────────────────


def test_weighted_cost_matches_cache_discount_convention() -> None:
    """未命中 ×1 + 命中 ×0.2 + 输出 ×4：与服务端 total_tokens 口径不同，命中要打折。"""
    assert weighted_cost(1000, 400, 100) == 600 + 400 * 0.2 + 100 * 4.0


def test_current_weighted_cost_reads_context_handler() -> None:
    """读数来自上下文计数器：有 handler 折算三聚合，无 handler 归零。"""
    detach_handler()
    assert current_weighted_cost() == 0.0

    attach_handler()
    try:
        handler = get_handler()
        handler.input_tokens, handler.cached_tokens, handler.output_tokens = 1000, 400, 100
        assert current_weighted_cost() == weighted_cost(1000, 400, 100)
    finally:
        detach_handler()


def test_tool_internal_llm_tokens_count_into_weighted_cost() -> None:
    """工具内部发起的 LLM 调用（hyde 等）也计入成本读数——护栏不能漏掉工具里的花费。"""
    inner = GenericFakeChatModel(messages=iter([AIMessage(
        content="假设文档",
        usage_metadata={"input_tokens": 500, "output_tokens": 40, "total_tokens": 540,
                        "input_token_details": {"cache_read": 300}},
    )]))

    @tool
    async def tool_with_llm(q: str) -> str:
        """工具内调 LLM"""
        doc = await inner.ainvoke(q)
        return str(doc.content)

    detach_handler()
    assert current_weighted_cost() == 0.0
    attach_handler()
    try:
        asyncio.run(tool_with_llm.ainvoke({"q": "x"}))
        assert current_weighted_cost() == weighted_cost(500, 300, 40)
    finally:
        detach_handler()


# ── 折叠与重新基线 ────────────────────────────────────────


def test_fold_rebaselines_per_request() -> None:
    """跨请求重新基线：计数器请求级从 0 重启，新请求的花费不能被上一轮的 mark 吞掉。"""
    h = {"v": 0.0}
    mw = SessionBudgetMiddleware(cost_fn=lambda: h["v"])
    state: dict = {}

    start_request("s1", "q1")
    try:
        for v in (10.0, 30.0):
            h["v"] = v
            upd = asyncio.run(mw.aafter_model(state, None))
            if upd:
                state.update(upd)
        assert state["session_cost"] == 30.0
    finally:
        end_request()

    start_request("s1", "q2")  # 新请求：request_id 变化
    try:
        h["v"] = 5.0
        state.update(asyncio.run(mw.aafter_model(state, None)) or {})
        assert state["session_cost"] == 35.0, "新请求须以当前读数为整段增量，而非 max(h - 上轮 mark)"
    finally:
        end_request()


def test_fold_skips_without_observation_context() -> None:
    """未挂观测（如流式前的 resume 调用）：跳过折叠且不推进 mark。"""
    end_request()
    h = {"v": 99.0}
    mw = SessionBudgetMiddleware(cost_fn=lambda: h["v"])
    state = {"session_cost": 35.0, "cost_fold_mark": 5.0, "cost_fold_req": "old"}
    assert asyncio.run(mw.aafter_model(state, None)) is None
    assert state == {"session_cost": 35.0, "cost_fold_mark": 5.0, "cost_fold_req": "old"}


# ── 降级行为 ──────────────────────────────────────────────


def _request(state: dict, tools: list) -> ModelRequest:
    """按 test_graph.py 同款手法手工构造 ModelRequest（dataclass(init=False)，字段先全填 None）。"""
    kwargs = {f.name: None for f in dataclasses.fields(ModelRequest)}
    kwargs.update(
        messages=[HumanMessage(content="问题")],
        tools=tools,
        state=state,
        system_message=SystemMessage(content="SHELL"),
    )
    return ModelRequest(**kwargs)


def test_under_limit_keeps_everything_intact() -> None:
    """未越线：before_model 不写 state，wrap_model_call 原样放行（工具表与消息都不动）。"""
    mw = SessionBudgetMiddleware(cost_fn=lambda: 0.0)
    state = {"session_cost": 100.0, "deep": True}
    assert asyncio.run(mw.abefore_model(state, None)) is None

    seen: dict = {}

    async def handler(req: ModelRequest) -> ModelResponse:
        seen["tools"] = list(req.tools)
        seen["messages"] = list(req.messages)
        return ModelResponse(result=[])

    tools = [search_textbook, ask_clarification]
    asyncio.run(mw.awrap_model_call(_request(state, tools), handler))
    assert seen["tools"] == tools
    assert seen["messages"] == [HumanMessage(content="问题")]


def test_over_limit_sets_flag_and_disables_deep(monkeypatch) -> None:
    """越线：写 budget_exhausted 与 deep=False（search_textbook 经 InjectedState 读取）。"""
    monkeypatch.setattr(budget_config, "limit", 100.0)
    mw = SessionBudgetMiddleware(cost_fn=lambda: 0.0)
    state = {"session_cost": 150.0, "deep": True}

    assert asyncio.run(mw.abefore_model(state, None)) == {"budget_exhausted": True, "deep": False}

    # 已降级后不重复返回（避免每次模型调用都产生无谓的 checkpoint 写入）
    state.update({"budget_exhausted": True, "deep": False})
    assert asyncio.run(mw.abefore_model(state, None)) is None


def test_over_limit_strips_clarify_and_appends_addendum(monkeypatch) -> None:
    """越线：模型可见工具表摘掉澄清工具，消息尾部追加收尾说明，system 壳不动。"""
    monkeypatch.setattr(budget_config, "limit", 100.0)
    mw = SessionBudgetMiddleware(cost_fn=lambda: 0.0)
    seen: dict = {}

    async def handler(req: ModelRequest) -> ModelResponse:
        seen["tools"] = [getattr(t, "name", "") for t in req.tools]
        seen["messages"] = list(req.messages)
        seen["system"] = req.system_message.content
        return ModelResponse(result=[])

    asyncio.run(
        mw.awrap_model_call(
            _request({"session_cost": 150.0}, [search_textbook, ask_clarification]), handler
        )
    )

    assert "ask_clarification" not in seen["tools"] and "search_textbook" in seen["tools"]
    last = seen["messages"][-1]
    assert isinstance(last, SystemMessage), "收尾说明应追加在消息末尾"
    assert last.additional_kwargs.get("lc_source") == "budget_exhausted"
    assert "额度" in last.content
    assert seen["system"] == "SHELL", "静态壳不能被动，前缀缓存靠它"


def test_wrap_tool_call_blocks_clarify_when_over_budget(monkeypatch) -> None:
    """真闸门：只摘工具表拦不住在途调用（工具体照常执行），wrap_tool_call 必须短路。"""
    monkeypatch.setattr(budget_config, "limit", 100.0)
    mw = SessionBudgetMiddleware(cost_fn=lambda: 0.0)
    called = {"n": 0}

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        called["n"] += 1
        return ToolMessage(content="不该走到这里", tool_call_id="c1")

    request = ToolCallRequest(
        tool_call={"name": "ask_clarification", "args": {}, "id": "c1"},
        tool=None,
        state={"session_cost": 150.0},
        runtime=None,
    )
    out = asyncio.run(mw.awrap_tool_call(request, handler))

    assert called["n"] == 0, "越线后不得进入工具体（interrupt 入口）"
    assert isinstance(out, ToolMessage) and out.tool_call_id == "c1"
    assert "额度已用完" in out.content


def test_wrap_tool_call_lets_calls_through_under_limit() -> None:
    """未越线：澄清等一切工具调用原样放行。"""
    mw = SessionBudgetMiddleware(cost_fn=lambda: 0.0)
    called = {"n": 0}

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        called["n"] += 1
        return ToolMessage(content="ok", tool_call_id="c2")

    request = ToolCallRequest(
        tool_call={"name": "ask_clarification", "args": {}, "id": "c2"},
        tool=None,
        state={},
        runtime=None,
    )
    out = asyncio.run(mw.awrap_tool_call(request, handler))

    assert called["n"] == 1 and out.content == "ok"


# ── 装配位置 ──────────────────────────────────────────────


def test_budget_middleware_registered_last(monkeypatch) -> None:
    """装配必须在列表最后（HITL 之后）：after_model 逆注册序执行，成本先于 interrupt 折叠。"""
    monkeypatch.setattr(budget_config, "enabled", True)
    captured: dict = {}
    monkeypatch.setattr(graph_mod, "create_agent", lambda **kw: captured.update(kw) or "stub_graph")

    assert graph_mod.build_graph() == "stub_graph"
    chain = captured["middleware"]
    assert isinstance(chain[-1], SessionBudgetMiddleware)
    assert isinstance(chain[-2], HumanInTheLoopMiddleware), "成本护栏必须排在 HITL 之后"


# ── 真实循环（脚本化模型 + 生产中间件）────────────────────


class _RecordingScriptedModel(GenericFakeChatModel):
    """脚本化模型：照单吐预置消息，并记录每次调用实际看到的消息（断言转瞬说明是否送达）。"""

    seen: list = []

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003 - 与 test_tool_guard 同款对齐
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_mid_turn_degradation_in_real_loop(monkeypatch) -> None:
    """真实 create_agent 循环里第一轮调用后越线：第二轮调用就该吃到降级，随后正常收尾。

    收尾说明是转瞬的：它必须已送达模型，但不得落进历史。
    """
    monkeypatch.setattr(budget_config, "limit", 75.0)
    ticks = {"n": 0}

    def _fake_cost() -> float:
        ticks["n"] += 1
        return 50.0 * ticks["n"]  # 第一次模型调用后即越过 75 的线

    async def _empty_hits(*_a, **_kw):
        return []

    monkeypatch.setattr(search_mod, "search_by_query", _empty_hits)
    monkeypatch.setattr(search_mod, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    model = _RecordingScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_textbook", "args": {"query": "二叉树遍历"}, "id": "c1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="收尾回答"),
            ]
        )
    )
    agent = create_agent(
        model=model,
        tools=[search_textbook, ask_clarification],
        middleware=[SessionBudgetMiddleware(cost_fn=_fake_cost)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-budget"}}

    start_request("s-budget", "q")
    try:
        out = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="二叉树遍历")]}, config=config))
    finally:
        end_request()

    # 折叠字段是 PrivateStateAttr（不进运行输出），从 checkpoint 读终态
    final = agent.get_state(config).values
    assert out["budget_exhausted"] is True and final["session_cost"] >= 75.0
    assert len(model.seen) == 2, "脚本只有两条回复，多出来的调用说明循环没收住"
    last = model.seen[1][-1]
    assert last.type == "system" and "额度" in str(last.content), "越线后的模型调用应带收尾说明"
    assert not any(
        (m.additional_kwargs or {}).get("lc_source") == "budget_exhausted" for m in out["messages"]
    ), "转瞬说明不得落进历史"
