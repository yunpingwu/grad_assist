"""工具护栏测试：限流与失败兜底全用官方中间件，这里验装配和两条关键行为。

覆盖：
1. 装配：graph.py 挂上两个 ToolCallLimitMiddleware（检索按轮封顶）与 ToolRetryMiddleware；
2. 工具抛裸异常不再打死整轮请求：官方重试耗尽后回一行说明文本给模型，图正常收尾；
3. 超额的工具调用在模型侧就被拦下（工具根本不执行）；
4. ``GraphBubbleUp``（澄清挂起/人工中断）不被官方重试中间件吞掉。
"""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest

import app.study_agent.graph as graph_mod
import app.study_agent.tools.search as search_mod
from app.study_agent.tools import search_textbook


class _FakeLLM:
    """只为让 build_graph 能装起来，不参与调用。"""

    _llm_type = "fake"

    def with_config(self, **_kw):
        return self


@pytest.fixture
def tool_rails(monkeypatch) -> list:
    """从 build_graph 抓生产装配，只留工具相关的护栏——测的就是线上那几件。"""
    captured: dict = {}
    # 生产里 build_graph 会带 enable_thinking 等实参调它，假实现要能吃下任意参数
    monkeypatch.setattr(graph_mod, "get_llm_client", lambda *_a, **_kw: _FakeLLM())
    monkeypatch.setattr(graph_mod, "create_agent", lambda **kwargs: captured.update(kwargs))

    graph_mod.build_graph()

    keep = (ToolCallLimitMiddleware, ToolRetryMiddleware)
    return [m for m in captured["middleware"] if isinstance(m, keep)]


class _ScriptedModel(GenericFakeChatModel):
    """脚本化模型：照单吐出预置的 AI 消息，只为把工具调用喂进图里。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


def _calls(*names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": {"query": "二叉树遍历"}, "id": f"c{i}", "type": "tool_call"}
            for i, n in enumerate(names)
        ],
    )


def _run_tools(rails: list, message: AIMessage) -> list:
    """跑一遍真实 create_agent（脚本化模型 + 生产护栏），返回所有 ToolMessage。"""
    agent = create_agent(
        model=_ScriptedModel(messages=iter([message, AIMessage(content="收尾")])),
        tools=[search_textbook],
        middleware=rails,
    )
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="二叉树遍历")]}))
    return [m for m in out["messages"] if m.type == "tool"]


def test_tool_rails_are_wired(tool_rails) -> None:
    """装配：两个检索工具各有按轮上限；失败兜底走官方重试（先重试一次，耗尽回说明）。"""
    limits = {m.tool_name: m.run_limit for m in tool_rails if isinstance(m, ToolCallLimitMiddleware)}
    assert limits == {"search_textbook": 4, "search_web": 2}

    retry = next(m for m in tool_rails if isinstance(m, ToolRetryMiddleware))
    assert retry.max_retries == 1 and retry.on_failure == "continue"


def test_bare_exception_does_not_kill_the_run(tool_rails, monkeypatch) -> None:
    """工具抛裸异常：重试耗尽后回说明文本，图照常收尾（裸异常曾直接打死整轮请求）。"""
    calls = {"n": 0}

    async def _boom(*_a, **_kw):
        calls["n"] += 1
        raise RuntimeError("milvus 连接超时")

    monkeypatch.setattr(search_mod, "search_by_query", _boom)
    results = _run_tools(tool_rails, _calls("search_textbook"))

    assert calls["n"] == 2, "应原地重试一次"
    assert len(results) == 1 and results[0].status == "error"
    assert "Tool 'search_textbook' failed after 2 attempts" in results[0].content
    assert "milvus 连接超时" in results[0].content, "说明文本必须带上原因，否则模型无从修正"


def test_tool_call_limit_blocks_excess_calls(tool_rails, monkeypatch) -> None:
    """超额调用在模型侧就被拦下：工具不执行，只回一行官方限流说明。"""
    calls = {"n": 0}

    async def _hits(*_a, **_kw):
        calls["n"] += 1
        return [{"id": "c1", "entity": {"text": "内容", "chapter": "第1章", "section": "1.1"}}]

    monkeypatch.setattr(search_mod, "search_by_query", _hits)
    monkeypatch.setattr(search_mod, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    results = _run_tools(tool_rails, _calls(*["search_textbook"] * 5))

    assert len(results) == 5
    assert calls["n"] == 4, f"上限之外的调用不该执行: {calls['n']}"
    blocked = [m for m in results if "Tool call limit exceeded" in m.content]
    assert len(blocked) == 1 and "search_textbook" in blocked[0].content


def test_interrupt_is_not_swallowed_by_retry() -> None:
    """澄清挂起/人工中断必须原样上抛：被重试中间件吞掉，ask_clarification 就再也回不了头。"""
    retry = ToolRetryMiddleware(max_retries=1, on_failure="continue")

    async def handler(_request):
        raise GraphBubbleUp

    request = ToolCallRequest(
        tool_call={"name": "ask_clarification", "args": {}, "id": "c1"}, tool=None, state={}, runtime=None
    )
    with pytest.raises(GraphBubbleUp):
        asyncio.run(retry.awrap_tool_call(request, handler))
