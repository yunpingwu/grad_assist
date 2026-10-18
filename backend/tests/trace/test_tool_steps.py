"""工具执行留痕测试：工具步带 args/结果/meta，工具内 LLM 靠上下文挂回工具体。

这些用例钉住的是实测过的框架行为，升 langchain 先跑这里：
- 同步回调默认走线程池（copy_context），``run_inline=True`` 后 ContextVar 才传得进工具体——
  ``test_tool_internal_llm_links_by_context_not_parent_run`` 是这条的防回归；
- ``prompt | llm | parser`` 的 chat run 直接父级是链 run 而不是工具 run，所以关联只能走上下文，
  不能靠 ``parent_run_id`` 单跳；
- 直调工具（``tool.ainvoke``，测试与 eval 的用法）同样触发回调——工具采集因此能统一走回调，
  不必依赖图装配的中间件。
"""

import asyncio
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from app.trace import astage, end_request, get_metrics, llm_usage, mark_tool, start_request
from app.trace.payload import digest


def _tool_steps(handler: Any) -> list[Any]:
    return [step for step in handler.steps if step.kind == "tool"]


def test_tool_step_records_args_result_and_meta() -> None:
    """一次工具执行：入参、结果摘要（字符数/指纹/预览）、mark_tool 挂的 meta、状态与耗时齐全。"""

    @tool
    async def lookup(query: str) -> str:
        """查点东西"""
        mark_tool(query=query, recall=3, chunk_ids=["c1", "c2"])
        return "结果正文"

    llm_usage.attach_handler()
    asyncio.run(lookup.ainvoke({"query": "指针"}))
    handler = llm_usage.get_handler()
    assert handler is not None
    step = _tool_steps(handler)[0]
    assert step.name == "lookup"
    assert step.args == {"query": "指针"}
    assert step.result == {
        "chars": len("结果正文"),
        "sha1": digest("结果正文"),
        "preview": "结果正文",
        "truncated": False,
    }
    assert step.meta == {"query": "指针", "recall": 3, "chunk_ids": ["c1", "c2"]}
    assert step.status == "ok" and step.ms is not None


def test_tool_error_marks_step() -> None:
    """工具抛异常：该步落 status/error 并带上耗时，异常照原语义向上抛（不吞）。"""

    @tool
    async def boom(query: str) -> str:
        """一调就炸"""
        raise RuntimeError("milvus 连接超时")

    llm_usage.attach_handler()
    try:
        asyncio.run(boom.ainvoke({"query": "x"}))
    except RuntimeError:
        pass
    handler = llm_usage.get_handler()
    assert handler is not None
    step = _tool_steps(handler)[0]
    assert step.status == "error" and step.error == "RuntimeError"
    assert step.ms is not None


def test_tool_internal_llm_links_by_context_not_parent_run() -> None:
    """工具内部发起的 LLM 调用（``prompt | llm | parser``）靠上下文变量挂回该工具步。

    直接父 run 是 RunnableSequence 的链 run（工具 run 还在更上一层），所以 ``parent_id``
    只能由 on_tool_start 挂上的 ContextVar 认领——这条用例就是防止有人改回 parent_run_id 单跳。
    """
    inner = GenericFakeChatModel(messages=iter([AIMessage(content="假设文档")]))
    prompt = ChatPromptTemplate.from_template("{q}")

    @tool
    async def fake_search(q: str) -> str:
        """检索"""
        chain = prompt | inner | StrOutputParser()
        return await chain.ainvoke({"q": q}, config={"tags": ["trace_role:hyde"]})

    llm_usage.attach_handler()
    asyncio.run(fake_search.ainvoke({"q": "指针"}))
    handler = llm_usage.get_handler()
    assert handler is not None
    tool_step, hyde_step = handler.steps
    assert tool_step.kind == "tool" and hyde_step.kind == "llm"
    assert hyde_step.role == "hyde"
    assert hyde_step.parent_id == tool_step.call_id, "工具内的 LLM 调用必须挂在所属工具步下"
    assert hyde_step.start_ms >= tool_step.start_ms, "子调用不可能早于宿主工具开始"


def test_parallel_tools_keep_distinct_steps_and_meta() -> None:
    """并行工具各写各的步与 meta（各自的 task 上下文），不互相串台。"""

    @tool
    async def slow(x: str) -> str:
        """慢工具"""
        await asyncio.sleep(0.03)
        mark_tool(which="slow", x=x)
        return "慢"

    @tool
    async def fast(x: str) -> str:
        """快工具"""
        mark_tool(which="fast", x=x)
        await asyncio.sleep(0.01)
        return "快"

    async def _run() -> None:
        await asyncio.gather(slow.ainvoke({"x": "a"}), fast.ainvoke({"x": "b"}))

    llm_usage.attach_handler()
    asyncio.run(_run())
    handler = llm_usage.get_handler()
    assert handler is not None
    steps = {step.name: step for step in _tool_steps(handler)}
    assert set(steps) == {"slow", "fast"}
    assert steps["slow"].meta == {"which": "slow", "x": "a"}
    assert steps["fast"].meta == {"which": "fast", "x": "b"}


def test_stage_inside_tool_lands_in_tool_step_phases() -> None:
    """工具内部阶段（hyde/rerank 这类）归到该工具步的 meta.phases，同轮多次检索各自归因。"""

    @tool
    async def with_stage(name: str) -> str:
        """带内部阶段的工具"""
        async with astage("hyde_ms"):
            await asyncio.sleep(0.01)
        async with astage("hyde_ms"):
            await asyncio.sleep(0.01)
        return "ok"

    start_request("s1", "q")
    asyncio.run(with_stage.ainvoke({"name": "a"}))
    metrics = get_metrics()
    assert metrics is not None
    step = _tool_steps(llm_usage.get_handler())[0]
    assert set(step.meta["phases"]) == {"hyde_ms"}
    assert step.meta["phases"]["hyde_ms"] > 0
    assert "hyde_ms" not in metrics.stage_ms, "工具内部阶段不该再混进请求级 stage_ms"
    end_request()


def test_stage_outside_tool_stays_request_level() -> None:
    """工具之外的阶段（总耗时/intent 等）仍落请求级 stage_ms，读数口径不变。"""

    @tool
    async def plain() -> str:
        """无阶段工具"""
        return "ok"

    async def _run() -> None:
        await plain.ainvoke({})
        async with astage("intent_ms"):
            await asyncio.sleep(0.005)

    metrics = start_request("s1", "q")
    asyncio.run(_run())
    assert metrics.stage_ms["intent_ms"] > 0
    step = _tool_steps(llm_usage.get_handler())[0]
    assert "phases" not in step.meta
    end_request()
