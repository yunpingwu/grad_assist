"""真实 agent 循环里的留痕集成：假模型 + 假工具走 create_agent 真图（不打网络）。

钉住的是实测过的框架行为，升 langchain 先跑这里：
- 一次「模型 → 工具 → 模型」跑完，时间线按发起顺序排列、两步用 tool_call_id/parent_id 互相挂住；
- 工具重试是**新的 tool run**：失败那次只发 ``on_tool_error`` 不发 ``on_tool_end``，
  于是留下「error + ok」两条同 ``tool_call_id`` 的步（读的时候别当重复记账）。
"""

import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.trace import llm_usage


class _ToolFakeModel(GenericFakeChatModel):
    """支持 bind_tools 的假 chat 模型（GenericFakeChatModel 本身不实现工具绑定）。"""

    def bind_tools(self, tools: object, **kwargs: object) -> "_ToolFakeModel":
        return self


def test_agent_loop_timeline_links_llm_and_tool_steps() -> None:
    """时间线：llm(发起) → tool(失败) → tool(重试成功) → llm(作答)，关联键齐全。"""
    attempts = {"n": 0}

    @tool
    async def flaky(q: str) -> str:
        """第一次故障、第二次成功"""
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("瞬时故障")
        return "第二次的结果"

    model = _ToolFakeModel(messages=iter([
        AIMessage(content="先取数", tool_calls=[{"name": "flaky", "args": {"q": "x"}, "id": "c1"}]),
        AIMessage(content="最终回答"),
    ]))
    agent = create_agent(
        model=model,
        tools=[flaky],
        middleware=[ToolRetryMiddleware(max_retries=1)],
    )

    llm_usage.attach_handler()
    asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="问题")]}))
    handler = llm_usage.get_handler()
    assert handler is not None
    steps = handler.steps
    assert [step.kind for step in steps] == ["llm", "tool", "tool", "llm"]

    first_llm, failed_tool, retried_tool, final_llm = steps
    assert first_llm.output["tool_calls"] == [
        {"id": "c1", "name": "flaky", "args": {"q": "x"}}
    ]
    assert failed_tool.status == "error" and failed_tool.error == "RuntimeError"
    assert retried_tool.status == "ok" and retried_tool.result["preview"] == "第二次的结果"
    assert failed_tool.tool_call_id == retried_tool.tool_call_id == "c1", "两次尝试是同一个逻辑调用"
    assert failed_tool.parent_id == retried_tool.parent_id == first_llm.call_id
    assert final_llm.output["content"] == "最终回答"
    assert [view.type for view in final_llm.input][:2] == ["human", "ai"]
    assert any(view.type == "tool" for view in final_llm.input), "第二次调用要看得见工具结果"
