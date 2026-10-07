"""LLM 采集回调测试：逐调用明细要带角色、output token 与耗时（假模型走真实回调链路，不打网络）。

角色靠调用点的 config tags 传递：tags 能被 ``on_llm_start`` 收到这一点在 langchain 1.5 上是实测过的，
本文件把它钉住——探针的前后对比只读 agent 首轮那次调用，读错调用就等于读噪声。
耗时同理要验到 ``on_llm_new_token`` 真能被流式调用触发：首屏归因全靠它把一次调用切成
prefill 等待与生成两段，事件收不到就是收到 0 次，不能假设框架会送。
"""

import asyncio
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from app.trace import llm_usage

_USAGE: dict[str, Any] = {
    "input_tokens": 500,
    "output_tokens": 40,
    "total_tokens": 540,
    "input_token_details": {"cache_read": 300},
}


class _StreamFakeChatModel(GenericFakeChatModel):
    """逐块吐出、末块带 usage 的假模型：贴近 langchain-openai 流式在收尾给出用量的形态。"""

    def _stream(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for word in ("甲", "乙"):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=dict(_USAGE)))


def _reply(text: str) -> AIMessage:
    """造一条带 usage 的假回复：假模型默认不带 usage_metadata，回调会整条跳过。"""
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 64},
        },
    )


def test_calls_detail_carries_role_and_output() -> None:
    """打了 trace_role: 标签的调用记该角色，没打的留空（即 agent 主体调用），并逐条记 output。"""
    llm_usage.attach_handler()
    model = GenericFakeChatModel(messages=iter([_reply("甲"), _reply("乙")]))

    async def _run() -> None:
        await model.with_config(tags=["trace_role:intent"]).ainvoke("问一")
        await model.ainvoke("问二")

    asyncio.run(_run())
    handler = llm_usage.get_handler()
    assert handler is not None
    assert [c["role"] for c in handler.calls_detail] == ["intent", ""]
    assert handler.calls_detail[0]["output"] == 30
    assert handler.calls_detail[0]["cached"] == 64


def test_role_tag_does_not_leak_across_calls() -> None:
    """角色按 run_id 认领：一次打了标签，不该粘到后续未打标签的调用上。"""
    llm_usage.attach_handler()
    model = GenericFakeChatModel(messages=iter([_reply("甲"), _reply("乙"), _reply("丙")]))

    async def _run() -> None:
        await model.ainvoke("q")
        await model.with_config(tags=["trace_role:hyde"]).ainvoke("q")
        await model.ainvoke("q")

    asyncio.run(_run())
    handler = llm_usage.get_handler()
    assert handler is not None
    assert [c["role"] for c in handler.calls_detail] == ["", "hyde", ""]


def test_calls_detail_carries_timing_split_by_first_chunk() -> None:
    """逐调用耗时：流式调用记 first_ms（发起→首 chunk，即 prefill 段），非流式为 None，两者都有 ms。"""
    llm_usage.attach_handler()
    # 流式那次的块由 _stream 自己吐，非流式那次仍走 _generate，所以要给它备一条消息
    model = _StreamFakeChatModel(messages=iter([AIMessage(content="直 答", usage_metadata=dict(_USAGE))]))

    async def _run() -> None:
        async for _ in model.with_config(tags=["trace_role:intent"]).astream("流式一问"):
            pass
        await model.ainvoke("非流式一问")

    asyncio.run(_run())
    handler = llm_usage.get_handler()
    assert handler is not None
    streamed, plain = handler.calls_detail
    assert [c["role"] for c in handler.calls_detail] == ["intent", ""]
    assert streamed["first_ms"] is not None, "流式调用必须记到首 chunk 的等待，否则首屏无法归因"
    assert streamed["first_ms"] <= streamed["ms"], "首 chunk 不可能晚于调用结束"
    assert plain["first_ms"] is None, "非流式调用没有 token 事件，不能编造首 chunk 时间"
    assert streamed["ms"] > 0 and plain["ms"] > 0, "每次调用都要有总时长"
    assert plain["start_ms"] >= streamed["start_ms"], "start_ms 应随调用次序推进"

