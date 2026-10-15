"""LLM 采集回调测试：每条 LLM 步要带输入视图、输出与耗时（假模型走真实回调链路，不打网络）。

角色靠调用点的 config tags 传递：tags 能被 ``on_chat_model_start`` 收到这一点在生产链路上
实测过，本文件把它钉住——探针的前后对比只读 agent 首次调用，读错调用就等于读噪声。
耗时同理要验到 ``on_llm_new_token`` 真能被流式调用触发：首屏归因全靠它把一次调用切成
prefill 等待与生成两段，事件收不到就是收到 0 次，不能假设框架会送。
"""

import asyncio
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk

from app.config import trace_config
from app.trace import llm_usage
from app.trace.entities import LLMStep

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


class _BoomChatModel(GenericFakeChatModel):
    """一调就炸的假模型：钉住失败路径也要关步（不能整条丢）。"""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("上游 500")


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


def _llm_steps(handler: Any) -> list[LLMStep]:
    """时间线里的 LLM 步（工具步另有 tests/trace/test_tool_steps.py 覆盖）。"""
    return [step for step in handler.steps if step.kind == "llm"]


def test_llm_steps_carry_role_and_tokens() -> None:
    """打了 trace_role: 标签的调用记该角色，没打的留空（即 agent 主体调用），并逐条记 token。"""
    llm_usage.attach_handler()
    model = GenericFakeChatModel(messages=iter([_reply("甲"), _reply("乙")]))

    async def _run() -> None:
        await model.with_config(tags=["trace_role:intent"]).ainvoke("问一")
        await model.ainvoke("问二")

    asyncio.run(_run())
    handler = llm_usage.get_handler()
    assert handler is not None
    steps = _llm_steps(handler)
    assert [step.role for step in steps] == ["intent", ""]
    assert steps[0].tokens == {"in": 120, "cached": 64, "out": 30}
    assert handler.cached_tokens == 128 and handler.llm_calls == 2


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
    assert [step.role for step in _llm_steps(handler)] == ["", "hyde", ""]


def test_llm_steps_carry_timing_split_by_first_chunk() -> None:
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
    streamed, plain = _llm_steps(handler)
    assert [step.role for step in (streamed, plain)] == ["intent", ""]
    assert streamed.first_ms is not None, "流式调用必须记到首 chunk 的等待，否则首屏无法归因"
    assert streamed.first_ms <= streamed.ms, "首 chunk 不可能晚于调用结束"
    assert plain.first_ms is None, "非流式调用没有 token 事件，不能编造首 chunk 时间"
    assert streamed.ms > 0 and plain.ms > 0, "每次调用都要有总时长"
    assert plain.start_ms >= streamed.start_ms, "start_ms 应随调用次序推进"


def test_llm_step_records_input_views_and_digest() -> None:
    """输入留痕：每条 message 的 type/字符数/指纹/预览齐全；同一输入两次调用指纹相同。"""
    llm_usage.attach_handler()
    model = GenericFakeChatModel(messages=iter([_reply("甲"), _reply("乙"), _reply("丙")]))
    messages = [SystemMessage(content="系统提示"), HumanMessage(content="指针是什么")]

    async def _run() -> None:
        await model.ainvoke(messages)
        await model.ainvoke(messages)
        await model.ainvoke([SystemMessage(content="系统提示"), HumanMessage(content="换个问题")])

    asyncio.run(_run())
    handler = llm_usage.get_handler()
    assert handler is not None
    first, second, third = _llm_steps(handler)
    assert [view.type for view in first.input] == ["system", "human"]
    assert first.input[1].chars == len("指针是什么")
    assert first.input[1].preview == "指针是什么" and first.input[1].truncated is False
    assert first.input_digest == second.input_digest, "同一输入两次调用应得同一整串指纹"
    assert third.input_digest != first.input_digest, "换了消息内容指纹必须变"


def test_llm_step_records_output_content_and_tool_calls() -> None:
    """输出留痕：content 与 tool_calls（id/name/args）都要落，供时间线 join 工具步。"""
    llm_usage.attach_handler()
    ai = AIMessage(
        content="先查教材",
        tool_calls=[{"name": "search_textbook", "args": {"query": "补码乘法"}, "id": "call_s1"}],
    )
    model = GenericFakeChatModel(messages=iter([ai]))

    asyncio.run(model.ainvoke("q"))
    handler = llm_usage.get_handler()
    assert handler is not None
    step = _llm_steps(handler)[0]
    assert step.output is not None
    assert step.output["content"] == "先查教材"
    assert step.output["tool_calls"] == [
        {"id": "call_s1", "name": "search_textbook", "args": {"query": "补码乘法"}}
    ]
    assert step.status == "ok"


def test_llm_error_closes_step_with_error() -> None:
    """失败路径：模型抛错也要关步并留异常类名（旧实现整条明细直接丢）。"""
    llm_usage.attach_handler()
    model = _BoomChatModel(messages=iter([_reply("不会用到")]))

    try:
        asyncio.run(model.ainvoke("q"))
    except RuntimeError:
        pass
    handler = llm_usage.get_handler()
    assert handler is not None
    step = _llm_steps(handler)[0]
    assert step.status == "error" and step.error == "RuntimeError"
    assert step.ms is not None, "失败调用同样要有耗时"


def test_payload_mode_controls_preview(monkeypatch) -> None:
    """保真度开关：digest 档只留指纹与长度，full 档落全文，preview 档首尾摘要并标截断。"""
    long_text = "开" * 200

    def _view_after_one_call() -> Any:
        """每档各挂一次新 handler 跑一次调用，取该次输入第一条消息的视图（下标恒为 0）。"""
        llm_usage.attach_handler()
        model = GenericFakeChatModel(messages=iter([_reply("甲")]))

        async def _run() -> None:
            await model.ainvoke([HumanMessage(content=long_text)])

        asyncio.run(_run())
        handler = llm_usage.get_handler()
        assert handler is not None
        return _llm_steps(handler)[0].input[0]

    monkeypatch.setattr(trace_config, "preview_head", 10)
    monkeypatch.setattr(trace_config, "preview_tail", 5)

    monkeypatch.setattr(trace_config, "payload", "digest")
    view = _view_after_one_call()
    assert view.preview == "" and view.truncated is True and view.chars == 200
    assert view.sha1, "最省的档位也要留指纹，前缀缓存判读靠它"

    monkeypatch.setattr(trace_config, "payload", "preview")
    view = _view_after_one_call()
    assert view.preview == "开" * 10 + "…" + "开" * 5 and view.truncated is True

    monkeypatch.setattr(trace_config, "payload", "full")
    view = _view_after_one_call()
    assert view.preview == long_text and view.truncated is False


def test_no_usage_metadata_still_records_output() -> None:
    """服务端没报用量时，tokens 取 None 但输出照落——不能因为缺 usage 把整条调用丢掉。"""
    llm_usage.attach_handler()
    model = GenericFakeChatModel(messages=iter([AIMessage(content="裸答")]))

    asyncio.run(model.ainvoke("q"))
    handler = llm_usage.get_handler()
    assert handler is not None
    step = _llm_steps(handler)[0]
    assert step.tokens is None
    assert step.output is not None and step.output["content"] == "裸答"
