"""首屏埋点测试：首个可见事件与首个正文 token 分开计时（假图，不触达 PostgreSQL）。"""

import asyncio
from types import SimpleNamespace
from typing import Any

from app.services import study as svc
from app.trace import end_request, start_request


def _model_chunk(*, content: str = "", reasoning: str | None = None, tool_calls: list | None = None):
    """造一个 model 节点的流式 chunk：字段缺省即该 chunk 不携带这部分内容。"""
    return SimpleNamespace(content=content, reasoning_content=reasoning, tool_call_chunks=tool_calls or [])


def _tool_result(name: str):
    """造一个 tools 节点的 chunk：只有带 name 的工具结果块才会推给前端。"""
    return SimpleNamespace(name=name, content="检索到 3 条")


class _Snap:
    """收尾状态：不在任何中断上（next 为空），走 done 分支。"""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {"messages": []}
        self.tasks: tuple = ()
        self.next = None


class _FakeGraph:
    def __init__(self, pairs: list) -> None:
        self._pairs = pairs

    async def aget_state(self, config) -> _Snap:
        return _Snap()

    async def astream(self, run_input, *, config, stream_mode, durability):
        for pair in self._pairs:
            await asyncio.sleep(0.08)  # 让每个事件在时间轴上分开，计时才会把「只记一次」与「逐次累加」区分开
            yield pair


async def _collect() -> list[str]:
    return [ev["type"] async for ev in svc.stream_agent({}, {"configurable": {}}, "s1")]


def _run(monkeypatch, pairs: list) -> tuple[list[str], dict[str, float]]:
    """用假图跑一轮 stream_agent，回传事件类型序列与该请求的阶段耗时。"""
    monkeypatch.setattr(svc.lifecycle, "study_graph", _FakeGraph(pairs))
    metrics = start_request("s1", "这一章的重点是什么")
    try:
        events = asyncio.run(_collect())
    finally:
        end_request()
    return events, dict(metrics.stage_ms)


def test_first_event_marks_thought_ahead_of_answer_token(monkeypatch) -> None:
    """推理先出屏：first_event 记在 thought 上且不晚于 first_token，两条都落盘。"""
    pairs = [
        (_model_chunk(reasoning="先想想问的是哪一章"), {"langgraph_node": "model"}),
        (_model_chunk(content="重点是第 3 章"), {"langgraph_node": "model"}),
    ]

    types, stage = _run(monkeypatch, pairs)

    assert types == ["thought", "token", "done"]
    assert "llm_first_event_ms" in stage, "首个可见事件没有计时"
    assert stage["llm_first_event_ms"] <= stage["llm_first_token_ms"], "thought 比正文先出屏，不该反过来"


def test_first_event_covers_request_without_answer_token(monkeypatch) -> None:
    """只出工具结果、没吐正文 token 的请求：first_event 有值而 first_token 缺——这正是原先的指标盲区。"""
    pairs = [
        (
            _model_chunk(tool_calls=[{"name": "search_textbook", "args": {}, "id": "c1"}]),
            {"langgraph_node": "model"},
        ),
        (_tool_result("search_textbook"), {"langgraph_node": "tools"}),
    ]

    types, stage = _run(monkeypatch, pairs)

    assert types == ["tool", "done"]
    assert "llm_first_event_ms" in stage, "工具事件同样上屏，必须计入首屏等待"
    assert "llm_first_token_ms" not in stage, "没有正文 token 就不该有首 token 延迟"


def test_first_event_counted_once_not_accumulated(monkeypatch) -> None:
    """三个可见事件也只记第一个：mark_stage 是累加语义，少了守卫就会把等待算成三倍。"""
    pairs = [
        (_model_chunk(reasoning="想"), {"langgraph_node": "model"}),
        (_model_chunk(reasoning="再想"), {"langgraph_node": "model"}),
        (_model_chunk(content="正文"), {"langgraph_node": "model"}),
    ]

    _types, stage = _run(monkeypatch, pairs)

    first = stage["llm_first_event_ms"]
    assert first < 150, f"首屏耗时应只含第一段 80ms 间隔，实际 {first:.0f}ms——说明逐事件重复计时了"
