"""chat 端点的 deep 透传：深度检索档由前端请求带入图输入 state（假图，不触达 PostgreSQL）。

deep 从模型可见参数里摘掉后，这条 state 通道就是唯一来源——前端开关到工具之间的链路
断了不会有任何报错，只会静默退化成快检索，所以必须在服务层钉住。
"""

import asyncio
from typing import Any

from app.api.routes import study as route
from app.api.schemas import ChatRequest
from app.services import study as svc


class _Snap:
    """不在任何中断上（next 为空）→ 端点按新问题走 state 输入。"""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {"messages": []}
        self.tasks: tuple = ()
        self.next = None


class _FakeGraph:
    async def aget_state(self, config) -> _Snap:
        return _Snap()


def _run(monkeypatch, req: ChatRequest) -> list:
    """打桩跑一轮 chat 端点，回传 stream_agent 收到的图输入序列。"""
    seen: list = []

    async def _fake_stream(agent_input, config, session_id):
        seen.append(agent_input)
        yield {"type": "done", "session_id": session_id}

    monkeypatch.setattr(svc, "study_graph", _FakeGraph())
    monkeypatch.setattr(svc, "stream_agent", _fake_stream)
    monkeypatch.setattr(svc, "flush_metrics", svc.end_request)
    resp = asyncio.run(route.chat(req, user_id="u1"))
    asyncio.run(_drain(resp))
    return seen


async def _drain(resp) -> None:
    async for _ in resp.body_iterator:
        pass


def test_chat_defaults_to_fast_retrieval() -> None:
    """请求不带 deep 时默认关：深度档增量近 5s，不能让缺省请求白付。"""
    assert ChatRequest(textbook_name="C语言程序设计", query="指针是什么").deep is False


def test_chat_passes_deep_into_graph_state(monkeypatch) -> None:
    """前端传 deep=True 时，图输入的 state 必须带上同名的 deep，工具才取得到。"""
    seen = _run(monkeypatch, ChatRequest(textbook_name="C语言程序设计", query="指针是什么", deep=True))

    assert len(seen) == 1
    agent_input = seen[0]
    assert isinstance(agent_input, dict), "新问题应是 state 输入而非 Command"
    assert agent_input["deep"] is True, f"state 未透传 deep：{sorted(agent_input)}"
