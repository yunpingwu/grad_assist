"""多用户评测的延迟计时口径测试：区分排队等待、纯服务耗时与首 token 延迟。

背景：旧实现把 ``started`` 放在 ``async with semaphore`` 之前，``latency_ms`` 含
并发排队时间；50 会话抢 10 槽时实测该值约为真实服务耗时的 4 倍，导致评测报告
把排队误读成 Agent 慢。
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from app.eval import run_multi_user_eval as runner


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        url="http://test/study/chat",
        request_retries=0,
        timeout=30.0,
        concurrency=2,
        dataset_root=Path("data/eval_test"),
        seed=1,
        shuffle=False,
    )


def _session() -> dict[str, Any]:
    return {
        "user_id": "u1",
        "session_id": "s1",
        "textbook": "DS",
        "questions": [{"id": "q1", "question": "栈和队列的区别"}],
    }


def _delayed_sse_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")


def test_service_ms_excludes_semaphore_queue_wait() -> None:
    """排队等信号量的时间计入 queue_ms，不得混进 service_ms。"""
    service_seconds = 0.02
    queue_hold_seconds = 0.25

    async def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            await asyncio.sleep(service_seconds)
            yield _sse({"type": "done", "request_id": "r1", "answer_content": "答案", "retrieved_chunk_ids": []})
        return httpx.Response(200, content=body())

    async def main() -> dict[str, Any]:
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()  # 先占满槽位，让本轮必须排队
        client = _delayed_sse_client(handler)
        async with client:
            released = asyncio.create_task(_release_after(semaphore, queue_hold_seconds))
            row = await runner._request_turn(
                client, semaphore, _args(), _session(), {"id": "q1", "question": "栈和队列的区别"}, 1
            )
            await released
        return row

    row = asyncio.run(main())

    assert row["queue_ms"] >= queue_hold_seconds * 1000 * 0.8, f"排队时间未单独记录: {row.get('queue_ms')}"
    assert row["service_ms"] < queue_hold_seconds * 1000, f"service_ms 混入了排队时间: {row['service_ms']}"
    assert abs(row["service_ms"] - service_seconds * 1000) < service_seconds * 1000 * 1.5, (
        f"service_ms 应接近纯服务耗时 {service_seconds * 1000}ms，实际 {row['service_ms']}ms"
    )
    assert abs(row["latency_ms"] - (row["queue_ms"] + row["service_ms"])) < 1.0, (
        f"latency_ms 应等于排队与服务之和: {row['latency_ms']} vs {row['queue_ms']}+{row['service_ms']}"
    )


async def _release_after(semaphore: asyncio.Semaphore, seconds: float) -> None:
    await asyncio.sleep(seconds)
    semaphore.release()


def test_first_token_ms_measured_on_first_answer_token_not_tool() -> None:
    """首 token 延迟取第一个 token 事件（最终回答正文），工具事件不算。"""
    tool_at = 0.02
    token_at = 0.12

    async def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            await asyncio.sleep(tool_at)
            yield _sse({"type": "tool", "name": "search_textbook", "content": "片段..."})
            await asyncio.sleep(token_at - tool_at)
            yield _sse({"type": "token", "content": "栈"})
            yield _sse({"type": "done", "request_id": "r2", "answer_content": "栈是...", "retrieved_chunk_ids": []})
        return httpx.Response(200, content=body())

    async def main() -> dict[str, Any]:
        client = _delayed_sse_client(handler)
        async with client:
            return await runner._request_turn(
                client,
                asyncio.Semaphore(10),
                _args(),
                _session(),
                {"id": "q1", "question": "栈和队列的区别"},
                1,
            )

    row = asyncio.run(main())

    assert row["first_token_ms"] is not None, "缺少 first_token_ms 字段"
    assert row["first_token_ms"] >= token_at * 1000 * 0.8, (
        f"首 token 延迟不应按 tool 事件计（应 >= {token_at * 1000}ms），实际 {row['first_token_ms']}ms"
    )
    assert row["first_token_ms"] <= row["service_ms"], "首 token 延迟不可能大于服务耗时"


def test_latency_summary_reports_service_queue_and_first_token_percentiles() -> None:
    """汇总给出服务/排队/首 token 三条 p50、p90，不再只有单一 latency 均值。"""
    rows = [
        {"service_ms": 30_000, "queue_ms": 100, "first_token_ms": 15_000, "latency_ms": 30_100, "status": "done"},
        {"service_ms": 38_000, "queue_ms": 200, "first_token_ms": 17_000, "latency_ms": 38_200, "status": "done"},
        {"service_ms": 40_000, "queue_ms": 150_000, "first_token_ms": 26_000, "latency_ms": 190_000, "status": "done"},
        {"service_ms": 78_000, "queue_ms": 160_000, "first_token_ms": 50_000, "latency_ms": 238_000, "status": "done"},
        {"service_ms": 5_000, "queue_ms": 10_000, "first_token_ms": None, "latency_ms": 15_000, "status": "error"},
    ]

    summary = runner._latency_summary(rows)

    assert summary["turns"] == 5
    assert summary["service_ms"]["p50"] == 38_000, f"p50 应为升序中位数（排除 error 轮？不应排除），实际 {summary['service_ms']}"
    assert summary["service_ms"]["max"] == 78_000
    assert summary["queue_ms"]["p50"] == 10_000, "排队时间应单列，便于识别并发瓶颈"
    assert summary["queue_ms"]["p90"] == 150_000
    assert summary["first_token_ms"]["p50"] == 17_000, "首 token 延迟只统计非空值"
    assert summary["first_token_ms"]["turns"] == 4, "None 的首 token 不参与统计"


def test_run_payload_reports_wall_clock_and_latency_summary(monkeypatch) -> None:
    """整体运行给出真实墙钟与延迟汇总，避免再用「客户端耗时总和/并发」推算总时长。"""
    async def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            await asyncio.sleep(0.01)
            yield _sse({"type": "token", "content": "答案"})
            yield _sse({"type": "done", "request_id": "r1", "answer_content": "答案", "retrieved_chunk_ids": []})
        return httpx.Response(200, content=body())

    real_factory = httpx.AsyncClient

    def fake_factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_factory(transport=httpx.MockTransport(handler), base_url="http://test", **kwargs)

    monkeypatch.setattr(runner.httpx, "AsyncClient", fake_factory)
    monkeypatch.setattr(runner, "_build_user_sessions", lambda args: [_session()])

    payload = asyncio.run(runner._run(_args()))

    assert payload["wall_ms"] > 0, "结果里应有真实墙钟 wall_ms"
    assert "latency_summary" in payload, "结果里应有 latency_summary"
    assert payload["latency_summary"]["service_ms"]["p50"] > 0
    assert payload["per_turn"][0]["service_ms"] > 0
