"""落库视图测试：flush 出来的记录形状（steps 时间线、派生字段、resolved 标记）。

落盘仍是「缩进 JSON 块 + 空行」追加到 ``logs/<日期>/metrics.log``（历史数据与人工翻查都在这
一侧），这里把记录形状钉住：旧字段消失、证据字段派生、被拦下的工具调用有标记可循。
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

import app.trace.metrics as metrics_mod
from app.trace import end_request, llm_usage, mark_tool, start_request
from app.trace.llm_usage import weighted_cost

_USAGE: dict[str, Any] = {
    "input_tokens": 1000,
    "output_tokens": 50,
    "total_tokens": 1050,
    "input_token_details": {"cache_read": 800},
}


def _response(content: str = "", tool_calls: list[dict] | None = None) -> LLMResult:
    """造一份带 usage 的 LLMResult：直接喂 handler 的回调，不经过模型。"""
    message = AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=dict(_USAGE),
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _read_single_record(tmp_path: Path) -> dict[str, Any]:
    """读回落盘记录（当天目录下唯一一条）。"""
    path = next(tmp_path.rglob("metrics.log"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_flush_writes_steps_timeline(monkeypatch, tmp_path: Path) -> None:
    """一条记录 = 一条 steps 时间线：旧明细字段消失，证据与成本字段各就各位。"""
    monkeypatch.setattr(metrics_mod, "LOGS_DIR", tmp_path)

    @tool
    async def search_textbook(query: str) -> str:
        """检索"""
        mark_tool(query=query, recall=7, kept=2, chunk_ids=["c9", "c1", "c9"])
        return "片段正文"

    start_request("sess-1", "问题")
    handler = llm_usage.get_handler()
    assert handler is not None
    handler.on_chat_model_start(
        {},
        [[HumanMessage(content="问题")]],
        run_id="run-1",
        tags=None,
    )
    handler.on_llm_end(
        _response("先查教材", tool_calls=[{"id": "call_1", "name": "search_textbook", "args": {"query": "q"}}]),
        run_id="run-1",
    )
    asyncio.run(search_textbook.ainvoke({"query": "q"}))
    metrics_mod.flush_metrics()
    end_request()

    record = _read_single_record(tmp_path)
    assert "llm_calls_detail" not in record and "retrievals" not in record
    assert [step["kind"] for step in record["steps"]] == ["llm", "tool"]
    llm_step, tool_step = record["steps"]
    assert llm_step["output"]["tool_calls"][0]["resolved"] is True, "已执行的调用必须标 resolved"
    assert tool_step["tool_call_id"] == "call_1"
    assert tool_step["parent_id"] == llm_step["call_id"], "工具步要挂回发起它的 LLM 步"
    assert tool_step["meta"]["recall"] == 7
    assert record["retrieved_chunk_ids"] == ["c9", "c1"], "证据由工具步 meta 派生、去重保序"
    assert record["turn_cost"] == round(weighted_cost(1000, 800, 50), 1)
    assert record["prompt_tokens"] == 1000 and record["cached_tokens"] == 800


def test_flush_marks_unexecuted_tool_calls_unresolved(monkeypatch, tmp_path: Path) -> None:
    """模型发出但本轮没执行的调用（HITL 拦下/限流短路）标 resolved=false，供跨记录对照。"""
    monkeypatch.setattr(metrics_mod, "LOGS_DIR", tmp_path)

    start_request("sess-2", "写个文件")
    handler = llm_usage.get_handler()
    assert handler is not None
    handler.on_chat_model_start({}, [[HumanMessage(content="写个文件")]], run_id="run-9", tags=None)
    handler.on_llm_end(
        _response("准备写盘", tool_calls=[{"id": "call_w1", "name": "write_file", "args": {"filename": "a.md"}}]),
        run_id="run-9",
    )
    metrics_mod.flush_metrics()
    end_request()

    record = _read_single_record(tmp_path)
    assert [step["kind"] for step in record["steps"]] == ["llm"]
    assert record["steps"][0]["output"]["tool_calls"][0]["resolved"] is False
    assert record["retrieved_chunk_ids"] == []


def test_resolved_only_marks_agent_tool_calls(monkeypatch, tmp_path: Path) -> None:
    """非 agent 调用（intent 的结构化输出）不打 resolved 标记：那是 function_calling 载荷，不是工具执行。"""
    monkeypatch.setattr(metrics_mod, "LOGS_DIR", tmp_path)

    start_request("sess-3", "q")
    handler = llm_usage.get_handler()
    assert handler is not None
    handler.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id="run-i", tags=["trace_role:intent"])
    handler.on_llm_end(
        _response("", tool_calls=[{"id": "call_decision", "name": "_IntentDecision", "args": {}}]),
        run_id="run-i",
    )
    metrics_mod.flush_metrics()
    end_request()

    record = _read_single_record(tmp_path)
    call = record["steps"][0]["output"]["tool_calls"][0]
    assert "resolved" not in call
