"""统一教材助手服务路由：SSE 流式对话（含写盘确认续跑）+ 产物文件读取。

对话 thread_id = user_id:session_id，独立 checkpoint collection（study_checkpoints），
多用户按 X-User-Id 软隔离。多轮交互的消息结构调试信息输出到控制台并追加写
``logs/messages.log``。
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from starlette.responses import Response, StreamingResponse

from app.api.deps import get_user_id
from app.api.req import ChatRequest
from app.clients import mongo_client
from app.config import mongo_config
from app.core import (
    end_request,
    flush_metrics,
    logger,
    mark_stage,
    start_request,
)
from app.study_agent.graph import build_graph
from app.study_agent.state import StudyState
from app.study_agent.tools.files import MATERIAL_ROOT

router = APIRouter(tags=["study"])

# 消息结构 trace 的落盘目录（项目根 logs/）
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"

# 模块级编译一次：独立 checkpoint collection，避免与 query/textbook 图 thread 冲突
checkpointer = MongoDBSaver(
    mongo_client.get_client(),
    mongo_config.db,
    checkpoint_collection_name="study_checkpoints",
    writes_collection_name="study_checkpoint_writes",
)
study_graph = build_graph(checkpointer=checkpointer)


@router.post(
    "/study/chat",
    summary="统一助手对话（含写盘确认续跑）",
    description="SSE 事件流：tool / thought / token / ask_confirm / done / error；多轮经 session_id 续接；"
    "对停留在写盘确认的会话传 decision=approve|reject 统一放行/拒绝",
)
async def chat(req: ChatRequest, user_id: str = Depends(get_user_id)) -> StreamingResponse:
    """统一对话入口：新问题直接跑；上轮停在写盘确认时按 decision 放行/拒绝后续跑同一线程。"""
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    state: StudyState = {
        "user_id": user_id,
        "task_id": session_id,
        "textbook_name": req.textbook_name,
        "requirement": req.query,
        "messages": [HumanMessage(content=req.query)],
    }

    # 判断线程是否停在 HumanInTheLoop 写盘确认中断
    snapshot = await study_graph.aget_state(config)
    pending_interrupt = bool(getattr(snapshot, "next", None)) and any(
        task for task in getattr(snapshot, "tasks", ()) if getattr(task, "interrupts", ())
    )
    agent_input: dict | Command = state
    if pending_interrupt:
        pending_count = max(len(getattr(snapshot, "tasks", ())), 1)  # 无快照时兜底 1
        if req.decision in ("approve", "reject"):
            # 用户回应上轮确认：对全部 pending 中断统一生效，续跑同一线程
            agent_input = Command(resume={"decisions": [{"type": req.decision}] * pending_count})
        else:
            # 方案A：停在中断却发来新问题 → 自动拒绝本次写盘（解锁线程），再正常跑新问题
            logger.info(f"session {session_id} 停在写盘确认但收到新问题，自动拒绝本次写盘")
            await study_graph.ainvoke(
                Command(resume={"decisions": [{"type": "reject"}] * pending_count}), config
            )
            agent_input = state

    async def event_gen():
        metrics = start_request(session_id, req.query)
        t_start = time.perf_counter()
        try:
            async for ev in stream_agent(agent_input, config, session_id):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            # 教材未登记等业务错误
            metrics.error_type = type(exc).__name__
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        except Exception as exc:
            metrics.error_type = type(exc).__name__
            logger.exception(f"统一对话执行失败: {exc}")
            payload = json.dumps({"type": "error", "message": "回答生成失败"}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        finally:
            # 总耗时 = 请求进入到最后一条事件（含收尾落盘前），随后聚合落库
            mark_stage("total_ms", (time.perf_counter() - t_start) * 1000)
            flush_metrics()
            end_request()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def stream_agent(agent_input: dict | Command, config: dict, session_id: str):
    """以 SSE 事件流消费统一 agent 的一轮执行（消息级 + 断点续跑共用）。

    事件协议（按前端展示规则）：
    - ``thought``：模型在调工具前后的思考/推理内容（能看清大模型的思路），单独推送；
    - ``token``：仅最终回答正文；
    - ``tool``：工具执行结果（name + content），全部推送，前端按时间线穿插展示；
    - ``ask_confirm`` / ``done``：写盘确认 / 收尾。

    Args:
        agent_input: 本轮输入（新会话传 state 字典；续跑传 Command(resume=...)）。
        config: 含 thread_id 的运行时配置。
        session_id: 会话 ID，随 done / ask_confirm 回传。

    Yields:
        形如 {"type": "thought"|"token"|"tool"|"ask_confirm"|"done", ...} 的事件字典。
    """
    t0 = time.perf_counter()
    first_token = False
    async for chunk, meta in study_graph.astream(
        agent_input, config=config, stream_mode="messages", durability="sync"
    ):
        node = (meta or {}).get("langgraph_node")
        if node == "model":
            # 工具调用回合：模型先"说"的一段（如「教材已定位到…我再取回代码」）放在 content，
            real_calls = [tc for tc in (getattr(chunk, "tool_call_chunks", None) or []) if tc.get("name")]
            speech = getattr(chunk, "content", "") or ""
            reasoning = getattr(chunk, "reasoning_content", None)
            if not reasoning:
                reasoning = (getattr(chunk, "additional_kwargs", {}) or {}).get("reasoning_content")
            if real_calls:
                seg = (str(reasoning) if reasoning else "") + ("\n" if reasoning and speech else "") + speech
                if seg.strip():
                    yield {"type": "thought", "content": seg.strip()}
                continue
            # 纯正文回合：推理 → thought；正文 → 最终回答 token
            if reasoning:
                yield {"type": "thought", "content": str(reasoning)}
            if speech:
                if not first_token:
                    # 首个回答正文 token 距开始执行的延迟（首 token 延迟）
                    mark_stage("llm_first_token_ms", (time.perf_counter() - t0) * 1000)
                    first_token = True
                yield {"type": "token", "content": speech}
        elif node == "tools":
            # 工具结果：前端展示
            tm_name = getattr(chunk, "name", None)
            if tm_name:
                yield {"type": "tool","status": "done","name": tm_name,"content": getattr(chunk, "content", "") or "",}

    # 回答生成结束（含工具调用与最终回答），记录生成阶段总耗时
    mark_stage("llm_total_ms", (time.perf_counter() - t0) * 1000)

    # 检测是否停留在写盘确认中断（HumanInTheLoop 暂停后的线程 next 非空且 tasks 带 interrupts）
    snapshot = await study_graph.aget_state(config)
    # 调试：多轮交互消息的全量结构只在后端控制台/日志文件输出（不推送前端）
    _all = _dump_messages((snapshot.values or {}).get("messages", []))
    _write_message_trace(session_id, _all)
    # 停在 HumanInTheLoop 写盘中断（next 非空且 tasks 带 interrupts）→ 等确认；否则正常收尾
    pending_interrupt = bool(getattr(snapshot, "next", None)) and any(
        task for task in getattr(snapshot, "tasks", ()) if getattr(task, "interrupts", ())
    )
    if pending_interrupt:
        yield {
            "type": "ask_confirm",
            "session_id": session_id,
            "message": "助手准备写入文件以生成学习资料，是否允许？",
        }
    else:
        yield {"type": "done", "session_id": session_id}


def _dump_messages(messages: list) -> list[dict]:
    """把线程消息抽成可读结构，供调试查看 agent 与 LLM 的多轮交互。

    每种消息保留关键字段：type（human/ai/tool）、content（全量）、tool_calls（模型要调的工具）、
    tool_call_id（工具回填对应哪个调用）、name（工具名）。字段不存在则不出现。

    Args:
        messages: 线程状态里的 messages 列表（LangChain BaseMessage）。

    Returns:
        结构化的消息字典列表。
    """
    out: list[dict] = []
    for m in messages:
        item: dict = {"type": m.type, "content": str(m.content)}
        reasoning = getattr(m, "reasoning_content", None)
        if not reasoning:
            reasoning = (getattr(m, "additional_kwargs", {}) or {}).get("reasoning_content")
        if reasoning:
            item["reasoning_content"] = str(reasoning)
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")} for tc in tool_calls
            ]
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            item["tool_call_id"] = tool_call_id
        name = getattr(m, "name", None)
        if name:
            item["name"] = name
        out.append(item)
    return out


def _write_message_trace(session_id: str, dump: list[dict]) -> None:
    """把一轮消息结构按 JSONL 追加到 logs/messages.log（失败仅告警，不影响主链路）。

    Args:
        session_id: 会话 ID。
        dump: _dump_messages 的结构化消息列表。
    """
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"ts": datetime.now().isoformat(timespec="seconds"), "session_id": session_id, "messages": dump},
            ensure_ascii=False,
        )
        with (LOGS_DIR / "messages.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.warning(f"写入消息 trace 失败(session_id={session_id}): {exc}")


CHAT_THREAD_MARKER = ":study:"  # 资料任务线程前缀标记，用于会话列表排除


@router.get("/study/files", summary="任务产物文件列表", description="列出某任务目录内已生成的文件（相对路径 + 大小）")
async def files(task_id: str, textbook_name: str, user_id: str = Depends(get_user_id)) -> list[dict]:
    """列出某任务的全部落盘文件。"""
    root = (MATERIAL_ROOT / textbook_name / task_id).resolve()
    if not root.is_dir():
        return []
    return [
        {"filename": p.name, "rel_path": p.relative_to(root).as_posix(), "size": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(".tmp")
    ]


@router.get("/study/file", summary="读取产物文件内容", description="读取任务目录内一份文件（路径经沙箱校验）")
async def read_file(task_id: str, textbook_name: str, path: str) -> dict:
    """读取一份文件内容（前端预览用）。"""
    root = (MATERIAL_ROOT / textbook_name / task_id).resolve()
    # 路径防线：拒绝绝对路径 / `..` 逃逸 / symlink 逃逸（与 tools/files.py 同一策略）
    try:
        if not path or "\x00" in path or Path(path).is_absolute():
            raise ValueError(f"文件名非法（须为非空相对路径）: {path!r}")
        target = root
        for part in Path(path).parts:
            if part in ("", ".", ".."):
                raise ValueError(f"非法路径片段: {path!r}")
            target = (target / part).resolve(strict=False)
            if target != root and not target.is_relative_to(root):
                raise ValueError(f"路径越出任务沙箱: {path!r}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    return {"path": path, "content": target.read_text(encoding="utf-8")}


@router.get(
    "/sessions",
    summary="会话列表",
    description="按教材列出当前用户的对话会话摘要（从 study checkpoint 读取），最近更新在前，排除资料任务线程",
)
async def sessions(textbook_name: str | None = None, user_id: str = Depends(get_user_id)) -> list[dict]:
    """列出当前用户某教材的全部对话会话摘要（基于 study_checkpoints，倒序）。"""
    if not textbook_name:
        raise HTTPException(status_code=400, detail="缺少 textbook_name")

    prefix = f"{user_id}:"
    thread_ids = [
        t
        for t in mongo_client.get_collection("study_checkpoints").distinct("thread_id")
        if t.startswith(prefix) and CHAT_THREAD_MARKER not in t
    ]
    items: list[dict] = []
    for tid in thread_ids:
        snapshot = await study_graph.aget_state({"configurable": {"thread_id": tid}})
        values = snapshot.values or {}
        if values.get("textbook_name") != textbook_name:
            continue
        messages = values.get("messages", [])
        # 会话条目数只统计可回显的消息（用户 + 最终回答），过滤工具消息/中间轮次
        chat_count = sum(
            1
            for m in messages
            if m.type == "human" or (m.type == "ai" and not getattr(m, "tool_calls", None))
        )
        last_user = next((m for m in reversed(messages) if m.type == "human"), None)
        items.append(
            {
                "session_id": tid.split(":", 1)[1],  # 返回裸 session_id，前端配合 X-User-Id 使用
                "updated_at": snapshot.created_at or "",
                "message_count": chat_count,
                # 返回最新用户问题原文；截断等展示整形由前端负责
                "last_question": str(last_user.content) if last_user else "",
            }
        )
    items.sort(key=lambda s: s["updated_at"], reverse=True)
    return items


@router.get(
    "/history",
    summary="会话历史",
    description="获取当前用户某会话的全部消息（从 study checkpoint 读取），供历史回显",
)
async def history(session_id: str | None = None, user_id: str = Depends(get_user_id)) -> list[dict]:
    """获取某会话的结构化消息列表。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    snapshot = await study_graph.aget_state({"configurable": {"thread_id": f"{user_id}:{session_id}"}})
    messages = (snapshot.values or {}).get("messages", [])
    # 历史回显：只保留「用户 + 最终助手回答」；思考与检索结果按消息真实顺序折进对应回答的 steps 时间线
    items: list[dict] = []
    pending_steps: list[dict] = []
    for m in messages:
        if m.type == "human":
            items.append({"role": "user", "content": str(m.content)})
        elif m.type == "ai":
            reasoning = getattr(m, "reasoning_content", None)
            if not reasoning:
                reasoning = (getattr(m, "additional_kwargs", {}) or {}).get("reasoning_content")
            # 带 tool_calls 的中间回合：推理也进时间线（穿插在工具结果之前）
            if reasoning:
                pending_steps.append({"type": "thought", "content": str(reasoning)})
            if getattr(m, "tool_calls", None):
                continue  # 中间的工具调用回合，不单独展示
            items.append({"role": "assistant", "content": str(m.content), "steps": pending_steps[:]})
            pending_steps = []
        elif m.type == "tool":
            tm_name = getattr(m, "name", None)
            if tm_name:
                pending_steps.append({"type": "tool", "name": tm_name, "content": str(m.content)})
    return items


@router.delete(
    "/sessions/{session_id}",
    summary="删除会话",
    description="删除当前用户的单个对话会话（删除其全部 checkpoint）",
)
async def delete_session(session_id: str, user_id: str = Depends(get_user_id)) -> Response:
    """删除单个会话（幂等：会话不存在也返回 204）。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    checkpointer.delete_thread(f"{user_id}:{session_id}")
    return Response(status_code=204)
