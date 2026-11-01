"""统一教材助手的对话编排：中断判定、SSE 事件流与一轮图执行。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.core import logger
from app.core.safety import detect_prompt_leak, strip_boundary
from app.services.study.lifecycle import get_graph
from app.agent.state import StudyState
from app.agent.tools.clarify import CANCEL_MARKER
from app.trace import end_request, flush_metrics, get_metrics, mark_stage, start_request


async def prepare_chat(
    *,
    user_id: str,
    session_id: str | None,
    textbook_name: str,
    query: str,
    deep: bool,
    decision: str | None = None,
    clarify_answers: list[dict] | None = None,
) -> tuple[dict | Command, dict, str]:
    """判定线程是否停在中断，生成本轮图输入与运行配置。

    续跑用 ``Command(resume=...)`` 推进同一线程；停在中断却发来新问题时先自动解锁再重跑。
    返回 ``(agent_input, config, session_id)``。
    """
    session_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    state: StudyState = {
        "user_id": user_id,
        "task_id": session_id,
        "textbook_name": textbook_name,
        "requirement": query,
        "deep": deep,
        "messages": [HumanMessage(content=query)],
    }

    # 判断线程是否停在中断（写盘确认 或 澄清反问）
    snapshot = await get_graph().aget_state(config)
    pending_interrupt = bool(getattr(snapshot, "next", None)) and any(
        task for task in getattr(snapshot, "tasks", ()) if getattr(task, "interrupts", ())
    )
    agent_input: dict | Command = state
    if pending_interrupt:
        # 取首个挂起中断的值（用其 type 区分写盘确认与澄清反问）
        interrupt_value: dict | None = None
        for task in getattr(snapshot, "tasks", ()):
            for intr in getattr(task, "interrupts", ()):
                value = getattr(intr, "value", None)
                if isinstance(value, dict):
                    interrupt_value = value
                    break
            if interrupt_value is not None:
                break
        pending_count = _hanging_write_count(snapshot)
        if interrupt_value and interrupt_value.get("type") == "clarify":
            if clarify_answers is not None:
                # 用户回应上轮澄清反问：按问题逐一作答，续跑同一线程
                agent_input = Command(resume={"answers": clarify_answers})
            else:
                # 停在澄清却发来新问题 → 自动取消本轮澄清（返回空答案解锁线程），再正常跑新问题
                logger.info(f"session {session_id} 停在澄清反问但收到新问题，自动取消本轮澄清")
                await get_graph().ainvoke(Command(resume={CANCEL_MARKER: True}), config)
                agent_input = state
        elif decision in ("approve", "reject"):
            # 用户回应上轮确认：对该中断里全部挂起的写盘调用统一生效，续跑同一线程
            agent_input = Command(resume={"decisions": [{"type": decision}] * pending_count})
        else:
            # 方案A：停在写盘确认却发来新问题 → 自动拒绝本次写盘（解锁线程），再正常跑新问题
            logger.info(f"session {session_id} 停在写盘确认但收到新问题，自动拒绝本次写盘")
            await get_graph().ainvoke(
                Command(resume={"decisions": [{"type": "reject"}] * pending_count}), config
            )
            agent_input = state

    return agent_input, config, session_id


async def chat_events(agent_input: dict | Command, config: dict, session_id: str, query: str) -> AsyncIterator[str]:
    """把一轮对话编码成 SSE 帧，并负责请求级 metrics 的收尾。"""
    metrics = start_request(session_id, query)
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
        # 输出侧留痕（不拦截：流式已把 token 发给前端，事后无法撤回）：
        # 答案疑似复述系统提示词或漏出配置形态时告警并落进 metrics 记录
        if detect_prompt_leak(metrics.answer_content):
            metrics.output_leak_suspected = True
            logger.warning(f"回答疑似泄露系统提示词/敏感形态（session {session_id}）")
        flush_metrics()
        end_request()


async def stream_agent(agent_input: dict | Command, config: dict, session_id: str):
    """以 SSE 事件流消费 agent 的一轮执行（新会话与断点续跑共用）。

    事件协议：``thought``（思考）、``token``（回答正文）、``tool``（工具结果）、
    ``ask_confirm`` / ``done``（写盘确认 / 收尾）。
    """
    t0 = time.perf_counter()
    first_token = False
    first_event = False

    def mark_first_event() -> None:
        """首个用户可见事件（thought / tool / token 任一）的延迟：前端真开始出现字的时刻。"""
        nonlocal first_event
        if not first_event:
            mark_stage("llm_first_event_ms", (time.perf_counter() - t0) * 1000)
            first_event = True

    async for chunk, meta in get_graph().astream(
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
                    mark_first_event()
                    yield {"type": "thought", "content": seg.strip()}
                continue
            # 纯正文回合：推理 → thought；正文 → 最终回答 token
            if reasoning:
                mark_first_event()
                yield {"type": "thought", "content": str(reasoning)}
            if speech:
                mark_first_event()
                if not first_token:
                    # 首个回答正文 token 距开始执行的延迟（首 token 延迟）
                    mark_stage("llm_first_token_ms", (time.perf_counter() - t0) * 1000)
                    first_token = True
                metrics = get_metrics()
                if metrics is not None:
                    metrics.answer_content += str(speech)
                yield {"type": "token", "content": speech}
        elif node == "tools":
            # 工具结果：前端展示。展示走 artifact 的干净原文 / 剥掉边界标记（与 /history 回显同一策略）：
            # 模型侧 content 已包外部资料边界、带随机串，不该漏给用户看（见 app/core/safety.py）
            tm_name = getattr(chunk, "name", None)
            if tm_name:
                mark_first_event()
                artifact = getattr(chunk, "artifact", None) or {}
                display = artifact.get("content") or strip_boundary(getattr(chunk, "content", "") or "")
                yield {"type": "tool", "status": "done", "name": tm_name, "content": display}

    # 回答生成结束（含工具调用与最终回答），记录生成阶段总耗时
    mark_stage("llm_total_ms", (time.perf_counter() - t0) * 1000)

    # 检测是否停留在中断（HumanInTheLoop 写盘确认 / 澄清反问），按类型分发事件
    snapshot = await get_graph().aget_state(config)
    # 调试：多轮交互消息的全量结构仅输出到控制台（不推送前端、不落盘）
    logger.info(
        f"消息结构(session {session_id}): "
        f"{json.dumps(_dump_messages((snapshot.values or {}).get('messages', [])), ensure_ascii=False)}"
    )
    # 停在中断（next 非空且 tasks 带 interrupts）→ 等确认/作答；否则正常收尾
    pending_interrupt = bool(getattr(snapshot, "next", None)) and any(
        task for task in getattr(snapshot, "tasks", ()) if getattr(task, "interrupts", ())
    )
    if pending_interrupt:
        # 取首个挂起中断的值（用其 type 区分写盘确认与澄清反问）
        interrupt_value: dict | None = None
        for task in getattr(snapshot, "tasks", ()):
            for intr in getattr(task, "interrupts", ()):
                value = getattr(intr, "value", None)
                if isinstance(value, dict):
                    interrupt_value = value
                    break
            if interrupt_value is not None:
                break
        if interrupt_value and interrupt_value.get("type") == "clarify":
            # 澄清反问：把问题（及校验失败提示）原样抛给前端
            yield {
                "type": "ask_clarify",
                "session_id": session_id,
                "questions": interrupt_value.get("questions") or [],
                "max_questions": interrupt_value.get("max_questions", 3),
                "error": interrupt_value.get("error") or "",
            }
        else:
            yield {
                "type": "ask_confirm",
                "session_id": session_id,
                "message": "助手准备写入文件以生成学习资料，是否允许？",
            }
    else:
        metrics = get_metrics()
        done_event = {"type": "done", "session_id": session_id}
        if metrics is not None:
            # 附带观测字段便于评测脚本与客户端直接关联回答和检索证据；
            # 同一份数据随后仍会由 flush_metrics 写入 metrics.log。
            done_event.update(
                {
                    "request_id": metrics.request_id,
                    "retrieved_chunk_ids": list(metrics.retrieved_chunk_ids),
                    "answer_content": metrics.answer_content,
                }
            )
        yield done_event


def _dump_messages(messages: list) -> list[dict]:
    """把线程消息抽成可读结构（type/content/tool_calls 等），供调试查看。"""
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


def _hanging_write_count(snapshot: Any) -> int:
    """挂起的写盘调用数 = 各中断 payload 的 ``action_requests`` 总和（无快照时兜底 1）。

    HITL 把一轮所有待确认调用打包进同一个 interrupt；resume 的 decisions 数必须与之一致。
    """
    count = 0
    for task in getattr(snapshot, "tasks", ()):
        for intr in getattr(task, "interrupts", ()):
            value = getattr(intr, "value", None)
            if isinstance(value, dict):
                count += len(value.get("action_requests") or ())
    return max(count, 1)
