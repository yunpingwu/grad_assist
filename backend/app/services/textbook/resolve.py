"""教材解析编排：首次/偏移续跑/断点续跑的判定与 SSE 推流。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from langgraph.types import Command

from app.core import logger
from app.services.textbook.lifecycle import get_graph
from app.textbook_flow.cache import toc_cache
from app.textbook_flow.state import TextBookState


class MissingTextbookPath(ValueError):
    """首次解析缺少 textbook_path。"""


class TextbookPathNotFound(FileNotFoundError):
    """textbook_path 指向的目录不存在。"""


@dataclass
class ResolvePlan:
    """resolve 的一次执行计划：prepare 阶段判定，events 阶段只负责推流。"""

    task_id: str
    config: dict
    run_input: Command | None | TextBookState
    resolved_path: str | None
    pending_offset: bool
    resumable: bool


async def prepare_resolve(
    *,
    textbook_path: str | None,
    task_id: str | None,
    user_id: str,
    offsets: list[dict] | None,
) -> ResolvePlan:
    """判定本次 resolve 是首次解析、偏移确认续跑还是断点续跑。

    校验放在事件流之前，HTTP 层才能在建 SSE 响应前返回 400/404。
    """
    # 任务标识：缺省则新建；传入则复用（用于断线重连/续跑）
    task_id = task_id or str(uuid.uuid4())
    thread_id = f"{user_id}:{task_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 借助 checkpoint 判断线程所处状态：
    # - 停在 offset 人工确认中断（tasks 带 interrupts，value.type=="offset_review"）
    # - 因断线停在节点边界、尚未完成（next 非空）
    snapshot = await get_graph().aget_state(config)
    prev = (snapshot.values or {}) if snapshot else {}

    pending_offset = any(
        (getattr(i, "value", None) or {}).get("type") == "offset_review"
        for task in (getattr(snapshot, "tasks", None) or ())
        for i in (getattr(task, "interrupts", None) or ())
    )
    resumable = bool(prev and not prev.get("ingestion_done") and getattr(snapshot, "next", None))

    if pending_offset:
        if offsets:
            run_input: Command | None | TextBookState = Command(resume={"offsets": offsets})
        else:
            # 前端凭 task_id 重连找回中断任务：不带 offsets 视为「重看」，重发 ask_offset 供再次确认
            run_input = None
        resolved_path = prev.get("textbook_path") if prev else None
    elif resumable:
        # 断点续跑：输入传 None，从上次完成节点继续，不做重头解析
        run_input = None
        resolved_path = prev.get("textbook_path") if prev else None
    else:
        if not textbook_path:
            raise MissingTextbookPath("缺少 textbook_path，请先调用 /upload 上传教材")
        if not Path(textbook_path).is_dir():
            raise TextbookPathNotFound(f"教材目录不存在: {textbook_path}")
        resolved_path = textbook_path
        run_input = {
            "textbook_exists": False,
            "user_id": user_id,
            "textbook_path": resolved_path,
        }

    return ResolvePlan(
        task_id=task_id,
        config=config,
        run_input=run_input,
        resolved_path=resolved_path,
        pending_offset=pending_offset,
        resumable=resumable,
    )


async def resolve_events(plan: ResolvePlan) -> AsyncIterator[str]:
    """按计划把教材摄入进度编码成 SSE 帧（info / 图事件 / done / error）。"""
    try:
        if plan.pending_offset or plan.resumable:
            info = {"type": "info", "task_id": plan.task_id, "resumed": True, "message": "检测到中断，正在续跑…"}
        else:
            info = {
                "type": "info",
                "task_id": plan.task_id,
                "resumed": False,
                "message": "开始解析",
                "progress": 0.0,
            }
        # 目录树快照：缓存命中直接返回，miss 时从 mineru_toc/full.md 懒重建；
        # 断线重进凭 info 帧即可恢复常驻树，不依赖流水线节点重放（尚未解析出树时不带该字段）
        if plan.resolved_path:
            tocs = await asyncio.to_thread(toc_cache.get_tocs, plan.task_id, plan.resolved_path)
            if tocs:
                info["tocs"] = tocs
        yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"
        async for event in get_graph().astream(
            plan.run_input, config=plan.config, stream_mode="custom", durability="sync"
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        # 流结束后确认是否停在 offset 确认中断：停在中断不发 done，由 ask_offset 事件收尾
        snap_after = await get_graph().aget_state(plan.config)
        awaiting_offset = any(
            (getattr(i, "value", None) or {}).get("type") == "offset_review"
            for task in (getattr(snap_after, "tasks", None) or ())
            for i in (getattr(task, "interrupts", None) or ())
        )
        if not awaiting_offset:
            toc_cache.pop_tocs(plan.task_id)  # 完成即删：树已随流送达，无需再留快照
            yield f"data: {json.dumps({'type': 'done', 'task_id': plan.task_id}, ensure_ascii=False)}\n\n"
    except ValueError as exc:
        # 教材未找到等业务错误
        toc_cache.pop_tocs(plan.task_id)
        payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
        yield f"data: {payload}"
    except Exception as exc:
        logger.exception(f"解析任务 {plan.task_id} 执行异常: {exc}")
        toc_cache.pop_tocs(plan.task_id)
        payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
        yield f"data: {payload}"
