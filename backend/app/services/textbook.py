"""教材摄入的应用编排层：上传落盘、断点续跑与 SSE 事件流。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from app.core import logger
from app.textbook_flow.cache import toc_cache
from app.textbook_flow.graph import build_graph
from app.textbook_flow.state import TextBookState
from app.utils import list_textbooks as list_textbooks  # 显式再导出：路由经服务层取教材列表


class MissingTextbookPath(ValueError):
    """首次解析缺少 textbook_path。"""


class TextbookPathNotFound(FileNotFoundError):
    """textbook_path 指向的目录不存在。"""


# 教材根目录（本文件位于 app/services/ 下，项目根为 parents[2]）。
# 每次上传在根目录下新建独立子目录（pdf-{特征值}），与测试数据 textbooks/pdf 隔离。
TEXTBOOK_ROOT = Path(__file__).resolve().parents[2] / "textbooks"

# 教材摄入图：独立 textbook schema，避免与对话图 thread_id 冲突；
# thread_id = user_id:task_id，多用户任务天然隔离。
# AsyncPostgresSaver 构造要求运行中的事件循环，图和 checkpointer 在 lifespan 内装配。
checkpointer: AsyncPostgresSaver | None = None
textbook_graph: Any | None = None


def init_graph(pool: AsyncConnectionPool) -> None:
    """在 lifespan 内装配摄入图。"""
    global checkpointer, textbook_graph
    checkpointer = AsyncPostgresSaver(pool)
    textbook_graph = build_graph(checkpointer=checkpointer)


async def setup_checkpointer() -> None:
    """首次启动建 checkpoint 表（幂等；迁移里有并发建索引，连接必须是 autocommit）。"""
    if checkpointer is None:
        raise RuntimeError("textbook checkpointer 尚未初始化")
    await checkpointer.setup()


def _graph() -> Any:
    """取已装配的图；未初始化时给出明确错误。"""
    if textbook_graph is None:
        raise RuntimeError("textbook_graph 尚未初始化（应在 FastAPI lifespan 中调用 init_graph）")
    return textbook_graph


def _new_textbook_dir() -> Path:
    """在 textbooks/ 下创建本次上传的独立目录（pdf-{时间戳}-{随机}）。

    每次上传都是干净目录，天然规避旧解析产物的幂等误判；
    向量化入库后该目录可整体删除（当前保留，便于验证解析效果）。
    """
    feature = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    textbook_dir = TEXTBOOK_ROOT / f"pdf-{feature}"
    textbook_dir.mkdir(parents=True, exist_ok=False)
    return textbook_dir


async def save_uploads(files: list[UploadFile]) -> tuple[Path, int]:
    """保存上传文件到独立目录，返回 ``(目录, 文件数)``。"""
    textbook_dir = _new_textbook_dir()
    saved = 0
    for file in files:
        with open(textbook_dir / file.filename, "wb") as f:
            f.write(await file.read())
        saved += 1
    return textbook_dir, saved


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
    snapshot = await _graph().aget_state(config)
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
        async for event in _graph().astream(
            plan.run_input, config=plan.config, stream_mode="custom", durability="sync"
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        # 流结束后确认是否停在 offset 确认中断：停在中断不发 done，由 ask_offset 事件收尾
        snap_after = await _graph().aget_state(plan.config)
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
