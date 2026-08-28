"""教材知识学习服务路由：SSE 流式执行 Agent + 产物文件读取。

契约对齐 query_service：thread_id = user_id:study:task_id，独立 checkpoint
collection（study_checkpoints），多用户按 X-User-Id 软隔离。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.api.deps import get_user_id
from app.clients import mongo_client
from app.config import mongo_config
from app.core import logger
from app.study_agent.graph import build_graph
from app.study_agent.state import StudyState
from app.study_agent.tools.files import MATERIAL_ROOT

router = APIRouter(tags=["study"])

# 缺省任务指令：用户未填 requirement 时的兜底目标
DEFAULT_REQUIREMENT = "结合教材内容，生成一份完整的知识点总结学习笔记。"

# 模块级编译一次：独立 checkpoint collection，避免与 query/textbook 图 thread 冲突
checkpointer = MongoDBSaver(
    mongo_client.get_client(),
    mongo_config.db,
    checkpoint_collection_name="study_checkpoints",
    writes_collection_name="study_checkpoint_writes",
)
study_graph = build_graph(checkpointer=checkpointer)


class RunRequest(BaseModel):
    """学习任务请求。"""

    textbook_name: str = Field(..., description="教材名（已解析入库）")
    requirement: str = Field(default="", description="任务要求（如生成哪些学习资料/回答什么问题），留空用默认指令")
    task_id: str | None = Field(default=None, description="任务 ID（断点续跑用，缺省自动生成）")


@router.post(
    "/study/run",
    summary="执行学习任务",
    description="SSE 事件流：stage 阶段提示 / tool 工具调用 / token 增量 / done 收尾 / error 异常",
)
async def run(req: RunRequest, user_id: str = Depends(get_user_id)) -> StreamingResponse:
    """执行一轮学习任务（Agent 自主检索教材并产出文件），以 SSE 实时推送过程事件。"""
    task_id = req.task_id or str(uuid.uuid4())
    state: StudyState = {
        "user_id": user_id,
        "task_id": task_id,
        "textbook_name": req.textbook_name,
        "requirement": req.requirement or DEFAULT_REQUIREMENT,
        "messages": [HumanMessage(content=req.requirement or DEFAULT_REQUIREMENT)],
    }
    config = {"configurable": {"thread_id": f"{user_id}:study:{task_id}"}}

    async def event_gen():
        try:
            async for event in study_graph.astream(state, config=config, stream_mode="custom", durability="sync"):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            # 教材未登记等业务错误
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        except Exception as exc:
            # LLM / Milvus / 文件系统等基础设施错误
            logger.exception(f"学习任务执行失败: {exc}")
            payload = json.dumps({"type": "error", "message": "任务执行失败"}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
