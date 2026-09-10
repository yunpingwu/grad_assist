import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from starlette.responses import JSONResponse, StreamingResponse

from app.api.deps import get_user_id
from app.clients import mongo_client
from app.config import mongo_config
from app.core import logger
from app.textbook_flow.graph import build_graph
from app.textbook_flow.state import TextBookState
from app.utils import list_textbooks

# 教材根目录（本文件位于 app/api/ 下，项目根为 parents[2]）。
# 每次上传在根目录下新建独立子目录（pdf-{特征值}），与测试数据 textbooks/pdf 隔离。
TEXTBOOK_ROOT = Path(__file__).resolve().parents[2] / "textbooks"


def _new_textbook_dir() -> Path:
    """在 textbooks/ 下创建本次上传的独立目录（pdf-{时间戳}-{随机}）。

    每次上传都是干净目录，天然规避旧解析产物的幂等误判；
    向量化入库后该目录可整体删除（当前保留，便于验证解析效果）。
    """
    feature = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    textbook_dir = TEXTBOOK_ROOT / f"pdf-{feature}"
    textbook_dir.mkdir(parents=True, exist_ok=False)
    return textbook_dir


# 教材路由
router = APIRouter(tags=["textbook"])

# 教材摄入图：独立 checkpointer collection，避免与 query 图的 thread_id 冲突；
# thread_id = user_id:task_id，多用户任务天然隔离。
checkpointer = MongoDBSaver(
    mongo_client.get_client(),
    mongo_config.db,
    checkpoint_collection_name="textbook_checkpoints",
    writes_collection_name="textbook_checkpoint_writes",
)
textbook_graph = build_graph(checkpointer=checkpointer)


@router.post("/upload", summary="上传教材", description="支持批量上传，接受格式当前为pdf")
async def upload_textbooks(files: list[UploadFile] = File(...)):
    """上传教材：每次上传在 textbooks/ 下创建独立目录 pdf-{特征值}/ 并保存文件。

    Args:
        files: 批量上传的教材 PDF 文件。

    Returns:
        textbook_path: 本次上传的独立目录，供 /resolve 使用。
    """
    textbook_dir = _new_textbook_dir()
    # 保存文件
    saved_files = []
    for file in files:
        with open(textbook_dir / file.filename, "wb") as f:
            f.write(await file.read())
        saved_files.append(file.filename)

    return JSONResponse(
        {
            "textbook_path": str(textbook_dir),
            "message": f"上传成功：{len(saved_files)} 个文件",
        }
    )


@router.post(
    "/resolve",
    summary="解析教材",
    description="解析教材并以 SSE 流返回实时进度（合并创建任务与进度订阅为单请求，支持断点续跑）",
)
async def resolve_textbooks(
    textbook_path: str | None = None,
    task_id: str | None = None,
    user_id: str = Depends(get_user_id),
    offsets: list[dict] | None = Body(default=None),
) -> StreamingResponse:
    """解析教材：在 /upload 返回的独立目录下执行摄入流水线（支持断点续跑）。

    合并为单个 SSE 请求：任务在请求协程内执行，实时推送进度事件
    （message 进度 / done 终态 / error 异常），checkpointer 持久化状态。
    durability="sync" 使每个节点完成后同步落盘，进程/连接中断后可恢复。

    断点续跑：传入已存在的 task_id（断线重连时复用），后台利用 checkpoint 内置能力
    （aget_state 判定中断、astream(None, ...) 从上次完成节点续跑），无需自建任务表。
    """
    # 任务标识：缺省则新建；传入则复用（用于断线重连/续跑）
    task_id = task_id or str(uuid.uuid4())
    thread_id = f"{user_id}:{task_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 借助 checkpoint 判断线程所处状态：
    # - 停在 offset 人工确认中断（tasks 带 interrupts，value.type=="offset_review"）
    # - 因断线停在节点边界、尚未完成（next 非空）
    snapshot = await textbook_graph.aget_state(config)
    prev = (snapshot.values or {}) if snapshot else {}

    pending_offset = any(
        (getattr(i, "value", None) or {}).get("type") == "offset_review"
        for task in (getattr(snapshot, "tasks", None) or ())
        for i in (getattr(task, "interrupts", None) or ())
    )
    resumable = bool(prev and not prev.get("ingestion_done") and getattr(snapshot, "next", None))

    if pending_offset:
        if not offsets:
            raise HTTPException(status_code=400, detail="缺少 offsets：请先确认章节页码偏移后再续跑")
        run_input = Command(resume={"offsets": offsets})
        resolved_path = prev.get("textbook_path") if prev else None
    elif resumable:
        # 断点续跑：输入传 None，从上次完成节点继续，不做重头解析
        run_input = None
        resolved_path = prev.get("textbook_path") if prev else None
    else:
        if not textbook_path:
            raise HTTPException(status_code=400, detail="缺少 textbook_path，请先调用 /upload 上传教材")
        if not Path(textbook_path).is_dir():
            raise HTTPException(status_code=404, detail=f"教材目录不存在: {textbook_path}")
        resolved_path = textbook_path
        run_input: TextBookState = {
            "textbook_exists": False,
            "user_id": user_id,
            "textbook_path": resolved_path,
        }

    async def event_gen():
        try:
            if pending_offset or resumable:
                payload = json.dumps(
                    {"type": "info", "task_id": task_id, "resumed": True, "message": "检测到中断，正在续跑…"},
                    ensure_ascii=False,
                )
            else:
                payload = json.dumps(
                    {"type": "info", "task_id": task_id, "resumed": False, "message": "开始解析", "progress": 0.0},
                    ensure_ascii=False,
                )
            yield f"data: {payload}\n\n"
            async for event in textbook_graph.astream(
                run_input, config=config, stream_mode="custom", durability="sync"
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            # 流结束后确认是否停在 offset 确认中断：停在中断不发 done，由 ask_offset 事件收尾
            snap_after = await textbook_graph.aget_state(config)
            awaiting_offset = any(
                (getattr(i, "value", None) or {}).get("type") == "offset_review"
                for task in (getattr(snap_after, "tasks", None) or ())
                for i in (getattr(task, "interrupts", None) or ())
            )
            if not awaiting_offset:
                yield f"data: {json.dumps({"type": "done", "task_id": task_id}, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            # 教材未找到等业务错误
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}"
        except Exception as exc:
            logger.exception(f"解析任务 {task_id} 执行异常: {exc}")
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}"
    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/list", summary="获取所有教材", description="获取所有教材（教材库全局共享，不按用户隔离），支持分页")
async def get_all_textbooks(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
):
    """分页获取教材"""
    result = list_textbooks(page=page, page_size=page_size)
    return JSONResponse(result)
