import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from langgraph.checkpoint.mongodb import MongoDBSaver
from starlette.responses import JSONResponse, StreamingResponse

from app.api.deps import get_user_id
from app.clients import mongo_client
from app.config import mongo_config
from app.core import logger
from app.textbook_agent.graph import build_graph
from app.textbook_agent.state import TextBookState
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

    # 借助 checkpoint 判断线程是否为「曾被中断且未完成」：next 非空 = 尚有未执行节点。
    snapshot = await textbook_graph.aget_state(config)
    prev = (snapshot.values or {}) if snapshot else {}
    interrupted = bool(prev and not prev.get("ingestion_done") and snapshot.next)
    # 续跑时沿用 checkpoint 中已持久化的路径；全新任务则校验本次传入路径
    resolved_path = (prev.get("textbook_path") if interrupted else None) or textbook_path
    if not interrupted:
        if not resolved_path:
            raise HTTPException(status_code=400, detail="缺少 textbook_path，请先调用 /upload 上传教材")
        if not Path(resolved_path).is_dir():
            raise HTTPException(status_code=404, detail=f"教材目录不存在: {resolved_path}")

    # 续跑时输入传 None：LangGraph 据此从断点继续，不做重头解析
    fresh_state: TextBookState = {
        "textbook_exists": False,
        "user_id": user_id,
        "textbook_path": resolved_path,
    }
    run_input = None if interrupted else fresh_state

    async def event_gen():
        try:
            if interrupted:
                payload = json.dumps(
                    {"type": "info", "task_id": task_id, "resumed": True, "message": "检测到中断，正在从断点续跑…"},
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


@router.get("/list", summary="获取所有教材", description="获取所有教材（教材库全局共享，不按用户隔离）")
async def get_all_textbooks():
    """获取所有教材"""
    textbooks = list_textbooks()
    return JSONResponse(textbooks)
