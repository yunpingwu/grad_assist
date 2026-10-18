"""教材摄入路由：HTTP 参数/依赖与响应封装，业务逻辑见 app/services/textbook.py。"""

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from starlette.responses import JSONResponse, StreamingResponse

from app.api.deps import get_user_id
from app.services import textbook as textbook_service

router = APIRouter(tags=["textbook"])


@router.post("/upload", summary="上传教材", description="支持批量上传，接受格式当前为pdf")
async def upload_textbooks(files: list[UploadFile] = File(...)):
    """上传教材：每次上传在 textbooks/ 下创建独立目录 pdf-{特征值}/ 并保存文件。

    Returns:
        textbook_path: 本次上传的独立目录，供 /resolve 使用。
    """
    textbook_dir, count = await textbook_service.save_uploads(files)
    return JSONResponse(
        {
            "textbook_path": str(textbook_dir),
            "message": f"上传成功：{count} 个文件",
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

    任务在请求协程内执行，实时推送进度事件（message 进度 / done 终态 / error 异常），
    checkpointer 持久化状态。durability="sync" 使每个节点完成后同步落盘，进程/连接中断后可恢复。

    断点续跑：传入已存在的 task_id（断线重连时复用），后台利用 checkpoint 内置能力
    （aget_state 判定中断、astream(None, ...) 从上次完成节点续跑），无需自建任务表。
    停在偏移确认时不带 offsets 的重连视为「重看」：重放并重新发出 ask_offset，供前端再次弹出确认面板。
    """
    try:
        plan = await textbook_service.prepare_resolve(
            textbook_path=textbook_path,
            task_id=task_id,
            user_id=user_id,
            offsets=offsets,
        )
    except textbook_service.MissingTextbookPath as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except textbook_service.TextbookPathNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        textbook_service.resolve_events(plan),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/list", summary="获取所有教材", description="获取所有教材（教材库全局共享，不按用户隔离），支持分页")
async def get_all_textbooks(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(8, ge=1, le=100, description="每页条数"),
):
    """分页获取教材"""
    return JSONResponse(textbook_service.list_page(page=page, page_size=page_size))
