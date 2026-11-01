"""统一教材助手路由：HTTP 参数/依赖与响应封装，业务逻辑见 app/services/study.py。"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import Response, StreamingResponse

from app.api.deps import get_user_id
from app.api.schemas import ChatRequest
from app.services import study as study_service

router = APIRouter(tags=["study"])

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post(
    "/study/chat",
    summary="统一助手对话（含写盘确认/澄清反问续跑）",
    description="SSE 事件流：tool / thought / token / ask_confirm / ask_clarify / done / error；多轮经 session_id 续接；"
    "对停留在写盘确认的会话传 decision=approve|reject 统一放行/拒绝；"
    "对停留在澄清反问的会话传 clarify_answers 续跑",
)
async def chat(req: ChatRequest, user_id: str = Depends(get_user_id)) -> StreamingResponse:
    """统一对话入口：新问题直接跑；上轮停在写盘确认时按 decision 放行/拒绝后续跑同一线程。"""
    agent_input, config, session_id = await study_service.prepare_chat(
        user_id=user_id,
        session_id=req.session_id,
        textbook_name=req.textbook_name,
        query=req.query,
        deep=req.deep,
        decision=req.decision,
        clarify_answers=(
            [a.model_dump() for a in req.clarify_answers] if req.clarify_answers is not None else None
        ),
    )
    return StreamingResponse(
        study_service.chat_events(agent_input, config, session_id, req.query),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/study/files", summary="任务产物文件列表", description="列出某任务目录内已生成的文件（相对路径 + 大小）")
async def files(task_id: str, textbook_name: str, user_id: str = Depends(get_user_id)) -> list[dict]:
    """列出某任务的全部落盘文件。"""
    return study_service.files(task_id, textbook_name)


@router.get("/study/file", summary="读取产物文件内容", description="读取任务目录内一份文件（路径经沙箱校验）")
async def read_file(task_id: str, textbook_name: str, path: str) -> dict:
    """读取一份文件内容（前端预览用）。"""
    try:
        return study_service.read_file(task_id, textbook_name, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/sessions",
    summary="会话列表",
    description="按教材列出当前用户的对话会话摘要（从 study checkpoint 读取），最近更新在前，排除资料任务线程",
)
async def sessions(textbook_name: str | None = None, user_id: str = Depends(get_user_id)) -> list[dict]:
    """列出当前用户某教材的全部对话会话摘要（基于 study schema 的 checkpoints 表，倒序）。"""
    if not textbook_name:
        raise HTTPException(status_code=400, detail="缺少 textbook_name")
    return await study_service.sessions(textbook_name, user_id)


@router.get(
    "/history",
    summary="会话历史",
    description="获取当前用户某会话的全部消息（从 study checkpoint 读取），供历史回显",
)
async def history(session_id: str | None = None, user_id: str = Depends(get_user_id)) -> list[dict]:
    """获取某会话的结构化消息列表。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    return await study_service.history(session_id, user_id)


@router.delete(
    "/sessions/{session_id}",
    summary="删除会话",
    description="删除当前用户的单个对话会话（删除其全部 checkpoint）",
)
async def delete_session(session_id: str, user_id: str = Depends(get_user_id)) -> Response:
    """删除单个会话（幂等：会话不存在也返回 204）。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    await study_service.delete_session(session_id, user_id)
    return Response(status_code=204)
