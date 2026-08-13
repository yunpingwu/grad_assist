
import json

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.core import logger
from app.query_agent.graph import build_graph
from app.query_agent.state import QueryState
from app.utils.chat_util import delete_session as remove_session_record, get_session_messages, list_sessions

router = APIRouter(tags=["query"])


class ChatRequest(BaseModel):
    textbook_name: str = Field(..., description="教材名")
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID，缺省时图内自动生成")
    is_web_search: bool = Field(default=True, description="是否启用联网搜索（前端按钮控制）")


# 模块级编译一次，LangGraph 编译图无共享可变状态，可并发复用
query_graph = build_graph()


@router.post("/chat/stream", summary="流式问答", description="SSE 事件流：stage 阶段提示 / token 答案增量 / done 收尾(含 session_id) / error 异常")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """执行一轮问答，以 SSE 实时推送过程事件。

    事件（data 为 JSON）：
    - {"type": "stage", "stage": …, "message": …}  图推进阶段提示
    - {"type": "token", "content": …}              答案增量（打字机）
    - {"type": "done", "session_id": …}            生成完成，前端保存会话 ID
    - {"type": "error", "message": …}              业务/基础设施错误
    """
    state: QueryState = {
        "session_id": req.session_id,
        "textbook_name": req.textbook_name,
        "original_query": req.query,
        "is_web_search": req.is_web_search,
    }

    async def event_gen():
        try:
            async for chunk in query_graph.astream(state, stream_mode="custom"):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            # 教材未登记、问题为空等业务错误
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        except Exception as exc:
            # Milvus / LLM / embedding 等基础设施错误（Mongo 故障已由 chat_util 降级）
            logger.exception(f"问答失败: {exc}")
            payload = json.dumps({"type": "error", "message": "生成答案失败"}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions", summary="会话列表", description="按教材列出会话摘要，最近更新在前，供左侧会话栏使用")
async def sessions(textbook_name: str | None = None) -> list[dict]:
    """获取某教材的全部会话摘要（倒序）。"""
    if not textbook_name:
        raise HTTPException(status_code=400, detail="缺少 textbook_name")
    return list_sessions(textbook_name)


@router.get("/history", summary="会话历史", description="获取某会话的全部消息，供点击历史会话时回显")
async def history(session_id: str | None = None) -> list[dict]:
    """获取某会话的结构化消息列表。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    return get_session_messages(session_id)


@router.delete("/sessions/{session_id}", summary="删除会话", description="删除单个历史会话及其全部消息")
async def delete_session(session_id: str) -> Response:
    """删除单个历史会话。

    - 成功: 204 No Content
    - 会话不存在: 404（幂等删除，重复删除同一会话也会得到 404）
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    if not remove_session_record(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return Response(status_code=204)
