import json
import uuid

from fastapi import APIRouter, HTTPException, Response
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.clients import mongo_client
from app.config import mongo_config
from app.core import logger
from app.query_agent.graph import build_graph
from app.query_agent.state import QueryState

router = APIRouter(tags=["query"])


class ChatRequest(BaseModel):
    textbook_name: str = Field(..., description="教材名")
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID，缺省时后端生成")
    is_web_search: bool = Field(default=False, description="是否启用联网搜索（前端按钮控制）")


# 模块级编译一次。
# 注意：MongoDBSaver.from_conn_string 是 context manager（generator），
# 直接实例化 MongoDBSaver(client, db) 更符合 FastAPI 常驻进程的用法；
# 复用项目 MongoClient 单例，认证已由 uri / username / password 配置处理。
checkpointer = MongoDBSaver(mongo_client.get_client(), mongo_config.db)
query_graph = build_graph(checkpointer=checkpointer)


@router.post(
    "/chat/stream",
    summary="流式问答",
    description="SSE 事件流：stage 阶段提示 / token 答案增量 / done 收尾(含 session_id) / error 异常",
)
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """执行一轮问答，以 SSE 实时推送过程事件。

    多轮会话由 LangGraph checkpointer 管理：thread_id = session_id，
    每轮把用户问题放入 messages（add_messages 追加），checkpointer 自动持久化到 MongoDB。
    """
    session_id = req.session_id or str(uuid.uuid4())
    state: QueryState = {
        "session_id": session_id,
        "textbook_name": req.textbook_name,
        "original_query": req.query,
        "is_web_search": req.is_web_search,
        "messages": [HumanMessage(content=req.query)],
    }
    config = {"configurable": {"thread_id": session_id}}

    async def event_gen():
        try:
            async for chunk in query_graph.astream(state, config=config, stream_mode="custom"):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            # 教材未登记、问题为空等业务错误
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        except Exception as exc:
            # Milvus / LLM / embedding 等基础设施错误
            logger.exception(f"问答失败: {exc}")
            payload = json.dumps({"type": "error", "message": "生成答案失败"}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions", summary="会话列表", description="按教材列出会话摘要（从 checkpoint 读取），最近更新在前")
async def sessions(textbook_name: str | None = None) -> list[dict]:
    """列出某教材的全部会话摘要（基于 checkpoints 集合，倒序）。"""
    if not textbook_name:
        raise HTTPException(status_code=400, detail="缺少 textbook_name")

    thread_ids = mongo_client.get_collection("checkpoints").distinct("thread_id")
    items: list[dict] = []
    for tid in thread_ids:
        snapshot = await query_graph.aget_state({"configurable": {"thread_id": tid}})
        values = snapshot.values or {}
        if values.get("textbook_name") != textbook_name:
            continue
        messages = values.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.type == "human"), None)
        items.append(
            {
                "session_id": tid,
                "updated_at": snapshot.created_at or "",
                "message_count": len(messages),
                # 返回最新用户问题原文；截断等展示整形由前端负责
                "last_question": str(last_user.content) if last_user else "",
            }
        )
    items.sort(key=lambda s: s["updated_at"], reverse=True)
    return items


@router.get("/history", summary="会话历史", description="获取某会话的全部消息（从 checkpoint 读取），供历史回显")
async def history(session_id: str | None = None) -> list[dict]:
    """获取某会话的结构化消息列表。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    snapshot = await query_graph.aget_state({"configurable": {"thread_id": session_id}})
    messages = (snapshot.values or {}).get("messages", [])

    return [
        {
            "role": "user" if m.type == "human" else "assistant",
            "content": str(m.content),
        }
        for m in messages
    ]


@router.delete("/sessions/{session_id}", summary="删除会话", description="删除单个会话（删除其全部 checkpoint）")
async def delete_session(session_id: str) -> Response:
    """删除单个会话（幂等：会话不存在也返回 204）。"""
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    checkpointer.delete_thread(session_id)
    return Response(status_code=204)
