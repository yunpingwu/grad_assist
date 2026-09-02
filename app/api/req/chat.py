"""统一助手的对话请求实体。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """统一助手的对话请求。

    - 新问题：只传 textbook_name / query（可带 session_id 续接多轮）；
    - 回应上轮写盘确认：传 session_id + decision（approve/reject），query 可留空。
    """

    textbook_name: str = Field(..., description="教材名（已解析入库）")
    query: str = Field(..., min_length=0, description="用户问题或资料生成要求；仅在回应写盘确认时可留空")
    session_id: str | None = Field(default=None, description="会话 ID（裸 ID，缺省时后端生成）")
    decision: str | None = Field(
        default=None,
        description='"approve"|"reject"：回应上轮停在写盘确认的线程（对该线程全部挂起写盘统一生效）；缺省视为发新问题',
    )
