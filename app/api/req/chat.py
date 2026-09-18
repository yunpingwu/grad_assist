"""统一助手的对话请求实体。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClarifyAnswerIn(BaseModel):
    """人工对单个澄清问题的作答（回应上轮澄清反问时传）。"""

    id: str = Field(..., description="澄清问题 id（对应 ask_clarify 事件里的 questions[].id）")
    answer: str = Field(..., description="选中的建议选项 method 原文，或用户自定义回答文本")


class ChatRequest(BaseModel):
    """统一助手的对话请求。

    - 新问题：只传 textbook_name / query（可带 session_id 续接多轮）；
    - 回应上轮写盘确认：传 session_id + decision（approve/reject），query 可留空；
    - 回应上轮澄清反问：传 session_id + clarify_answers（每个问题一条），query 可留空。
    """

    textbook_name: str = Field(..., description="教材名（已解析入库）")
    query: str = Field(..., min_length=0, description="用户问题或资料生成要求；仅在回应写盘确认/澄清反问时可留空")
    session_id: str | None = Field(default=None, description="会话 ID（裸 ID，缺省时后端生成）")
    decision: str | None = Field(
        default=None,
        description='"approve"|"reject"：回应上轮停在写盘确认的线程（对该线程全部挂起写盘统一生效）；缺省视为发新问题',
    )
    clarify_answers: list[ClarifyAnswerIn] | None = Field(
        default=None,
        description="回应上轮停在澄清的线程：每个澄清问题的作答；缺省视为发新问题",
    )
