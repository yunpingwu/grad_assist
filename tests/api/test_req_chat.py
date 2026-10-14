"""ChatRequest 入参边界测试：超长输入在校验层直接拒绝（防超长注入与滥用），空 query 仍可用于确认/澄清回应。"""

import pytest
from pydantic import ValidationError

from app.api.req.chat import ChatRequest, ClarifyAnswerIn


def test_overlong_query_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(textbook_name="T", query="长" * 2001)


def test_overlong_textbook_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(textbook_name="名" * 201, query="q")


def test_overlong_clarify_answer_rejected() -> None:
    with pytest.raises(ValidationError):
        ClarifyAnswerIn(id="q1", answer="答" * 2001)


def test_empty_query_still_accepted_for_resume_flows() -> None:
    """回应写盘确认/澄清反问时 query 可留空：长度下限保持 0。"""
    req = ChatRequest(textbook_name="T", query="", decision="approve")
    assert req.query == ""
