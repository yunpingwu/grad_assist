"""ask_clarification 工具测试：只覆盖"问题数超限"的抛点（人机回环本身由服务层测）。"""

import asyncio

import pytest

from app.study_agent.tools.clarify import MAX_CLARIFY_QUESTIONS, ask_clarification


def test_too_many_questions_raises() -> None:
    """超过问题上限即报错，说明文本回给模型让它自行收敛。"""
    questions = [
        {"id": f"q{i}", "question": f"问题{i}", "type": "open"} for i in range(MAX_CLARIFY_QUESTIONS + 1)
    ]
    with pytest.raises(ValueError, match=f"最多提出 {MAX_CLARIFY_QUESTIONS} 个"):
        asyncio.run(ask_clarification.ainvoke({"questions": questions}))
