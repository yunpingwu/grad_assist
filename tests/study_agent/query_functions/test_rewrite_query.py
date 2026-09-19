"""rewrite_query 测试：format_questions 纯函数 + rewrite 桩验证历史拼装。"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.study_agent.query_functions import rewrite_query


def test_format_questions_keeps_human_history() -> None:
    history = [HumanMessage(content="什么是指针?"), AIMessage(content="指针是一种…")]
    assert rewrite_query.format_questions(history) == "用户: 什么是指针?"


def test_format_questions_empty_without_history() -> None:
    assert rewrite_query.format_questions([]) == ""


def test_rewrite_passes_history_into_prompt(monkeypatch) -> None:
    """桩掉 LLM 重写，验证多轮历史已拼入重写输入。"""

    async def _fake_rewrite(original_query: str, textbook_name: str, questions_history: str) -> str:
        assert "什么是指针?" in questions_history, f"多轮历史未拼进重写输入: {questions_history!r}"
        return f"{original_query}（针对教材 {textbook_name} 重写）"

    monkeypatch.setattr(rewrite_query, "rewrite", _fake_rewrite)
    history = [HumanMessage(content="什么是指针?"), AIMessage(content="指针是一种…")]
    rewritten = asyncio.run(rewrite_query.rewrite(
        "如何使用它?", "C语言程序设计", rewrite_query.format_questions(history)
    ))
    assert rewritten and "如何使用它?" in rewritten
