"""list_chapters 工具格式化测试：桩掉 Milvus 聚合，只验证输出结构。"""

import asyncio

from app.study_agent.tools import chapters


def test_list_chapters_formatting(monkeypatch) -> None:
    def _fake_list(textbook_name: str) -> list[dict]:
        assert textbook_name == "C语言程序设计"
        return [
            {"chapter": "第1章 绪论", "sections": ["1.1 概述"]},
            {"chapter": "第2章 数据类型", "sections": []},
        ]

    monkeypatch.setattr(chapters, "_list_chapters", _fake_list)
    out = asyncio.run(chapters.list_chapters.coroutine(textbook_name="C语言程序设计"))
    assert "- 第1章 绪论（1.1 概述）" in out
    assert "- 第2章 数据类型" in out


def test_list_chapters_empty(monkeypatch) -> None:
    monkeypatch.setattr(chapters, "_list_chapters", lambda _: [])
    out = asyncio.run(chapters.list_chapters.coroutine(textbook_name="任意教材"))
    assert "未查到章节结构" in out
