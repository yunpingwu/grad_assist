"""list_chapters 工具格式化测试：桩掉 Milvus 聚合，只验证输出结构与失败抛点。"""

import asyncio

import pytest

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


def test_list_chapters_upstream_failure_raises(monkeypatch) -> None:
    """Milvus 不可用抛带说明的异常，由图上官方中间件兜成模型可见文本。"""

    def _boom(_name: str) -> list[dict]:
        raise RuntimeError("milvus 连接超时")

    monkeypatch.setattr(chapters, "_list_chapters", _boom)
    with pytest.raises(RuntimeError, match="章节结构查询失败"):
        asyncio.run(chapters.list_chapters.ainvoke({"textbook_name": "C语言程序设计"}))
