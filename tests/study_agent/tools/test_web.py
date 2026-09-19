"""search_web 工具格式化测试：桩掉 MCP 联网，只验证输出渲染。"""

import asyncio

from app.study_agent.tools import web


def test_search_web_formatting(monkeypatch) -> None:
    async def _fake_search(query: str, count: int | None = None) -> list[dict]:
        return [{"title": "示例", "url": "https://example.com", "content": f"关于 {query} 的内容"}]

    monkeypatch.setattr(web, "_search_web", _fake_search)
    out = asyncio.run(web.search_web.ainvoke({"query": "二叉树遍历"}))
    assert "示例" in out
    assert "https://example.com" in out


def test_search_web_empty(monkeypatch) -> None:
    async def _fake_search(query: str, count: int | None = None) -> list[dict]:
        return []

    monkeypatch.setattr(web, "_search_web", _fake_search)
    out = asyncio.run(web.search_web.ainvoke({"query": "不存在的内容"}))
    assert "未搜索到相关结果" in out
