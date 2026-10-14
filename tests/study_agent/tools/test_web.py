"""search_web 工具格式化测试：桩掉 MCP 联网，只验证输出渲染与失败抛点。"""

import asyncio

import pytest

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


def test_search_web_upstream_failure_raises(monkeypatch) -> None:
    """联网链路故障与"搜到了但没内容"必须可区分：前者抛异常，由图上官方中间件给说明文本。"""

    async def _fake_search(query: str, count: int | None = None) -> list[dict]:
        raise RuntimeError("MCP 连接失败")

    monkeypatch.setattr(web, "_search_web", _fake_search)
    with pytest.raises(RuntimeError, match="MCP 连接失败"):
        asyncio.run(web.search_web.ainvoke({"query": "二叉树遍历"}))

    # 空结果仍是正常结果（上面 test_search_web_empty 覆盖），两者不能混为一谈
