"""search_web 工具：通过百炼 WebSearch MCP 联网搜索（降级兜底，旁路增强）。"""

from __future__ import annotations

from langchain_core.tools import tool

from app.core import logger
from app.query_flow.nodes.web_search import _search_web


@tool
async def search_web(query: str) -> str:
    """联网搜索补充外部资料（教材未覆盖的内容时才用）。

    Args:
        query: 搜索关键词（简短有效，如「二叉树的遍历方式 2026」）。

    Returns:
        搜索结果摘要（含来源链接），无结果时返回提示。
    """
    results = await _search_web(query)
    if not results:
        return "（未搜索到相关结果）"

    lines: list[str] = []
    for i, item in enumerate(results, start=1):
        title = item.get("title") or ""
        url = item.get("url") or ""
        content = item.get("content") or ""
        lines.append(f"{i}. [{title}]({url})\n   {content}")

    logger.info(f"search_web({query!r}) → {len(results)} 条")
    return "\n\n".join(lines)


# 冒烟测试：桩掉 MCP 联网
if __name__ == "__main__":
    import asyncio

    async def _fake_search(query: str, count: int | None = None) -> list[dict]:
        return [
            {"title": "示例", "url": "https://example.com", "content": f"关于 {query} 的内容"},
        ]

    _search_web = _fake_search

    async def _run() -> None:
        out = await search_web.ainvoke({"query": "二叉树遍历"})
        assert "示例" in out, out
        assert "https://example.com" in out, out
        print(out)
        print("search_web 测试通过")

    asyncio.run(_run())
