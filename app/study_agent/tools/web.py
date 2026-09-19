"""search_web 工具：通过百炼 WebSearch MCP 联网搜索（降级兜底，旁路增强）。

从 query_functions 收编而来：原 web_search 节点的搜索逻辑并入本工具模块，
联网搜索作为 agent 的旁路补充能力，失败降级为空结果、不阻断主链路。
"""

from __future__ import annotations

import json

import httpx2
from langchain_core.tools import tool
from mcp.client.streamable_http import streamable_http_client

from app.config import llm_config, web_search_config
from app.core import logger
from mcp import ClientSession


async def _search_web(query: str, count: int | None = None) -> list[dict]:
    """调用百炼 WebSearch MCP 搜索，返回 [{title, url, content}]。

    Args:
        query: 搜索关键词。
        count: 返回条数（缺省用配置 search_count）。

    Returns:
        形如 [{title, url, content}] 的结果列表；失败或未配置 API Key 返回空列表。
    """
    if not llm_config.api_key:
        logger.warning("未配置 ALIBABA_API_KEY，跳过联网搜索")
        return []
    count = count or web_search_config.search_count
    http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {llm_config.api_key}"})
    try:
        async with streamable_http_client(web_search_config.mcp_url, http_client=http_client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(web_search_config.tool_name, {"query": query, "count": count})
                text = "".join(getattr(b, "text", "") for b in result.content)
                data = json.loads(text)
                return [
                    {
                        "title": p.get("title", ""),
                        "url": p.get("url", ""),
                        "content": p.get("snippet", ""),
                    }
                    for p in data.get("pages", [])
                    if p.get("snippet")
                ]
    except Exception as exc:
        logger.warning(f"联网搜索失败(query={query!r}): {exc}")
        return []
    finally:
        await http_client.aclose()


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
