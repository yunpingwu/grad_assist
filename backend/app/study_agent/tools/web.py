"""search_web 工具：通过百炼 WebSearch MCP 联网搜索；失败降级为空结果，不阻断主链路。"""

from __future__ import annotations

import json

import httpx2
from langchain_core.tools import tool
from mcp.client.streamable_http import streamable_http_client

from app.config import llm_config, web_search_config
from app.core import logger
from app.study_agent.safety import wrap_untrusted
from mcp import ClientSession


async def _search_web(query: str, count: int | None = None) -> list[dict]:
    """调用百炼 WebSearch MCP，返回 [{title, url, content}]；链路故障抛 RuntimeError。"""
    if not llm_config.api_key:
        logger.warning("未配置 ALIBABA_API_KEY，联网搜索不可用")
        raise RuntimeError("联网搜索未启用：未配置 ALIBABA_API_KEY")
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
        raise RuntimeError(f"联网搜索失败: {type(exc).__name__}: {exc}") from exc
    finally:
        await http_client.aclose()


@tool
async def search_web(query: str) -> str:
    """联网搜索补充外部资料（教材未覆盖时才用）；query 宜简短有效。"""
    results = await _search_web(query)
    if not results:
        return "（未搜索到相关结果）"

    lines: list[str] = []
    for i, item in enumerate(results, start=1):
        # 标题/链接是单行元数据：折叠其中的换行，防脏数据把条目形状撑破（摘要正文保真，进边界）
        title = " ".join((item.get("title") or "").split())
        url = " ".join((item.get("url") or "").split())
        content = item.get("content") or ""
        lines.append(f"{i}. [{title}]({url})\n   {content}")

    logger.info(f"search_web({query!r}) → {len(results)} 条")
    # 网页摘要不可控，整块进外部资料边界（见 app/study_agent/safety.py）
    return wrap_untrusted("\n\n".join(lines), "网页")
