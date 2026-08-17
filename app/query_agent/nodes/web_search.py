"""联网搜索节点：通过百炼 WebSearch MCP 检索互联网，补充教材检索不到的实时/外部信息。

- 仅在 state.is_web_search 为真且检索结果不足时由条件边路由到本节点；
- MCP 连接与搜索逻辑直接写在节点内（不单独建工具类）；
- 任何失败均降级为空结果，不阻断主链路。
"""

import json

import httpx2
from langgraph.types import StreamWriter
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.config import llm_config, web_search_config
from app.core import log_node, logger
from app.query_agent.state import QueryState


async def _search_web(query: str, count: int | None = None) -> list[dict]:
    """调用百炼 WebSearch MCP 搜索，返回 [{title, url, content}]。

    返回结构来自 bailian_web_search：{"pages": [{title, url, snippet, ...}]}。
    失败降级为空列表（联网搜索是旁路增强，不阻断问答主链路）。
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


@log_node
async def web_search(state: QueryState, *, writer: StreamWriter) -> dict:
    """根据重写问题联网搜索，结果写入 web_chunks 供生成节点合并。"""
    writer({"type": "stage", "stage": "web_search", "message": "正在联网搜索…"})
    query = state.get("rewritten_query") or state.get("original_query", "")
    web_chunks = await _search_web(query)
    logger.info(f"联网搜索 {query!r} → {len(web_chunks)} 条结果")
    return {"web_chunks": web_chunks}


# 冒烟测试：桩掉 MCP 联网搜索，验证查询选择与 web_chunks 写回
if __name__ == "__main__":
    import asyncio

    async def _fake_search(query: str, count: int | None = None) -> list[dict]:
        return [
            {"title": "示例结果", "url": "https://example.com", "content": f"关于 {query} 的内容"},
        ]

    _search_web = _fake_search  # 覆盖真实 MCP 调用

    def writer(chunk):
        print(f"  [writer] {chunk}")

    async def _run() -> tuple[list[dict], list[dict]]:
        # 1) rewritten_query 优先
        r1 = await web_search(
            {"session_id": "test", "original_query": "原始问题", "rewritten_query": "重写后的问题"},
            writer=writer,
        )
        assert r1["web_chunks"][0]["content"].startswith("关于 重写后的问题"), r1
        # 2) 无 rewritten_query 时回退 original_query
        r2 = await web_search(
            {"session_id": "test", "original_query": "只有原始问题"},
            writer=writer,
        )
        assert r2["web_chunks"][0]["content"].startswith("关于 只有原始问题"), r2
        return r1["web_chunks"], r2["web_chunks"]

    w1, w2 = asyncio.run(_run())
    print(f"web_search 测试通过：rewritten 优先 {len(w1)} 条，original 回退 {len(w2)} 条")
