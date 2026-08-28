"""search_textbook 工具：复用问答流的混合检索，按教材（可选章节）语义检索。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.core import logger
from app.query_flow.nodes.embedding_search import rewrite_query_search

# 单片段最大字符数（压缩护栏：防止原文过长挤爆上下文）
MAX_CHARS_PER_HIT = 600


@tool
async def search_textbook(
    query: str,
    chapter: str | None = None,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
) -> str:
    """在教材库中按语义检索相关知识点片段（混合检索：稠密 + 稀疏）。

    撰写任何基于教材的内容前都应先用本工具拿到教材原文依据；可限定章节缩小范围。

    Args:
        query: 检索查询，用完整问句描述想找的知识点，如「栈的先进后出特性是什么」。
        chapter: 限定章节名（可选），与 list_chapters 返回的 chapter 完全一致时才传。

    Returns:
        编号的 TOP-K 片段（章节 + 小节 + 正文截断），无结果时返回提示。
    """
    chunks = await rewrite_query_search(textbook_name=textbook_name or "", rewrite_query=query, chapter=chapter)
    if not chunks:
        return "（未检索到相关片段，请换个问法或去掉章节限定）"

    parts: list[str] = []
    for i, hit in enumerate(chunks, start=1):
        entity = hit.get("entity") or hit
        text = (entity.get("text") or "").strip()
        chapter_name = entity.get("chapter") or ""
        section = entity.get("section") or ""
        location = " > ".join(x for x in (chapter_name, section) if x)
        truncated = text if len(text) <= MAX_CHARS_PER_HIT else text[:MAX_CHARS_PER_HIT] + "…"
        parts.append(f"[片段{i}｜{location or '未标注位置'}] {truncated}")

    logger.info(f"search_textbook({query!r}, chapter={chapter!r}) → {len(parts)} 条")
    return "\n\n".join(parts)


# 冒烟测试：桩掉真实检索（依赖 Milvus + embedding 模型），只验证片段拼接
if __name__ == "__main__":
    import asyncio

    async def _fake_search(textbook_name: str, rewrite_query: str, chapter: str | None = None) -> list[dict]:
        assert textbook_name == "C语言程序设计", textbook_name
        assert chapter is None or chapter.startswith("第"), chapter
        return [
            {"id": "c1", "distance": 0.8, "entity": {"text": "指针是C语言的核心概念。", "chapter": "第3章", "section": "3.1"}},
            {"id": "c2", "distance": 0.7, "entity": {"text": "数组是相同类型元素的集合。", "chapter": "第3章", "section": "3.2"}},
        ]

    # 覆盖模块级绑定，只验证拼接逻辑
    rewrite_query_search = _fake_search

    async def _run() -> None:
        # 直接调用工具底层协程（ToolNode 注入链路在真实图执行中生效）
        out = await search_textbook.coroutine(query="什么是指针?", chapter=None, textbook_name="C语言程序设计")
        assert "[片段1｜第3章 > 3.1]" in out, out
        assert "[片段2" in out, out
        print(out)
        print("search_textbook 测试通过")

    asyncio.run(_run())
