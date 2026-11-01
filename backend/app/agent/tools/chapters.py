"""list_chapters 工具：列出教材章节结构，供 Agent 了解教材骨架与规划检索范围。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.core import logger
from app.core.safety import wrap_untrusted
from app.retrieval.store import list_chapters as _list_chapters


@tool
async def list_chapters(
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
) -> str:
    """列出教材「章 → 节」结构，供目录查询或按章规划检索范围。

    单个知识点问答应直接调用 search_textbook；chapter 过滤参数必须使用本工具返回的原文。
    """
    try:
        chapters = _list_chapters(textbook_name or "")
    except Exception as exc:
        # 章节聚合依赖 Milvus：不可用时交给图上的官方 ToolRetryMiddleware 出说明文本
        logger.warning(f"list_chapters({textbook_name!r}) 失败: {exc}")
        raise RuntimeError(f"章节结构查询失败: {type(exc).__name__}: {exc}") from exc
    if not chapters:
        return "（未查到章节结构，可能教材尚未解析入库）"

    lines: list[str] = []
    for item in chapters:
        chapter = item["chapter"]
        sections = item.get("sections") or []
        lines.append(f"- {chapter}" + (f"（{'、'.join(sections)}）" if sections else ""))
    logger.info(f"list_chapters: 共 {len(chapters)} 章")
    # 章节名来自 PDF 解析产物（不可信文本），与检索片段同规格包边界
    return wrap_untrusted("\n".join(lines), "章节")
