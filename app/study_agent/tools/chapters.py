"""list_chapters 工具：列出教材章节结构，供 Agent 了解教材骨架与规划检索范围。"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.core import logger
from app.utils.milvus_util import list_chapters as _list_chapters


@tool
async def list_chapters(
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
) -> str:
    """按需列出教材章节结构（章节 + 小节）。

    仅用于目录/章节结构查询、跨章节学习计划，或需要先确定整章/多章范围的任务。
    单个知识点问答应直接调用 search_textbook；search_textbook 不传 chapter 时可检索全书。
    如果要按章节过滤，chapter 参数必须使用本工具返回的原文。

    Returns:
        教材「章 → 节」结构清单。
    """
    chapters = _list_chapters(textbook_name or "")
    if not chapters:
        return "（未查到章节结构，可能教材尚未解析入库）"

    lines: list[str] = []
    for item in chapters:
        chapter = item["chapter"]
        sections = item.get("sections") or []
        lines.append(f"- {chapter}" + (f"（{'、'.join(sections)}）" if sections else ""))
    logger.info(f"list_chapters: 共 {len(chapters)} 章")
    return "\n".join(lines)
