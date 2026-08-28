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
    """列出教材的完整章节结构（章节 + 小节），了解教材骨架。

    适合在开工前调用，确定按哪些章节检索与组织内容；search_textbook 的
    chapter 参数必须传本工具返回的 chapter 原文。

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


# 冒烟测试：桩掉 Milvus 聚合，验证格式化
if __name__ == "__main__":
    import asyncio

    def _fake_list(textbook_name: str) -> list[dict]:
        assert textbook_name == "C语言程序设计", textbook_name
        return [
            {"chapter": "第1章 绪论", "sections": ["1.1 概述"]},
            {"chapter": "第2章 数据类型", "sections": []},
        ]

    # 覆盖模块级绑定，只验证输出格式
    _list_chapters = _fake_list

    async def _run() -> None:
        out = await list_chapters.coroutine(textbook_name="C语言程序设计")
        assert "- 第1章 绪论（1.1 概述）" in out, out
        assert "- 第2章 数据类型" in out, out
        print(out)
        print("list_chapters 测试通过")

    asyncio.run(_run())
