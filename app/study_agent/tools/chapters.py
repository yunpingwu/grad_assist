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
