"""search_textbook 工具：叠加强化检索（重写消歧 + 多路召回 + 精排），供 agent 自主调用。

将 query_flow 的核心能力折叠进单一检索工具，最终回答统一由 agent 模型生成，本工具只返回检索片段：
- 始终先做问题重写（利用对话历史消歧），提升召回针对性；
- ``deep=False`` 快速路径：混合召回（稠密+稀疏）后直接返回 TOP-K，适合反复取素材；
- ``deep=True`` 深度路径：再叠加 HyDE 假设文档召回 → RRF 融合 → 交叉编码精排，适合严谨作答。
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.config import rerank_config
from app.core import logger
from app.query_flow.nodes.embedding_search import rewrite_query_search
from app.query_flow.nodes.hyde_embedding_search import hyde_doc_generate, hyde_doc_search
from app.query_flow.nodes.merge_recalls import rrf_merge
from app.query_flow.nodes.rerank import rerank_chunks
from app.query_flow.nodes.rewrite_query import format_questions, rewrite

# 单片段最大字符数（压缩护栏：防止原文过长挤爆上下文）
MAX_CHARS_PER_HIT = 600


def _format_hits(chunks: list[dict]) -> str:
    """将 hit 列表格式化为编号的 TOP-K 片段文本。

    Args:
        chunks: 检索引擎返回的 hit 列表（含 entity 或扁平字段）。

    Returns:
        编号的片段文本；供 deep 快速/深度两路返回复用。
    """
    parts: list[str] = []
    for i, hit in enumerate(chunks, start=1):
        entity = hit.get("entity") or hit
        text = (entity.get("text") or "").strip()
        chapter = entity.get("chapter") or ""
        section = entity.get("section") or ""
        location = " > ".join(x for x in (chapter, section) if x)
        truncated = text if len(text) <= MAX_CHARS_PER_HIT else text[:MAX_CHARS_PER_HIT] + "…"
        parts.append(f"[片段{i}｜{location or '未标注位置'}] {truncated}")
    return "\n\n".join(parts)


def _collect_image_candidates(chunks: list[dict]) -> list[str]:
    """从召回片段中收集图片候选（简介 + 可复制 markdown 引用），供 agent 插入回答。

    Args:
        chunks: 检索引擎返回的 hit 列表（实体须含 ``metadata_json`` 字段）。

    Returns:
        形如 ["- 简介: …\n  引用: ![图注](url)"] 的行列表；无候选返回空列表。
    """
    lines: list[str] = []
    seen: set[str] = set()
    for hit in chunks:
        entity = hit.get("entity") or hit
        meta_raw = entity.get("metadata_json")
        if not meta_raw:
            continue
        try:
            meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for img in meta.get("images", []) or []:
            url = img.get("url")
            desc = (img.get("description") or "").strip()
            if not url or not desc or url in seen:
                continue
            seen.add(url)
            lines.append(f"- 简介: {desc}\n  引用: ![{desc}]({url})")
    return lines


@tool
async def search_textbook(
    query: str,
    deep: bool = False,
    chapter: str | None = None,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    messages: Annotated[list, InjectedState("messages")] = None,
) -> str:
    """取教材依据：回答基于教材的问题或整理资料前，先按语义检索相关知识点片段（自动消歧 + 可选深度召回精排）。

    每次检索都会自动用对话历史消歧当前问句；默认快速混合召回（稠密+稀疏），如需更高质量（如严谨作答、
    快速检索无果时扩大召回）请设 deep=True：多做一次 HyDE 假设文档召回，再经 RRF
    融合与交叉编码精排——结果更准但更慢。

    Args:
        query: 检索查询，用完整问句描述想找的知识点，如「栈的先进后出特性是什么」。
        deep: 是否启用深度召回（HyDE + 精排），默认 False；对答案质量要求高或快检索无果时设为 True。
        chapter: 限定章节名（可选），与 list_chapters 返回的 chapter 完全一致时才传。

    Returns:
        编号的 TOP-K 片段（章节 + 小节 + 正文截断），无结果或失败时返回友好提示。
    """
    textbook_name = textbook_name or ""
    try:
        # 1) 问题重写消歧：多轮指代/省略由 agent 上下文保留，此处重写为独立检索问句
        questions_history = format_questions(messages or [])
        rewritten = await rewrite(query, textbook_name, questions_history)
        logger.info(f"search_textbook 重写: {query!r} → {rewritten!r}")

        if not deep:
            # 快速路径：单路混合召回（稠密 + 稀疏）后截取 TOP-K
            chunks = await rewrite_query_search(textbook_name, rewritten, chapter)
            hits = chunks[:rerank_config.top_k]
        else:
            # 深度路径：两路召回（混合 + HyDE）→ RRF 融合 → 精排 TOP-K
            embedding_chunks = await rewrite_query_search(textbook_name, rewritten, chapter)
            hyde_doc = await hyde_doc_generate(rewritten)
            hyde_chunks = await hyde_doc_search(hyde_doc, rewritten, textbook_name)
            merged = await rrf_merge(embedding_chunks, hyde_chunks)
            merged_hits = [entry["hit"] for entry in merged]
            try:
                hits = rerank_chunks(rewritten, merged_hits, min(rerank_config.top_k, len(merged_hits)))
            except Exception as exc:  # 精排降级：回退 RRF 融合结果，不阻断
                logger.warning(f"deep 检索精排失败，回退 RRF：{exc}")
                hits = merged_hits[:rerank_config.top_k]
    except Exception as exc:
        # 检索失败降级：只告警并返回友好提示，交由 agent 决定回退 web 或如实说明
        logger.warning(f"search_textbook({query!r}, deep={deep}) 失败: {exc}")
        return f"（检索教材失败：{exc}）"

    if not hits:
        return "（未检索到相关片段，请换个问法或去掉章节限定）"

    logger.info(f"search_textbook({query!r}, deep={deep}) → {len(hits)} 条")
    result = _format_hits(hits)
    # 附带图片候选：简介 + 可复制引用，供 agent 决定是否在回答中插入（图片指令见 system prompt）
    img_lines = _collect_image_candidates(hits)
    if img_lines:
        result += "\n\n【图片候选】\n" + "\n".join(img_lines)
    return result


# 冒烟测试：桩掉 LLM/检索/精排（依赖真实模型/Milvus），只验证编排与拼接
if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage

    async def _fake_rewrite(original_query: str, textbook_name: str, questions_history: str) -> str:
        assert "什么是指针?" in questions_history, f"多轮历史未拼进重写: {questions_history!r}"
        return original_query

    async def _fake_hybrid(textbook_name: str, rewrite_query: str, chapter: str | None = None) -> list[dict]:
        return [
            {"id": "c1", "distance": 0.8, "entity": {
                "text": "指针是C语言的核心概念。", "chapter": "第3章", "section": "3.1",
                "metadata_json": json.dumps({"images": [{"url": "https://x/y.png", "description": "指针示意图"}]}),
            }},
            {"id": "c2", "distance": 0.7, "entity": {"text": "数组是相同类型元素的集合。", "chapter": "第3章", "section": "3.2"}},
        ]

    async def _fake_hyde_generate(rewritten_query: str) -> str:
        return f"假设性文档: {rewritten_query}"

    async def _fake_hyde_search(hyde_doc: str, rewritten_query: str, textbook_name: str) -> list[dict]:
        return [{"id": "h1", "distance": 0.5, "entity": {"text": "HyDE 补充片段", "chapter": "第3章", "section": "3.3"}}]

    async def _fake_rrf(embedding_chunks: list[dict], hyde_chunks: list[dict], k: int = 60) -> list[dict]:
        return [{"rrf_score": 1.0, "hit": embedding_chunks[0]}, {"rrf_score": 0.8, "hit": hyde_chunks[0]}]

    def _fake_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
        item = dict(chunks[0])
        item["rerank_score"] = 0.99
        return [item]

    # 覆盖模块级绑定，只验证工具内部编排
    rewrite = _fake_rewrite
    rewrite_query_search = _fake_hybrid
    hyde_doc_generate = _fake_hyde_generate
    hyde_doc_search = _fake_hyde_search
    rrf_merge = _fake_rrf
    rerank_chunks = _fake_rerank

    async def _run() -> None:
        # 模拟一轮已作答的历史：保证消歧历史被拼进重写输入
        history = [HumanMessage(content="什么是指针?"), AIMessage(content="指针是…")]

        # 快速路径：重写 + 混合召回，返回 TOP-K 片段
        fast = await search_textbook.coroutine(
            query="它有什么用途?", textbook_name="C语言程序设计", messages=history
        )
        assert "[片段1｜第3章 > 3.1]" in fast and "[片段2" in fast, fast
        assert "【图片候选】" in fast and "指针示意图" in fast, "图片候选未装配"
        print("--- fast ---\n" + fast)

        # 深度路径：触发 HyDE + 融合 + 精排，取精排后片段
        deep = await search_textbook.coroutine(
            query="它有什么用途?", deep=True, textbook_name="C语言程序设计", messages=history
        )
        assert "指针是C语言的核心概念" in deep, deep
        print("--- deep ---\n" + deep)
        print("search_textbook 测试通过")

    asyncio.run(_run())
