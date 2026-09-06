"""search_textbook 工具：叠加强化检索（重写消歧 + 多路召回 + 精排），供 agent 自主调用。

将 query_functions 的核心能力折叠进单一检索工具，最终回答统一由 agent 模型生成，本工具只返回检索片段：
- 始终先做问题重写（利用对话历史消歧），提升召回针对性；
- ``deep=False`` 快速路径：混合召回（稠密+稀疏）后直接返回 TOP-K，适合反复取素材；
- ``deep=True`` 深度路径：再叠加 HyDE 假设文档召回 → RRF 融合 → 交叉编码精排，适合严谨作答。
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.config import rerank_config
from app.core import logger
from app.utils import get_collection_by_name, query_section_codes
from app.study_agent.query_functions.embedding_search import rewrite_query_search
from app.study_agent.query_functions.hyde_embedding_search import hyde_doc_generate, hyde_doc_search
from app.study_agent.query_functions.merge_recalls import rrf_merge
from app.study_agent.query_functions.rerank import rerank_chunks
from app.study_agent.query_functions.rewrite_query import format_questions, rewrite

# 正文内嵌的图片标记：切块时图片行被替换为「【图: 简介】」，此处按简介回绑 url
_FIGURE_MARK_PATTERN = re.compile(r"【图: (.*?)】")

def _collect_hit_images(hit: dict) -> dict[str, str]:
    """提取单个片段附带的图片映射：简介(正文【图】标记 alt) → MinIO url。

    Args:
        hit: 检索引擎返回的单条 hit（实体须含 ``metadata_json`` 字段）。

    Returns:
        {简介: url} 映射；无图片或缺 url 返回空 dict。
    """
    entity = hit.get("entity") or hit
    meta_raw = entity.get("metadata_json")
    if not meta_raw:
        return {}
    try:
        meta = json.loads(meta_raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    url_by_desc: dict[str, str] = {}
    for img in meta.get("images", []) or []:
        url = img.get("url")
        desc = (img.get("description") or "").strip()
        if not url or not desc or desc in url_by_desc:
            continue
        url_by_desc[desc] = url
    return url_by_desc


def _render_text_entity(entity: dict, url_by_desc: dict[str, str]) -> str:
    """回绑图片 url 到正文图标记并返回正文文本。

    Args:
        entity: 单条 hit 的 entity（含 text 字段）。
        url_by_desc: 简介 → url 映射（来自 metadata_json.images）。

    Returns:
        回绑后的正文文本。
    """
    text = (entity.get("text") or "").strip()
    if url_by_desc:
        def _replace(m: re.Match) -> str:
            desc = m.group(1).strip()
            url = url_by_desc.get(desc)
            # 简介未匹配到 url（旧教材/元数据缺失）时保留原图标记
            return f"【图: {desc}】({url})" if url else m.group(0)

        text = _FIGURE_MARK_PATTERN.sub(_replace, text)
    return text


def _format_hits(chunks: list[dict]) -> str:
    """将 hit 列表格式化为编号的 TOP-K 片段文本。

    处理顺序：
    1. 渲染文本片段（含图片 url 回绑到「【图: 简介】」标记处），记录各
       (chapter, section) 对应的片段位置；
    2. 将代码块（block_type="code"）拼回同 (chapter, section) 的正文片段末尾，
       无对应正文时作为独立片段展示（标注「代码」）。

    Args:
        chunks: 检索引擎返回的 hit 列表（含 entity 或扁平字段）。

    Returns:
        编号的片段文本（章节 + 小节 + 正文，代码块已拼回所属小节）。
    """
    parts: list[str] = []
    section_to_part: dict[tuple[str, str], int] = {}
    code_entities: list[dict] = []

    for hit in chunks:
        entity = hit.get("entity") or hit
        # 代码块独立成段：先收集，待正文渲染完成后按其小节位置拼回
        if entity.get("block_type") == "code":
            code_entities.append(entity)
            continue
        chapter = entity.get("chapter") or ""
        section = entity.get("section") or ""
        location = " > ".join(x for x in (chapter, section) if x)
        text = _render_text_entity(entity, _collect_hit_images(hit))
        parts.append(f"[片段{len(parts) + 1}｜{location or '未标注位置'}] {text}")
        # 记录该小节首个正文片段的索引，供后续代码块拼回定位
        key = (chapter, section)
        if key not in section_to_part:
            section_to_part[key] = len(parts) - 1

    for entity in code_entities:
        chapter = entity.get("chapter") or ""
        section = entity.get("section") or ""
        code_text = (entity.get("text") or "").strip()
        key = (chapter, section)
        if key in section_to_part:
            parts[section_to_part[key]] += "\n\n" + code_text
        else:
            location = " > ".join(x for x in (chapter, section) if x)
            parts.append(f"[代码｜{location or '未标注位置'}] {code_text}")

    return "\n\n".join(parts)


# 召回后二次补拉代码块的上限（控制检索延迟与 prompt 体积）
_MAX_PULL_SECTIONS = 3        # 最多补拉前几个正文命中节
_MAX_CODES_PER_SECTION = 20   # 每节最多补拉的代码块条数
_MAX_CODE_CHARS = 2000        # 单个代码块超长时截断


def _enrich_hits_with_section_codes(textbook_name: str, hits: list[dict]) -> list[dict]:
    """正文命中后，按 (chapter, section) 二次补拉该节代码块，实现粗粒度代码协同召回。

    只补拉正文 hit 所在节、且 primary 召回中未出现的代码块（按 id 去重）；
    任一步失败都降级为仅返回原 hits，不阻断主流程。

    Args:
        textbook_name: 教材名（须已登记）。
        hits: 检索引擎返回的 hit 列表（含 id 与 entity）。

    Returns:
        原 hits + 补拉的代码块 hit（结构一致，entity.block_type="code"）。
    """
    if not hits:
        return hits

    try:
        collection_name = get_collection_by_name(textbook_name)
    except Exception as exc:  # 定位集合失败不影响主召回
        logger.warning(f"二次补拉代码块：定位集合失败，跳过: {exc}")
        return hits
    if not collection_name:
        return hits

    # primary 召回已直接带出的代码块 id（补拉时跳过，避免重复拼接）
    seen_code_ids = {
        str(h.get("id"))
        for h in hits
        if (h.get("entity") or h).get("block_type") == "code"
    }

    # 正文 hit 的 distinct (chapter, section)，保序、去空章
    sections: list[tuple[str, str]] = []
    seen_sections: set[tuple[str, str]] = set()
    for h in hits:
        entity = h.get("entity") or h
        if entity.get("block_type") == "code":
            continue
        chapter = (entity.get("chapter") or "").strip()
        section = (entity.get("section") or "").strip()
        key = (chapter, section)
        if not chapter or key in seen_sections:
            continue
        seen_sections.add(key)
        sections.append(key)
    sections = sections[:_MAX_PULL_SECTIONS]

    extra: list[dict] = []
    for chapter, section in sections:
        try:
            rows = query_section_codes(collection_name, chapter, section, _MAX_CODES_PER_SECTION)
        except Exception as exc:
            logger.warning(f"二次补拉代码块失败({chapter} > {section})，跳过: {exc}")
            continue
        for row in rows:
            row_id = str(row.get("id"))
            if not row_id or row_id in seen_code_ids:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            if len(text) > _MAX_CODE_CHARS:
                text = text[:_MAX_CODE_CHARS] + "\n# …（代码过长，已截断）"
            seen_code_ids.add(row_id)
            extra.append(
                {
                    "id": row_id,
                    "entity": {
                        "text": text,
                        "chapter": chapter,
                        "section": section,
                        "block_type": "code",
                    },
                }
            )

    if extra:
        logger.info(f"二次补拉代码块: {len(sections)} 节 → 补 {len(extra)} 个代码块")
    return hits + extra


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
        编号的 TOP-K 片段（章节 + 小节 + 正文，图片 url 已附加到正文图标记处），无结果或失败时返回友好提示。
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
    # 正文命中后按节补拉代码块（粗粒度协同召回），使自然语言问代码能取回代码原文
    hits = _enrich_hits_with_section_codes(textbook_name, hits)
    # 图片引用已内联到所属片段末尾（配图行），供 agent 决定是否在回答中插入
    return _format_hits(hits)


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
                "text": "指针是C语言的核心概念。【图: 指针示意图】", "chapter": "第3章", "section": "3.1",
                "metadata_json": json.dumps({"images": [{"url": "https://x/y.png", "description": "指针示意图"}]}),
            }},
            {"id": "c3_code", "distance": 0.6, "entity": {
                "text": "```c\nint main() { return 0; }\n```", "chapter": "第3章", "section": "3.1",
                "block_type": "code",
            }},
            {"id": "c2", "distance": 0.7, "entity": {"text": "数组是相同类型元素的集合。", "chapter": "第3章", "section": "3.2"}},
            {"id": "c4_code", "distance": 0.5, "entity": {
                "text": "```python\nprint(1)\n```", "chapter": "第4章", "section": "4.1",
                "block_type": "code",
            }},
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

    def _fake_enrich(textbook_name: str, hits: list[dict]) -> list[dict]:
        return hits  # 冒烟测试不依赖真实 Milvus，二次补拉原样透传

    # 覆盖模块级绑定，只验证工具内部编排
    rewrite = _fake_rewrite
    rewrite_query_search = _fake_hybrid
    hyde_doc_generate = _fake_hyde_generate
    hyde_doc_search = _fake_hyde_search
    rrf_merge = _fake_rrf
    rerank_chunks = _fake_rerank
    _enrich_hits_with_section_codes = _fake_enrich

    async def _run() -> None:
        # 模拟一轮已作答的历史：保证消歧历史被拼进重写输入
        history = [HumanMessage(content="什么是指针?"), AIMessage(content="指针是…")]

        # 快速路径：重写 + 混合召回，返回 TOP-K 片段
        fast = await search_textbook.coroutine(
            query="它有什么用途?", textbook_name="C语言程序设计", messages=history
        )
        assert "[片段1｜第3章 > 3.1]" in fast and "[片段2" in fast, fast
        assert "【图: 指针示意图】(https://x/y.png)" in fast, "url 未回绑到正文图标记"
        assert "int main()" in fast, "代码块未拼回所属小节正文"
        assert "[代码｜第4章 > 4.1]" in fast and "print(1)" in fast, "无对应正文的代码块未独立展示"
        assert "配图:" not in fast and "【图片候选】" not in fast, "不应再有独立图片行"
        print("--- fast ---\n" + fast)

        # 深度路径：触发 HyDE + 融合 + 精排，取精排后片段
        deep = await search_textbook.coroutine(
            query="它有什么用途?", deep=True, textbook_name="C语言程序设计", messages=history
        )
        assert "指针是C语言的核心概念" in deep, deep
        print("--- deep ---\n" + deep)
        print("search_textbook 测试通过")

    asyncio.run(_run())
