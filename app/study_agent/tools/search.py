"""search_textbook 工具：叠加强化检索（重写消歧 + 多路召回 + 精排），供 agent 自主调用。

将 query_functions 的核心能力折叠进单一检索工具，最终回答统一由 agent 模型生成，本工具只返回检索片段：
- 始终先做问题重写（利用对话历史消歧），提升召回针对性；
- ``deep=False`` 快速路径：混合召回（稠密+稀疏）后直接返回 TOP-K，适合反复取素材；
- ``deep=True`` 深度路径：再叠加 HyDE 假设文档召回 → RRF 融合 → 交叉编码精排
  （精排分数与 RRF 分数加权融合，只微调排序而非覆盖，适合严谨作答）。
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.config import rerank_config
from app.core import astage, get_metrics, logger
from app.study_agent.query_functions.embedding_search import rewrite_query_search, search_by_vectors
from app.study_agent.query_functions.hyde_embedding_search import hyde_doc_generate
from app.study_agent.query_functions.merge_recalls import rrf_merge
from app.study_agent.query_functions.rerank import arerank_chunks_weighted
from app.utils import agenerate_embeddings, get_collection_by_name, query_section_codes

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


def _record_retrieved_chunk_ids(chunks: list[dict]) -> None:
    """把本次检索最终返回的 chunk ID 按顺序写入请求级指标。"""
    metrics = get_metrics()
    if metrics is None:
        return
    for hit in chunks:
        entity = hit.get("entity") or hit
        chunk_id = hit.get("id") or entity.get("id")
        if chunk_id is None:
            continue
        chunk_id = str(chunk_id)
        if chunk_id not in metrics.retrieved_chunk_ids:
            metrics.retrieved_chunk_ids.append(chunk_id)


# 召回后二次补拉代码块的上限（控制检索延迟与 prompt 体积）
_MAX_PULL_SECTIONS = 3        # 最多补拉前几个正文命中节
_MAX_CODES_PER_SECTION = 5   # 每节最多补拉的代码块条数
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
    query: str | None = None,
    deep: bool = False,
    chapter: str | None = None,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    rewritten_query: Annotated[str, InjectedState("rewritten_query")] = None,
) -> str:
    """取教材依据：回答基于教材的问题或整理资料前，先按语义检索相关知识点片段（可选深度召回精排）。

    检索问句已由系统前置消歧（rewritten_query，基于对话历史重写）；本工具默认直接用该问句，
    也可用 query 显式覆盖检索词。默认快速混合召回（稠密+稀疏），如需更高质量（如严谨作答、
    快速检索无果时扩大召回）请设 deep=True：多做一次 HyDE 假设文档召回，再经 RRF
    融合与交叉编码精排——结果更准但更慢。

    Args:
        query: 检索查询（可选），显式指定想检索的内容时传；缺省用系统消歧后的问句。
        deep: 是否启用深度召回（HyDE + 精排），默认 False；对答案质量要求高或快检索无果时设为 True。
        chapter: 限定章节名（可选），与 list_chapters 返回的 chapter 完全一致时才传。

    Returns:
        编号的 TOP-K 片段（章节 + 小节 + 正文，图片 url 已附加到正文图标记处），无结果或失败时返回友好提示。
    """
    textbook_name = textbook_name or ""
    metrics = get_metrics()
    try:
        # 检索问句：优先用 LLM 显式传入的 query，缺省用前置 query 理解消歧好的 rewritten_query
        rewritten = query or rewritten_query or ""
        if not rewritten:
            return "（未提供检索问句，请重试）"
        if metrics is not None:
            metrics.rewrite_query = rewritten
            metrics.deep = bool(deep)
        logger.info(f"search_textbook 检索问句: {rewritten!r}")

        if not deep:
            # 快速路径：单路混合召回（稠密 + 稀疏）后截取 TOP-K
            chunks = await rewrite_query_search(textbook_name, rewritten, chapter)
            if metrics is not None:
                metrics.recall_count = len(chunks)
            hits = chunks[:rerank_config.top_k]
        else:
            # 深度路径：两路召回（混合 + HyDE，各扩候选池）→ RRF 融合 → 精排 TOP-K
            async with astage("hyde_ms"):
                hyde_doc = await hyde_doc_generate(rewritten)
            # 一次批量：query 与 hyde_doc 的向量合并计算，省一次模型前向（embedding_ms 只累加一次）
            embeddings = await agenerate_embeddings([rewritten, hyde_doc + rewritten])
            embedding_chunks = await search_by_vectors(
                textbook_name, embeddings["dense"][0], embeddings["sparse"][0], chapter,
                limit=rerank_config.candidate_pool,
            )
            hyde_chunks = await search_by_vectors(
                textbook_name, embeddings["dense"][1], embeddings["sparse"][1],
                limit=rerank_config.candidate_pool,
            )
            merged = await rrf_merge(embedding_chunks, hyde_chunks)
            merged_hits = [entry["hit"] for entry in merged]
            if metrics is not None:
                metrics.recall_count = len(merged_hits)
            try:
                async with astage("rerank_ms"):
                    # 交叉编码打分后与 RRF 分数加权融合（精排微调而非覆盖），只对返回结果截断
                    hits = await arerank_chunks_weighted(
                        rewritten, merged, rerank_config.top_k, rerank_config.fusion_alpha
                    )
            except Exception as exc:  # 精排降级：回退 RRF 融合结果，不阻断
                logger.warning(f"deep 检索精排失败，回退 RRF：{exc}")
                hits = merged_hits[:rerank_config.top_k]
    except Exception as exc:
        # 检索失败降级：只告警并返回友好提示，交由 agent 决定回退 web 或如实说明
        logger.warning(f"search_textbook({query!r}, deep={deep}) 失败: {exc}")
        return f"（检索教材失败：{exc}）"

    if metrics is not None:
        # 精排后保留的片段数（仅深度路径精排；快速路径无精排，记最终返回条数）
        metrics.rerank_count = len(hits) if deep else 0

    if not hits:
        return "（未检索到相关片段，请换个问法或去掉章节限定）"

    logger.info(f"search_textbook({query!r}, deep={deep}) → {len(hits)} 条")
    # 正文命中后按节补拉代码块（粗粒度协同召回），使自然语言问代码能取回代码原文
    hits = _enrich_hits_with_section_codes(textbook_name, hits)
    _record_retrieved_chunk_ids(hits)
    # 图片引用已内联到所属片段末尾（配图行），供 agent 决定是否在回答中插入
    return _format_hits(hits)


# 冒烟测试已迁移至 tests/study_agent/tools/test_search.py
