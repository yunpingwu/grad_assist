"""search_textbook 工具：快速路径做混合召回，深度路径叠加 HyDE → RRF → 精排。"""

from __future__ import annotations

import json
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.config import rerank_config
from app.core import logger
from app.ml import agenerate_embeddings
from app.study_agent.query_functions.embedding_search import search_by_query, search_by_vectors
from app.study_agent.query_functions.hyde_embedding_search import hyde_doc_generate
from app.study_agent.query_functions.merge_recalls import rrf_merge
from app.study_agent.query_functions.rerank import arerank_chunks_weighted
from app.study_agent.safety import wrap_untrusted
from app.trace import astage, mark_tool
from app.utils import (
    get_collection_by_name,
    query_section_codes,
)

# 正文内嵌的图片标记：切块时图片行被替换为「【图: 简介】」，此处按简介回绑 url
_FIGURE_MARK_PATTERN = re.compile(r"【图: (.*?)】")

def _collect_hit_images(hit: dict) -> dict[str, str]:
    """提取片段附带的图片映射：正文图标记的简介 → MinIO url。"""
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
    """把图片 url 回绑到正文的「【图: 简介】」标记处。"""
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
    """把 hit 列表格式化为编号片段；代码块拼回同一章节的正文片段末尾。"""
    parts: list[str] = []
    section_to_part: dict[tuple[str, str], int] = {}
    code_entities: list[tuple[str, dict]] = []

    for hit in chunks:
        entity = hit.get("entity") or hit
        chunk_id = str(hit.get("id") or entity.get("id") or "")
        # 代码块独立成段：先收集，待正文渲染完成后按其小节位置拼回
        if entity.get("block_type") == "code":
            code_entities.append((chunk_id, entity))
            continue
        chapter = entity.get("chapter") or ""
        section = entity.get("section") or ""
        location = " > ".join(x for x in (chapter, section) if x)
        head = f"片段{len(parts) + 1}｜{chunk_id}｜" if chunk_id else f"片段{len(parts) + 1}｜"
        text = _render_text_entity(entity, _collect_hit_images(hit))
        parts.append(f"[{head}{location or '未标注位置'}] {text}")
        # 记录该小节首个正文片段的索引，供后续代码块拼回定位
        key = (chapter, section)
        if key not in section_to_part:
            section_to_part[key] = len(parts) - 1

    for cid, entity in code_entities:
        chapter = entity.get("chapter") or ""
        section = entity.get("section") or ""
        code_text = (entity.get("text") or "").strip()
        key = (chapter, section)
        if key in section_to_part:
            parts[section_to_part[key]] += "\n\n" + code_text
        else:
            location = " > ".join(x for x in (chapter, section) if x)
            head = f"代码｜{cid}｜" if cid else "代码｜"
            parts.append(f"[{head}{location or '未标注位置'}] {code_text}")

    return "\n\n".join(parts)


def _build_artifact(
    textbook_name: str, query: str, deep: bool, chunks: list[dict], content: str
) -> dict:
    """组装 ToolMessage.artifact：原文与片段定位随消息落库，但不发给模型。

    发给模型的只有 ``content``（实测 ChatOpenAI 序列化请求时不带 artifact），历史降级把
    content 换成一行记录后，回显与审计仍能从这里的 ``content`` 取回原文。
    """
    records = []
    for hit in chunks:
        entity = hit.get("entity") or hit
        location = " > ".join(x for x in (entity.get("chapter") or "", entity.get("section") or "") if x)
        records.append(
            {
                "id": str(hit.get("id") or entity.get("id") or ""),
                "location": location,
                "block_type": entity.get("block_type") or "text",
            }
        )
    return {
        "textbook_name": textbook_name,
        "query": query,
        "deep": bool(deep),
        "chunks": records,
        "content": content,
    }


def _hit_chunk_ids(chunks: list[dict]) -> list[str]:
    """取本次检索最终返回的 chunk ID（保序、去重、跳过缺 ID 的条目）。

    写进本次工具步的 meta 后，请求级的 ``retrieved_chunk_ids`` 由它派生（去重保序），
    证据只落一处，不会出现「steps 里记了、请求级忘了记」的分家。
    """
    ids: dict[str, None] = {}
    for hit in chunks:
        entity = hit.get("entity") or hit
        chunk_id = hit.get("id") or entity.get("id")
        if chunk_id is not None:
            ids.setdefault(str(chunk_id), None)
    return list(ids)


# 召回后二次补拉代码块的上限（控制检索延迟与 prompt 体积）
_MAX_PULL_SECTIONS = 3        # 最多补拉前几个正文命中节
_MAX_CODES_PER_SECTION = 5   # 每节最多补拉的代码块条数
_MAX_CODE_CHARS = 2000        # 单个代码块超长时截断


def _enrich_hits_with_section_codes(textbook_name: str, hits: list[dict]) -> list[dict]:
    """按命中片段所在节补拉未出现的代码块；失败时原样返回 hits。"""
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


@tool(response_format="content_and_artifact")
async def search_textbook(
    query: str | None = None,
    deep: Annotated[bool, InjectedState("deep")] = False,
    chapter: str | None = None,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    requirement: Annotated[str, InjectedState("requirement")] = None,
) -> tuple[str, dict]:
    """取教材依据：回答教材问题或整理资料前，先按语义检索相关片段。

    query 必须自包含（把「它」「这一章」等指代结合上文补全），缺省回退本轮原问题；
    快检索还是深度召回由请求决定，不由你选择。召回为空或不够时，换问法或如实说明没查到。
    """
    textbook_name = textbook_name or ""
    try:
        # 检索问句：优先用 LLM 显式传入的自包含 query，缺省回退本轮用户原问题
        search_query = (query or requirement or "").strip()
        if not search_query:
            return "（未提供检索问句，请重试）", {}
        # 观测挂到本次工具步的 meta 上（并发或连续多次检索各写各的步，互不覆盖）；
        # 失败不再自己写 error：on_tool_error 会把 status/error 落在同一条工具步上
        mark_tool(query=search_query, deep=bool(deep))
        logger.info(f"search_textbook 检索问句: {search_query!r}")

        if not deep:
            # 快速路径：单路混合召回（稠密 + 稀疏）后截取 TOP-K
            chunks = await search_by_query(textbook_name, search_query, chapter)
            mark_tool(recall=len(chunks))
            hits = chunks[:rerank_config.top_k]
        else:
            # 深度路径：两路召回（混合 + HyDE，各扩候选池）→ RRF 融合 → 精排 TOP-K
            async with astage("hyde_ms"):
                hyde_doc = await hyde_doc_generate(search_query)
            # 一次批量：query 与 hyde_doc 的向量合并计算，省一次模型前向（embedding_ms 只累加一次）
            embeddings = await agenerate_embeddings([search_query, hyde_doc + search_query])
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
            mark_tool(recall=len(merged_hits))
            try:
                async with astage("rerank_ms"):
                    # 交叉编码打分后与 RRF 分数加权融合（精排微调而非覆盖），只对返回结果截断
                    hits = await arerank_chunks_weighted(
                        search_query, merged, rerank_config.top_k, rerank_config.fusion_alpha
                    )
            except Exception as exc:  # 精排降级：回退 RRF 融合结果，不阻断
                logger.warning(f"deep 检索精排失败，回退 RRF：{exc}")
                hits = merged_hits[:rerank_config.top_k]
    except Exception as exc:
        logger.warning(f"search_textbook({query!r}, deep={deep}) 失败: {exc}")
        # 交给图上的官方 ToolRetryMiddleware：先原地重试一次，耗尽后由它把说明文本回给模型
        raise RuntimeError(f"教材检索失败: {type(exc).__name__}: {exc}") from exc

    # 送入模型前截断后的条数（补拉的代码块不计入，另由 meta.chunk_ids 体现）
    mark_tool(kept=len(hits))

    if not hits:
        return "（未检索到相关片段，请把 query 补全为自包含问句或去掉章节限定）", {}

    logger.info(f"search_textbook({query!r}, deep={deep}) → {len(hits)} 条")
    # 正文命中后按节补拉代码块（粗粒度协同召回），使自然语言问代码能取回代码原文
    hits = _enrich_hits_with_section_codes(textbook_name, hits)
    # 证据随本次工具步落盘，请求级 retrieved_chunk_ids 由它派生（见 RequestMetrics 属性）
    mark_tool(chunk_ids=_hit_chunk_ids(hits))
    # 图片引用已内联到所属片段末尾（配图行），供 agent 决定是否在回答中插入
    content = _format_hits(hits)
    # artifact 存干净原文（/history 回显与往轮降级记录都读它）；只有给模型的 content 包边界，
    # 边界与随机串的设计理由见 app/study_agent/safety.py
    return wrap_untrusted(content, "教材"), _build_artifact(textbook_name, search_query, deep, hits, content)


