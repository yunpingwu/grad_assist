"""检索打点：复用检索函数，逐级采集各配置的排序结果与耗时。

评测直接以自包含问句作为检索输入（跳过 search_textbook 里的问题重写步骤），
以隔离“查询重写”这一变量，聚焦检索组件本身的消融：

- dense          ：单路稠密检索（基线，验证 dense 单独排序质量）
- sparse         ：单路稀疏/词面检索（基线，验证 sparse 单独排序质量）
- hybrid(w)      ：dense + sparse，WeightedRanker(w, 1-w) 后截取 TOP-K（权重扫描）
- run_fast       ：hybrid 默认权重 (0.8, 0.2) —— 快速路径
- run_hyde_rrf   ：HyDE + RRF（混合召回 + HyDE 第二路 → RRF 融合后截断，不做精排）
- run_deep       ：HyDE + RRF → 交叉编码精排（精排变体，仅返回 rerank 排序）
"""

from __future__ import annotations

import time

from app.config import rerank_config
from app.core import logger
from app.study_agent.query_functions.embedding_search import rewrite_query_search
from app.study_agent.query_functions.hyde_embedding_search import hyde_doc_generate, hyde_doc_search
from app.study_agent.query_functions.merge_recalls import rrf_merge
from app.study_agent.query_functions.rerank import arerank_chunks_weighted
from app.utils.batch_manager.embedder import agenerate_embeddings
from app.utils.milvus_util import (
    dense_search,
    escape_expr_value,
    get_collection_by_name,
    hybrid_search,
    sparse_search,
)


async def run_dense(query: str, textbook: str, chapter: str | None = None) -> dict:
    """dense-only 单路稠密检索，截取 TOP-K。"""
    start = time.perf_counter()
    embeddings = await agenerate_embeddings([query])
    collection = get_collection_by_name(textbook)
    if not collection:
        raise ValueError(f"教材未登记: {textbook}")
    expr = None
    if chapter:
        safe = escape_expr_value(chapter)
        expr = f'chapter == "{safe}"'
    hits = dense_search(
        embeddings["dense"][0], collection, expr=expr, limit=rerank_config.top_k
    )
    latency = time.perf_counter() - start
    return {"retrieved_ids": [str(hit.get("id")) for hit in hits], "latency_s": latency}


async def run_sparse(query: str, textbook: str, chapter: str | None = None) -> dict:
    """sparse-only 单路词面检索，截取 TOP-K。"""
    start = time.perf_counter()
    embeddings = await agenerate_embeddings([query])
    collection = get_collection_by_name(textbook)
    if not collection:
        raise ValueError(f"教材未登记: {textbook}")
    expr = None
    if chapter:
        safe = escape_expr_value(chapter)
        expr = f'chapter == "{safe}"'
    hits = sparse_search(
        embeddings["sparse"][0], collection, expr=expr, limit=rerank_config.top_k
    )
    latency = time.perf_counter() - start
    return {"retrieved_ids": [str(hit.get("id")) for hit in hits], "latency_s": latency}


async def run_hybrid(
    query: str,
    textbook: str,
    weights: tuple[float, float] = (0.8, 0.2),
    chapter: str | None = None,
) -> dict:
    """加权混合检索（dense + sparse，WeightedRanker(weights)），截取 TOP-K。"""
    start = time.perf_counter()
    embeddings = await agenerate_embeddings([query])
    collection = get_collection_by_name(textbook)
    if not collection:
        raise ValueError(f"教材未登记: {textbook}")
    expr = None
    if chapter:
        safe = escape_expr_value(chapter)
        expr = f'chapter == "{safe}"'
    hits = hybrid_search(
        embeddings["dense"][0],
        embeddings["sparse"][0],
        collection,
        expr=expr,
        limit=rerank_config.top_k,
        weights=weights,
    )
    latency = time.perf_counter() - start
    return {"retrieved_ids": [str(hit.get("id")) for hit in hits], "latency_s": latency}


async def run_fast(query: str, textbook: str, chapter: str | None = None) -> dict:
    """快速路径：混合召回（dense + sparse，默认权重 0.8/0.2）后截取 TOP-K。"""
    logger.info(f"混合召回(0.8/0.2): {query[:32]}...")
    return await run_hybrid(query, textbook, chapter=chapter)


async def run_hyde_rrf(
    query: str, textbook: str, top_k: int | None = None, candidate_pool: int | None = None
) -> dict:
    """HyDE + RRF：混合召回 + HyDE 第二路 → RRF 融合后截断（不做精排）。

    深度路径的独立阶段：产出 RRF 融合排序（含全候选与分阶段耗时），
    供 ``run_deep`` 复用同一批融合候选做交叉编码精排。

    Args:
        query: 检索问句（已自包含）。
        textbook: 教材名（须已登记）。
        top_k: 最终截断条数，缺省用 rerank_config.top_k。
        candidate_pool: 每路召回的候选条数（RRF 融合前）。缺省用 rerank_config.candidate_pool，
            给后续交叉编码精排更大候选空间。

    Returns:
        {
            "rrf_hyde": {"retrieved_ids": [...], "n_candidates": int},  # RRF 融合序 TOP-K
            "merged": [...],         # RRF 融合全候选（含 hit / 分数），供精排复用
            "latency": {...},        # embed / hyde / rrf / total 各阶段耗时（秒）
        }
    """
    top_k = top_k or rerank_config.top_k
    candidate_pool = candidate_pool or rerank_config.candidate_pool
    timing: dict[str, float] = {}
    total_start = time.perf_counter()

    # 第一路：混合召回（候选池扩容，供融合/精排）
    start = time.perf_counter()
    embedding_chunks = await rewrite_query_search(textbook, query, limit=candidate_pool)
    timing["embed"] = time.perf_counter() - start

    # 第二路：HyDE 假设文档召回
    start = time.perf_counter()
    hyde_doc = await hyde_doc_generate(query)
    hyde_chunks = await hyde_doc_search(hyde_doc, query, textbook, limit=candidate_pool)
    timing["hyde"] = time.perf_counter() - start

    # RRF 融合两路
    start = time.perf_counter()
    merged = await rrf_merge(embedding_chunks, hyde_chunks)
    merged_hits = [entry["hit"] for entry in merged]
    timing["rrf"] = time.perf_counter() - start

    timing["total"] = time.perf_counter() - total_start
    logger.info(
        f"HyDE+RRF: 候选 {len(merged_hits)} 条 → TOP{top_k}, 耗时 {timing['total']:.2f}s"
    )
    return {
        "rrf_hyde": {"retrieved_ids": [str(hit.get("id")) for hit in merged_hits[:top_k]], "n_candidates": len(merged_hits)},
        "merged": merged,
        "latency": timing,
    }


async def run_deep(
    query: str,
    textbook: str,
    top_k: int | None = None,
    candidate_pool: int | None = None,
    base: dict | None = None,
) -> dict:
    """深度路径（精排变体）：HyDE+RRF 融合（可复用外部 ``run_hyde_rrf`` 结果）→ 交叉编码精排。

    仅返回精排后的 ``rerank`` 排序。需与融合序对照（如消融评测）时，先调用
    ``run_hyde_rrf`` 拿到 ``base`` 再传入本函数，保证 rerank 与 rrf_hyde 基于
    同一批融合候选；不传 ``base`` 则内部自行跑一遍 HyDE+RRF。

    Args:
        query: 检索问句（已自包含）。
        textbook: 教材名（须已登记）。
        top_k: 最终截断条数，缺省用 rerank_config.top_k。
        candidate_pool: 每路召回的候选条数（RRF 融合前）。缺省用 rerank_config.candidate_pool，
            给交叉编码精排更大候选空间（精排输入为全候选池，仅在返回时截断至 TOP-K）。
        base: ``run_hyde_rrf`` 的返回结果（含 ``merged`` 融合候选与分阶段耗时）；
            传入后不再重复执行 HyDE+RRF。

    Returns:
        含 ``rerank`` / ``latency`` 两段的字典：
        - rerank: {"retrieved_ids": [...], "n_candidates": int}（精排序）
        - latency: embed/hyde/rrf/rerank/total 各阶段耗时（秒）
    """
    top_k = top_k or rerank_config.top_k
    if base is None:
        base = await run_hyde_rrf(query, textbook, top_k=top_k, candidate_pool=candidate_pool)

    # rerank：全候选池交叉编码打分 → 与 RRF 分数加权融合（精排微调而非覆盖），截取 TOP-K
    start = time.perf_counter()
    rerank_hits = await arerank_chunks_weighted(
        query, base["merged"], top_k, rerank_config.fusion_alpha
    )
    timing = {**base["latency"], "rerank": time.perf_counter() - start}
    timing["total"] = base["latency"]["total"] + timing["rerank"]

    logger.info(
        f"深度检索: 候选 {base['rrf_hyde']['n_candidates']} 条 → 精排 TOP{top_k}, "
        f"总耗时 {timing['total']:.2f}s"
    )
    return {
        "rerank": {"retrieved_ids": [str(hit.get("id")) for hit in rerank_hits], "n_candidates": base["rrf_hyde"]["n_candidates"]},
        "latency": timing,
    }