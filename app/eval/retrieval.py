"""检索打点：复用检索函数，逐级采集各配置的排序结果与耗时。

评测直接以自包含问句作为检索输入（跳过 search_textbook 里的问题重写步骤），
以隔离“查询重写”这一变量，聚焦检索组件本身的消融：

- C3（快速混合召回）：dense + sparse，WeightedRanker(0.8, 0.2) 后截取 TOP-K
- C4（RRF 融合）    ：混合召回 + HyDE 第二路 → RRF(k=60) 融合后截取 TOP-K
- C5（精排）        ：对 C4 的融合候选做交叉编码精排，截取 TOP-K
"""

from __future__ import annotations

import time

from app.config import rerank_config
from app.core import logger
from app.study_agent.query_functions.embedding_search import rewrite_query_search
from app.study_agent.query_functions.hyde_embedding_search import hyde_doc_generate, hyde_doc_search
from app.study_agent.query_functions.merge_recalls import rrf_merge
from app.study_agent.query_functions.rerank import rerank_chunks


def _ids(hits: list[dict]) -> list[str]:
    """从 hit 列表提取主键 id 的有序列表（保持检索返回顺序）。"""
    return [str(hit.get("id")) for hit in hits]


async def run_fast(query: str, textbook: str, chapter: str | None = None) -> dict:
    """C3 快速路径：混合召回（dense + sparse）后截取 TOP-K。

    Args:
        query: 检索问句（已自包含，无需重写）。
        textbook: 教材名（须已登记）。
        chapter: 可选章节过滤。

    Returns:
        ``{"retrieved_ids": [...], "latency_s": ...}``。
    """
    start = time.perf_counter()
    chunks = await rewrite_query_search(textbook, query, chapter)
    latency = time.perf_counter() - start
    hit_ids = _ids(chunks)[: rerank_config.top_k]
    logger.info(f"C3 混合召回: {len(chunks)} 条候选, TOP{rerank_config.top_k}, 耗时 {latency:.2f}s")
    return {"retrieved_ids": hit_ids, "latency_s": latency}


async def run_deep(query: str, textbook: str, top_k: int | None = None) -> dict:
    """C4 + C5 深度路径：HyDE 第二路召回 → RRF 融合 →（可选）交叉编码精排。

    一次计算同时产出 C4（融合后截断）与 C5（精排后）两套排序，避免重复跑模型。

    Args:
        query: 检索问句（已自包含）。
        textbook: 教材名（须已登记）。
        top_k: 最终截断条数，缺省用 rerank_config.top_k。

    Returns:
        含 ``c4`` / ``c5`` / ``latency`` 三段的字典：
        - c4: ``{"retrieved_ids": [...], "n_candidates": int}``（RRF 融合序）
        - c5: ``{"retrieved_ids": [...], "n_candidates": int}``（精排序）
        - latency: embed/hyde/rrf/rerank/total 各阶段耗时（秒）
    """
    top_k = top_k or rerank_config.top_k
    timing: dict[str, float] = {}
    total_start = time.perf_counter()

    # 第一路：混合召回
    start = time.perf_counter()
    embedding_chunks = await rewrite_query_search(textbook, query)
    timing["embed"] = time.perf_counter() - start

    # 第二路：HyDE 假设文档召回
    start = time.perf_counter()
    hyde_doc = await hyde_doc_generate(query)
    hyde_chunks = await hyde_doc_search(hyde_doc, query, textbook)
    timing["hyde"] = time.perf_counter() - start

    # RRF 融合两路
    start = time.perf_counter()
    merged = await rrf_merge(embedding_chunks, hyde_chunks)
    merged_hits = [entry["hit"] for entry in merged]
    timing["rrf"] = time.perf_counter() - start

    # C4：融合后直接截断
    c4_hits = merged_hits[:top_k]

    # C5：交叉编码精排
    start = time.perf_counter()
    c5_hits = rerank_chunks(query, merged_hits, min(top_k, len(merged_hits)))
    timing["rerank"] = time.perf_counter() - start

    timing["total"] = time.perf_counter() - total_start
    logger.info(
        f"C4/C5 深度检索: 候选 {len(merged_hits)} 条 → 融合/精排 TOP{top_k}, "
        f"总耗时 {timing['total']:.2f}s"
    )
    return {
        "c4": {"retrieved_ids": _ids(c4_hits), "n_candidates": len(merged_hits)},
        "c5": {"retrieved_ids": _ids(c5_hits), "n_candidates": len(merged_hits)},
        "latency": timing,
    }
