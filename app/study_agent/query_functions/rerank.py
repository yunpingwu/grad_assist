"""Rerank 精排工具函数：对 RRF 融合后的候选片段做交叉编码重排序，截取 TOP-K。

从 query_functions 收编而来：仅保留纯函数 rerank_chunks（原节点封装已移除）。

- 打分由 ``app.utils.reranker_util.compute_rerank_scores`` 提供（BGE-Reranker），
  模型单例在 util 模块内部维护，本函数只拿分数、不接触模型实例；
- 候选仅 ≤10 条，本地 CPU 精排约 0.5~2s，CUDA 更快；任何失败由调用方降级回退。
"""

from __future__ import annotations

from app.core import logger
from app.utils.batch_manager.reranker import arerank_scores
from app.utils.reranker_util import compute_rerank_scores


def _extract_texts(chunks: list[dict]) -> list[str]:
    """提取片段文本（Milvus hit 的 entity.text）。"""
    texts = []
    for hit in chunks:
        entity = hit.get("entity") or hit
        texts.append((entity.get("text") or "").strip())
    return texts


def _rank_and_truncate(chunks: list[dict], scores: list[float], top_k: int) -> list[dict]:
    """把分数挂到对应 hit → 降序排序 → 截取 TOP-K。"""
    ranked = []
    for hit, score in zip(chunks, scores, strict=True):
        item = dict(hit)  # 浅拷贝，避免污染调用方的原始 hit
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]


def _minmax(rrf_scores: list[float]) -> tuple[float, float]:
    """求 RRF 分数最小/最大值（避免除零：全等时返回 0/1 占位）。"""
    lo, hi = min(rrf_scores), max(rrf_scores)
    return (0.0, 1.0) if hi == lo else (lo, hi)


def _fuse_and_truncate(
    chunks: list[dict],
    rerank_scores: list[float],
    rrf_scores: list[float],
    top_k: int,
    alpha: float,
) -> list[dict]:
    """精排分数与 RRF 分数加权融合 → 降序 → 截取 TOP-K。

    final = alpha * rerank(已 sigmoid 归一化) + (1-alpha) * minmax(rrf)
    alpha 越大越信任精排，越小越信任 RRF 融合序。
    """
    lo, hi = _minmax(rrf_scores)
    fused = []
    for hit, rs, rr in zip(chunks, rerank_scores, rrf_scores, strict=True):
        item = dict(hit)  # 浅拷贝
        item["rerank_score"] = float(rs)
        item["rrf_score"] = float(rr)
        norm_rrf = (rr - lo) / (hi - lo) if hi > lo else 0.0
        item["fusion_score"] = alpha * float(rs) + (1.0 - alpha) * norm_rrf
        fused.append(item)
    fused.sort(key=lambda h: h["fusion_score"], reverse=True)
    return fused[:top_k]


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """用交叉编码模型对候选片段精排。

    Args:
        query: 检索问句（与召回的语义对齐）。
        chunks: RRF 融合后的候选 hit 列表。
        top_k: 精排后保留的 TOP-K 片段数。

    Returns:
        按精排分数降序的 TOP-K hit 列表（附 rerank_score）。

    Raises:
        ValueError: 无候选或模型不可用。
    """
    if not chunks:
        raise ValueError("无候选片段可精排")
    if not query:
        raise ValueError("精排查询为空")

    texts = _extract_texts(chunks)
    scores = compute_rerank_scores(query, texts)
    logger.info(f"Rerank 完成：{len(chunks)} 条候选 → TOP{top_k}，分数范围 {min(scores):.3f} ~ {max(scores):.3f}")
    return _rank_and_truncate(chunks, scores, top_k)


async def arerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """异步精排：打分走动态攒批器（``arerank_scores``），避免阻塞事件循环。

    多个并发 deep 检索的交叉编码请求会被攒成一批 (query, text) 对，一次前向，
    再由 drainer 按各请求候选数拆回分数。耗时埋点由调用方在协程层用 ``astage`` 记录。

    Args:
        query: 检索问句。
        chunks: RRF 融合后的候选 hit 列表。
        top_k: 精排后保留的 TOP-K 片段数。

    Returns:
        按精排分数降序的 TOP-K hit 列表。
    """
    if not chunks:
        raise ValueError("无候选片段可精排")
    if not query:
        raise ValueError("精排查询为空")

    texts = _extract_texts(chunks)
    scores = await arerank_scores(query, texts)
    logger.info(f"Rerank 完成：{len(chunks)} 条候选 → TOP{top_k}，分数范围 {min(scores):.3f} ~ {max(scores):.3f}")
    return _rank_and_truncate(chunks, scores, top_k)


def rerank_chunks_weighted(
    query: str,
    merged: list[dict],
    top_k: int,
    alpha: float,
) -> list[dict]:
    """用交叉编码模型打分后与 RRF 分数加权融合（同步版）。

    merged 必须是 ``rrf_merge`` 的产物（每项含 ``rrf_score`` 与 ``hit``），
    解决"精排把多路共识的正确 chunk 挤掉"的问题：只做排序微调而非覆盖。

    Args:
        query: 检索问句。
        merged: RRF 融合结果列表（项含 rrf_score / hit）。
        top_k: 融合后保留的 TOP-K 片段数。
        alpha: 精排分权重（0~1），越大越信精排，越小越信 RRF。

    Returns:
        按融合分降序的 TOP-K hit 列表（附 rerank_score / rrf_score / fusion_score）。
    """
    if not merged:
        raise ValueError("无候选片段可精排")
    if not query:
        raise ValueError("精排查询为空")

    hits = [entry["hit"] for entry in merged]
    rrf_scores = [float(entry["rrf_score"]) for entry in merged]
    texts = _extract_texts(hits)
    rerank_scores = compute_rerank_scores(query, texts)
    logger.info(
        f"Rerank+RRF 融合完成：{len(merged)} 条候选 → TOP{top_k}，"
        f"rerank {min(rerank_scores):.3f}~{max(rerank_scores):.3f}"
    )
    return _fuse_and_truncate(hits, rerank_scores, rrf_scores, top_k, alpha)


async def arerank_chunks_weighted(
    query: str,
    merged: list[dict],
    top_k: int,
    alpha: float,
) -> list[dict]:
    """异步版加权融合精排：打分走动态攒批器，再与 RRF 分数融合（推荐供生产/评测使用）。"""
    if not merged:
        raise ValueError("无候选片段可精排")
    if not query:
        raise ValueError("精排查询为空")

    hits = [entry["hit"] for entry in merged]
    rrf_scores = [float(entry["rrf_score"]) for entry in merged]
    texts = _extract_texts(hits)
    rerank_scores = await arerank_scores(query, texts)
    logger.info(
        f"Rerank+RRF 融合完成：{len(merged)} 条候选 → TOP{top_k}，"
        f"rerank {min(rerank_scores):.3f}~{max(rerank_scores):.3f}"
    )
    return _fuse_and_truncate(hits, rerank_scores, rrf_scores, top_k, alpha)
