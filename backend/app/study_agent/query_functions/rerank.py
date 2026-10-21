"""Rerank 精排：对 RRF 融合后的候选做交叉编码重排序并截取 TOP-K。"""

from __future__ import annotations

from app.core import logger
from app.ml.batch_manager.reranker import arerank_scores
from app.ml.reranker_util import compute_rerank_scores


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
    lo, hi = min(rrf_scores), max(rrf_scores)
    if hi == lo:  # 全等时用 0/1 占位，避免除零
        lo, hi = 0.0, 1.0
    fused = []
    for hit, rs, rr in zip(chunks, rerank_scores, rrf_scores, strict=True):
        item = dict(hit)  # 浅拷贝
        item["rerank_score"] = float(rs)
        item["rrf_score"] = float(rr)
        norm_rrf = (rr - lo) / (hi - lo)
        item["fusion_score"] = alpha * float(rs) + (1.0 - alpha) * norm_rrf
        fused.append(item)
    fused.sort(key=lambda h: h["fusion_score"], reverse=True)
    return fused[:top_k]


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """交叉编码精排，返回按分数降序的 TOP-K（附 rerank_score）。"""
    if not chunks:
        raise ValueError("无候选片段可精排")
    if not query:
        raise ValueError("精排查询为空")

    texts = _extract_texts(chunks)
    scores = compute_rerank_scores(query, texts)
    logger.info(f"Rerank 完成：{len(chunks)} 条候选 → TOP{top_k}，分数范围 {min(scores):.3f} ~ {max(scores):.3f}")
    return _rank_and_truncate(chunks, scores, top_k)


async def arerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """异步精排：打分走动态攒批器，避免阻塞事件循环。"""
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
    """精排分数与 RRF 分数加权融合后截取 TOP-K；merged 须为 ``rrf_merge`` 的产物。"""
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
