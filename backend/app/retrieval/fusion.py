"""RRF 融合工具函数：供 search_textbook 深度路径融合两路召回。"""

from __future__ import annotations

from app.core import logger

# RRF 平滑常数 k
RRF_K = 10


async def rrf_merge(
    embedding_chunks: list[dict],
    hyde_chunks: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """按 Reciprocal Rank Fusion 融合两路召回，按 doc_id 去重后降序排序。"""
    rrf_scores: dict[str, dict] = {}
    for hits in (embedding_chunks, hyde_chunks):
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit.get("id"))
            entry = rrf_scores.setdefault(doc_id, {"rrf_score": 0.0, "hit": hit})
            entry["rrf_score"] += 1.0 / (k + rank)

    # 降序排序：rrf_score 高的在前（同时被两路召回的片段分数叠加，排最前）
    merged = sorted(
        (entry for entry in rrf_scores.values()),
        key=lambda e: e["rrf_score"],
        reverse=True,
    )
    logger.info(f"RRF 融合完成：普通 {len(embedding_chunks)} 条 + HyDE {len(hyde_chunks)} 条 → {len(merged)} 条")
    return merged
