"""RRF 融合工具函数：供 search_textbook 深度路径融合两路召回。

从 query_functions 收编而来：仅保留纯函数 rrf_merge（原节点封装已移除）。
"""

from __future__ import annotations

from app.core import logger

# RRF 平滑常数 k
RRF_K = 10


async def rrf_merge(
    embedding_chunks: list[dict],
    hyde_chunks: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """按 Reciprocal Rank Fusion 融合两路召回，按 doc_id 去重并降序排序。

    Args:
        embedding_chunks: 普通向量检索结果（hit 列表，含 id/entity/distance）。
        hyde_chunks: HyDE 检索结果（hit 列表，结构同上）。
        k: RRF 平滑常数，默认 60。

    Returns:
        融合后的 hit 列表（含 RRF 分数），按融合分从高到低。
    """
    rrf_scores: dict[str, dict] = {}

    def accumulate(hits: list[dict]) -> None:
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit.get("id"))
            entry = rrf_scores.setdefault(doc_id, {"rrf_score": 0.0, "hit": hit})
            entry["rrf_score"] += 1.0 / (k + rank)

    accumulate(embedding_chunks)
    accumulate(hyde_chunks)

    # 降序排序：rrf_score 高的在前（同时被两路召回的片段分数叠加，排最前）
    merged = sorted(
        (entry for entry in rrf_scores.values()),
        key=lambda e: e["rrf_score"],
        reverse=True,
    )
    logger.info(f"RRF 融合完成：普通 {len(embedding_chunks)} 条 + HyDE {len(hyde_chunks)} 条 → {len(merged)} 条")
    return merged
