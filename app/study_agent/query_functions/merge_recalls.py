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


# 冒烟测试：rrf_merge 为纯函数，构造两路召回即可确定性验证
if __name__ == "__main__":
    import asyncio

    def _hit(doc_id: str, distance: float, text: str) -> dict:
        return {"id": doc_id, "distance": distance, "entity": {"text": text}}

    embedding = [
        _hit("doc_a", 0.8, "普通召回A"),
        _hit("doc_b", 0.7, "普通召回B"),
        _hit("doc_c", 0.6, "普通召回C"),
    ]
    hyde = [
        _hit("doc_b", 0.75, "HyDE召回B"),  # 与普通召回重复 → 分数叠加
        _hit("doc_d", 0.5, "HyDE召回D"),
    ]

    merged = asyncio.run(rrf_merge(embedding, hyde))
    assert len(merged) == 4, f"两路共 5 条召回、doc_b 重复，去重应为 4 条，实际 {len(merged)}"
    ids = [e["hit"]["id"] for e in merged]
    assert ids[0] == "doc_b", "doc_b 被两路召回，RRF 分数叠加应排第一"
    assert set(ids) == {"doc_a", "doc_b", "doc_c", "doc_d"}, f"id 集不正确: {ids}"
    assert merged[0]["hit"]["distance"] == 0.7, "应保留原 hit 的 distance（供精排使用）"
    print(f"融合顺序: {ids}")
    print("rrf_merge 测试通过")
