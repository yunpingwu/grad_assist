"""rrf_merge 纯函数测试：两路召回去重、分数叠加与排序。"""

import asyncio

from app.study_agent.query_functions.merge_recalls import rrf_merge


def _hit(doc_id: str, distance: float, text: str) -> dict:
    return {"id": doc_id, "distance": distance, "entity": {"text": text}}


def test_rrf_merge_dedup_orders_by_fused_score() -> None:
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
