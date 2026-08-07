from langgraph.types import StreamWriter

from app.core import log_node, logger
from app.query_agent.state import QueryState

# RRF 平滑常数 k（默认 60，越小排名权重越突出）
RRF_K = 60


async def rrf_merge(embedding_chunks: list[dict],hyde_chunks: list[dict],k: int = RRF_K,) -> list[dict]:
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
    return merged


@log_node
async def merge_recalls(state: QueryState, *, writer: StreamWriter) -> dict:
    """用 RRF 融合普通检索与 HyDE 检索两路召回。

    Args:
        state: 含 ``embedding_chunks`` 与 ``hyde_embedding_chunks`` 两路召回。
        writer: 流式 writer，推送融合阶段提示。

    Returns:
        写回 ``merged_chunks``（RRF 融合结果），同时保留 ``distance`` 供后续 rerank 使用。
    """
    writer({"type": "stage", "stage": "merge", "message": "正在融合检索结果…"})
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_chunks = state.get("hyde_embedding_chunks", [])

    merged = await rrf_merge(embedding_chunks, hyde_chunks)
    # 仅把原始 hit 写入 state（RRF 分数不进 graph state，避免污染下游），
    # 但保留 hit 自带的 distance 字段（阶段2 reranker 需要）
    merged_hits = [entry["hit"] for entry in merged]
    logger.info(f"RRF 融合完成：普通 {len(embedding_chunks)} 条 + HyDE {len(hyde_chunks)} 条 → {len(merged)} 条")
    # 只返回本节点写入的字段（风格与并行检索节点统一）
    return {"merged_chunks": merged_hits}


# 单元测试
if __name__ == '__main__':
    import asyncio

    # 构造两路召回的模拟 hit（含 id/distance/entity）
    def _hit(doc_id: str, distance: float, text: str) -> dict:
        return {"id": doc_id, "distance": distance, "entity": {"text": text}}

    emb = [
        _hit("doc_a", 0.8, "普通召回A"),
        _hit("doc_b", 0.7, "普通召回B"),
        _hit("doc_c", 0.6, "普通召回C"),
    ]
    hyd = [
        _hit("doc_b", 0.75, "HyDE召回B"),  # 与普通召回重复 → 分数叠加
        _hit("doc_d", 0.5, "HyDE召回D"),
    ]

    test_state: QueryState = {
        "session_id": "test",
        "textbook_name": "test",
        "original_query": "test",
        "rewritten_query": "test",
        "embedding_chunks": emb,
        "hyde_embedding_chunks": hyd,
    }
    result = asyncio.run(merge_recalls(test_state))
    merged = result["merged_chunks"]
    assert len(merged) == 4, f"两路共 5 条召回、doc_b 重复，去重应为 4 条，实际 {len(merged)}"
    ids = [h["id"] for h in merged]
    assert ids[0] == "doc_b", "doc_b 被两路召回，RRF 分数叠加应排第一"
    assert set(ids) == {"doc_a", "doc_b", "doc_c", "doc_d"}, f"id 集不正确: {ids}"
    print(f"融合顺序: {ids}")
    print("merge_recalls 测试通过")
