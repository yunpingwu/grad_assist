"""rerank 精排测试：桩掉打分 API，验证排序、截断与分数挂载。"""

import pytest

from app.study_agent.query_functions import rerank


def _hit(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "distance": 0.5, "entity": {"text": text}}


def test_rerank_chunks_ranks_and_truncates(monkeypatch) -> None:
    def _fake_scores(query: str, texts: list[str]) -> list[float]:
        return [0.9, 0.1, 0.8]

    monkeypatch.setattr(rerank, "compute_rerank_scores", _fake_scores)
    chunks = [
        _hit("doc_a", "指针是C语言中用于存储变量地址的变量。"),
        _hit("doc_b", "数组是一组相同类型元素的集合。"),
        _hit("doc_c", "通过指针可以直接访问内存地址。"),
    ]
    ranked = rerank.rerank_chunks("什么是指针？", chunks, top_k=2)
    assert len(ranked) == 2, f"应截取 TOP2，实际 {len(ranked)}"
    assert [h["id"] for h in ranked] == ["doc_a", "doc_c"], f"排序不正确: {ranked}"
    assert ranked[0]["rerank_score"] == 0.9, "rerank_score 未正确挂载"


def test_rerank_chunks_rejects_empty(monkeypatch) -> None:
    monkeypatch.setattr(rerank, "compute_rerank_scores", lambda q, texts: [])
    with pytest.raises(ValueError, match="无候选片段可精排"):
        rerank.rerank_chunks("什么是指针？", [], top_k=2)
    with pytest.raises(ValueError, match="精排查询为空"):
        rerank.rerank_chunks("", [_hit("doc_a", "x")], top_k=2)


def test_rerank_chunks_weighted_fuses_rrf(monkeypatch) -> None:
    """加权融合：alpha 越高越信精排，fusion_score 由 rerank/rrf 共同决定。"""
    def _fake_scores(query: str, texts: list[str]) -> list[float]:
        return [0.9, 0.1]

    monkeypatch.setattr(rerank, "compute_rerank_scores", _fake_scores)
    merged = [
        {"rrf_score": 1.0, "hit": _hit("doc_a", "指针是C语言核心概念。")},
        {"rrf_score": 0.5, "hit": _hit("doc_b", "数组是相同类型元素集合。")},
    ]
    top = rerank.rerank_chunks_weighted("什么是指针？", merged, top_k=2, alpha=1.0)
    assert [h["id"] for h in top] == ["doc_a", "doc_b"]
    # alpha=1.0 时 fusion_score 即精排分；alpha=0 时退化为 RRF 归一化序
    low = rerank.rerank_chunks_weighted("什么是指针？", merged, top_k=2, alpha=0.0)
    assert [h["id"] for h in low] == ["doc_a", "doc_b"]
