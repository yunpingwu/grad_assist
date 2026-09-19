"""search_textbook 工具编排测试：桩掉 LLM/检索/精排/补拉，只验证编排与拼接。"""

import asyncio
import json

from app.study_agent.tools import search

_HYBRID_HITS = [
    {"id": "c1", "distance": 0.8, "entity": {
        "text": "指针是C语言的核心概念。【图: 指针示意图】", "chapter": "第3章", "section": "3.1",
        "metadata_json": json.dumps({"images": [{"url": "https://x/y.png", "description": "指针示意图"}]}),
    }},
    {"id": "c3_code", "distance": 0.6, "entity": {
        "text": "```c\nint main() { return 0; }\n```", "chapter": "第3章", "section": "3.1",
        "block_type": "code",
    }},
    {"id": "c2", "distance": 0.7, "entity": {"text": "数组是相同类型元素的集合。", "chapter": "第3章", "section": "3.2"}},
    {"id": "c4_code", "distance": 0.5, "entity": {
        "text": "```python\nprint(1)\n```", "chapter": "第4章", "section": "4.1",
        "block_type": "code",
    }},
]


def test_search_textbook_fast_path(monkeypatch) -> None:
    """快速路径：重写问句直接混合召回，验证图片回绑与代码块拼回。"""
    monkeypatch.setattr(search, "rewrite_query_search", _fake_hybrid)

    fast = asyncio.run(search.search_textbook.coroutine(
        textbook_name="C语言程序设计", rewritten_query="指针有什么用途?"
    ))
    assert "[片段1｜第3章 > 3.1]" in fast and "[片段2" in fast, fast
    assert "【图: 指针示意图】(https://x/y.png)" in fast, "url 未回绑到正文图标记"
    assert "int main()" in fast, "代码块未拼回所属小节正文"
    assert "[代码｜第4章 > 4.1]" in fast and "print(1)" in fast, "无对应正文的代码块未独立展示"
    assert "配图:" not in fast and "【图片候选】" not in fast, "不应再有独立图片行"


def test_search_textbook_deep_path(monkeypatch) -> None:
    """深度路径：HyDE + 融合 + 精排，取精排后片段。"""
    monkeypatch.setattr(search, "rewrite_query_search", _fake_hybrid)
    monkeypatch.setattr(search, "hyde_doc_generate", _fake_hyde_generate)
    monkeypatch.setattr(search, "agenerate_embeddings", _fake_generate_embeddings)
    monkeypatch.setattr(search, "search_by_vectors", _fake_search_by_vectors)
    monkeypatch.setattr(search, "rrf_merge", _fake_rrf)
    monkeypatch.setattr(search, "arerank_chunks_weighted", _fake_arerank)
    monkeypatch.setattr(search, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    deep = asyncio.run(search.search_textbook.coroutine(
        deep=True, textbook_name="C语言程序设计", rewritten_query="指针有什么用途?"
    ))
    assert "指针是C语言的核心概念" in deep, deep


async def _fake_hybrid(textbook_name: str, rewrite_query: str, chapter: str | None = None) -> list[dict]:
    return _HYBRID_HITS


async def _fake_hyde_generate(rewritten_query: str) -> str:
    return f"假设性文档: {rewritten_query}"


async def _fake_generate_embeddings(texts: list[str]) -> dict:
    n = len(texts)
    return {"dense": [[0.0] for _ in range(n)], "sparse": [{0: 1.0} for _ in range(n)]}


_vector_search_calls = {"n": 0}


async def _fake_search_by_vectors(textbook_name, dense_vec, sparse_vec, chapter=None, limit=5):
    _vector_search_calls["n"] += 1
    if _vector_search_calls["n"] == 1:
        return await _fake_hybrid(textbook_name, "", chapter)
    return [{"id": "h1", "distance": 0.5, "entity": {"text": "HyDE 补充片段", "chapter": "第3章", "section": "3.3"}}]


async def _fake_rrf(embedding_chunks: list[dict], hyde_chunks: list[dict], k: int = 60) -> list[dict]:
    return [{"rrf_score": 1.0, "hit": embedding_chunks[0]}, {"rrf_score": 0.8, "hit": hyde_chunks[0]}]


async def _fake_arerank(query: str, merged: list[dict], top_k: int, alpha: float) -> list[dict]:
    item = dict(merged[0]["hit"])
    item["rerank_score"] = 0.99
    item["fusion_score"] = 0.99
    return [item]
