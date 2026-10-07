"""search_textbook 工具编排测试：桩掉 LLM/检索/精排/补拉，只验证编排与拼接。"""

import asyncio
import json
from types import SimpleNamespace

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
    """快速路径：缺省用本轮原问题做混合召回，验证图片回绑与代码块拼回。"""
    monkeypatch.setattr(search, "search_by_query", _fake_hybrid)

    fast, _ = asyncio.run(search.search_textbook.coroutine(
        textbook_name="C语言程序设计", requirement="指针有什么用途?"
    ))
    assert "[片段1｜c1｜第3章 > 3.1]" in fast and "[片段2" in fast, fast
    assert "【图: 指针示意图】(https://x/y.png)" in fast, "url 未回绑到正文图标记"
    assert "int main()" in fast, "代码块未拼回所属小节正文"
    assert "[代码｜c4_code｜第4章 > 4.1]" in fast and "print(1)" in fast, "无对应正文的代码块未独立展示"
    assert "配图:" not in fast and "【图片候选】" not in fast, "不应再有独立图片行"


def test_search_textbook_deep_path(monkeypatch) -> None:
    """深度路径：HyDE + 融合 + 精排，取精排后片段。"""
    monkeypatch.setattr(search, "search_by_query", _fake_hybrid)
    monkeypatch.setattr(search, "hyde_doc_generate", _fake_hyde_generate)
    monkeypatch.setattr(search, "agenerate_embeddings", _fake_generate_embeddings)
    monkeypatch.setattr(search, "search_by_vectors", _fake_search_by_vectors)
    monkeypatch.setattr(search, "rrf_merge", _fake_rrf)
    monkeypatch.setattr(search, "arerank_chunks_weighted", _fake_arerank)
    monkeypatch.setattr(search, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    deep, _ = asyncio.run(search.search_textbook.coroutine(
        deep=True, textbook_name="C语言程序设计", requirement="指针有什么用途?"
    ))
    assert "指针是C语言的核心概念" in deep, deep


def test_search_textbook_prefers_agent_query(monkeypatch) -> None:
    """多轮指代由模型自组自包含问句：显式 query 优先于本轮原问题，并写进检索打点。"""
    used: list[str] = []

    async def _capture(textbook_name: str, search_query: str, chapter: str | None = None) -> list[dict]:
        used.append(search_query)
        return _HYBRID_HITS

    record = SimpleNamespace(retrievals=[], retrieved_chunk_ids=[])
    monkeypatch.setattr(search, "search_by_query", _capture)
    monkeypatch.setattr(search, "get_metrics", lambda: record)
    monkeypatch.setattr(search, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    asyncio.run(search.search_textbook.coroutine(
        query="C 语言指针常量的用途", textbook_name="C语言程序设计", requirement="那它呢?"
    ))
    assert used == ["C 语言指针常量的用途"], f"未优先采用模型自组的检索问句: {used}"
    assert record.retrievals[0]["query"] == "C 语言指针常量的用途"


def test_search_textbook_keeps_every_retrieval(monkeypatch) -> None:
    """一次请求内多次检索逐条留痕，后一次不得覆盖前一次。

    线上出现过 deep=False 却带 rerank 耗时的自相矛盾记录：旧实现把问句/深度/召回数
    写成请求级标量，第二次检索（含入口写完就抛异常提前返回的路径）会污染第一次的读数。
    """
    record = SimpleNamespace(retrievals=[], retrieved_chunk_ids=[])
    fast_calls = {"n": 0}

    async def _fast_or_boom(textbook_name: str, query_arg: str, chapter: str | None = None) -> list[dict]:
        # 第一次快检索正常返回，第二次抛错，用于覆盖降级分支的打点
        fast_calls["n"] += 1
        if fast_calls["n"] >= 2:
            raise RuntimeError("milvus 连接超时")
        return _HYBRID_HITS

    monkeypatch.setattr(search, "get_metrics", lambda: record)
    monkeypatch.setattr(search, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)
    monkeypatch.setattr(search, "search_by_query", _fast_or_boom)
    monkeypatch.setattr(search, "hyde_doc_generate", _fake_hyde_generate)
    monkeypatch.setattr(search, "agenerate_embeddings", _fake_generate_embeddings)
    monkeypatch.setattr(search, "search_by_vectors", _fake_vectors)
    monkeypatch.setattr(search, "rrf_merge", _fake_rrf)
    monkeypatch.setattr(search, "arerank_chunks_weighted", _fake_arerank)

    asyncio.run(search.search_textbook.coroutine(
        query="指针的用途", deep=True, textbook_name="C语言程序设计"
    ))
    asyncio.run(search.search_textbook.coroutine(
        query="数组的定义", textbook_name="C语言程序设计"
    ))
    failed_content, failed_artifact = asyncio.run(search.search_textbook.coroutine(
        query="越界的问句", textbook_name="C语言程序设计"
    ))
    assert "检索教材失败" in failed_content and failed_artifact == {}

    assert [(e["query"], e["deep"], e["recall"], e["kept"]) for e in record.retrievals] == [
        ("指针的用途", True, 2, 1),
        ("数组的定义", False, 4, 4),
        ("越界的问句", False, 0, 0),
    ], record.retrievals
    assert [e["error"] for e in record.retrievals] == [None, None, "RuntimeError"]


async def _fake_vectors(_tb, _dense, _sparse, chapter=None, limit=5) -> list[dict]:
    return _HYBRID_HITS


def test_search_textbook_falls_back_to_requirement(monkeypatch) -> None:
    """模型没传 query 时回退本轮原问题，不再依赖前置重写产物。"""
    used: list[str] = []

    async def _capture(textbook_name: str, search_query: str, chapter: str | None = None) -> list[dict]:
        used.append(search_query)
        return _HYBRID_HITS

    monkeypatch.setattr(search, "search_by_query", _capture)
    monkeypatch.setattr(search, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    asyncio.run(search.search_textbook.coroutine(textbook_name="C语言程序设计", requirement="什么是数组?"))
    assert used == ["什么是数组?"]


def test_search_textbook_hides_injected_state_args() -> None:
    """注入给工具的 state 字段（教材名、检索问句回退值）不得出现在模型可见的调用 schema 里。"""
    assert set(search.search_textbook.tool_call_schema.model_fields) == {"query", "deep", "chapter"}


def test_search_textbook_returns_content_and_artifact(monkeypatch) -> None:
    """工具输出分两通道：content 上线给模型，artifact 存原文给系统（实测 artifact 不进请求）。

    降级与历史回显都靠 artifact 拿到原文，content 事后被换成一行记录也不影响这两条链路；
    少了 artifact，「原文只活在模型上下文里」这个耦合就回来了。
    """
    monkeypatch.setattr(search, "search_by_query", _fake_hybrid)
    monkeypatch.setattr(search, "_enrich_hits_with_section_codes", lambda _tb, hits: hits)

    content, artifact = asyncio.run(
        search.search_textbook.coroutine(textbook_name="C语言程序设计", requirement="指针有什么用途?")
    )

    assert "[片段1｜c1｜第3章 > 3.1]" in content, content
    assert artifact["query"] == "指针有什么用途?" and artifact["deep"] is False
    assert artifact["textbook_name"] == "C语言程序设计"
    assert "指针是C语言的核心概念" in artifact["content"], "原文未随 artifact 落库"
    assert [c["id"] for c in artifact["chunks"]] == ["c1", "c3_code", "c2", "c4_code"]
    assert artifact["chunks"][0]["location"] == "第3章 > 3.1"


def test_search_textbook_failure_still_returns_pair(monkeypatch) -> None:
    """失败与空结果同样成对返回：content_and_artifact 下返回裸字符串会让 ToolNode 解包报错。"""

    async def _boom(textbook_name: str, search_query: str, chapter: str | None = None) -> list[dict]:
        raise RuntimeError("milvus 连接超时")

    monkeypatch.setattr(search, "search_by_query", _boom)

    content, artifact = asyncio.run(
        search.search_textbook.coroutine(textbook_name="C语言程序设计", requirement="指针有什么用途?")
    )

    assert "检索教材失败" in content and artifact == {}


async def _fake_hybrid(textbook_name: str, search_query: str, chapter: str | None = None) -> list[dict]:
    return _HYBRID_HITS


async def _fake_hyde_generate(search_query: str) -> str:
    return f"假设性文档: {search_query}"


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
