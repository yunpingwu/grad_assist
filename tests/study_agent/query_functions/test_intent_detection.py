"""intent_detection 测试：桩掉 embedding 与 LLM，验证三档漏斗编排与关键词层。"""

import asyncio

import pytest

from app.study_agent.entity.intent import IntentResult
from app.study_agent.query_functions import intent_detection


@pytest.fixture(autouse=True)
def _stub_external_bindings(monkeypatch) -> None:
    """桩掉 embedding/LLM 与意图描述向量缓存，只验证编排逻辑。"""

    async def _fake_embeddings(texts: list[str]) -> dict:
        # 意图描述预计算是 5 条的多文本批次、分类只编码 query 单条：query=[1.0,0.0]
        # 与描述=[0.4,0.0] 的余弦=0.4，落在中等置信区间 → 交 LLM 兜底
        vec = [1.0, 0.0] if len(texts) == 1 else [0.4, 0.0]
        return {"dense": [list(vec) for _ in texts], "sparse": [{0: 1.0} for _ in texts]}

    async def _fake_llm(text: str) -> IntentResult:
        return IntentResult("generate", 0.8, "llm")

    monkeypatch.setattr(intent_detection, "agenerate_embeddings", _fake_embeddings)
    monkeypatch.setattr(intent_detection, "_classify_by_llm", _fake_llm)
    # 清理意图描述向量进程级缓存，避免跨测试污染
    monkeypatch.setattr(intent_detection, "_precomputed_desc_vectors", None)


def test_keyword_unique_hit() -> None:
    r = intent_detection._classify_by_keyword("帮我出几道选择题")
    assert r is not None and r.intent == "quiz" and r.source == "keyword"


def test_keyword_ambiguous_falls_through() -> None:
    r = intent_detection._classify_by_keyword("帮我生成第三章的练习题")
    assert r is None, "多意图并列（生成+练习题）应判定为歧义交下一档"


def test_detect_intent_empty_input() -> None:
    r = asyncio.run(intent_detection.detect_intent(""))
    assert r.intent == "unclear" and r.source == "fallback"


def test_embedding_medium_confidence_delegates_to_llm() -> None:
    r = asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    assert r.intent == "generate" and r.source == "llm"
