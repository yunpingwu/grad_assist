"""intent_detection 测试：桩掉 embedding 与 LLM，验证三档漏斗编排、历史传递与关键词层。"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.core import end_request, start_request
from app.study_agent.entity.intent import _IntentDecision
from app.study_agent.query_functions import intent_detection


class _RecordingLLM:
    """记录发给 LLM 兜底层的 prompt 文本，并返回预置判定结果。"""

    def __init__(self, sink: list[str], decision: _IntentDecision) -> None:
        self.sink = sink
        self.decision = decision

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, prompt_text: str) -> _IntentDecision:
        self.sink.append(prompt_text)
        return self.decision


@pytest.fixture(autouse=True)
def _stub_external_bindings(monkeypatch) -> dict:
    """桩掉例句池向量、query 编码与 LLM，只验证编排逻辑。

    返回的 dict 供测试读取副作用：``llm_prompts`` 为兜底层收到的 prompt 文本，
    ``embedded`` 为送去编码的文本列表，``sim`` 为本轮 query 向量与例句向量的点积。
    """
    state: dict = {"llm_prompts": [], "embedded": [], "sim": 0.4, "llm_kwargs": {}}

    async def _example_vectors() -> list[list[list[float]]]:
        # 每个意图的例句池向量取同一方向：query 向量 [sim, 0] 与任一条的点积即 sim，
        # top-3 均值也即 sim，让编排类测试只反映阈值分档而不掺池内形状因素
        return [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]] for _ in intent_detection.INTENT_ORDER]

    async def _embeddings(texts: list[str]) -> dict:
        state["embedded"].extend(texts)
        return {"dense": [[state["sim"], 0.0] for _ in texts], "sparse": [{0: 1.0} for _ in texts]}

    llm = _RecordingLLM(state["llm_prompts"], _IntentDecision(intent="generate", confidence=0.8))

    def _client(**kwargs):
        state["llm_kwargs"] = kwargs
        return llm

    monkeypatch.setattr(intent_detection, "_get_intent_example_vectors", _example_vectors)
    monkeypatch.setattr(intent_detection, "agenerate_embeddings", _embeddings)
    monkeypatch.setattr(intent_detection, "get_llm_client", _client)
    return state


def test_keyword_unique_hit() -> None:
    r = intent_detection._classify_by_keyword("帮我出几道选择题")
    assert r is not None and r.intent == "quiz" and r.source == "keyword"


def test_keyword_ambiguous_falls_through() -> None:
    r = intent_detection._classify_by_keyword("帮我生成第三章的练习题")
    assert r is None, "多意图并列（生成+练习题）应判定为歧义交下一档"


def test_format_history_keeps_human_history() -> None:
    messages = [HumanMessage(content="什么是指针?"), AIMessage(content="指针是一种…"), HumanMessage(content="如何使用它?")]
    assert intent_detection.format_history(messages) == "用户: 什么是指针?"


def test_format_history_empty_without_history() -> None:
    assert intent_detection.format_history([]) == ""


def test_detect_intent_empty_input() -> None:
    r = asyncio.run(intent_detection.detect_intent(""))
    assert r.intent == "unclear" and r.source == "fallback"


def test_embedding_medium_confidence_delegates_to_llm(_stub_external_bindings) -> None:
    r = asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    assert r.intent == "generate" and r.source == "llm"


def test_intent_examples_pool_shape() -> None:
    """每个意图都要有足够多且不重复的例句：少于 4 条时 top-3 均值退化成个别例句的相似度。"""
    pools = intent_detection.INTENT_EXAMPLES
    assert set(pools) == set(intent_detection.INTENT_ORDER)
    for intent, pool in pools.items():
        assert len(pool) >= 4, f"{intent} 例句池过少"
        assert all(q.strip() for q in pool), f"{intent} 含空例句"
    flat = [q for pool in pools.values() for q in pool]
    assert len(flat) == len(set(flat)), "同一例句不允许挂在多个意图下"


def test_pool_scores_uses_top3_mean() -> None:
    """意图得分 = query 与池内例句 cosine 的 top-3 均值，抗个别离群例句。"""
    pools = [
        [[0.9, 0.0], [0.5, 0.0], [0.4, 0.0], [0.3, 0.0]],  # top3 = 0.9/0.5/0.4
        [[0.8, 0.0], [0.8, 0.0], [0.7, 0.0], [0.1, 0.0], [0.0, 0.0]],
        [[0.5, 0.0], [0.2, 0.0]],  # 不足 3 条时按实际条数取均值
    ]
    scores = intent_detection._pool_scores([1.0, 0.0], pools)
    assert scores[0] == pytest.approx(0.6)
    assert scores[1] == pytest.approx((0.8 + 0.8 + 0.7) / 3)
    assert scores[2] == pytest.approx(0.35)


def test_embedding_direct_hit_prefers_top3_mean_over_max(monkeypatch, _stub_external_bindings) -> None:
    """直判定胜者用池均值而非最大相似度：quiz 池有 0.9 的离群单例，但整池不均（均值 0.70），
    应输给更整齐的 explain 池（均值 0.767）——旧「单描述最大 cosine」逻辑会误选 quiz。"""
    state = _stub_external_bindings
    state["sim"] = 1.0
    off = [[0.0, 1.0]] * 4
    pools = {
        "explain": [[0.8, 0.0], [0.8, 0.0], [0.7, 0.0], [0.2, 0.0]],
        "generate": off,
        "quiz": [[0.9, 0.0], [0.9, 0.0], [0.3, 0.0], [0.2, 0.0]],
        "plan": off,
        "chat": off,
    }

    async def _vectors():
        return [pools[i] for i in intent_detection.INTENT_ORDER]

    monkeypatch.setattr(intent_detection, "_get_intent_example_vectors", _vectors)
    r = asyncio.run(intent_detection.detect_intent("指针和地址到底是什么关系"))
    assert r.source == "embedding", f"0.767 已过 _EMBED_HIGH 且 margin 达标，应直判: {r}"
    assert r.intent == "explain", f"应取 top-3 均值最高的池而非最大单例: {r}"
    assert r.confidence == pytest.approx((0.8 + 0.8 + 0.7) / 3)


def test_embedding_high_sim_thin_margin_defers_to_llm(monkeypatch, _stub_external_bindings) -> None:
    """top1 过线但与次冠池几乎并列（消融案发区形态），不直判，下沉 LLM 消歧。"""
    state = _stub_external_bindings
    state["sim"] = 1.0
    pools = {
        "explain": [[0.83, 0.0], [0.83, 0.0], [0.83, 0.0], [0.83, 0.0]],  # 0.83
        "generate": [[0.80, 0.0], [0.80, 0.0], [0.80, 0.0], [0.80, 0.0]],  # 0.80, margin 0.03 < 0.05
        "quiz": [[0.0, 1.0]] * 4,
        "plan": [[0.0, 1.0]] * 4,
        "chat": [[0.0, 1.0]] * 4,
    }

    async def _vectors():
        return [pools[i] for i in intent_detection.INTENT_ORDER]

    monkeypatch.setattr(intent_detection, "_get_intent_example_vectors", _vectors)
    r = asyncio.run(intent_detection.detect_intent("把错题整理一下"))
    assert r.source == "llm", f"margin 0.03 未达 _EMBED_MARGIN，应下沉兜底: {r}"


def _pools_only(quiz: float, *, others: float = 0.1) -> dict[str, list[list[float]]]:
    """构造「quiz 池得分 quiz、其余意图池得分 others」的向量桩（query 取 [1,0] 时点积即得分）。"""
    base = [[others, 0.0]] * 4
    return {"explain": base, "generate": base, "quiz": [[quiz, 0.0]] * 4, "plan": base, "chat": base}


def test_keyword_hit_confirmed_by_pool_returns_fast(monkeypatch, _stub_external_bindings) -> None:
    """关键词命中且例句池 top1 同意 → 直返关键词档，不碰 LLM。"""
    state = _stub_external_bindings
    state["sim"] = 1.0
    pools = _pools_only(0.45)

    async def _vectors():
        return [pools[i] for i in intent_detection.INTENT_ORDER]

    monkeypatch.setattr(intent_detection, "_get_intent_example_vectors", _vectors)
    r = asyncio.run(intent_detection.detect_intent("帮我出几道选择题"))
    assert r.source == "keyword" and r.intent == "quiz", f"池 top1 与关键词一致应放行快车道: {r}"
    assert not state["llm_prompts"], "核验放行不该花 LLM 调用"


def test_keyword_hit_demoted_when_pool_top1_disagrees(monkeypatch, _stub_external_bindings) -> None:
    """真实错例形态（「生成树」被「生成」误拦）：池 top1 不同意 → 视同未命中续走漏斗到 LLM。"""
    state = _stub_external_bindings
    state["sim"] = 1.0
    pools = _pools_only(0.20)
    pools["explain"] = [[0.45, 0.0]] * 4  # 池认为更像讲解，且未过高阈值

    async def _vectors():
        return [pools[i] for i in intent_detection.INTENT_ORDER]

    monkeypatch.setattr(intent_detection, "_get_intent_example_vectors", _vectors)
    r = asyncio.run(intent_detection.detect_intent("为什么生成树算法要用贪心"))
    assert r.source == "llm" and r.intent == "generate", f"核验不通过应下沉兜底层判定: {r}"
    assert state["embedded"] == ["为什么生成树算法要用贪心"], "核验已编码一次，embedding 档不得重复编码"


def test_keyword_demoted_with_strong_pool_falls_to_embedding(monkeypatch, _stub_external_bindings) -> None:
    """降级后续走漏斗：若池对另一意图高置信直判，仍归 embedding 档定夺，不必劳 LLM。"""
    state = _stub_external_bindings
    state["sim"] = 1.0
    pools = _pools_only(0.20)
    pools["explain"] = [[0.75, 0.0]] * 4  # top1 0.75 ≥ HIGH，margin 0.55 ≥ MARGIN

    async def _vectors():
        return [pools[i] for i in intent_detection.INTENT_ORDER]

    monkeypatch.setattr(intent_detection, "_get_intent_example_vectors", _vectors)
    r = asyncio.run(intent_detection.detect_intent("为什么生成树算法要用贪心"))
    assert r.source == "embedding" and r.intent == "explain", f"降级样本带着已算分数续走 embedding 档: {r}"
    assert state["embedded"] == ["为什么生成树算法要用贪心"], "只许编码一次"


def test_keyword_verification_failure_trusts_keyword(monkeypatch, _stub_external_bindings) -> None:
    """核验编码失败不阻断：回退到现状——信任关键词档直返。"""
    state = _stub_external_bindings

    async def _boom(texts: list[str]) -> dict:
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(intent_detection, "agenerate_embeddings", _boom)
    r = asyncio.run(intent_detection.detect_intent("帮我出几道选择题"))
    assert r.source == "keyword" and r.intent == "quiz", "embedding 服务故障时关键词档应照旧生效"
    assert not state["llm_prompts"]


def test_llm_layer_receives_history(_stub_external_bindings) -> None:
    """有历史时兜底层输入与重写原输入同构：历史进【历史问题】、当前问句进【最新问题】。"""
    state = _stub_external_bindings
    asyncio.run(intent_detection.detect_intent("那再来十道", history="用户: 帮我出五道选择题"))
    prompt = state["llm_prompts"][-1]
    # 提示词正文里也提到这两个段名，按最后一次出现的分节标记切段再断言
    history_part, latest_part = prompt.split("【历史问题】")[-1].split("【最新问题】")
    assert history_part.strip() == "用户: 帮我出五道选择题", prompt
    assert latest_part.strip() == "那再来十道", "两段占位符可能被互换"


def test_front_layers_only_encode_current_query(_stub_external_bindings) -> None:
    """关键词与 embedding 两档只看当前问句，历史不进向量。"""
    state = _stub_external_bindings
    asyncio.run(intent_detection.detect_intent("随便讲讲这本书", history="用户: 上一轮的长问题" * 20))
    assert state["embedded"] == ["随便讲讲这本书"]


def test_llm_layer_marks_first_turn(_stub_external_bindings) -> None:
    """无历史时兜底层显式标注首次提问，不留空档。"""
    state = _stub_external_bindings
    asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    assert "（无历史问题，本次为首次提问）" in state["llm_prompts"][-1]


def test_low_similarity_without_history_stays_unclear(_stub_external_bindings) -> None:
    """单轮低相似仍是「领域外」，不额外花一次 LLM 调用。"""
    _stub_external_bindings["sim"] = 0.1
    r = asyncio.run(intent_detection.detect_intent("今天天气怎么样"))
    assert r.intent == "unclear" and r.source == "embedding"


def test_low_similarity_with_history_falls_to_llm(_stub_external_bindings) -> None:
    """多轮短句易低相似，须下沉兜底层用历史消歧，而非就地判 unclear。"""
    _stub_external_bindings["sim"] = 0.1
    r = asyncio.run(intent_detection.detect_intent("那再来十道", history="用户: 帮我出五道选择题"))
    assert r.source == "llm"


def test_llm_layer_uses_tight_timeout(_stub_external_bindings) -> None:
    """兜底层收紧时延：短超时 + 不自动重试，且关思考（判错只回退 explain，不值得拖整轮）。"""
    asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    kwargs = _stub_external_bindings["llm_kwargs"]
    assert kwargs["max_retries"] == 0 and kwargs["enable_thinking"] is False, kwargs
    assert 0 < kwargs["timeout"] <= 10, f"意图分类超时未收口: {kwargs['timeout']}s"


def test_llm_failure_falls_back_to_explain(monkeypatch, _stub_external_bindings) -> None:
    """兜底 LLM 拿不到结果（超时/异常）时回退 explain，不向上抛。"""

    def _boom(**_kw):
        raise RuntimeError("请求超时")

    monkeypatch.setattr(intent_detection, "get_llm_client", _boom)
    r = asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    assert r.intent == "explain" and r.source == "fallback"


def test_embedding_failure_delegates_to_llm(monkeypatch, _stub_external_bindings) -> None:
    """embedding 服务不可用不阻断识别，降级到兜底层。"""

    async def _boom(texts: list[str]) -> dict:
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(intent_detection, "agenerate_embeddings", _boom)
    r = asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    assert r.source == "llm" and r.intent == "generate"


def test_funnel_records_intent_ms(_stub_external_bindings) -> None:
    """整段漏斗计入 intent_ms，前置理解的时延在打点里可见。"""
    metrics = start_request("s1", "随便讲讲这本书")
    try:
        asyncio.run(intent_detection.detect_intent("随便讲讲这本书"))
    finally:
        end_request()
    assert metrics.stage_ms.get("intent_ms") is not None, metrics.stage_ms
