"""intent.run_threshold_ablation 路由推演纯函数单测：模拟各档位在三档漏斗中的走向。"""

from __future__ import annotations

from app.eval.intent.run_threshold_ablation import pick_best, route_predict


def test_keyword_confirmed_by_pool_takes_precedence() -> None:
    """关键词命中且池冠同意 → 快车道直返，与阈值档无关。"""
    pred = route_predict(0.45, 0.25, "quiz", keyword_intent="quiz", multi_turn=False, llm_intent="explain", high=0.9)
    assert pred == {"intent": "quiz", "source": "keyword"}


def test_keyword_demoted_by_pool_continues_funnel() -> None:
    """池核验降级（「生成树」误拦形态）：视同未命中续走漏斗，embedding 档过闸即直判。"""
    pred = route_predict(0.62, 0.20, "generate", keyword_intent="quiz", multi_turn=False, llm_intent="explain", high=0.6, min_margin=0.1)
    assert pred == {"intent": "generate", "source": "embedding"}


def test_keyword_demoted_mid_band_sinks_to_llm() -> None:
    pred = route_predict(0.45, 0.25, "explain", keyword_intent="generate", multi_turn=False, llm_intent="explain", high=0.55)
    assert pred == {"intent": "explain", "source": "llm"}


def test_keyword_demoted_single_turn_low_sim_stays_unclear() -> None:
    pred = route_predict(0.1, 0.05, "explain", keyword_intent="generate", multi_turn=False, llm_intent="explain", high=0.55)
    assert pred == {"intent": "unclear", "source": "embedding"}


def test_sim_and_margin_both_meeting_high_uses_embedding_argmax() -> None:
    pred = route_predict(0.62, 0.20, "generate", keyword_intent=None, multi_turn=False, llm_intent="explain", high=0.6, min_margin=0.1)
    assert pred == {"intent": "generate", "source": "embedding"}


def test_high_sim_but_thin_margin_defers_to_llm() -> None:
    """top1 过线但领先亚军不足（如 0.86 判 quiz 那类案发区），宁缺毋滥下沉兜底层。"""
    pred = route_predict(0.86, 0.04, "quiz", keyword_intent=None, multi_turn=False, llm_intent="generate", high=0.65, min_margin=0.10)
    assert pred == {"intent": "generate", "source": "llm"}


def test_margin_defaults_to_gate_off() -> None:
    """不扫 margin 档位时判据退化为纯绝对阈值（旧行为）。"""
    pred = route_predict(0.86, 0.0, "quiz", keyword_intent=None, multi_turn=False, llm_intent="generate", high=0.65)
    assert pred == {"intent": "quiz", "source": "embedding"}


def test_single_turn_low_sim_stays_unclear_below_high() -> None:
    pred = route_predict(0.1, 0.05, "generate", keyword_intent=None, multi_turn=False, llm_intent="explain", high=0.6)
    assert pred["intent"] == "unclear" and pred["source"] == "embedding"


def test_mid_band_and_multi_turn_low_delegate_to_llm() -> None:
    mid = route_predict(0.4, 0.1, "generate", keyword_intent=None, multi_turn=False, llm_intent="quiz", high=0.6)
    low_multi = route_predict(0.1, 0.1, "generate", keyword_intent=None, multi_turn=True, llm_intent="quiz", high=0.6)
    assert mid["source"] == "llm" and low_multi["source"] == "llm"
    assert mid["intent"] == "quiz" == low_multi["intent"]


def test_missing_llm_prediction_falls_back_defensively() -> None:
    pred = route_predict(0.4, 0.1, "generate", keyword_intent=None, multi_turn=False, llm_intent=None, high=0.6)
    assert pred == {"intent": "explain", "source": "fallback"}


def test_pick_best_maxes_embedding_share_within_accuracy_floor() -> None:
    table = {
        "A0.50|B0.00": {"accuracy": 0.96, "embedding_share": 0.50, "llm_share": 0.20},
        "A0.60|B0.10": {"accuracy": 0.98, "embedding_share": 0.30, "llm_share": 0.40},
        "A0.70|B0.20": {"accuracy": 0.99, "embedding_share": 0.10, "llm_share": 0.55},
    }
    assert pick_best(table) == "A0.60|B0.10"


def test_pick_best_without_any_above_floor_uses_accuracy_then_less_llm() -> None:
    table = {
        "wide": {"accuracy": 0.90, "embedding_share": 0.80, "llm_share": 0.05},
        "safe": {"accuracy": 0.95, "embedding_share": 0.10, "llm_share": 0.60},
    }
    assert pick_best(table) == "safe"


def test_pick_best_breaks_ties_toward_stricter_threshold() -> None:
    """分流与准确率相同的档位取更高的 A：对新分布之外的数据更保守，且选点须可复现。"""
    table = {
        "A0.50|B0.05": {"accuracy": 0.991, "embedding_share": 0.22, "llm_share": 0.46},
        "A0.55|B0.05": {"accuracy": 0.991, "embedding_share": 0.22, "llm_share": 0.46},
    }
    assert pick_best(table) == "A0.55|B0.05"
