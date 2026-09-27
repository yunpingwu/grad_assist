"""embedding 档路由判据消融：例句池 top-3 均值分布上扫「绝对阈值 A × 次冠差 B」网格，离线模拟漏斗路由。

原理：三档漏斗里受判据影响的是第 2 档「高置信直判」与第 1 档的池核验。新打分（INTENT_EXAMPLES
池的 top-3 均值，见 intent_detection）下，直判条件为 ``top1 >= A 且 top1-top2 >= B``：
绝对阈值管「够不够像」，次冠差管「赢没赢干净」——0.86 把错题整理判成 quiz 那类自信误判
正是「过线但没拉开」的形态。关键词档与生产同口径：命中还须 **池冠=关键词意图** 才直返，
否则视同未命中续走漏斗。探测阶段对每条样本拿到与判据无关的三样事实：①关键词档结果、
②五池完整得分（推出 top1/margin/argmax，关键词命中样本同样要编码供核验）、③LLM 档判定。
**LLM 探测覆盖全部非关键词直返样本**（含被核验降级者与 margin 档位下高 top1 低 margin 的
样本，否则无法模拟其下沉走向）。之后任意 (A,B) 的预测都由纯函数 route_predict 推演，
比较 36 个档位无需重复烧 LLM 调用。

选点口径 pick_best：准确率 ≥ _ACC_FLOOR 的档位里取 embedding 直判占比最大者（分流越多、
LLM 越省）；若无档位过线，退回「最高准确率、同分少花 LLM」。

用法（项目根目录，需要 embedding 服务与 LLM 中转可用）::

    .venv\\Scripts\\python.exe -m app.eval.intent.run_threshold_ablation
    .venv\\Scripts\\python.exe -m app.eval.intent.run_threshold_ablation --leak-check   # 只查例句池与评测集的泄漏
    .venv\\Scripts\\python.exe -m app.eval.intent.run_threshold_ablation --include-anchors

输出：top1/margin 分布分位数、各档位 准确率 / embedding 直判量 / 下沉 LLM 量 对照表、
推荐点错例明细；结果 JSON 写入 data/eval/intent/threshold_ablation.json。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from app.eval.intent.metrics import abstention_rate, accuracy, is_correct
from app.eval.intent.run_eval import _parse_args as _eval_args
from app.eval.intent.run_eval import build_cases, filter_cases
from app.study_agent.query_functions.intent_detection import (
    _EMBED_LOW,
    INTENT_EXAMPLES,
    INTENT_ORDER,
    _classify_by_keyword,
    _classify_by_llm,
    _get_intent_example_vectors,
    _pool_scores,
    logger,
)
from app.utils import agenerate_embeddings

# 扫描网格：A 覆盖池均值的动态范围，B 含 0（退化为纯绝对阈值，兼容旧判据对照）
_SWEEP_A = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
_SWEEP_B = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)

# 选点的准确率底线：低于此值的档位再多分流也不接受（评测集 114 条，0.98 ≈ 至多错 2 条）
_ACC_FLOOR = 0.98

# 泄漏检查的报告线：例句与评测 query 的 cosine 高于此值会污染消融定标
_LEAK_LINE = 0.85


def route_predict(
    top1: float,
    margin: float,
    argmax_intent: str,
    *,
    keyword_intent: str | None,
    multi_turn: bool,
    llm_intent: str | None,
    high: float,
    min_margin: float = 0.0,
    low: float = _EMBED_LOW,
) -> dict[str, str]:
    """按候选判据 (high, min_margin) 推演单条样本的漏斗路由（纯函数，供扫描复用）。

    与生产 detect_intent 同口径：关键词命中须经例句池 top1 核验（池冠=关键词意图才直返），
    核验降级样本视同未命中，带着同一份池分数续走双闸门→unclear→LLM 的后续档。
    """
    if keyword_intent is not None and argmax_intent == keyword_intent:
        return {"intent": keyword_intent, "source": "keyword"}
    if top1 >= high and margin >= min_margin:
        return {"intent": argmax_intent, "source": "embedding"}
    if top1 < low and not multi_turn:
        return {"intent": "unclear", "source": "embedding"}
    if llm_intent is not None:
        return {"intent": llm_intent, "source": "llm"}
    # 扫描期没调 LLM 的样本不会落到这里（探测已覆盖全部非关键词样本），防御性回退
    return {"intent": "explain", "source": "fallback"}


def pick_best(table: dict[str, dict[str, float]]) -> str:
    """按「准确率≥底线内分流最大」选档；无一过线时取最高准确率、同分少花 LLM。

    档位键形如 ``A0.55|B0.05``；分流与准确率的平票取更高 A（更保守）且消除
    set 迭代顺序带来的不可复现性。
    """
    def _a_value(key: str) -> float:
        """从 ``A0.55|B0.05`` 形态的档位键取 A 值；非该形态的键（测试用语义名）按 0 处理。"""
        try:
            return float(key[1:5])
        except ValueError:
            return 0.0

    passing = {key for key, row in table.items() if row["accuracy"] >= _ACC_FLOOR}
    if passing:
        return max(passing, key=lambda key: (table[key]["embedding_share"], table[key]["accuracy"], _a_value(key)))
    return max(table, key=lambda key: (table[key]["accuracy"], -table[key]["llm_share"], _a_value(key)))


def _row(
    case: dict[str, Any],
    scores: list[float],
    keyword_intent: str | None,
    llm_intent: str | None,
) -> dict[str, Any]:
    """把探测到的事实压成一行：五池得分、top1/margin/argmax、关键词与 LLM 判定。"""
    ordered = sorted(scores, reverse=True)
    multi_turn = bool((case.get("history") or "").strip())
    argmax = INTENT_ORDER[max(range(len(scores)), key=scores.__getitem__)]
    return {
        **case,
        "scores": scores,
        "sim": ordered[0],
        "margin": ordered[0] - ordered[1],
        "argmax": argmax,
        "keyword": keyword_intent,
        "llm": llm_intent,
        "multi_turn": multi_turn,
    }


def _rows_at_threshold(rows: list[dict[str, Any]], high: float, min_margin: float) -> list[dict[str, Any]]:
    """把探测行按判据 (A, B) 展开成 retrieval.metrics 与 intent.metrics 均可评的行结构。"""
    out = []
    for row in rows:
        pred = route_predict(
            row["sim"], row["margin"], row["argmax"],
            keyword_intent=row["keyword"], multi_turn=row["multi_turn"], llm_intent=row["llm"],
            high=high, min_margin=min_margin,
        )
        out.append({**row, **pred})
    return out


async def _leak_check(cases: list[dict[str, Any]]) -> None:
    """例句池 × 评测 query 的最大 cosine 泄漏检查：高于 _LEAK_LINE 的组合逐条报告。

    评测集是消融与终验的「考卷」，例句池是「教材」；两者近重复会让直判准确率虚高。
    """
    texts = [q for pool in INTENT_EXAMPLES.values() for q in pool]
    emb = await agenerate_embeddings(texts + [c["query"] for c in cases])
    example_vecs, query_vecs = emb["dense"][: len(texts)], emb["dense"][len(texts):]
    hits = 0
    for case, vec in zip(cases, query_vecs, strict=True):
        worst_text, worst_sim = "", -1.0
        for text, ex_vec in zip(texts, example_vecs, strict=True):
            sim = sum(x * y for x, y in zip(vec, ex_vec, strict=True))
            if sim > worst_sim:
                worst_text, worst_sim = text, sim
        if worst_sim >= _LEAK_LINE:
            hits += 1
            print(f"  [泄漏] {case['id']} {case['query'][:36]} ≈ {worst_text[:36]} cos {worst_sim:.3f}")
    print(f"泄漏检查完成：{hits} / {len(cases)} 条评测 query 与例句池 cosine ≥ {_LEAK_LINE}")


async def _probe(rows_src: list[dict[str, Any]], concurrency: int) -> list[dict[str, Any]]:
    """逐条拿与判据无关的事实：关键词结果、五池完整得分、（未放行者的）LLM 判定。

    池核验入产后关键词命中也不能免编码：核验本身就要用五池得分选冠。LLM 探测
    覆盖所有「非关键词直返」样本——含被核验降级续走漏斗者，与生产的下沉面一致。
    """
    pools = await _get_intent_example_vectors()
    emb = await agenerate_embeddings([c["query"] for c in rows_src])
    score_map = {id(c): _pool_scores(v, pools) for c, v in zip(rows_src, emb["dense"], strict=True)}
    kw_map = {id(c): (_kw.intent if (_kw := _classify_by_keyword(c["query"])) else None) for c in rows_src}

    def _argmax(case: dict[str, Any]) -> str:
        scores = score_map[id(case)]
        return INTENT_ORDER[max(range(len(scores)), key=scores.__getitem__)]

    # margin 判据下高 top1 样本也可能下沉 LLM，故兜底探测覆盖全部非关键词直返样本
    semaphore = asyncio.Semaphore(concurrency)
    llm_intents: dict[int, str] = {}

    async def _ask(case: dict[str, Any]) -> None:
        async with semaphore:
            try:
                llm_intents[id(case)] = (await _classify_by_llm(case["query"], case.get("history", ""))).intent
            except Exception as exc:  # noqa: BLE001 — 与生产 detect_intent 同口径：兜底失败回退 explain
                logger.warning(f"消融 LLM 判定失败，按 explain 处理: {exc}")
                llm_intents[id(case)] = "explain"

    not_confirmed = [c for c in rows_src if kw_map[id(c)] is None or _argmax(c) != kw_map[id(c)]]
    await asyncio.gather(*(_ask(c) for c in not_confirmed))

    rows: list[dict[str, Any]] = []
    for case in rows_src:
        rows.append(_row(case, score_map[id(case)], kw_map[id(case)], llm_intents.get(id(case))))
    return rows


def _pct(values: list[float], q: float) -> float:
    """最近秩分位数（评测集小，无需插值）。"""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return ordered[rank - 1]


async def _run(args: argparse.Namespace) -> None:
    cases = filter_cases(build_cases(args), args.groups, args.limit)
    if not cases:
        print("没有匹配的样本", file=sys.stderr)
        raise SystemExit(2)
    if args.leak_check:
        await _leak_check(cases)
        return
    print(f"探测 {len(cases)} 条样本（编码 1 次/条，LLM 探测全部非关键词直返样本），网格 A×B = {len(_SWEEP_A)}×{len(_SWEEP_B)}")
    rows = await _probe(cases, args.concurrency)

    # 分布只统计会真正过 embedding 闸门的样本：关键词核验直返者不参赛，混入会虚抬分位
    gate_rows = [r for r in rows if r["keyword"] is None or r["argmax"] != r["keyword"]]
    top1s = [r["sim"] for r in gate_rows]
    margins = [r["margin"] for r in gate_rows]
    print(
        f"  分布（过闸候选 {len(top1s)} 条）top1 p10/p50/p90 = "
        f"{_pct(top1s, 0.1):.2f}/{_pct(top1s, 0.5):.2f}/{_pct(top1s, 0.9):.2f}"
        f"，margin p10/p50/p90 = {_pct(margins, 0.1):.3f}/{_pct(margins, 0.5):.3f}/{_pct(margins, 0.9):.3f}"
    )

    table: dict[str, dict[str, float]] = {}
    for high in _SWEEP_A:
        for min_margin in _SWEEP_B:
            predicted = _rows_at_threshold(rows, high, min_margin)
            by_source: dict[str, int] = {}
            for r in predicted:
                by_source[r["source"]] = by_source.get(r["source"], 0) + 1
            n = len(predicted)
            acc = accuracy(predicted)
            table[f"A{high:.2f}|B{min_margin:.2f}"] = {
                "accuracy": acc,
                "abstention": abstention_rate(predicted),
                "embedding_direct": by_source.get("embedding", 0),
                "embedding_share": by_source.get("embedding", 0) / n,
                "to_llm": by_source.get("llm", 0),
                "llm_share": by_source.get("llm", 0) / n,
            }

    for key, row in sorted(table.items(), key=lambda kv: (-kv[1]["embedding_share"], -kv[1]["accuracy"])):
        flag = "✓" if row["accuracy"] >= _ACC_FLOOR else " "
        print(
            f" {flag} {key}  acc={row['accuracy']:.3f}  弃权={row['abstention']:.3f}"
            f"  直判 {row['embedding_direct']:3d}（{row['embedding_share']:.0%}）  下沉LLM {row['to_llm']:3d}（{row['llm_share']:.0%}）"
        )

    best = pick_best(table)
    high, min_margin = float(best[1:5]), float(best[7:])
    predicted = _rows_at_threshold(rows, high, min_margin)
    wrong = [r for r in predicted if not is_correct(r)]
    print(f"\n推荐 {best}（准确率≥{_ACC_FLOOR} 内分流最大），错例 {len(wrong)} 条：")
    for r in wrong:
        print(f"  [{r['id']}] {r['query'][:40]} -> {r['intent']}/{r['source']} (top1 {r['sim']:.2f}, margin {r['margin']:.3f}) 期望 {r['expected']}")

    keys = ("id", "group", "query", "history", "expected", "intent", "source", "sim", "margin", "argmax", "scores")
    payload = {
        "n": len(rows),
        "acc_floor": _ACC_FLOOR,
        "sweep": table,
        "best": {"key": best, "high": high, "min_margin": min_margin},
        "wrong_at_best": [{k: r.get(k) for k in keys} for r in wrong],
        "per_case": rows,
    }
    output = args.output.with_name("threshold_ablation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {output}")


def main(argv: list[str] | None = None) -> int:
    # 复用评测脚本的全部样本装配参数（--extra/--include-anchors/--groups/--limit/--concurrency/--leak-check）；
    # --output 的值忽略不计，本脚本固定落盘 threshold_ablation.json
    args = _eval_args(argv if argv is not None else sys.argv[1:])
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
