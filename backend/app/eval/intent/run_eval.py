"""意图识别准确性评测：真漏斗（关键词/embedding/LLM）跑标注样本，输出分层指标报告。

默认口径只跑手写补充集 ``data/eval/intent/intent_extra.json``（114 条，覆盖
quiz/generate/plan/chat/unclear 与跨意图省略句）。``--include-anchors`` 追加两套零标注
成本的 explain 锚点：六教材 ``data/eval/*/qa_set.json`` 单轮题、``data/eval_multi_turn``
各轮（前轮问句拼 history，与线上 format_history 同构）——检索评测集问句无行为关键词，
期望意图清一色 explain，用来测错误改判与错误弃权。

用法（项目根目录，需要 embedding 服务与 LLM 中转可用）::

    .venv\\Scripts\\python.exe -m app.eval.intent.run_eval
    .venv\\Scripts\\python.exe -m app.eval.intent.run_eval --include-anchors
    .venv\\Scripts\\python.exe -m app.eval.intent.run_eval --groups extra_multiturn,extra_quiz
    .venv\\Scripts\\python.exe -m app.eval.intent.run_eval --repeats 3 --limit 50

指标口径：expected 为可接受集合、默认单值精确匹配；unclear 判到非 unclear 金标上记「弃权」，
单独报弃权率。--repeats>1 时对同一条取多数票并标记不稳定样本（LLM 随机性体检）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.eval.intent.metrics import evaluate, is_correct
from app.study_agent.query_functions.intent_detection import detect_intent
from app.utils import agenerate_embeddings

_ROOT = Path(__file__).resolve().parents[3]

# 检索评测集问句 → 意图锚点：全部是「讲解/答疑/对比/计算」，无生成/出题/规划诉求
_ANCHOR_EXPECTED = ["explain"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="意图识别准确性评测（真 embedding + 真 LLM 兜底）")
    parser.add_argument("--single", type=Path, default=None, help="单轮锚点集根目录，默认 data/eval（递归找 qa_set.json）")
    parser.add_argument("--multi", type=Path, default=_ROOT / "data" / "eval_multi_turn" / "qa_set.json", help="多轮锚点集")
    parser.add_argument("--extra", type=Path, default=_ROOT / "data" / "eval" / "intent" / "intent_extra.json", help="手写补充集")
    parser.add_argument("--include-anchors", action="store_true", help="追加检索评测集派生的 explain 锚点样本（默认只跑补充集）")
    parser.add_argument("--groups", default="", help="只跑这些组（逗号分隔），默认全部")
    parser.add_argument("--limit", type=int, default=0, help="每组最多 N 条；0 表示不限")
    parser.add_argument("--repeats", type=int, default=1, help="每条重复检测次数，>1 取多数票并标记不稳定，默认 1")
    parser.add_argument("--concurrency", type=int, default=4, help="并发检测数，默认 4")
    parser.add_argument("--output", type=Path, default=_ROOT / "data" / "eval" / "intent" / "intent_results.json", help="结果 JSON")
    # 供消融脚本复用同一装配管线时的开关（本脚本不读取）
    parser.add_argument("--leak-check", action="store_true", help="消融脚本专用：只跑例句池与评测集的泄漏检查")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于 0")
    if args.repeats < 1:
        parser.error("--repeats 必须大于 0")
    if args.limit < 0:
        parser.error("--limit 不能为负数")
    return args


# ── 样本装配（纯函数，可单测） ─────────────────────────────


def load_single_anchor(root: Path) -> list[dict[str, Any]]:
    """六教材单轮题 → explain 锚点样本。"""
    cases: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/qa_set.json")):
        questions = json.loads(path.read_text(encoding="utf-8"))["questions"]
        cases.extend(
            {
                "id": f"S-{q.get('id', idx)}",
                "group": "anchor_single",
                "query": q["question"],
                "history": "",
                "expected": list(_ANCHOR_EXPECTED),
            }
            for idx, q in enumerate(questions, start=1)
        )
    return cases


def load_multi_anchor(path: Path) -> list[dict[str, Any]]:
    """多轮集轮次 → explain 锚点样本；history 用前轮问句按线上「用户: …」逐行拼接。"""
    conversations = json.loads(path.read_text(encoding="utf-8"))["conversations"]
    cases: list[dict[str, Any]] = []
    for conv in conversations:
        questions = [t["question"] for t in conv["turns"]]
        for idx, turn in enumerate(conv["turns"]):
            role = "anchor_multi_initial" if turn.get("turn_role") == "initial" else "anchor_multi_follow"
            cases.append(
                {
                    "id": f"M-{turn.get('id', idx)}",
                    "group": role,
                    "query": turn["question"],
                    "history": "\n".join(f"用户: {q}" for q in questions[:idx]),
                    "expected": list(_ANCHOR_EXPECTED),
                    "anaphora": turn.get("anaphora"),
                }
            )
    return cases


def load_extra(path: Path) -> list[dict[str, Any]]:
    """手写补充集原样加载（已带 group/expected/history）。"""
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return [{**case, "id": f"E-{case['id']}"} for case in cases]


def filter_cases(cases: list[dict[str, Any]], groups: str, limit: int) -> list[dict[str, Any]]:
    """按组白名单过滤并按组截断（--groups 空表示全收；--limit 为每组上限）。"""
    wanted = {g.strip() for g in groups.split(",") if g.strip()}
    picked = [c for c in cases if not wanted or c["group"] in wanted]
    if limit <= 0:
        return picked
    seen: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in picked:
        if seen[case["group"]] < limit:
            out.append(case)
            seen[case["group"]] += 1
    return out


def resolve_majority(case: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """多次检测 → 多数票行：票数平局取首次出现的预测；ms 取全体中位数；标记不稳定。"""
    votes = Counter(run["intent"] for run in runs)
    winner = max(votes, key=lambda intent: (votes[intent], -next(i for i, r in enumerate(runs) if r["intent"] == intent)))
    chosen = next(r for r in runs if r["intent"] == winner)
    return {
        **case,
        "intent": chosen["intent"],
        "source": chosen["source"],
        "confidence": chosen["confidence"],
        "ms": st.median(float(r["ms"]) for r in runs),
        "unstable": len(votes) > 1,
        "runs": runs,
    }


# ── 执行 ──────────────────────────────────────────────


async def _detect_one(case: dict[str, Any], semaphore: asyncio.Semaphore, repeats: int) -> dict[str, Any]:
    """跑一条样本的漏斗（重复 repeats 次），单次异常记 error 行不阻断整场评测。"""
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        try:
            async with semaphore:
                t0 = time.perf_counter()
                result = await detect_intent(case["query"], case.get("history", ""))
                runs.append(
                    {
                        "intent": result.intent,
                        "source": result.source,
                        "confidence": result.confidence,
                        "ms": (time.perf_counter() - t0) * 1000,
                    }
                )
        except Exception as exc:  # noqa: BLE001 — 评测要的是全量覆盖，单条失败入表
            runs.append({"intent": "error", "source": "error", "confidence": 0.0, "ms": 0.0, "error": f"{type(exc).__name__}: {exc}"})
            break
    return resolve_majority(case, runs) if repeats > 1 else {**case, **runs[0]}


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """按入参装配样本：默认仅补充集；--include-anchors 时追加单轮/多轮 explain 锚点。"""
    cases = load_extra(args.extra)
    if args.include_anchors:
        cases = load_single_anchor(args.single or _ROOT / "data" / "eval") + load_multi_anchor(args.multi) + cases
    return cases


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cases = filter_cases(build_cases(args), args.groups, args.limit)
    if not cases:
        print("没有匹配的样本，检查 --groups 与数据路径", file=sys.stderr)
        raise SystemExit(2)
    print(f"样本 {len(cases)} 条（组分布 {dict(Counter(c['group'] for c in cases))}），每条检测 {args.repeats} 次，并发 {args.concurrency}")

    # 预热意图描述向量与 embedding 服务，避免冷加载算进首样本时延
    await agenerate_embeddings(["预热"])
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = await asyncio.gather(*(_detect_one(case, semaphore, args.repeats) for case in cases))

    summary = evaluate(rows)
    wrong = [r for r in rows if not is_correct(r)]
    payload = {
        "total": len(rows),
        "summary": summary,
        "wrong": [
            {
                "id": r["id"],
                "group": r["group"],
                "query": r["query"],
                "history": r.get("history", ""),
                "expected": r["expected"],
                "intent": r["intent"],
                "source": r["source"],
                "confidence": r.get("confidence"),
                "note": r.get("note"),
            }
            for r in wrong
        ],
        "per_case": rows,
    }
    return payload


def _print_report(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lat = s["latency_ms"]
    print(
        f"\n### 整体: n={s['n']}  准确率 {s['accuracy']:.3f}  弃权率 {s['abstention_rate']:.3f}"
        f"  时延 p50 {lat['p50']:.0f}ms / p95 {lat['p95']:.0f}ms"
    )

    print("\n### 分组准确率")
    for group, g in sorted(s["by_group"].items(), key=lambda kv: kv[1]["accuracy"]):
        print(f"  {group:<22} n={g['n']:3d}  acc={g['accuracy']:.3f}")

    print("\n### 按命中档位（哪档在干活、哪档在出错）")
    for source, g in sorted(s["by_source"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {source:<10} n={g['n']:3d}  占比 {g['n'] / s['n']:.2f}  acc={g['accuracy']:.3f}")

    print("\n### 混淆矩阵（行=金标，列=预测）")
    labels = sorted({g for row in s["confusion"].values() for g in row})
    print("  " + " " * 10 + "".join(f"{name:<10}" for name in labels))
    for gold, preds in s["confusion"].items():
        cells = "".join(f"{preds.get(name, 0):<10}" for name in labels)
        print(f"  {gold:<10}{cells}")

    print(f"\n### 错例 {len(payload['wrong'])} 条")
    for row in payload["wrong"]:
        hist = (row["history"] or "").replace("\n", " | ")
        extra = f"  note: {row['note']}" if row.get("note") else ""
        print(
            f"  [{row['id']}] {row['query'][:40]} (史:{hist[:30]})"
            f" -> {row['intent']}/{row['source']} 期望 {row['expected']}{extra}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(payload)
    print(f"\n结果已写入 {args.output}")
    return 0 if payload["summary"]["accuracy"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
