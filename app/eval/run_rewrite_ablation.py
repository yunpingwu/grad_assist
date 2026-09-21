"""问题重写消融：同一轮分别用「原问句」和「线上重写问句」各走一遍生产检索，比 hit@5。

用法（在项目根目录，需要 service 打点日志与一次多轮评测结果）::

    .venv\\Scripts\\python.exe -m app.eval.run_rewrite_ablation

数据源为只读输入，不改动任何线上代码：
- ``logs/metrics.log``：每轮的 ``rewrite_mode`` / ``rewrite_query``（按 request_id 关联）；
- 评测结果 JSON：每轮的原问题、turn_role、gold_chunk_ids 与线上实际召回。

输出三列对照：单用原问句检索、单用重写问句检索、两者 RRF(k=60) 融合，
外加「重写掉出 top5 / 救回 top5」的翻转明细。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from app.eval.dataset import gold_chunk_ids
from app.eval.metrics import compute_metrics
from app.eval.retrieval import run_fast
from app.utils.batch_manager.embedder import agenerate_embeddings

_DECODER = json.JSONDecoder()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="问题重写离线检索消融")
    parser.add_argument("--metrics", type=Path, default=root / "logs" / "metrics.log")
    parser.add_argument(
        "--results",
        type=Path,
        default=root / "data" / "eval_multi_turn" / "multi_user_agent_results.json",
    )
    return parser.parse_args(argv)


def _load_metrics(path: Path) -> dict[str, dict]:
    """按 request_id 索引打点记录（metrics.log 是连续 JSON 对象拼接的流）。"""
    text = path.read_text(encoding="utf-8")
    index = 0
    out: dict[str, dict] = {}
    while index < len(text):
        while index < len(text) and text[index] in " \n\r\t":
            index += 1
        if index >= len(text):
            break
        record, index = _DECODER.raw_decode(text, index)
        out[record["request_id"]] = record
    return out


def _rrf_merge(*id_lists: list[str], k: int = 60, top: int = 5) -> list[str]:
    """两路召回做 RRF 融合后截断（与线上深度路径的融合口径一致）。"""
    scores: dict[str, float] = {}
    for ids in id_lists:
        for position, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    return [chunk_id for chunk_id, _ in ranked[:top]]


async def _retrieve(query: str, textbook: str, cache: dict[tuple[str, str], list[str]]) -> list[str]:
    """走生产快速路径（dense/sparse 0.8/0.2，top_k=5），同一问句只查一次。"""
    key = (query, textbook)
    if key not in cache:
        cache[key] = (await run_fast(query, textbook))["retrieved_ids"]
    return cache[key]


async def _run(args: argparse.Namespace) -> None:
    metrics_by_request = _load_metrics(args.metrics)
    turns = json.loads(args.results.read_text(encoding="utf-8"))["per_turn"]
    # 预热 embedding 模型，避免冷加载污染首条耗时
    await agenerate_embeddings(["预热"])

    samples = []
    for turn in turns:
        record = metrics_by_request.get(turn.get("request_id"))
        if not record or record.get("rewrite_mode") != "llm":
            continue
        rewritten = (record.get("rewrite_query") or "").strip()
        if rewritten:
            samples.append((turn, rewritten))
    print(f"参与消融的轮数（rewrite_mode=llm 且重写非空）: {len(samples)}")

    cache: dict[tuple[str, str], list[str]] = {}
    columns: dict[str, dict[str, list[float]]] = {
        role: defaultdict(list) for role in ("原问句", "重写问句", "RRF融合", "线上实际")
    }
    mrr: dict[str, list[float]] = defaultdict(list)
    lost, rescued = [], []

    for turn, rewritten in samples:
        role = turn.get("turn_role") or "initial"
        gold = set(gold_chunk_ids(turn))
        original_ids = await _retrieve(turn["question"], turn["textbook"], cache)
        rewritten_ids = await _retrieve(rewritten, turn["textbook"], cache)
        rows = {
            "原问句": compute_metrics(gold, original_ids),
            "重写问句": compute_metrics(gold, rewritten_ids),
            "RRF融合": compute_metrics(gold, _rrf_merge(original_ids, rewritten_ids)),
            "线上实际": compute_metrics(gold, turn["retrieved_chunk_ids"]),
        }
        for name, row in rows.items():
            columns.setdefault(name, defaultdict(list))[f"{role}"].append(row["hit@5"])
        mrr[f"{role} 原问句"].append(rows["原问句"]["mrr"])
        mrr[f"{role} 重写问句"].append(rows["重写问句"]["mrr"])
        if rows["原问句"]["hit@5"] and not rows["重写问句"]["hit@5"]:
            lost.append((turn, rewritten))
        if rows["重写问句"]["hit@5"] and not rows["原问句"]["hit@5"]:
            rescued.append((turn, rewritten))

    print("\n### hit@5：单用原问句 / 单用重写问句 / 两路 RRF / 线上实际")
    for role in sorted({key.split()[0] for key in mrr}):
        sizes = len(columns["原问句"][role])
        print(
            f"  {role:<12} n={sizes:3d}  "
            + "  ".join(f"{name} {st.mean(columns[name][role]):.3f}" for name in columns)
            + f"   MRR 原 {st.mean(mrr[f'{role} 原问句']):.3f} / 重写 {st.mean(mrr[f'{role} 重写问句']):.3f}"
        )

    print(f"\n### 翻转明细：重写掉出 top5 {len(lost)} 例 / 救回 top5 {len(rescued)} 例")
    for label, group in (("丢", lost), ("救", rescued)):
        for turn, rewritten in group[:6]:
            print(f"  [{label}] {turn['question'][:50]} | {rewritten[:50]}")


def main(argv: list[str] | None = None) -> int:
    asyncio.run(_run(_parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
