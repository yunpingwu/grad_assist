"""消融实验入口：跑 C3（混合）→ C4（RRF+HyDE）→ C5（精排）并输出指标表。

用法（在项目根目录）::

    .venv\\Scripts\\python.exe -m app.eval.run_ablation

结果同时打印到控制台，并写入 ``data/eval/results.json``（含逐题明细与均值）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.core import logger
from app.eval.dataset import gold_chunk_ids, load_dataset
from app.eval.metrics import compute_metrics
from app.eval.retrieval import run_deep, run_fast

_RESULT_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "results.json"

# 参与对比的配置：{配置名: 结果来源键}（c3 走快速路径，c4/c5 走深度路径）
_CONFIGS: list[tuple[str, str]] = [
    ("C3_hybrid", "c3"),
    ("C4_rrf_hyde", "c4"),
    ("C5_rerank", "c5"),
]

# 逐题行中要展示/写入的指标
_METRICS = ["recall@1", "recall@3", "recall@5", "hit@5", "mrr", "ndcg@5"]
# 深度路径分阶段耗时字段
_STAGES = ["embed", "hyde", "rrf", "rerank", "total"]


async def _eval_one(question: dict, textbook: str) -> dict:
    """对单条问题跑三个配置，返回逐题指标行（含分阶段耗时）。"""
    query = question["question"]
    gold = gold_chunk_ids(question)

    fast = await run_fast(query, textbook)
    deep = await run_deep(query, textbook)

    row: dict = {"query_id": question["id"]}
    for label, key in _CONFIGS:
        retrieved = fast["retrieved_ids"] if key == "c3" else deep[key]["retrieved_ids"]
        for name, value in compute_metrics(gold, retrieved).items():
            row[f"{label}_{name}"] = value

    row["latency_c3"] = fast["latency_s"]
    for stage in _STAGES:
        row[f"latency_{stage}"] = deep["latency"][stage]
    return row


def _mean(rows: list[dict], key: str) -> float:
    """取某字段跨问题的均值。"""
    return sum(r[key] for r in rows) / len(rows)


def _print_table(rows: list[dict]) -> None:
    """按配置分组打印指标均值对比表与平均耗时。"""
    print("\n================ RAG 离线评测（检索侧）================\n")
    header = "指标".ljust(10) + "".join(f"{label:>16}" for label, _ in _CONFIGS)
    print(header)
    print("-" * len(header))

    for metric in _METRICS:
        line = metric.ljust(10)
        for label, _ in _CONFIGS:
            line += f"{_mean(rows, f'{label}_{metric}'):>16.4f}"
        print(line)
    print("-" * len(header))

    print("\n平均耗时（秒）:")
    print(f"  C3 端到端     : {_mean(rows, 'latency_c3'):.2f}")
    stage_names = {"embed": "混合召回", "hyde": "HyDE 生成+召回", "rrf": "RRF 融合", "rerank": "精排", "total": "深度路径总计"}
    for stage in _STAGES:
        print(f"  {stage_names[stage]:<10}: {_mean(rows, f'latency_{stage}'):>8.2f}")


def _build_summary(rows: list[dict]) -> dict:
    """由逐题行计算各配置的指标均值，作为 JSON 的 summary 段。"""
    summary: dict = {}
    for label, _ in _CONFIGS:
        summary[label] = {metric: _mean(rows, f"{label}_{metric}") for metric in _METRICS}
        summary[label]["n"] = len(rows)
    return summary


async def main() -> None:
    data = load_dataset()
    textbook = data["textbook"]
    questions = data["questions"]
    logger.info(f"开始消融评测: 教材「{textbook}」, 共 {len(questions)} 条问题")

    rows: list[dict] = []
    for question in questions:
        try:
            rows.append(await _eval_one(question, textbook))
        except Exception as exc:  # 单条失败不阻断整体，记录后继续
            logger.error(f"问题 {question['id']} 评测失败: {exc}")

    if not rows:
        logger.error("无可用评测结果，终止")
        return

    _print_table(rows)

    payload = {
        "textbook": textbook,
        "configs": [label for label, _ in _CONFIGS],
        "per_query": rows,
        "summary": _build_summary(rows),
    }
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已写入 {_RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
