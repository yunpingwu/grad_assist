"""检索侧消融实验入口：dense/sparse/权重扫描/HyDE/RRF/rerank 逐项对比。

用法（在项目根目录）::

    .venv\\Scripts\\python.exe -m app.eval.run_ablation             # 默认评测集（见下方 _DEFAULT_DATASET_PATH 常量）
    .venv\\Scripts\\python.exe -m app.eval.run_ablation "data/eval/数据结构 (陈越、何钦铭、徐镜春、魏宝刚、杨枨编)/qa_set.json"          # 指定评测集
    .venv\\Scripts\\python.exe -m app.eval.run_ablation "data/eval/C语言程序设计（第五版）_(谭浩强)/qa_set.json"  # C语言困难集

结果同时打印到控制台，并写入评测集所在目录下的 ``results.json``
（例如 ``data/eval/XXX/qa_set.json`` 的结果写到 ``data/eval/XXX/results.json``，
结果文件名可改下方 ``_RESULT_FILENAME`` 常量）。

配置链（同一评测集、同一教材集合上跑）：
- dense       : 单路稠密检索（= 权重 1.0/0.0 的极端）
- sparse      : 单路稀疏/词面检索（= 权重 0.0/1.0 的极端）
- hyb_50/70/80/90 : dense+sparse 加权混合，权重 (0.5,0.5)/(0.7,0.3)/(0.8,0.2)/(0.9,0.1)
- rrf_hyde      : 混合召回 + HyDE 第二路 → RRF(k=60) 融合后截断
- rerank        : 对 rrf_hyde 融合候选池（每路 limit=20）做交叉编码精排后截断

由此可直接读出：dense vs sparse vs 混合、0.8/0.2 是否最优、HyDE+RRF 增益、
rerank 前后命中率变化，以及各配置平均检索延迟。

注意点：
- 正式跑题前会先 ``_warmup`` 预热 BGE-M3 / BGE-Reranker 冷加载，避免模型
  首次数秒加载污染第一个配置（dense）的耗时统计；
- 深度路径的候选池远大于生产（生产 limit=5 融合池 ≤10），精排候选空间更大。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.core import logger
from app.eval.dataset import gold_chunk_ids, load_dataset
from app.eval.metrics import compute_metrics
from app.eval.retrieval import run_deep, run_dense, run_hybrid, run_hyde_rrf, run_sparse
from app.utils.batch_manager.embedder import agenerate_embeddings
from app.utils.reranker_util import compute_rerank_scores

# ===== 评测集路径常量：改这里即可切换默认评测的套集 =====
# 也可运行时传第一个位置参数（如 python -m app.eval.run_ablation "data/eval/xxx/qa_set.json"）
_DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "eval" / "操作系统：精髓与设计原理（第8版）_(斯托林斯)" / "qa_set.json"
)

# 结果文件名：每次运行的结果写到评测集所在目录下（数据与结果不混目录）
_RESULT_FILENAME = "results.json"

# 参与对比的配置标签（顺序即输出列顺序）
_CONFIG_LABELS = [
    "dense",
    "sparse",
    "hyb_50",
    "hyb_70",
    "hyb_80",
    "hyb_90",
    "rrf_hyde",
    "rerank",
]
# 混合权重扫描点：(标签, dense_weight)；sparse 权重 = 1 - dense_weight
_WEIGHT_SWEEPS: list[tuple[str, float]] = [
    ("hyb_50", 0.5),
    ("hyb_70", 0.7),
    ("hyb_80", 0.8),
    ("hyb_90", 0.9),
]
# 逐题行中要展示/写入的指标（仅 @5 档）
_METRICS = ["recall@5", "mrr@5", "ndcg@5", "hit@5"]
# 深度路径配置标签；key 与 _CONFIG_LABELS 尾部一致（模块加载即校验）
_DEEP_STAGE_LABELS = ["rrf_hyde", "rerank"]
assert _CONFIG_LABELS[-len(_DEEP_STAGE_LABELS):] == _DEEP_STAGE_LABELS, (
    "_CONFIG_LABELS 尾部标签与 _DEEP_STAGE_LABELS 不一致，请同步修改"
)
# 深度路径分阶段耗时字段
_STAGES = ["embed", "hyde", "rrf", "rerank", "total"]


def _compute_at5(gold: set[str], retrieved: list[str]) -> dict:
    """计算单配置指标，只保留 @5 档（compute_metrics 默认按 DEFAULT_KS 出 @1/@3/@5，
    还附带全表 mrr；评测只消费 @5，剔除其余字段避免 results.json 冗余）。"""
    return {
        name: value
        for name, value in compute_metrics(gold, retrieved, ks=(5,)).items()
        if name.endswith("@5")
    }


async def _eval_one(question: dict, textbook: str) -> dict:
    """对单条问题跑全部配置，返回逐题指标行（含各配置延迟）。"""
    query = question["question"]
    gold = gold_chunk_ids(question)
    row: dict = {"query_id": question["id"], "gold_size": len(gold)}

    # dense / sparse 单路基线
    dense = await run_dense(query, textbook)
    sparse = await run_sparse(query, textbook)
    for label, res in (("dense", dense), ("sparse", sparse)):
        for name, value in _compute_at5(gold, res["retrieved_ids"]).items():
            row[f"{label}_{name}"] = value
        row[f"{label}_latency_s"] = res["latency_s"]

    # 加权混合权重扫描
    for label, dense_w in _WEIGHT_SWEEPS:
        res = await run_hybrid(query, textbook, weights=(dense_w, 1.0 - dense_w))
        for name, value in _compute_at5(gold, res["retrieved_ids"]).items():
            row[f"{label}_{name}"] = value
        row[f"{label}_latency_s"] = res["latency_s"]

    # 深度路径：先跑 HyDE+RRF，再把同一批融合候选传给 rerank，
    # 保证 rrf_hyde 与 rerank 基于相同候选集（rerank 只重排，不换文档）
    rrf_hyde = await run_hyde_rrf(query, textbook)
    deep = await run_deep(query, textbook, base=rrf_hyde)
    sources = {
        "rrf_hyde": rrf_hyde["rrf_hyde"]["retrieved_ids"],
        "rerank": deep["rerank"]["retrieved_ids"],
    }
    for label in _DEEP_STAGE_LABELS:
        for name, value in _compute_at5(gold, sources[label]).items():
            row[f"{label}_{name}"] = value
    for stage in _STAGES:
        row[f"deep_{stage}_s"] = deep["latency"][stage]
    return row


def _mean(rows: list[dict], key: str) -> float:
    """取某字段跨问题的均值。"""
    return sum(r[key] for r in rows) / len(rows)


async def _warmup(textbook: str) -> None:
    """预热本地模型冷加载，避免污染首个配置的耗时统计。

    BGE-M3 与 BGE-Reranker 均为懒加载单例：第一次前向耗时数秒（CUDA 上约
    1~4s）。在正式评测计时前各触发一次，让其只出现在进程冷启动阶段，
    不计入任何配置的指标行。
    """
    logger.info("预热本地模型（不计入评测指标）...")
    # 触发 BGE-M3 加载 + 一次真实检索路径（Milvus/集合定位也跟着就绪）
    await agenerate_embeddings(["预热文本"])
    # 触发 BGE-Reranker 加载（同步打分一次）
    compute_rerank_scores("预热文本", ["预热文本"])
    logger.info("预热完成")


def _print_table(rows: list[dict]) -> None:
    """按配置分组打印指标均值对比表与平均耗时。"""
    print("\n============ RAG 离线评测（检索侧消融）============\n")
    header = "指标".ljust(10) + "".join(f"{label:>14}" for label in _CONFIG_LABELS)
    print(header)
    print("-" * len(header))
    for metric in _METRICS:
        line = metric.ljust(10)
        for label in _CONFIG_LABELS:
            line += f"{_mean(rows, f'{label}_{metric}'):>14.4f}"
        print(line)
    print("-" * len(header))

    print("\n平均检索耗时（秒，含 embedding；深度路径另列分阶段）:")
    for label in _CONFIG_LABELS[:6]:  # dense/sparse/hyb_*
        print(f"  {label:<12}: {_mean(rows, f'{label}_latency_s'):.3f}")
    stage_names = {
        "embed": "深度-混合召回",
        "hyde": "深度-HyDE 生成+召回",
        "rrf": "深度-RRF 融合",
        "rerank": "深度-精排",
        "total": "深度-端到端",
    }
    for stage in _STAGES:
        print(f"  {stage_names[stage]:<12}: {_mean(rows, f'deep_{stage}_s'):.3f}")


def _build_summary(rows: list[dict]) -> dict:
    """由逐题行计算各配置的指标均值，作为 JSON 的 summary 段。"""
    summary: dict = {}
    for label in _CONFIG_LABELS:
        summary[label] = {metric: _mean(rows, f"{label}_{metric}") for metric in _METRICS}
        if label in ("dense", "sparse") or label.startswith("hyb_"):
            summary[label]["latency_s"] = _mean(rows, f"{label}_latency_s")
        summary[label]["n"] = len(rows)
    summary["deep_latency"] = {stage: _mean(rows, f"deep_{stage}_s") for stage in _STAGES}
    return summary


async def main(dataset_path: str | None = None) -> None:
    ds_path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET_PATH
    data = load_dataset(ds_path)
    textbook = data["textbook"]
    questions = data["questions"]
    logger.info(f"开始检索侧消融评测: 教材「{textbook}」, 共 {len(questions)} 条问题")

    await _warmup(textbook)

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
        "configs": _CONFIG_LABELS,
        "per_query": rows,
        "summary": _build_summary(rows),
    }
    result_path = ds_path.parent / _RESULT_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已写入 {result_path}")


if __name__ == "__main__":
    # 可选参数：评测集路径，缺省用 _DEFAULT_DATASET_PATH
    ds_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(ds_path))
