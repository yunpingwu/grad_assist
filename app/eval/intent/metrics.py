"""意图识别评测指标：准确率 / 混淆矩阵 / 分来源与分组统计 / 弃权率 / 时延分位数。

与 ``metrics.py``（检索排序指标）刻意分文件：那边只依赖「黄金 id 集合 + 有序召回列表」，
这里依赖「可接受意图集合 + 单标签预测」，行结构与关注点都不同，合并只会互相牵制。

所有函数都是纯函数，输入统一为「评测行」字典列表，行约定：
    ``expected``: 可接受意图列表（多数样本单值；边界样本双值如 ["explain", "generate"]）
    ``intent`` / ``source`` / ``confidence`` / ``ms``: 检测器产出的预测意图、命中档位、置信度与耗时
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# 报告展示顺序：意图全集 + 兜底降级（source 维度另行统计）
LABELS = ("explain", "generate", "quiz", "plan", "chat", "unclear")


def is_correct(row: dict[str, Any]) -> bool:
    """单行判定：预测意图是否落在可接受集合内。"""
    return row.get("intent") in set(row.get("expected") or [])


def accuracy(rows: list[dict[str, Any]]) -> float:
    """整体准确率：正确行数 / 总行数；空列表返回 0。"""
    if not rows:
        return 0.0
    return sum(is_correct(r) for r in rows) / len(rows)


def confusion_matrix(rows: list[dict[str, Any]], labels: tuple[str, ...] = LABELS) -> dict[str, dict[str, int]]:
    """混淆计数：``{金标: {预测: 次数}}``，仅统计首个可接受意图作为金标（多值边界行记第一值）。

    预测/金标出现 labels 之外的值时原样保留键名，避免统计丢行。
    """
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        gold = (row.get("expected") or ["?"])[0]
        pred = row.get("intent") or "?"
        matrix.setdefault(gold, Counter())[pred] += 1
    order = [label for label in labels if label in matrix] + [k for k in matrix if k not in labels]
    return {gold: dict(matrix[gold]) for gold in order}


def group_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """按行内某字段（如 ``group`` / ``source``）分组的 n / correct / accuracy。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key) or "?"), []).append(row)
    return {
        name: {"n": len(sub), "correct": sum(is_correct(r) for r in sub), "accuracy": accuracy(sub)}
        for name, sub in buckets.items()
    }


def per_class_stats(rows: list[dict[str, Any]], labels: tuple[str, ...] = LABELS) -> dict[str, dict[str, float]]:
    """逐意图 precision / recall / F1（金标取可接受集合的并，预测为单标签）。

    双值边界行会同时进入两个类别的金标集（都判对才算各自命中），与单标签多分类的
    micro 惯例一致；样本量小时看混淆矩阵更直观，此表只做速览。
    """
    stats: dict[str, dict[str, float]] = {}
    seen = {label for row in rows for label in (row.get("expected") or [])} | {r["intent"] for r in rows if r.get("intent")}
    names = [label for label in labels if label in seen] + sorted(seen - set(labels))
    for label in names:
        gold_rows = [r for r in rows if label in (r.get("expected") or [])]
        tp = sum(1 for r in gold_rows if r.get("intent") == label)
        pred_rows = [r for r in rows if r.get("intent") == label]
        precision = tp / len(pred_rows) if pred_rows else 0.0
        recall = tp / len(gold_rows) if gold_rows else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[label] = {
            "gold_n": len(gold_rows),
            "pred_n": len(pred_rows),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return stats


def abstention_rate(rows: list[dict[str, Any]]) -> float:
    """弃权率：金标不含 unclear 却被判成 unclear 的比例（错误拒答，最值得盯的坏味道）。"""
    should_answer = [r for r in rows if "unclear" not in (r.get("expected") or [])]
    if not should_answer:
        return 0.0
    return sum(1 for r in should_answer if r.get("intent") == "unclear") / len(should_answer)


def latency_percentiles(values: list[float], ps: tuple[int, ...] = (50, 95)) -> dict[str, float]:
    """最近秩法分位数（评测样本小，不必插值）；空列表返回各档位 0。"""
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for p in ps:
        rank = max(1, min(len(ordered), -(-p * len(ordered) // 100)))  # ceil(p%n) 最近秩
        out[f"p{p}"] = ordered[rank - 1]
    return out


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """一次性聚合全部指标：整体准确率、分组/分来源、逐类、混淆矩阵、弃权率、时延。"""
    return {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "abstention_rate": abstention_rate(rows),
        "by_group": group_accuracy(rows, "group"),
        "by_source": group_accuracy(rows, "source"),
        "per_class": per_class_stats(rows),
        "confusion": confusion_matrix(rows),
        "latency_ms": latency_percentiles([float(r["ms"]) for r in rows if r.get("ms") is not None]),
    }
