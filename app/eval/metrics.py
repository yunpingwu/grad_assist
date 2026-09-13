"""检索评测指标：Recall@k / Hit@k / MRR / NDCG@k 的纯函数实现与聚合。

所有函数与召回内容无关，仅依赖“黄金 id 集合 + 检索返回的 id 有序列表”，
便于单测与复现。分级相关性按二值处理（命中黄金 id 记 1，否则 0）。
"""

from __future__ import annotations

import math

# 默认评测截断档位（Top-K 深度）
DEFAULT_KS = (1, 3, 5)


def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Recall@k：Top-k 中命中的黄金文档数 / 黄金文档总数。

    Args:
        gold: 黄金 chunk id 集合。
        retrieved: 检索返回的 id 有序列表。
        k: 截断深度。

    Returns:
        [0, 1] 之间的召回率；黄金集为空返回 0。
    """
    if not gold:
        return 0.0
    top = retrieved[:k]
    return sum(1 for gid in gold if gid in top) / len(gold)


def hit_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Hit@k：Top-k 中是否至少命中一条黄金文档（0/1）。"""
    return 1.0 if any(gid in retrieved[:k] for gid in gold) else 0.0


def mean_reciprocal_rank(gold: set[str], retrieved: list[str]) -> float:
    """MRR：首个相关结果排名的倒数的均值（单条问题即 1 / 首个命中排名）。"""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """NDCG@k：二值相关性的折扣累计增益。

    相关性 rel=1 表示该文档命中黄金集。DCG 使用 ``gain / log2(i+1)`` 折扣，
    理想排序为黄金文档全部排在最前的排列（截断到 min(|gold|, k)）。

    Args:
        gold: 黄金 chunk id 集合。
        retrieved: 检索返回的 id 有序列表。
        k: 截断深度。

    Returns:
        [0, 1] 之间的归一化折扣累计增益。
    """
    top = retrieved[:k]
    gains = [1.0 if doc_id in gold else 0.0 for doc_id in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = [1.0] * min(len(gold), k)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(gold: set[str], retrieved: list[str], ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """计算单条问题的全部指标（各 k 档 Recall/Hit + MRR + NDCG@k）。

    Args:
        gold: 黄金 chunk id 集合。
        retrieved: 检索返回的 id 有序列表。
        ks: 要计算的截断档位。

    Returns:
        指标字典，如 ``{"recall@1": ..., "hit@5": 1.0, "mrr": ..., "ndcg@5": ...}``。
    """
    out: dict = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(gold, retrieved, k)
        out[f"hit@{k}"] = hit_at_k(gold, retrieved, k)
        out[f"ndcg@{k}"] = ndcg_at_k(gold, retrieved, k)
    out["mrr"] = mean_reciprocal_rank(gold, retrieved)
    return out


def aggregate_metrics(rows: list[dict]) -> dict:
    """对多条问题的指标行求均值，并附样本数。

    Args:
        rows: 每条问题一行的指标字典（compute_metrics 的返回值）。

    Returns:
        聚合字典：各指标均值 + ``"n"`` 样本数。
    """
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if k not in {"query_id"}]
    agg = {"n": len(rows)}
    for key in keys:
        agg[key] = sum(r[key] for r in rows) / len(rows)
    return agg
