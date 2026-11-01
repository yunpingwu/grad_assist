"""检索评测指标：Recall@k / Hit@k / MRR / NDCG@k 的纯函数与聚合。"""

from __future__ import annotations

import math

# 默认评测截断档位（Top-K 深度）
DEFAULT_KS = (1, 3, 5)


def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Recall@k：Top-k 命中的黄金文档数 / 黄金文档总数；黄金集为空返回 0。"""
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


def mean_reciprocal_rank_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """MRR@k：首个相关结果排名倒数的均值，仅统计前 k 位（超出 k 记 0）。"""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """NDCG@k：二值相关性（命中黄金集记 1）的归一化折扣累计增益。"""
    top = retrieved[:k]
    gains = [1.0 if doc_id in gold else 0.0 for doc_id in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = [1.0] * min(len(gold), k)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(gold: set[str], retrieved: list[str], ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """计算单条问题的全部指标（各 k 档 Recall/Hit/NDCG + MRR）。"""
    out: dict = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(gold, retrieved, k)
        out[f"hit@{k}"] = hit_at_k(gold, retrieved, k)
        out[f"ndcg@{k}"] = ndcg_at_k(gold, retrieved, k)
        out[f"mrr@{k}"] = mean_reciprocal_rank_at_k(gold, retrieved, k)
    out["mrr"] = mean_reciprocal_rank(gold, retrieved)
    return out


def aggregate_metrics(rows: list[dict]) -> dict:
    """对多条问题的指标行求均值，并附样本数 ``n``。"""
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if k not in {"query_id"}]
    agg = {"n": len(rows)}
    for key in keys:
        agg[key] = sum(r[key] for r in rows) / len(rows)
    return agg
