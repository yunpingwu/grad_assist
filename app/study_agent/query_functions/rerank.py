"""Rerank 精排工具函数：对 RRF 融合后的候选片段做交叉编码重排序，截取 TOP-K。

从 query_functions 收编而来：仅保留纯函数 rerank_chunks（原节点封装已移除）。

- 打分由 ``app.utils.reranker_util.compute_rerank_scores`` 提供（BGE-Reranker），
  模型单例在 util 模块内部维护，本函数只拿分数、不接触模型实例；
- 候选仅 ≤10 条，本地 CPU 精排约 0.5~2s，CUDA 更快；任何失败由调用方降级回退。
"""

from __future__ import annotations

import asyncio

from app.core import logger
from app.utils.reranker_util import compute_rerank_scores


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """用交叉编码模型对候选片段精排。

    Args:
        query: 重写后的查询（与召回的语义对齐）。
        chunks: RRF 融合后的候选 hit 列表。
        top_k: 精排后保留的 TOP-K 片段数。

    Returns:
        按精排分数降序的 TOP-K hit 列表（附 rerank_score）。

    Raises:
        ValueError: 无候选或模型不可用。
    """
    if not chunks:
        raise ValueError("无候选片段可精排")
    if not query:
        raise ValueError("精排查询为空")

    # 提取片段文本（Milvus hit 的 entity.text）
    texts = []
    for hit in chunks:
        entity = hit.get("entity") or hit
        texts.append((entity.get("text") or "").strip())

    # 交叉编码打分（sigmoid 归一化，越大越相关；模型单例封装在 util 公共 API 内）
    scores = compute_rerank_scores(query, texts)
    logger.info(f"Rerank 完成：{len(chunks)} 条候选 → TOP{top_k}，分数范围 {min(scores):.3f} ~ {max(scores):.3f}")

    # 把分数挂到对应 hit 上 → 降序排序 → 截取 TOP-K
    ranked = []
    for hit, score in zip(chunks, scores, strict=True):
        item = dict(hit)  # 浅拷贝，避免污染调用方的原始 hit
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]


async def arerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """异步版 rerank_chunks：把交叉编码精排丢到线程池，避免阻塞事件循环。

    与 ``rerank_chunks`` 结果一致，仅把同步计算移出事件循环；模型前向的线程串行化
    由 ``compute_rerank_scores`` 内部的 ``_rerank_lock`` 保证。耗时埋点由调用方在
    协程层用 ``astage`` 记录。

    Args:
        query: 重写后的查询。
        chunks: RRF 融合后的候选 hit 列表。
        top_k: 精排后保留的 TOP-K 片段数。

    Returns:
        按精排分数降序的 TOP-K hit 列表。
    """
    return await asyncio.to_thread(rerank_chunks, query, chunks, top_k)


# 单元测试：用桩分数替换公共打分 API，避免单测加载真实模型
if __name__ == "__main__":
    def _fake_scores(query: str, texts: list[str]) -> list[float]:
        return [0.9, 0.1, 0.8]

    compute_rerank_scores = _fake_scores  # 覆盖模块级导入的绑定，只验证内联精排逻辑

    def _hit(doc_id: str, text: str) -> dict:
        return {"id": doc_id, "distance": 0.5, "entity": {"text": text}}

    chunks = [
        _hit("doc_a", "指针是C语言中用于存储变量地址的变量。"),
        _hit("doc_b", "数组是一组相同类型元素的集合。"),
        _hit("doc_c", "通过指针可以直接访问内存地址。"),
    ]
    # 模拟交叉编码打分：与"指针"相关的 doc_a/doc_c 应排到前面
    ranked = rerank_chunks("什么是指针？", chunks, top_k=2)
    assert len(ranked) == 2, f"应截取 TOP2，实际 {len(ranked)}"
    assert [h["id"] for h in ranked] == ["doc_a", "doc_c"], f"排序不正确: {ranked}"
    assert ranked[0]["rerank_score"] == 0.9, "rerank_score 未正确挂载"
    print(f"精排顺序: {[h['id'] for h in ranked]}")
    print("rerank 测试通过")