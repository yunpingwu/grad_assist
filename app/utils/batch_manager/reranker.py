"""BatchReranker：动态攒批精排器

把并发到来的交叉编码请求攒成一批 (query, text) 对，一次前向再按序拆分回填。
底层打分由 ``app.utils.reranker_util._score_pairs`` 提供（BGE-Reranker，
模型单例在 util 模块内部维护）。
"""

from __future__ import annotations

import asyncio

from app.utils.reranker_util import _score_pairs

# 动态攒批参数：单卡 RTX 5060（8GB）+ BGE-Reranker（fp16）。max_batch 按 (query, text)
# 对数量计；精排单请求候选通常少于 max_batch，满批时直接单发不截断。
_MAX_BATCH = 32
_MAX_WAIT_S = 0.01


class BatchReranker:
    """把并发到来的交叉编码请求攒成一批 (query, text) 对，一次前向再按序拆分回填。

    单实例模型只有一个 drainer 任务串行调用 ``_score_pairs``，天然线程安全，
    因此 ``_score_pairs`` 不再需要锁。调用方视角：提交 (query, texts) → await 得到
    与 texts 等长的分数列表。
    """

    def __init__(self, max_batch: int = _MAX_BATCH, max_wait_s: float = _MAX_WAIT_S):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._max_batch = max_batch
        self._max_wait = max_wait_s
        self._task: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        """幂等启动常驻 drainer（绑定到调用方所在事件循环）。"""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())

    async def encode(self, query: str, texts: list[str]) -> list[float]:
        """提交一个精排请求（单 query + 多个候选），挂起等待自己的分数。"""
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((query, texts, fut))
        return await fut

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            query, texts, fut = await self._queue.get()   # 拿到第一条即开窗
            items: list[tuple[str, list[str], asyncio.Future]] = [(query, texts, fut)]
            flat_pairs: list[tuple[str, str]] = [(query, t) for t in texts]
            deadline = loop.time() + self._max_wait
            while len(flat_pairs) < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                # 带超时的阻塞抓取：等满 max_wait 凑批，错峰请求也能并入
                try:
                    q, ts, f = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                items.append((q, ts, f))
                flat_pairs.extend((q, t) for t in ts)

            scores = await asyncio.to_thread(_score_pairs, flat_pairs)  # 一次前向
            off = 0
            for q, ts, f in items:                       # 按各请求候选数拆分回填
                n = len(ts)
                f.set_result(scores[off : off + n])
                off += n


# 模块级单例：全进程共享一个攒批器（与 _get_reranker 的模型单例一一对应）
_reranker_batcher = BatchReranker()


async def arerank_scores(query: str, texts: list[str]) -> list[float]:
    """异步版 compute_rerank_scores：提交到动态攒批器，攒批后一次前向。

    多用户并发精排时，把多个请求的 (query, text) 对合并成一批，避免逐请求前向。
    返回与 ``texts`` 等长的分数列表；空 ``texts`` 抛 ValueError。
    """
    if not texts:
        raise ValueError("texts 必须是非空列表")
    return await _reranker_batcher.encode(query, texts)
