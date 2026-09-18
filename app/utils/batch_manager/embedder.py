"""BatchEmbedder：动态攒批器

把并发小请求攒成一批，一次前向再按序拆分回填（单实例 + 动态 batching）。
底层前向由 ``app.utils.embedding_util.generate_embeddings`` 提供（BGE-M3，
模型单例在 util 模块内部维护）。
"""

from __future__ import annotations

import asyncio

from app.core import astage
from app.utils.embedding_util import generate_embeddings

# 动态攒批参数：单卡 RTX 5060（8GB）+ BGE-M3（fp16，权重约 2.2GB）。
# max_batch 控制「并发小请求攒到多少条文本就前向」；摄入的 64 条满批不受此上限
# 截断（drainer 拿到第一条即满批，直接单发）。max_wait 是攒批等待上限（10ms），
# 用几毫秒延迟换取 GPU 吞吐。
_MAX_BATCH = 32
_MAX_WAIT_S = 0.01


class BatchEmbedder:
    """把并发小请求攒成一批，一次前向再按序拆分回填（单实例 + 动态 batching）。

    单实例模型只有一个 drainer 任务串行调用 ``generate_embeddings``，天然线程安全，
    因此 ``generate_embeddings`` 不再需要锁。调用方视角与旧版 ``agenerate_embeddings``
    完全一致：提交文本 → await 返回自己的 dense/sparse。
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

    async def encode(self, texts: list[str]) -> dict:
        """提交一批文本，挂起等待本请求被拆分回填的结果。"""
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((texts, fut))
        return await fut

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            texts, fut = await self._queue.get()          # 拿到第一条即开窗
            items: list[tuple[list[str], asyncio.Future]] = [(texts, fut)]
            flat: list[str] = list(texts)
            deadline = loop.time() + self._max_wait
            while len(flat) < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                # 带超时的阻塞抓取：真正「等满 max_wait」凑批，错峰几毫秒的请求也能并入
                try:
                    t, f = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                items.append((t, f))
                flat.extend(t)

            emb = await asyncio.to_thread(generate_embeddings, flat)  # 一次 GPU 前向
            off = 0
            for t, f in items:                             # 按各请求边界拆分回填
                n = len(t)
                f.set_result(
                    {
                        "dense": emb["dense"][off : off + n],
                        "sparse": emb["sparse"][off : off + n],
                    }
                )
                off += n


# 模块级单例：全进程共享一个攒批器（与 _get_model 的模型单例一一对应）
_embedder = BatchEmbedder()


async def agenerate_embeddings(texts: list[str]) -> dict:
    """异步版 generate_embeddings：提交到动态攒批器，攒批后一次前向。

    适用多用户并发场景——把并发到达的小请求（1~6 条）攒成一批再一次 forward，
    避免逐条前向造成的 GPU 利用率低下。模型前向仍由 ``to_thread`` 移出事件循环；
    串行化不再依赖锁，改由 drainer 单任务串行调用 ``generate_embeddings`` 保证。

    注意：耗时埋点 ``embedding_ms`` 在协程层记录（``astage``），涵盖「排队 + 前向」；
    ``to_thread`` 会隔离 contextvar，同步函数内无法写入请求指标上下文。

    Args:
        texts: 待编码文本列表。

    Returns:
        同 generate_embeddings 的 ``{"dense": ..., "sparse": ...}`` 结构。
    """
    async with astage("embedding_ms"):
        return await _embedder.encode(texts)
