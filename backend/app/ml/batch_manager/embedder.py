"""BatchEmbedder：把并发小请求攒批后一次前向，再按请求边界拆分回填。"""

from __future__ import annotations

import asyncio

from app.ml.embedding_util import generate_embeddings
from app.trace import astage

# 攒批参数：max_batch 是并发小请求的合批上限（满批直接单发），max_wait 是等待窗口，
# 用几毫秒延迟换取 GPU 吞吐。
_MAX_BATCH = 32
_MAX_WAIT_S = 0.01


class BatchEmbedder:
    """把并发小请求攒成一批，一次前向再按序拆分回填。

    单实例只有一个 drainer 串行调用模型，天然线程安全，无需额外加锁。
    """

    def __init__(self, max_batch: int = _MAX_BATCH, max_wait_s: float = _MAX_WAIT_S):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._max_batch = max_batch
        self._max_wait = max_wait_s
        self._task: asyncio.Task | None = None

    async def encode(self, texts: list[str]) -> dict:
        """提交一批文本，挂起等待本请求被拆分回填的结果。"""
        # 幂等启动常驻 drainer（绑定到调用方所在事件循环）
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())
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
                except TimeoutError:
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

    耗时埋点 ``embedding_ms`` 在协程层记录，涵盖排队 + 前向。
    """
    async with astage("embedding_ms"):
        return await _embedder.encode(texts)
