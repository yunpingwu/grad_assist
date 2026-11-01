"""BGE-Reranker 交叉编码打分：单例加载、同步打分，以及并发请求的动态攒批。

攒批器与模型单例都在本模块内，二者一一对应；并发精排走 ``arerank_scores``，
预热等单处调用走 ``compute_rerank_scores``。
"""

from __future__ import annotations

import asyncio
import os
import time

from FlagEmbedding import FlagReranker

from app.config import rerank_config
from app.core import logger
from app.trace import mark_stage

# 动态攒批参数：单卡 RTX 5060（8GB）+ BGE-Reranker（fp16）。max_batch 按 (query, text)
# 对数量计；精排单请求候选通常少于 max_batch，满批时直接单发不截断。
_MAX_BATCH = 32
_MAX_WAIT_S = 0.01

_reranker = None


def _get_reranker() -> FlagReranker:
    """BGE-Reranker 模型单例（模块私有，仅供本模块公共函数调用）。

    加载优先级: config.model_path (本地) → config.model_name (HuggingFace ID)。
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    path = rerank_config.model_path
    if path and os.path.isdir(path):
        model_name_or_path = path
    elif "/" in rerank_config.model_name:
        model_name_or_path = rerank_config.model_name
    else:
        raise ValueError(f"BGE-Reranker 模型路径不存在且模型名不是 HF ID: {path!r} / {rerank_config.model_name!r}")

    device = rerank_config.device
    use_fp16 = device != "cpu"
    logger.info(f"加载 BGE-Reranker: {model_name_or_path} (device={device}, fp16={use_fp16})")
    t0 = time.perf_counter()
    _reranker = FlagReranker(
        model_name_or_path,
        use_fp16=use_fp16,
        devices=device,
        normalize=True,  # sigmoid 归一化，分数落在 (0,1) 便于理解与后续阈值
    )
    # 记入当前请求指标（冷启动首个精排请求暴露加载耗时）；无上下文时静默跳过
    mark_stage("rerank_model_load_ms", (time.perf_counter() - t0) * 1000)
    logger.info(f"BGE-Reranker 模型加载完成，耗时 {(time.perf_counter() - t0) * 1000:.0f}ms")
    return _reranker


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """批量交叉编码打分（纯同步前向，由 BatchReranker 串行调用）。"""
    reranker = _get_reranker()
    scores = reranker.compute_score(pairs)
    if isinstance(scores, float):  # 单对输入时返回标量
        scores = [scores]
    return [float(s) for s in scores]


def compute_rerank_scores(query: str, texts: list[str]) -> list[float]:
    """计算 query 与各候选的交叉编码分数（sigmoid 归一化，越大越相关）。

    同步版，供预热等单处调用；并发精排走 ``arerank_scores``。
    """
    if not texts:
        raise ValueError("texts 必须是非空列表")
    return score_pairs([(query, text) for text in texts])


class BatchReranker:
    """把并发精排请求攒成一批 (query, text) 对，一次前向再按序拆分回填。

    单实例只有一个 drainer 串行打分，天然线程安全，无需额外加锁。
    """

    def __init__(self, max_batch: int = _MAX_BATCH, max_wait_s: float = _MAX_WAIT_S):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._max_batch = max_batch
        self._max_wait = max_wait_s
        self._task: asyncio.Task | None = None

    async def encode(self, query: str, texts: list[str]) -> list[float]:
        """提交一个精排请求（单 query + 多个候选），挂起等待自己的分数。"""
        # 幂等启动常驻 drainer（绑定到调用方所在事件循环）
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())
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
                except TimeoutError:
                    break
                items.append((q, ts, f))
                flat_pairs.extend((q, t) for t in ts)

            scores = await asyncio.to_thread(score_pairs, flat_pairs)  # 一次前向
            off = 0
            for _q, ts, f in items:                      # 按各请求候选数拆分回填
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
