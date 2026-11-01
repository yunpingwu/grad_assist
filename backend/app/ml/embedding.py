"""BGE-M3 嵌入：单例加载、同步生成，以及并发请求的动态攒批。

攒批器与模型单例都在本模块内，二者一一对应；调用方通过
``agenerate_embeddings``（并发路径）或 ``generate_embeddings``（同步预热）进入。
"""

from __future__ import annotations

import asyncio
import os
import time

from FlagEmbedding import BGEM3FlagModel

from app.config import embedding_config
from app.core import logger
from app.core.decorators import retry
from app.trace import astage, mark_stage

# 攒批参数：max_batch 是并发小请求的合批上限（满批直接单发），max_wait 是等待窗口，
# 用几毫秒延迟换取 GPU 吞吐。
_MAX_BATCH = 32
_MAX_WAIT_S = 0.01

_model = None


def _get_model():
    """BGE-M3 模型单例"""
    global _model
    if _model is not None:
        return _model

    # 优先本地路径，否则 HuggingFace ID
    path = embedding_config.model_path
    if not path or not os.path.isdir(path):
        raise ValueError(f"BGE-M3 模型路径不存在: {path}")

    device = embedding_config.device
    use_fp16 = device != "cpu"

    logger.info(f"加载 BGE-M3: {path} (device={device}, fp16={use_fp16})")
    t0 = time.perf_counter()
    _model = BGEM3FlagModel(str(path), use_fp16=use_fp16, devices=[device])
    # 记入当前请求指标（冷启动首个请求暴露加载耗时，便于量化预热收益）；无上下文时静默跳过
    mark_stage("embedding_model_load_ms", (time.perf_counter() - t0) * 1000)
    logger.info(f"BGE-M3 模型加载完成，耗时 {(time.perf_counter() - t0) * 1000:.0f}ms")
    return _model


@retry(attempts=3, base_delay=0.3, max_delay=2.0, name="embedding_generate")
def generate_embeddings(texts: list[str]) -> dict:
    """生成 dense + sparse 向量，返回 ``{"dense": [...], "sparse": [...]}``。

    BGE-M3 前向失败按指数退避重试 3 次；空列表等参数错误不重试。
    """
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("texts 必须是非空列表")

    model = _get_model()
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        batch_size=32,
        max_length=8192,
    )

    # dense: ndarray → list
    dense = output["dense_vecs"].tolist()

    # sparse: 兼容两种格式 → [{int: float}, ...]
    # - 新版 FlagEmbedding: lexical_weights 为 dict/defaultdict（token_id → 权重）
    # - 旧版: scipy CSR 稀疏矩阵（有 .indices / .data）
    sparse: list[dict[int, float]] = []
    if "lexical_weights" in output and output["lexical_weights"] is not None:
        for sp in output["lexical_weights"]:
            if hasattr(sp, "indices"):  # scipy CSR
                sparse.append(dict(zip(sp.indices.tolist(), sp.data.astype("float32").tolist(), strict=True)))
            else:  # dict / defaultdict
                sparse.append({int(k): float(v) for k, v in sp.items()})

    return {"dense": dense, "sparse": sparse}


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
