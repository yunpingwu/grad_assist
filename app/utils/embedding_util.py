"""
Embedding 工具

基于 FlagEmbedding BGE-M3 生成 dense + sparse 向量，可直接存入 Milvus。

模型加载优先级: config.model_path (本地) → config.model_name (HuggingFace ID)
"""

import asyncio
import os
import threading
import time

from FlagEmbedding import BGEM3FlagModel

from app.config import embedding_config
from app.core import astage, logger, mark_stage

_model = None

# 串行化模型前向：PyTorch 推理在多线程并发下不安全（encode 内含 model.to/float 等状态变更），
# 多用户并发时用锁排队，避免同一单例被并发调用
_encode_lock = threading.Lock()


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


def generate_embeddings(texts: list[str]) -> dict:
    """为文本列表生成 dense + sparse 向量，返回 Milvus 可直接入库的格式。

    返回:
        {
            "dense": [[float * EMBEDDING_DIM], ...],  # 稠密向量，已 L2 归一化
            "sparse": [{int: float}, ...],             # 稀疏向量，{token_id: 权重}
        }
    """
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("texts 必须是非空列表")

    with _encode_lock:
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


async def agenerate_embeddings(texts: list[str]) -> dict:
    """异步版 generate_embeddings：把同步 encode 丢到线程池，避免阻塞事件循环。

    适用多用户并发场景——同步 ``generate_embeddings`` 是 CPU/GPU 密集的 PyTorch 前向，
    在 async 调用链里直接执行会占住事件循环，串行等待其他请求。本函数借 to_thread 把
    计算移出事件循环；模型前向的线程串行化由 ``generate_embeddings`` 内部的
    ``_encode_lock`` 保证。

    注意：耗时埋点 ``embedding_ms`` 在此协程层记录（用 ``astage``），因为 to_thread
    会隔离 contextvar，同步函数内无法写入请求指标上下文。

    Args:
        texts: 待编码文本列表。

    Returns:
        同 generate_embeddings 的 ``{"dense": ..., "sparse": ...}`` 结构。
    """
    async with astage("embedding_ms"):
        return await asyncio.to_thread(generate_embeddings, texts)
