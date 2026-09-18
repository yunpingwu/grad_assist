"""
Embedding 工具

基于 FlagEmbedding BGE-M3 生成 dense + sparse 向量，可直接存入 Milvus。

模型加载优先级: config.model_path (本地) → config.model_name (HuggingFace ID)
"""

from __future__ import annotations

import os
import time

from FlagEmbedding import BGEM3FlagModel

from app.config import embedding_config
from app.core import logger, mark_stage

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


# 动态攒批（BatchEmbedder / agenerate_embeddings）已迁至 app.utils.batch_manager.embedder