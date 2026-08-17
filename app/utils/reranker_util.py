"""Rerank 工具

基于 FlagEmbedding BGE-Reranker 对 (查询, 片段) 对做交叉编码打分。

- 对外只暴露打分结果（``compute_rerank_scores``），不暴露模型实例；
- 模型单例由模块内部维护（``_get_reranker``），全进程仅加载一份、懒加载、可复用。

模型加载优先级: config.model_path (本地) → config.model_name (HuggingFace ID)
"""

import os

from FlagEmbedding import FlagReranker

from app.config import rerank_config
from app.core import logger

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
        raise ValueError(
            f"BGE-Reranker 模型路径不存在且模型名不是 HF ID: {path!r} / {rerank_config.model_name!r}"
        )

    device = rerank_config.device
    use_fp16 = device != "cpu"
    logger.info(f"加载 BGE-Reranker: {model_name_or_path} (device={device}, fp16={use_fp16})")
    _reranker = FlagReranker(
        model_name_or_path,
        use_fp16=use_fp16,
        devices=device,
        normalize=True,  # sigmoid 归一化，分数落在 (0,1) 便于理解与后续阈值
    )
    return _reranker


def compute_rerank_scores(query: str, texts: list[str]) -> list[float]:
    """计算查询与每个候选片段的交叉编码分数（sigmoid 归一化，越大越相关）。

    Args:
        query: 查询文本。
        texts: 候选片段文本列表，顺序与返回分数一一对应。

    Returns:
        与 ``texts`` 等长的分数列表。

    Raises:
        ValueError: ``texts`` 为空。
    """
    if not texts:
        raise ValueError("texts 必须是非空列表")

    reranker = _get_reranker()
    scores = reranker.compute_score([(query, text) for text in texts])
    if isinstance(scores, float):  # 单对输入时返回标量
        scores = [scores]
    return [float(s) for s in scores]
