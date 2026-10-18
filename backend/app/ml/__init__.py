"""本地模型推理层：embedding / reranker 的单例加载与动态攒批。

与 ``app.utils`` 的分界：这里持有模型与 GPU 资源，utils 只放无状态工具。
"""

from app.ml.batch_manager.embedder import BatchEmbedder, agenerate_embeddings
from app.ml.batch_manager.reranker import BatchReranker, arerank_scores
from app.ml.embedding_util import generate_embeddings
from app.ml.reranker_util import compute_rerank_scores

__all__ = [
    "BatchEmbedder",
    "BatchReranker",
    "agenerate_embeddings",
    "arerank_scores",
    "compute_rerank_scores",
    "generate_embeddings",
]
