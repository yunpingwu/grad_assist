"""离线评测子包：``retrieval``（检索消融）与 ``intent``（意图识别）两条独立线。

入口按模块运行，如 ``python -m app.eval.intent.run_eval``。
"""

from app.eval.retrieval.channels import (
    run_deep,
    run_dense,
    run_fast,
    run_hybrid,
    run_hyde_rrf,
    run_sparse,
)
from app.eval.retrieval.dataset import load_dataset
from app.eval.retrieval.metrics import aggregate_metrics, compute_metrics

__all__ = [
    "load_dataset",
    "compute_metrics",
    "aggregate_metrics",
    "run_fast",
    "run_dense",
    "run_sparse",
    "run_hybrid",
    "run_hyde_rrf",
    "run_deep",
]
