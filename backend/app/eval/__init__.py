"""离线评测（RAG Offline Evaluation）子包：按评测对象分两条互不依赖的线。

子包：
- retrieval: 检索效果线——评测集加载（dataset）、召回指标（metrics）、生产检索通道
  复用（channels）、方法/重写消融与各端到端评测入口；
- intent:    意图识别线——三档漏斗的标注集评测（run_eval）与例句池判据消融
  （run_threshold_ablation），指标为纯函数（metrics）。

运行入口按模块定位，例如 ``python -m app.eval.intent.run_eval``、
``python -m app.eval.retrieval.run_ablation``。
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
