"""离线评测（RAG Offline Evaluation）子包。

评估教材问答系统的检索链路质量，通过消融实验量化 HyDE、RRF、Rerank
等组件对召回/排序指标的贡献。当前为检索侧评测骨架（生成侧 judge 后续补充）。

模块：
- dataset:     评测集加载（黄金 chunk id 与答案要点）
- metrics:     Recall@k / Hit@k / MRR / NDCG@k 纯函数与聚合
- retrieval:   复用生产检索函数，逐级打点与计时（hybrid 混合 / rrf_hyde 融合 / rerank 精排）
- run_ablation:消融入口，跑各配置并输出指标表
"""

from app.eval.dataset import load_dataset
from app.eval.metrics import aggregate_metrics, compute_metrics
from app.eval.retrieval import run_deep, run_dense, run_fast, run_hybrid, run_hyde_rrf, run_sparse

__all__ = [
    "load_dataset",
    "compute_metrics",
    "aggregate_metrics",
    "run_fast",
    "run_hyde_rrf",
    "run_deep",
    "run_dense",
    "run_sparse",
    "run_hybrid",
]
