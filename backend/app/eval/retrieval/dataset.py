"""评测集加载：读取黄金 chunk id、答案要点与参考答案。

gold_chunk_ids 是稳定主键，同一版本教材重灌后无需按入库顺序重新对齐。
"""

from __future__ import annotations

import json
from pathlib import Path

# 项目根目录：app/eval/retrieval/dataset.py 向上两级即仓库根
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data" / "eval" / "qa_set.json"


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict:
    """加载评测集，返回 ``{"textbook": str, "questions": list[dict]}``。"""
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"textbook": dataset["textbook"], "questions": dataset["questions"]}


def gold_chunk_ids(item: dict) -> set[str]:
    """提取单条问题的黄金 chunk id 集合（已去重）。"""
    return set(item.get("gold_chunk_ids", []))
