"""评测集加载：黄金 chunk id 与答案要点。

评测集以 JSON 文件存放，顶层含 ``textbook`` 字段与 ``questions`` 列表。
每个问题包含 question / category / gold_chunk_ids / answer_points / reference_answer。

注意：gold_chunk_ids 使用稳定的 chunk 主键（教材版本、章节/小节、块类型、索引和
正文哈希组成），同一版本教材重新摄入后无需因为入库顺序变化而重新对齐。
"""

from __future__ import annotations

import json
from pathlib import Path

# 项目根目录：app/eval/dataset.py 向上两级即仓库根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data" / "eval" / "qa_set.json"


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict:
    """加载评测集，返回 ``{"textbook": str, "questions": list[dict]}``。

    Args:
        path: 评测集 JSON 路径，默认 ``data/eval/qa_set.json``。

    Returns:
        顶层字典：textbook 为教材名，questions 为问题列表。

    Raises:
        FileNotFoundError: 评测集文件不存在。
    """
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"textbook": dataset["textbook"], "questions": dataset["questions"]}


def gold_chunk_ids(item: dict) -> set[str]:
    """提取单条问题的黄金 chunk id 集合（已去重）。

    Args:
        item: 单条问题字典。

    Returns:
        gold_chunk_ids 去重后的集合。
    """
    return set(item.get("gold_chunk_ids", []))
