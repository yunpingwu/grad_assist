"""转储教材 chunk（id + 章节 + 文本），供构建评测集 / 核对 gold_chunk_ids。

 gold_chunk_ids 直接写死入库时的稳定主键（包含章节定位和正文哈希）。本脚本把某本教材的全部 chunk 落盘，
便于人工逐条对齐“问题 → 黄金 chunk”时直接复制真实 id。

用法（在项目根目录）::

    .venv\\Scripts\\python.exe -m app.eval.dump_chunks "C语言程序设计"

- ``pattern`` 为教材名子串（大小写不敏感），命中注册表中的一本教材；
- 输出 ``data/eval/chunks/{collection_name}.json``：
  ``{"textbook": ..., "collection": ..., "chapters": [...], "chunks": [...]}``
  chunk 元素含 id / chapter / section / block_type / text。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.clients import milvus_client
from app.utils.milvus_util import list_textbooks

_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "eval" / "chunks"


def _find_textbook(pattern: str) -> str | None:
    """按子串匹配注册表教材名，返回唯一命中的完整教材名；无/多命中返回 None。"""
    matches = [
        item["textbook_name"]
        for item in list_textbooks(1, 100)["items"]
        if pattern.lower() in item["textbook_name"].lower()
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def dump(textbook: str) -> dict:
    """拉取某本教材全部 chunk 并返回结构化字典（含章节汇总）。"""
    from app.utils.milvus_util import get_collection_by_name

    collection = get_collection_by_name(textbook)
    if not collection:
        raise ValueError(f"教材未登记: {textbook}")

    client = milvus_client.get_client()
    chunks: list[dict] = []
    iterator = client.query_iterator(
        collection,
        filter="",
        batch_size=100,
        output_fields=["id", "chapter", "section", "block_type", "text"],
    )
    while True:
        try:
            batch = iterator.next()
        except StopIteration:
            break
        if not batch:
            break
        for row in batch:
            chunks.append(
                {
                    "id": row["id"],
                    "chapter": row.get("chapter") or "",
                    "section": row.get("section") or "",
                    "block_type": row.get("block_type") or "",
                    "text": (row.get("text") or "").strip(),
                }
            )

    # 章节结构：按 (chapter, section) 统计 chunk 数（保序）
    order: list[tuple[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        key = (c["chapter"], c["section"])
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    chapters = [
        {"chapter": ch, "section": sec, "n_chunks": counts[(ch, sec)], "n_text": 0}
        for ch, sec in order
    ]
    # 每节的正文 chunk 数（block_type=text，供选题参考）
    text_counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        if c["block_type"] == "text":
            key = (c["chapter"], c["section"])
            text_counts[key] = text_counts.get(key, 0) + 1
    for entry in chapters:
        entry["n_text"] = text_counts.get((entry["chapter"], entry["section"]), 0)

    return {"textbook": textbook, "collection": collection, "chapters": chapters, "chunks": chunks}


def main() -> None:
    if len(sys.argv) != 2:
        print("用法: python -m app.eval.dump_chunks <教材名子串>")
        raise SystemExit(1)
    textbook = _find_textbook(sys.argv[1])
    if not textbook:
        print(f"未匹配到唯一教材: {sys.argv[1]!r}")
        raise SystemExit(1)

    payload = dump(textbook)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{payload['collection']}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"教材「{textbook}」→ {len(payload['chunks'])} chunk 已转储: {out_path}")
    print(f"章节数: {len(payload['chapters'])}")


if __name__ == "__main__":
    main()
