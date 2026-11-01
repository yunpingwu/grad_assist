"""稳定的教材 chunk 主键生成。"""

from __future__ import annotations

import hashlib
import re


def build_chunk_id(
    *,
    textbook_name: str,
    chapter: str,
    section: str,
    block_type: str,
    chunk_index: int,
    text: str,
    textbook_version: str = "v1",
) -> str:
    """根据教材版本、内容定位和正文内容生成可复现的 chunk ID。

    同一教材、同一版本、同一定位及同一正文会生成相同 ID；正文发生变化时，
    content hash 会变化。``textbook_version`` 用于教材版本升级，默认 ``v1``
    兼容当前尚未单独维护版本号的摄入流程。
    """
    if not textbook_name or not chapter or not block_type or chunk_index < 0:
        raise ValueError("textbook_name、chapter、block_type 不能为空，chunk_index 不能为负数")

    normalized_text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()

    def short_hash(value: str, length: int) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]

    textbook_key = short_hash(f"{textbook_version}|{textbook_name}", 10)
    chapter_key = short_hash(chapter, 8)
    section_key = short_hash(section or "__root__", 8)
    content_key = short_hash(normalized_text, 12)
    safe_block_type = re.sub(r"[^a-z0-9_]+", "_", block_type.lower()).strip("_") or "unknown"

    return (
        f"v1_tb_{textbook_key}_ch_{chapter_key}_sec_{section_key}_"
        f"{safe_block_type}_{chunk_index:04d}_{content_key}"
    )
