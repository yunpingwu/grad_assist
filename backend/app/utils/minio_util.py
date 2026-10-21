"""MinIO 图片上传：按 markdown 引用上传，构建相对路径 → object_name/url 映射。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from app.clients import minio_client
from app.config import minio_config
from app.core import logger

# object key 前缀，形如 textbook/{教材名}/{章节}/images/{文件名}
_OBJECT_PREFIX = "textbook"


def build_object_key(textbook_name: str, chapter: str, filename: str) -> str:
    """构建图片 object key：``textbook/{教材名}/{章节}/images/{文件名}``。"""
    return f"{_OBJECT_PREFIX}/{textbook_name}/{chapter}/images/{filename}"


def build_object_url(object_name: str) -> str:
    """拼接 object 的直读 URL。"""
    scheme = "https" if minio_config.secure else "http"
    return f"{scheme}://{minio_config.endpoint}/{minio_config.bucket}/{quote(object_name)}"


def upload_and_map(
    chapter_dir: Path,
    textbook_name: str,
    chapter: str,
    rel_paths: set[str],
) -> dict[str, dict]:
    """上传章节 markdown 引用的图片，返回 {相对路径: {object_name, url}}。"""
    if not rel_paths:
        return {}

    # ensure_bucket 幂等:存在即跳过创建,返回值仅表示"本次是否新建",与上传无关
    minio_client.ensure_bucket()

    mapping: dict[str, dict] = {}
    for rel in sorted(rel_paths):
        local_path = chapter_dir / rel
        if not local_path.is_file():
            logger.warning(f"[{textbook_name}/{chapter}] 引用的图片不存在，跳过: {rel}")
            continue

        object_name = build_object_key(textbook_name, chapter, local_path.name)
        try:
            if not minio_client.exists(object_name):
                minio_client.upload_file(local_path, object_name)
            mapping[rel] = {
                "object_name": object_name,
                "url": build_object_url(object_name),
            }
        except Exception:
            # 单图失败不中断整章入库，url 留空由读取侧兜底
            logger.exception(f"[{textbook_name}/{chapter}] 上传图片失败: {rel}")
            continue

    logger.info(f"[{textbook_name}/{chapter}] 图片上传完成: {len(mapping)}/{len(rel_paths)}")
    return mapping
