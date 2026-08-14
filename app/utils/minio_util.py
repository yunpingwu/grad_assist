"""
MinIO 图片上传工具

将 MinerU 解析出的章节图片上传到 MinIO，构建 相对路径 → object_name/url 映射，
供文本切割入库时填充 metadata_json 使用。

按 markdown 引用上传而非扫描目录：保证"上传的 = 入库引用的"，
避免上传 MinerU 产物中未被引用的冗余图片。
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from app.clients import minio_client
from app.config import minio_config
from app.core import logger

# object key 前缀，形如 textbook/{教材名}/{章节}/images/{文件名}
_OBJECT_PREFIX = "textbook"


def build_object_key(textbook_name: str, chapter: str, filename: str) -> str:
    """构建图片在 MinIO 中的 object key。

    Args:
        textbook_name: 教材名，作为顶层隔离目录。
        chapter: 章节目录名，同一教材内按章隔离，避免不同章同名图片冲突。
        filename: 图片文件名（含扩展名）。

    Returns:
        形如 textbook/{教材名}/{章节}/images/{文件名} 的 object key。
    """
    return f"{_OBJECT_PREFIX}/{textbook_name}/{chapter}/images/{filename}"


def build_object_url(object_name: str) -> str:
    """拼接 object 的直读 URL。

    Args:
        object_name: MinIO object key。

    Returns:
        形如 http://endpoint/bucket/object_name 的完整 URL。
    """
    scheme = "https" if minio_config.secure else "http"
    return f"{scheme}://{minio_config.endpoint}/{minio_config.bucket}/{quote(object_name)}"


def upload_and_map(
    chapter_dir: Path,
    textbook_name: str,
    chapter: str,
    rel_paths: set[str],
) -> dict[str, dict]:
    """将章节 markdown 引用的图片上传到 MinIO，返回相对路径到存储信息的映射。

    Args:
        chapter_dir: 章节目录（包含 full.md 与 images/ 子目录）。
        textbook_name: 教材名，用于 object key 顶层隔离。
        chapter: 章节目录名，用于 object key 按章隔离。
        rel_paths: markdown 中引用的图片相对路径集合，如 {"images/1.jpg"}。

    Returns:
        {相对路径: {"object_name": str, "url": str}}，
        本地缺失或上传失败的图片不会出现在结果中。
    """
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


# 单元测试
if __name__ == "__main__":
    key = build_object_key("C语言程序设计", "第1章", "1.jpg")
    print(key)
    print(build_object_url(key))
