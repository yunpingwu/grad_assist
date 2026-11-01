"""MinIO 客户端：连接、上传/下载/列举，以及教材图片的对象名与 URL 映射。"""

from pathlib import Path
from urllib.parse import quote

from minio import Minio
from minio.error import S3Error

from app.config import minio_config
from app.core import logger

# object key 前缀，形如 textbook/{教材名}/{章节}/images/{文件名}
_OBJECT_PREFIX = "textbook"

_client: Minio | None = None


def get_client() -> Minio:
    """延迟初始化 MinIO 客户端"""
    global _client
    if _client is None:
        _client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
        )
        logger.info(f"MinIO 已连接: {minio_config.endpoint}")
    return _client


def disconnect_minio() -> None:
    """断开 MinIO。

    Minio SDK 无显式 close()（底层 urllib3 连接池由进程退出时 OS 回收），
    此处仅释放单例引用，与 milvus/postgres 的 disconnect 保持对称语义。
    """
    global _client
    if _client is not None:
        _client = None
        logger.info("MinIO 已断开")


def ensure_bucket() -> bool:
    """确保目标 bucket 存在"""
    client = get_client()
    bucket = minio_config.bucket
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"MinIO bucket 已创建: {bucket}")
        return True
    else:
        logger.debug(f"MinIO bucket 已存在: {bucket}")
        return False


def upload_file(local_path: Path | str, object_name: str | None = None) -> str:
    """上传本地文件到 MinIO，返回 object_name"""
    client = get_client()
    local_path = Path(local_path)
    object_name = object_name or local_path.name

    client.fput_object(
        bucket_name=minio_config.bucket,
        object_name=object_name,
        file_path=str(local_path),
    )
    logger.debug(f"上传成功: {local_path.name} → {object_name}")
    return object_name


def download_file(object_name: str, local_path: Path | str) -> Path:
    """从 MinIO 下载文件到本地，返回本地路径"""
    client = get_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client.fget_object(
        bucket_name=minio_config.bucket,
        object_name=object_name,
        file_path=str(local_path),
    )
    logger.debug(f"下载成功: {object_name} → {local_path}")
    return local_path


def list_objects(prefix: str = "") -> list[str]:
    """列出指定前缀下的所有 object 名称"""
    client = get_client()
    objects = client.list_objects(minio_config.bucket, prefix=prefix)
    return [obj.object_name for obj in objects]


def delete_object(object_name: str) -> None:
    """删除指定 object"""
    client = get_client()
    client.remove_object(minio_config.bucket, object_name)
    logger.debug(f"已删除: {object_name}")


def exists(object_name: str) -> bool:
    """检查 object 是否存在"""
    client = get_client()
    try:
        client.stat_object(minio_config.bucket, object_name)
        return True
    except S3Error:
        return False


# ── 教材图片对象命名与上传映射 ──────────────────────────────


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
    ensure_bucket()

    mapping: dict[str, dict] = {}
    for rel in sorted(rel_paths):
        local_path = chapter_dir / rel
        if not local_path.is_file():
            logger.warning(f"[{textbook_name}/{chapter}] 引用的图片不存在，跳过: {rel}")
            continue

        object_name = build_object_key(textbook_name, chapter, local_path.name)
        try:
            if not exists(object_name):
                upload_file(local_path, object_name)
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
