"""
MinIO 客户端

负责上传/下载/列举文件，用于存储 MinerU 提取的图片等静态资源。
"""

from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.config import minio_config
from app.core import logger

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
    此处仅释放单例引用，与 milvus/mongo 的 disconnect 保持对称语义。
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
