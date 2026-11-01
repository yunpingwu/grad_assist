"""MinIO 图片对象名/URL 构建函数的单元测试（纯函数，无外部依赖）。"""

from urllib.parse import quote

from app.clients.minio_client import build_object_key, build_object_url
from app.config import minio_config


def test_build_object_key() -> None:
    key = build_object_key("C语言程序设计", "第1章", "1.jpg")
    assert key == "textbook/C语言程序设计/第1章/images/1.jpg"


def test_build_object_url() -> None:
    object_name = build_object_key("C语言程序设计", "第1章", "1.jpg")
    url = build_object_url(object_name)

    scheme = "https" if minio_config.secure else "http"
    expected_prefix = f"{scheme}://{minio_config.endpoint}/{minio_config.bucket}/"
    assert url.startswith(expected_prefix)
    assert url.endswith(quote(object_name))
