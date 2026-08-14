"""统一配置入口。

实例化一次 Settings，导出各域配置对象。为保持既有调用方式不变，
导出的对象名与旧版一致（``llm_config`` / ``milvus_config`` 等），
调用方无需改动 ``llm_config.model`` 之类的属性访问。

同时直接导出 ``settings``（顶层聚合对象），供需要整体访问的场景使用。
"""

from app.config.settings import Settings

settings = Settings()

# 兼容旧版扁平导出：各域配置对象（属性与旧 dataclass 一致）
llm_config = settings.llm
embedding_config = settings.embedding
milvus_config = settings.milvus
mineru_config = settings.mineru
minio_config = settings.minio
mongo_config = settings.mongo
web_search_config = settings.web_search

__all__ = [
    "Settings",
    "settings",
    "llm_config",
    "embedding_config",
    "milvus_config",
    "mineru_config",
    "minio_config",
    "mongo_config",
    "web_search_config",
]
