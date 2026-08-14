"""
Milvus 客户端

负责连接、建表、建索引、批量插入、检索等通用操作。
基于 MilvusClient（pymilvus 3.x 推荐 API，替代已废弃的 ORM Collection API）。

教材注册表、集合名分配等教材领域逻辑见 utils/milvus_util.py。
"""

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from pymilvus.milvus_client.index import IndexParams

from app.config import milvus_config
from app.core import logger

EMBEDDING_DIM = 1024

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    """延迟初始化全局 MilvusClient 单例。"""
    global _client
    if _client is None:
        token = milvus_config.token or None
        _client = MilvusClient(uri=milvus_config.uri, token=token)
        logger.info(f"Milvus 已连接: {milvus_config.uri}")
    return _client


# ── 连接管理 ──────────────────────────────────────────────


def connect_milvus() -> None:
    """连接 Milvus（幂等，重复调用复用单例）。"""
    get_client()


def disconnect_milvus() -> None:
    """断开 Milvus。"""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("Milvus 已断开")


# ── 数据集合管理 ──────────────────────────────────────────────


def collection_exists(collection_name: str) -> bool:
    return get_client().has_collection(collection_name)


def create_collection(collection_name: str) -> None:
    """创建教材 collection，含 dense + sparse 向量字段"""
    if collection_exists(collection_name):
        logger.info(f"集合 {collection_name} 已存在，跳过创建")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="block_type", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="textbook_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="total_chunks", dtype=DataType.INT64),
        FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="教材知识库")
    get_client().create_collection(collection_name, schema=schema)
    logger.info(f"集合 {collection_name} 创建成功")


def create_indexes(collection_name: str) -> None:
    """为 dense 向量和标量字段建索引"""
    client = get_client()

    params = IndexParams()
    # dense 向量索引
    params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        index_name="idx_embedding",
        params={"nlist": 128},
    )
    # sparse 向量索引（Milvus 自动选择 SPARSE_INVERTED_INDEX）
    params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        index_name="idx_sparse",
        params={"drop_ratio_build": 0.2},
    )
    # 标量索引
    params.add_index(field_name="block_type", index_type="TRIE", index_name="idx_block_type")
    params.add_index(field_name="textbook_name", index_type="TRIE", index_name="idx_textbook_name")

    client.create_index(collection_name, params)
    client.load_collection(collection_name)
    logger.info(f"集合 {collection_name} 索引创建完成，已加载到内存")


def batch_insert(collection_name: str, rows: list[dict]) -> int:
    """批量插入，返回实际插入行数"""
    if not rows:
        return 0
    client = get_client()
    result = client.insert(collection_name, rows)
    client.flush(collection_name)
    count = int(result.get("insert_count", len(rows)))
    logger.info(f"插入 {count} 行")
    return count


def drop_collection(collection_name: str) -> None:
    """删除集合（重建用）"""
    client = get_client()
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        logger.info(f"集合 {collection_name} 已删除")
