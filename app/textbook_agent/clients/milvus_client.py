"""
Milvus 客户端

负责连接、建表、建索引、批量插入、检索。
"""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.textbook_agent.config.milvus_config import milvus_config
from app.textbook_agent.core.logger import logger

COLLECTION_NAME = "textbook_knowledge"
EMBEDDING_DIM = 1024


def connect_milvus() -> None:
    """连接 Milvus"""
    token = milvus_config.token or None
    connections.connect(
        alias="default",
        uri=milvus_config.uri,
        token=token,
    )
    logger.info(f"Milvus 已连接: {milvus_config.uri}")


def disconnect_milvus() -> None:
    """断开 Milvus"""
    connections.disconnect("default")
    logger.info("Milvus 已断开")


def collection_exists() -> bool:
    return utility.has_collection(COLLECTION_NAME)


def create_collection() -> None:
    """创建 textbook_knowledge 集合，含 dense + sparse 向量字段"""
    if collection_exists():
        logger.info(f"集合 {COLLECTION_NAME} 已存在，跳过创建")
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
        FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=8192),
    ]

    schema = CollectionSchema(fields, description="教材知识库")
    Collection(COLLECTION_NAME, schema)
    logger.info(f"集合 {COLLECTION_NAME} 创建成功")


def create_indexes() -> None:
    """为 dense 向量和标量字段建索引"""
    col = Collection(COLLECTION_NAME)

    # dense 向量索引
    col.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )

    # sparse 向量索引（Milvus 自动选择 SPARSE_INVERTED_INDEX）
    col.create_index(
        field_name="sparse_embedding",
        index_params={
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        },
    )

    # 标量索引
    col.create_index("block_type", {"index_type": "TRIE"})
    col.create_index("textbook_name", {"index_type": "TRIE"})

    col.load()
    logger.info(f"集合 {COLLECTION_NAME} 索引创建完成，已加载到内存")


def batch_insert(rows: list[dict]) -> int:
    """批量插入，返回实际插入行数"""
    if not rows:
        return 0
    col = Collection(COLLECTION_NAME)
    result = col.insert(rows)
    col.flush()
    logger.info(f"插入 {len(result.primary_keys)} 行")
    return len(result.primary_keys)


def drop_collection() -> None:
    """删除集合（重建用）"""
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        logger.info(f"集合 {COLLECTION_NAME} 已删除")


def search_dense(
    query_vectors: list[list[float]],
    top_k: int = 5,
    expr: str | None = None,
) -> list[list[dict]]:
    """dense 向量检索，返回 [ [ {id, text, score, ...}, ... ] ]"""
    col = Collection(COLLECTION_NAME)
    output_fields = ["id", "text", "textbook_name", "chapter", "section", "block_type"]
    results = col.search(
        data=query_vectors,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr,
        output_fields=output_fields,
    )
    return [
        [{"id": h.id, "score": h.score, **h.entity.to_dict()} for h in hits]
        for hits in results
    ]
