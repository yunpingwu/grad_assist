"""
Milvus 教材域工具

封装教材注册表、集合名分配等教材业务逻辑，供节点调用。
底层通用 Milvus 操作（连接、建表、插入、检索）见 clients/milvus_client.py。
"""

import re
from datetime import datetime

from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
)
from pymilvus.milvus_client.index import IndexParams

from app.clients import milvus_client
from app.core import logger

# collection 名前缀
COLLECTION_PREFIX = "tb"
# 教材注册表集合名（全校教材共享一张表）
REGISTRY_COLLECTION = "textbook_registry"


def _ensure_registry() -> None:
    """确保教材注册表集合存在（幂等）。"""
    client = milvus_client.get_client()
    if client.has_collection(REGISTRY_COLLECTION):
        return

    fields = [
        FieldSchema(name="textbook_name", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
        FieldSchema(name="collection_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_count", dtype=DataType.INT64),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
        # Milvus 要求集合至少一个向量字段，注册表仅用标量查询，占位即可
        FieldSchema(name="dummy_embedding", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    schema = CollectionSchema(fields, description="教材名 → 集合名 映射注册表")
    client.create_collection(REGISTRY_COLLECTION, schema=schema)

    params = IndexParams()
    params.add_index(field_name="dummy_embedding", index_type="FLAT", metric_type="L2", index_name="idx_dummy")
    params.add_index(field_name="collection_name", index_type="TRIE", index_name="idx_collection_name")
    client.create_index(REGISTRY_COLLECTION, params)
    client.load_collection(REGISTRY_COLLECTION)
    logger.info(f"注册表集合 {REGISTRY_COLLECTION} 创建成功")


def next_collection_name() -> str:
    """分配下一个可用的集合名（tb_01、tb_02...）。

    从注册表取当前最大序号 +1，并跳过已被占用的集合名，
    避免中途失败产生的空洞集合被重复分配。
    """
    _ensure_registry()
    client = milvus_client.get_client()

    # 已占用序号：注册表中登记过的 + 已存在于 Milvus 的集合
    used: set[int] = set()
    res = client.query(REGISTRY_COLLECTION, filter="", output_fields=["collection_name"], limit=1000)
    for r in res:
        m = re.fullmatch(r"tb_(\d+)", r["collection_name"])
        if m:
            used.add(int(m.group(1)))

    for existing in client.list_collections():
        m = re.fullmatch(r"tb_(\d+)", existing)
        if m:
            used.add(int(m.group(1)))

    idx = 1
    while idx in used:
        idx += 1
    return f"{COLLECTION_PREFIX}_{idx:02d}"


def register_textbook(textbook_name: str, collection_name: str, chunk_count: int) -> None:
    """将教材登记到注册表（教材名 → 集合名）。

    Args:
        textbook_name: 教材完整名（主键）。
        collection_name: 数据集合名，如 tb_01。
        chunk_count: 入库的文本块数量。
    """
    _ensure_registry()
    client = milvus_client.get_client()
    client.upsert(
        REGISTRY_COLLECTION,
        [
            {
                "textbook_name": textbook_name,
                "collection_name": collection_name,
                "chunk_count": chunk_count,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # 占位向量：向量字段不支持 nullable，注册表仅用标量查询
                "dummy_embedding": [0.0, 0.0],
            }
        ],
    )
    client.flush(REGISTRY_COLLECTION)
    logger.info(f"注册教材: {textbook_name} → {collection_name}（{chunk_count} chunk）")


def get_collection_by_name(textbook_name: str) -> str | None:
    """按教材名查注册表，返回对应的集合名；未登记返回 None。"""
    _ensure_registry()
    client = milvus_client.get_client()
    # 转义教材名中的双引号，避免破坏 filter 表达式
    safe_name = textbook_name.replace('"', '\\"')
    res = client.query(
        REGISTRY_COLLECTION,
        filter=f'textbook_name like "%{safe_name}%"',
        output_fields=["collection_name"],
        limit=1,
    )
    return res[0]["collection_name"] if res else None


def list_textbooks() -> list[dict]:
    """列出注册表中所有教材（前端教材下拉列表用）。"""
    _ensure_registry()
    client = milvus_client.get_client()
    return client.query(
        REGISTRY_COLLECTION,
        filter="",
        output_fields=["textbook_name", "collection_name", "chunk_count", "created_at"],
        limit=1000,
    )


# ── 混合检索 ──────────────────────────────────────────────

# 向量字段名（与 clients/milvus_client.py 的 create_collection 保持一致）
DENSE_FIELD = "embedding"
SPARSE_FIELD = "sparse_embedding"


def create_hybrid_search_requests(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    limit: int = 10,
    expr: str | None = None,
) -> list[AnnSearchRequest]:
    """构造 dense + sparse 混合检索请求列表（供 MilvusClient.hybrid_search 使用）。

    Args:
        dense_vector: 查询的稠密向量（单条，如 BGE-M3 的 dense_vecs[0]）。
        sparse_vector: 查询的稀疏向量（单条，{token_id: weight} 字典）。
        limit: 单路底层检索返回的最大结果数（供 ranker 融合重排）。
        expr: 标量过滤表达式，如 ``textbook_name == "C语言"``，缺省不过滤。

    Returns:
        两个 AnnSearchRequest：dense 路（COSINE）+ sparse 路（IP），
        顺序与 ranker 权重列表一一对应。
    """
    if not dense_vector:
        raise ValueError("dense_vector 不能为空")
    if not sparse_vector:
        raise ValueError("sparse_vector 不能为空")

    return [
        AnnSearchRequest(
            # pymilvus 3.0 的 prepare 会对 data 逐实体调用 len()，
            # 稠密向量需以「列表套向量」形式传入，裸一维列表会触发 len(float) 错误
            data=[dense_vector],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=limit,
            expr=expr or "",
        ),
        AnnSearchRequest(
            # 同理，稀疏向量需以「列表套字典」形式传入
            data=[sparse_vector],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "IP"},
            limit=limit,
            expr=expr or "",
        ),
    ]
