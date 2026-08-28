"""
Milvus 教材域工具

封装教材注册表、集合名分配等教材业务逻辑，供节点调用。
底层通用 Milvus 操作（连接、建表、插入、检索）见 clients/milvus_client.py。
"""

import hashlib
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


def ensure_registry() -> None:
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


def deterministic_collection_name(textbook_name: str) -> str:
    """由教材名确定性生成集合名（tb_ + 16 位 sha1 摘要）。

    Milvus 集合名仅允许字母/数字/下划线且不能以数字开头，教材名含中文
    无法直接使用；用内容摘要得到稳定、合法的集合名，保证同一教材重跑时
    复用同一集合（配合 truncate 实现无孤儿续跑）。
    """
    digest = hashlib.sha1(textbook_name.encode("utf-8")).hexdigest()[:16]
    return f"{COLLECTION_PREFIX}_{digest}"


def register_textbook(textbook_name: str, collection_name: str, chunk_count: int) -> None:
    """将教材登记到注册表（教材名 → 集合名）。

    Args:
        textbook_name: 教材完整名（主键）。
        collection_name: 数据集合名，如 tb_01。
        chunk_count: 入库的文本块数量。
    """
    ensure_registry()
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
    ensure_registry()
    client = milvus_client.get_client()
    # 转义教材名中的双引号，避免破坏 filter 表达式
    safe_name = textbook_name.replace('"', '\\"')
    res = client.query(
        REGISTRY_COLLECTION,
        filter=f'textbook_name == "{safe_name}"',
        output_fields=["collection_name"],
        limit=1,
    )
    return res[0]["collection_name"] if res else None


def list_textbooks() -> list[dict]:
    """列出注册表中所有教材（前端教材下拉列表用）。"""
    ensure_registry()
    client = milvus_client.get_client()
    return client.query(
        REGISTRY_COLLECTION,
        filter="",
        output_fields=["textbook_name", "collection_name", "chunk_count", "created_at"],
        limit=1000,
    )


def list_chapters(textbook_name: str) -> list[dict]:
    """列出教材的章节结构（chapter/section 聚合去重，按首次出现顺序）。

    供复习资料生成 Agent 了解教材骨架、决定按章检索的范围。

    Args:
        textbook_name: 教材名（须已登记）。

    Returns:
        形如 [{"chapter": "第1章 绪论", "sections": ["1.1 概述", ...]}, ...] 的列表；
        无 section 的聚合块 sections 为空列表。

    Raises:
        ValueError: 教材未登记。
    """
    collection_name = get_collection_by_name(textbook_name)
    if not collection_name:
        raise ValueError(f"教材未登记: {textbook_name}")

    client = milvus_client.get_client()
    # 迭代器分批拉取全部 chapter/section（避免单次 query 1000 条上限截断）
    chapter_order: list[str] = []
    sections_by_chapter: dict[str, list[str]] = {}
    iterator = client.query_iterator(
        collection_name,
        filter="",
        batch_size=100,
        output_fields=["chapter", "section"],
    )
    while True:
        try:
            batch = iterator.next()
        except StopIteration:
            break
        if not batch:
            break
        for row in batch:
            chapter = (row.get("chapter") or "").strip()
            section = (row.get("section") or "").strip()
            if not chapter or not section:
                continue
            if chapter not in sections_by_chapter:
                sections_by_chapter[chapter] = []
                chapter_order.append(chapter)
            if section not in sections_by_chapter[chapter]:
                sections_by_chapter[chapter].append(section)

    logger.info(f"教材 {textbook_name} 章节结构: {len(chapter_order)} 章")
    return [{"chapter": ch, "sections": sections_by_chapter[ch]} for ch in chapter_order]


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
