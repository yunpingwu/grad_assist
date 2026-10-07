"""
Milvus 教材域工具

封装教材注册表、集合名分配等教材业务逻辑，供节点调用。
底层通用 Milvus 操作（连接、建表、插入、检索）见 clients/milvus_client.py。
"""

import hashlib
import threading
from datetime import datetime

from cachetools import TTLCache
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
)
from pymilvus.exceptions import (
    ConnectError,
    ConnectionConfigException,
    ConnectionNotExistException,
    MilvusException,
    MilvusUnavailableException,
)
from pymilvus.milvus_client.index import IndexParams

from app.clients import milvus_client
from app.core import logger
from app.core.decorators import retry

# collection 名前缀
COLLECTION_PREFIX = "tb"
# 教材注册表集合名（全校教材共享一张表）
REGISTRY_COLLECTION = "textbook_registry"

# 视为「瞬时故障」可重试的 Milvus 异常：连接类 + 服务端不可用。
# 业务错误（集合不存在 / 参数错误等）不在其中，直接抛出不浪费重试。
RETRYABLE_MILVUS_EXCEPTIONS = (
    MilvusException,
    MilvusUnavailableException,
    ConnectError,
    ConnectionConfigException,
    ConnectionNotExistException,
)


# 检索重试配置：attempts=3、退避 0.4→0.8→1.6s，覆盖网络抖动与 Milvus 瞬时不可用
RETRY_SEARCH = dict(attempts=3, base_delay=0.4, max_delay=3.0, exceptions=RETRYABLE_MILVUS_EXCEPTIONS)


# ── 进程级缓存（教材域）──────────────────────────────────────
# 教材名 → 集合名的映射短缓存：映射仅在重摄入 register_textbook 时变化，
# 60s TTL + 显式失效兜底，避免每次检索都打一次 Milvus 注册表查询。
# （cachetools TTLCache 内部无锁，本项目所有访问均发生在 uvicorn 事件循环单一线程内）
_COLLECTION_CACHE_TTL = 60.0
_collection_cache = TTLCache(maxsize=1024, ttl=_COLLECTION_CACHE_TTL)

# 注册表集合存在性探测缓存：确认存在后进程内常驻，不再重复 has_collection RPC。
# 注：注册表集合一旦创建便不再变化，缓存失效风险可忽略。
_registry_exists = False
_registry_lock = threading.Lock()

# 章节结构缓存：全表扫描结果（教材入库后不变），进程级 10 分钟 TTL，
# 超过进程寿命即失效重扫——TTL 内存缓存放热数据足够，不需要磁盘层。
_CHAPTERS_CACHE_TTL = 600.0
_chapters_cache = TTLCache(maxsize=64, ttl=_CHAPTERS_CACHE_TTL)


def ensure_registry() -> None:
    """确保教材注册表集合存在（幂等，存在性探测结果进程内缓存）。

    首次调用做一次 has_collection RPC；确认存在后缓存标记，后续调用零 RPC。
    """
    global _registry_exists
    if _registry_exists:
        return

    client = milvus_client.get_client()
    if not client.has_collection(REGISTRY_COLLECTION):
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

    with _registry_lock:
        _registry_exists = True


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
    # 登记即数据变更提交点：失效该教材的映射与章节结构缓存，检索侧即刻可见
    _invalidate_textbook_caches(textbook_name)
    logger.info(f"注册教材: {textbook_name} → {collection_name}（{chunk_count} chunk）")


def _invalidate_textbook_caches(textbook_name: str) -> None:
    """失效教材维度的进程缓存（重摄入后调用，保证检索侧新鲜）。"""
    _collection_cache.pop(textbook_name, None)
    _chapters_cache.pop(textbook_name, None)


def escape_expr_value(value: str) -> str:
    """转义 Milvus 表达式中的字符串字面量值。

    Milvus 布尔表达式仅支持 JSON 风格转义（\\\\ " / b f n r t），值里出现的
    反斜杠（如小节标题 ``\\* 8.5.4 ...``）与双引号都会导致解析失败，须先转义
    反斜杠、再转义双引号（顺序不可颠倒）。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def get_collection_by_name(textbook_name: str) -> str | None:
    """按教材名查注册表，返回对应的集合名；未登记返回 None。

    结果做进程级 60s TTL 缓存（含未登记负缓存，避免反复查询未知教材）；
    重摄入 register_textbook 时显式失效，保证映射新鲜。
    """
    cached = _collection_cache.get(textbook_name)
    if cached is not None:
        return cached or None
    ensure_registry()
    client = milvus_client.get_client()
    # 转义教材名中的反斜杠/双引号，避免破坏 filter 表达式
    safe_name = escape_expr_value(textbook_name)
    res = client.query(
        REGISTRY_COLLECTION,
        filter=f'textbook_name == "{safe_name}"',
        output_fields=["collection_name"],
        limit=1,
    )
    collection_name = res[0]["collection_name"] if res else ""
    _collection_cache[textbook_name] = collection_name
    return collection_name or None


def list_textbooks(page: int = 1, page_size: int = 20) -> dict:
    """分页列出注册表中的教材（前端教材下拉列表/书架用）。

    Args:
        page: 页码，从 1 开始。
        page_size: 每页条数。

    Returns:
        {"items": [...], "total": int, "page": int, "page_size": int}，
        items 元素含 textbook_name/collection_name/chunk_count/created_at。
    """
    ensure_registry()
    client = milvus_client.get_client()
    total = client.get_collection_stats(REGISTRY_COLLECTION).get("row_count", 0)
    offset = (page - 1) * page_size
    items = client.query(
        REGISTRY_COLLECTION,
        filter="",
        output_fields=["textbook_name", "collection_name", "chunk_count", "created_at"],
        offset=offset,
        limit=page_size,
    )
    return {"items": items, "total": total, "page": page, "page_size": page_size}


def list_chapters(textbook_name: str) -> list[dict]:
    """列出教材的章节结构（chapter/section 聚合去重，按首次出现顺序）。

    供复习资料生成 Agent 了解教材骨架、决定按章检索的范围。

    缓存策略（全表扫描结果，教材入库后不变）：进程级 10 分钟 TTL 缓存，
    命中即返回；MISS 才 query_iterator 全表扫描并回填。

    Args:
        textbook_name: 教材名（须已登记）。

    Returns:
        形如 [{"chapter": "第1章 绪论", "sections": ["1.1 概述", ...]}, ...] 的列表；
        无 section 的聚合块 sections 为空列表。

    Raises:
        ValueError: 教材未登记。
    """
    cached = _chapters_cache.get(textbook_name)
    if cached is not None:
        return cached

    chapters = _scan_chapters(textbook_name)
    _chapters_cache[textbook_name] = chapters
    return chapters


def _scan_chapters(textbook_name: str) -> list[dict]:
    """全表扫描教材集合聚合章节结构（无缓存的原始实现）。"""
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

# 混合检索默认返回字段（检索问答需要的标量字段，不含向量）
SEARCH_OUTPUT_FIELDS = ["text", "chapter", "section", "metadata_json", "block_type"]


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


@retry(**RETRY_SEARCH, name="milvus_hybrid_search")
def hybrid_search(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    collection_name: str,
    *,
    expr: str | None = None,
    limit: int = 5,
    weights: tuple[float, float] = (0.8, 0.2),
    output_fields: list[str] | None = None,
) -> list[dict]:
    """用预生成的 dense/sparse 向量在指定集合执行混合检索，返回 TOP 命中列表。

    与 ``create_hybrid_search_requests`` 配套：接收已算好的向量，忽略 embedding 生成，
    便于多路检索共享一次批量 embedding 的产物（见 search_textbook 深度路径）。

    Args:
        dense_vector: 查询的稠密向量（单条）。
        sparse_vector: 查询的稀疏向量（单条）。
        collection_name: 教材数据集合名。
        expr: 标量过滤表达式（如章节过滤），缺省不过滤。
        limit: 融合后返回的最大命中数。
        weights: WeightedRanker 的 (dense, sparse) 权重，默认 (0.8, 0.2)；
                 消融实验通过该参数做权重扫描。
        output_fields: 返回字段，缺省用 ``SEARCH_OUTPUT_FIELDS``。

    Returns:
        混合检索的 TOP 命中列表（``res[0]``），元素含 id/entity/distance。
    """
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        # 底层单路召回数至少覆盖融合返回数：快速路径 limit=5 时维持 10 条（原状），
        # 深度路径评测扩候选池（limit≥10）时按池子大小扩容，精排才有候选空间
        limit=max(limit, 10),
        expr=expr,
    )
    client = milvus_client.get_client()
    if not client:
        raise ValueError("Milvus 客户端无法连接")
    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=WeightedRanker(*weights),
        limit=limit,
        output_fields=output_fields or SEARCH_OUTPUT_FIELDS,
    )
    return res[0]


@retry(**RETRY_SEARCH, name="milvus_dense_search")
def dense_search(
    dense_vector: list[float],
    collection_name: str,
    *,
    expr: str | None = None,
    limit: int = 5,
    output_fields: list[str] | None = None,
) -> list[dict]:
    """仅用稠密向量做单路检索（COSINE），供消融实验对比 dense-only 基线的排序质量。

    Args:
        dense_vector: 查询的稠密向量（单条）。
        collection_name: 教材数据集合名。
        expr: 标量过滤表达式，缺省不过滤。
        limit: 返回的最大命中数。
        output_fields: 返回字段，缺省用 ``SEARCH_OUTPUT_FIELDS``。

    Returns:
        TOP 命中列表（``res[0]``），元素含 id/entity/distance。
    """
    if not dense_vector:
        raise ValueError("dense_vector 不能为空")
    client = milvus_client.get_client()
    res = client.search(
        collection_name=collection_name,
        data=[dense_vector],
        anns_field=DENSE_FIELD,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=limit,
        filter=expr or "",
        output_fields=output_fields or SEARCH_OUTPUT_FIELDS,
    )
    return res[0]


@retry(**RETRY_SEARCH, name="milvus_sparse_search")
def sparse_search(
    sparse_vector: dict[int, float],
    collection_name: str,
    *,
    expr: str | None = None,
    limit: int = 5,
    output_fields: list[str] | None = None,
) -> list[dict]:
    """仅用稀疏（词面）向量做单路检索（IP），供消融实验对比 sparse-only 基线的排序质量。

    Args:
        sparse_vector: 查询的稀疏向量（单条，{token_id: weight}）。
        collection_name: 教材数据集合名。
        expr: 标量过滤表达式，缺省不过滤。
        limit: 返回的最大命中数。
        output_fields: 返回字段，缺省用 ``SEARCH_OUTPUT_FIELDS``。

    Returns:
        TOP 命中列表（``res[0]``），元素含 id/entity/distance。
    """
    if not sparse_vector:
        raise ValueError("sparse_vector 不能为空")
    client = milvus_client.get_client()
    res = client.search(
        collection_name=collection_name,
        data=[sparse_vector],
        anns_field=SPARSE_FIELD,
        search_params={"metric_type": "IP"},
        limit=limit,
        filter=expr or "",
        output_fields=output_fields or SEARCH_OUTPUT_FIELDS,
    )
    return res[0]


def query_section_codes(
    collection_name: str,
    chapter: str,
    section: str,
    limit: int = 20,
) -> list[dict]:
    """查询某 (chapter, section) 下的代码块记录，供召回后补全正文配码。

    Args:
        collection_name: 教材数据集合名。
        chapter: 章节名。
        section: 小节名。
        limit: 最多返回的代码块条数。

    Returns:
        [{id, text, chapter, section, block_type}, ...]，无则返回空列表。
    """
    client = milvus_client.get_client()
    safe_chapter = escape_expr_value(chapter)
    safe_section = escape_expr_value(section)
    expr = (
        'block_type == "code" and '
        f'chapter == "{safe_chapter}" and section == "{safe_section}"'
    )
    return client.query(
        collection_name,
        filter=expr,
        output_fields=["id", "text", "chapter", "section", "block_type"],
        limit=limit,
    )
