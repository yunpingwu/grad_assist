"""Milvus 通用 MCP 服务器：对外暴露集合的元数据与数据查询能力。

环境变量：
    MILVUS_URI      必填，Milvus 服务地址，如 http://localhost:19530
    MILVUS_TOKEN    可选，鉴权 token
    MILVUS_DB_NAME  可选，目标数据库名
"""

import os
from datetime import date, datetime
from typing import Any

from mcp.server import MCPServer
from pymilvus import DataType, MilvusClient

mcp = MCPServer("milvus")

# 所有向量字段类型（不能作为 JSON 文本返回）
_VECTOR_TYPES = {
    DataType.BINARY_VECTOR,
    DataType.FLOAT_VECTOR,
    DataType.FLOAT16_VECTOR,
    DataType.BFLOAT16_VECTOR,
    DataType.SPARSE_FLOAT_VECTOR,
    DataType.INT8_VECTOR,
}

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    """惰性初始化 MilvusClient，配置来自环境变量。"""
    global _client
    if _client is None:
        uri = os.getenv("MILVUS_URI")
        if not uri:
            raise RuntimeError("未配置环境变量 MILVUS_URI")
        kwargs = {
            "uri": uri,
            "token": os.getenv("MILVUS_TOKEN") or "",
        }
        db_name = os.getenv("MILVUS_DB_NAME") or ""
        if db_name:
            kwargs["db_name"] = db_name
        _client = MilvusClient(**kwargs)
    return _client


def _default_output_fields(fields: list[dict]) -> list[str]:
    """按 schema 动态推断默认返回字段：排除向量。"""
    names: list[str] = []
    for field in fields:
        dtype = DataType(field.get("type"))
        if dtype in _VECTOR_TYPES:
            continue

        names.append(field["name"])
    return names


def _to_jsonable(value: Any) -> Any:
    """递归把结果转成可 JSON 序列化的纯 Python 结构。"""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, DataType):
        return value.name
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, int, float, bool, list, dict, tuple)):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return value


@mcp.tool()
def list_collections() -> list[dict]:
    """列出 Milvus 中的所有集合，含每个集合的记录数。"""
    client = get_client()
    result: list[dict] = []
    for name in client.list_collections():
        try:
            stats = client.get_collection_stats(name)
            row_count = int(stats.get("row_count", 0))
        except Exception:
            row_count = 0
        result.append({"name": name, "row_count": row_count})
    return result


@mcp.tool()
def describe_collection(collection_name: str) -> dict:
    """返回指定集合的描述信息、记录数与字段结构，供判断可查询的字段。

    Args:
        collection_name: 集合名称。
    """
    client = get_client()
    if not client.has_collection(collection_name):
        raise ValueError(f"集合不存在: {collection_name}")

    desc = client.describe_collection(collection_name)
    try:
        row_count = int(client.get_collection_stats(collection_name).get("row_count", 0))
    except Exception:
        row_count = 0

    fields: list[dict] = []
    for field in desc.get("fields", []):
        dtype = DataType(field.get("type"))
        summary: dict[str, Any] = {"name": field.get("name"), "type": dtype.name}
        if field.get("is_primary"):
            summary["is_primary"] = True
        elif dtype in _VECTOR_TYPES:
            summary["dim"] = int(field.get("params").get("dim", 0) or 0)
        fields.append(summary)

    return {
        "collection_name": desc.get("collection_name", collection_name),
        "description": desc.get("description", ""),
        "row_count": row_count,
        "enable_dynamic_field": desc.get("enable_dynamic_field", False),
        "fields": fields,
    }


@mcp.tool()
def query_collection(
    collection_name: str,
    filter: str = "",
    limit: int = 20,
    offset: int = 0,
    output_fields: list[str] | None = None,
) -> dict:
    """查询指定集合的数据，默认只返回标量字段（自动排除向量字段）。

    Args:
        collection_name: 集合名称。
        filter: Milvus 布尔表达式，如 `chapter == "第一章"`，空串表示不过滤。
        limit: 返回条数上限（默认 20）。
        offset: 分页偏移量。
        output_fields: 显式指定返回字段；缺省按 schema 自动推断。
    """
    client = get_client()
    if not client.has_collection(collection_name):
        raise ValueError(f"集合不存在: {collection_name}")

    fields = get_client().describe_collection(collection_name).get("fields", [])
    vector_names = {f["name"] for f in fields if DataType(f.get("type")) in _VECTOR_TYPES}

    # 确定最终返回字段，并剔除向量字段（无法作为 JSON 文本返回）
    selected = list(output_fields) if output_fields else _default_output_fields(fields)
    selected = [name for name in selected if name not in vector_names]

    rows = client.query(
        collection_name,
        filter=filter,
        output_fields=selected,
        limit=limit,
        offset=offset,
    )
    rows = _to_jsonable(rows)
    return {
        "rows": rows,
        "returned": len(rows),
        "limit": limit,
        "offset": offset,
        "truncated": len(rows) == limit,
    }


def _scalar_fields(collection_name: str, requested: list[str] | None = None) -> list[str]:
    """返回集合中可 JSON 化的字段，并校验显式字段名。"""
    fields = get_client().describe_collection(collection_name).get("fields", [])
    vector_names = {f["name"] for f in fields if DataType(f.get("type")) in _VECTOR_TYPES}
    available = {f["name"] for f in fields}
    selected = requested or _default_output_fields(fields)
    unknown = set(selected) - available
    if unknown:
        raise ValueError(f"字段不存在: {sorted(unknown)}")
    return [name for name in selected if name not in vector_names]


def _iterate_rows(collection_name: str, output_fields: list[str], batch_size: int = 100):
    """以 query_iterator 读取集合，避免 query limit/offset 的服务端截断差异。"""
    iterator = get_client().query_iterator(
        collection_name,
        filter="",
        batch_size=batch_size,
        output_fields=output_fields,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from batch
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()


@mcp.tool()
def export_chunks(
    collection_name: str,
    offset: int = 0,
    limit: int = 100,
    batch_size: int = 100,
    chapter: str | None = None,
    section: str | None = None,
    block_type: str | None = None,
) -> dict:
    """稳定分页导出教材 chunks，供构建评测集使用。

    与 query_collection 不同，本工具使用 query_iterator 全表扫描，因此不会因为
    Milvus query 的 offset/limit 实际返回量变化而漏数据。offset 是扫描结果中的
    逻辑偏移；返回 next_offset 供下一页继续读取。可按 chapter、section、block_type
    过滤，返回字段包含 id、text、章节定位和 chunk 索引。
    """
    if offset < 0 or limit < 1 or limit > 1000 or batch_size < 1 or batch_size > 1000:
        raise ValueError("offset 需非负，limit/batch_size 需在 1~1000 之间")
    if not get_client().has_collection(collection_name):
        raise ValueError(f"集合不存在: {collection_name}")

    fields = _scalar_fields(
        collection_name,
        ["id", "text", "chapter", "section", "block_type", "chunk_index", "total_chunks"],
    )
    rows: list[dict] = []
    scanned = 0
    matched = 0
    for row in _iterate_rows(collection_name, fields, batch_size):
        scanned += 1
        if chapter is not None and row.get("chapter") != chapter:
            continue
        if section is not None and row.get("section") != section:
            continue
        if block_type is not None and row.get("block_type") != block_type:
            continue
        if matched < offset:
            matched += 1
            continue
        if len(rows) < limit:
            rows.append(_to_jsonable(row))
            matched += 1
            continue
        matched += 1
        break

    has_more = len(rows) == limit and matched >= offset + limit + 1
    return {
        "rows": rows,
        "returned": len(rows),
        "offset": offset,
        "limit": limit,
        "next_offset": offset + len(rows) if has_more else None,
        "has_more": has_more,
        "filters": {"chapter": chapter, "section": section, "block_type": block_type},
        "scanned": scanned,
    }


@mcp.tool()
def search_chunks(
    collection_name: str,
    keyword: str,
    chapter: str | None = None,
    section: str | None = None,
    block_type: str | None = None,
    limit: int = 20,
) -> dict:
    """按教材 chunk 正文关键词定位片段，供选题和核对 gold ID 使用。

    关键词匹配在 MCP 侧完成，支持中文和代码片段；同时支持章节、小节及块类型
    过滤。结果按 query_iterator 的稳定遍历顺序返回，不执行向量检索。
    """
    if not keyword or limit < 1 or limit > 200:
        raise ValueError("keyword 不能为空，limit 需在 1~200 之间")
    if not get_client().has_collection(collection_name):
        raise ValueError(f"集合不存在: {collection_name}")
    fields = _scalar_fields(
        collection_name,
        ["id", "text", "chapter", "section", "block_type", "chunk_index", "total_chunks"],
    )
    needle = keyword.casefold()
    rows: list[dict] = []
    for row in _iterate_rows(collection_name, fields):
        if chapter is not None and row.get("chapter") != chapter:
            continue
        if section is not None and row.get("section") != section:
            continue
        if block_type is not None and row.get("block_type") != block_type:
            continue
        if needle in str(row.get("text") or "").casefold():
            rows.append(_to_jsonable(row))
            if len(rows) >= limit:
                break
    return {"rows": rows, "returned": len(rows), "keyword": keyword}


@mcp.tool()
def summarize_chunks(collection_name: str) -> dict:
    """统计教材 chunks 的章节、小节和块类型分布，辅助设计分层评测集。"""
    if not get_client().has_collection(collection_name):
        raise ValueError(f"集合不存在: {collection_name}")
    rows = _iterate_rows(collection_name, ["chapter", "section", "block_type"])
    chapters: dict[str, dict[str, int]] = {}
    sections: dict[str, int] = {}
    block_types: dict[str, int] = {}
    total = 0
    for row in rows:
        total += 1
        chapter = str(row.get("chapter") or "")
        section = str(row.get("section") or "")
        block_type = str(row.get("block_type") or "")
        chapters.setdefault(chapter, {})[section] = chapters.setdefault(chapter, {}).get(section, 0) + 1
        sections[f"{chapter} > {section}"] = sections.get(f"{chapter} > {section}", 0) + 1
        block_types[block_type] = block_types.get(block_type, 0) + 1
    return {"total": total, "chapters": chapters, "sections": sections, "block_types": block_types}


@mcp.tool()
def find_textbooks(keyword: str = "") -> dict:
    """按教材名查找注册表中的教材及对应 collection，避免 agent 手填集合名。"""
    if not get_client().has_collection("textbook_registry"):
        raise ValueError("教材注册表不存在: textbook_registry")
    rows = query_collection(
        "textbook_registry",
        limit=1000,
        output_fields=["textbook_name", "collection_name", "chunk_count", "created_at"],
    )["rows"]
    needle = keyword.casefold()
    rows = [row for row in rows if not needle or needle in str(row.get("textbook_name", "")).casefold()]
    return {"rows": rows, "returned": len(rows), "keyword": keyword}


@mcp.tool()
def get_chunks_by_ids(collection_name: str, chunk_ids: list[str]) -> dict:
    """按 chunk ID 精确核对评测集 gold_chunk_ids，并返回正文与定位信息。"""
    if not chunk_ids or len(chunk_ids) > 200:
        raise ValueError("chunk_ids 不能为空且最多支持 200 个")
    if not get_client().has_collection(collection_name):
        raise ValueError(f"集合不存在: {collection_name}")
    fields = _scalar_fields(
        collection_name,
        ["id", "text", "chapter", "section", "block_type", "chunk_index", "total_chunks"],
    )
    wanted = set(chunk_ids)
    rows = [
        _to_jsonable(row)
        for row in _iterate_rows(collection_name, fields)
        if row.get("id") in wanted
    ]
    found = {row["id"] for row in rows}
    return {
        "rows": rows,
        "returned": len(rows),
        "requested": len(wanted),
        "missing_ids": sorted(wanted - found),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
