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


if __name__ == "__main__":
    mcp.run(transport="stdio")