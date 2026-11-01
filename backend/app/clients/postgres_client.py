"""PostgreSQL 连接池与 checkpoint schema 管理（替代原 MongoDB 客户端）。

对话图和摄入图各用一个独立 schema（默认 ``study`` / ``textbook``），
通过连接的 ``search_path`` 把 ``AsyncPostgresSaver`` 建的表隔离开——
两套图的 thread_id 命名空间互不干扰，删除/列举会话也不会串到另一条流水线。
"""

from __future__ import annotations

import re

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import postgres_config
from app.core import logger

# schema 名只允许小写标识符：search_path 里不带引号，大小写会被 PostgreSQL 折叠
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

_study_pool: AsyncConnectionPool | None = None
_textbook_pool: AsyncConnectionPool | None = None


def _schema_name(value: str) -> str:
    """校验并返回 schema 名；配置写错时尽早报错，避免拼进 SQL 后才发现。"""
    if not _SCHEMA_RE.fullmatch(value):
        raise ValueError(f"非法 PostgreSQL schema 名: {value!r}（只允许小写字母/数字/下划线）")
    return value


def _pool_kwargs(schema: str) -> dict:
    """连接参数：autocommit 是 checkpoint 迁移里 CREATE INDEX CONCURRENTLY 的前提。"""
    return {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
        # 每条连接进入目标 schema；public 兜底，避免扩展/函数找不到
        "options": f"-c search_path={schema},public",
    }


async def _open_pool(schema: str) -> AsyncConnectionPool:
    """创建并打开一个连接池；open=False + 显式 open() 是 psycopg_pool 3.2 的推荐写法。"""
    pool = AsyncConnectionPool(
        conninfo=postgres_config.uri,
        open=False,
        min_size=postgres_config.pool_min_size,
        max_size=postgres_config.pool_max_size,
        kwargs=_pool_kwargs(schema),
    )
    await pool.open(wait=True)
    return pool


async def _ensure_schema(pool: AsyncConnectionPool, schema: str) -> None:
    """建 schema（幂等）；表由 AsyncPostgresSaver.setup() 负责。"""
    async with pool.connection() as conn:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')


async def open_pools() -> None:
    """初始化两条流水线的连接池与 schema（幂等；应用 lifespan 启动时调用）。"""
    global _study_pool, _textbook_pool
    if _study_pool is not None:
        return

    study_schema = _schema_name(postgres_config.study_schema)
    textbook_schema = _schema_name(postgres_config.textbook_schema)
    if study_schema == textbook_schema:
        raise ValueError("POSTGRES_STUDY_SCHEMA 与 POSTGRES_TEXTBOOK_SCHEMA 不能相同")

    try:
        _study_pool = await _open_pool(study_schema)
        _textbook_pool = await _open_pool(textbook_schema)
        await _ensure_schema(_study_pool, study_schema)
        await _ensure_schema(_textbook_pool, textbook_schema)
    except Exception:
        await close_pools()
        raise

    logger.info(f"PostgreSQL 连接池已就绪（schemas: {study_schema}, {textbook_schema}）")


async def close_pools() -> None:
    """关闭连接池（幂等）。"""
    global _study_pool, _textbook_pool
    if _study_pool is not None:
        await _study_pool.close()
        _study_pool = None
    if _textbook_pool is not None:
        await _textbook_pool.close()
        _textbook_pool = None
    logger.info("PostgreSQL 连接池已关闭")


def get_study_pool() -> AsyncConnectionPool:
    """取对话图的连接池；未初始化时给出明确错误（而不是 None 上的属性错误）。"""
    if _study_pool is None:
        raise RuntimeError("PostgreSQL 连接池尚未初始化（应在 FastAPI lifespan 中调用 open_pools）")
    return _study_pool


def get_textbook_pool() -> AsyncConnectionPool:
    """取摄入图的连接池。"""
    if _textbook_pool is None:
        raise RuntimeError("PostgreSQL 连接池尚未初始化（应在 FastAPI lifespan 中调用 open_pools）")
    return _textbook_pool


async def list_thread_ids(prefix: str) -> list[str]:
    """列出 study schema 中以 ``prefix`` 开头的 thread_id（会话列表用）。

    用 ``left(thread_id, n) = prefix`` 而不是 LIKE，避免 user_id 里的 ``%``/``_``
    被当成通配符。
    """
    schema = _schema_name(postgres_config.study_schema)
    async with get_study_pool().connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f'SELECT DISTINCT thread_id FROM "{schema}".checkpoints WHERE left(thread_id, %s) = %s',
                (len(prefix), prefix),
            )
            rows = await cur.fetchall()
    return [str(row["thread_id"]) for row in rows]
