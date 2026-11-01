"""摄入图的初始化与单例访问：AsyncPostgresSaver 只能在运行中的事件循环内构造。"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.textbook_flow.graph import build_graph

# 教材摄入图：独立 textbook schema，避免与对话图 thread_id 冲突；
# thread_id = user_id:task_id，多用户任务天然隔离。
# AsyncPostgresSaver 构造要求运行中的事件循环，图和 checkpointer 在 lifespan 内装配。
checkpointer: AsyncPostgresSaver | None = None
textbook_graph: Any | None = None


def init_graph(pool: AsyncConnectionPool) -> None:
    """在 lifespan 内装配摄入图。"""
    global checkpointer, textbook_graph
    checkpointer = AsyncPostgresSaver(pool)
    textbook_graph = build_graph(checkpointer=checkpointer)


async def setup_checkpointer() -> None:
    """首次启动建 checkpoint 表（幂等；迁移里有并发建索引，连接必须是 autocommit）。"""
    if checkpointer is None:
        raise RuntimeError("textbook checkpointer 尚未初始化")
    await checkpointer.setup()


def get_graph() -> Any:
    """取已装配的图；未初始化时给出明确错误。"""
    if textbook_graph is None:
        raise RuntimeError("textbook_graph 尚未初始化（应在 FastAPI lifespan 中调用 init_graph）")
    return textbook_graph
