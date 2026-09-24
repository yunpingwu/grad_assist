"""embedding_search 守卫测试：仅验证空查询拦截（真实检索依赖 Milvus + embedding）。"""

import asyncio

import pytest

from app.study_agent.query_functions.embedding_search import search_by_query


def test_search_by_query_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="检索问句为空"):
        asyncio.run(search_by_query("C语言程序设计", ""))
