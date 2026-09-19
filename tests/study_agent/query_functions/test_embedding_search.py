"""embedding_search 守卫测试：仅验证空查询拦截（真实检索依赖 Milvus + embedding）。"""

import asyncio

import pytest

from app.study_agent.query_functions.embedding_search import rewrite_query_search


def test_rewrite_query_search_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="问题重写为空"):
        asyncio.run(rewrite_query_search("C语言程序设计", ""))
