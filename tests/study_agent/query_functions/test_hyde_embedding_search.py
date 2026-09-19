"""hyde_embedding_search 守卫测试：仅验证空输入拦截（真实验证依赖 LLM/Milvus）。"""

import asyncio

import pytest

from app.study_agent.query_functions.hyde_embedding_search import hyde_doc_generate, hyde_doc_search


def test_hyde_doc_generate_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="问题重写为空"):
        asyncio.run(hyde_doc_generate(""))


def test_hyde_doc_search_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="假设性文档为空"):
        asyncio.run(hyde_doc_search("", "问题", "教材"))
    with pytest.raises(ValueError, match="问题重写为空"):
        asyncio.run(hyde_doc_search("文档", "", "教材"))
