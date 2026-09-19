"""split_text_and_store 节点集成测试：切块 + 向量化入库（真实 Milvus/MinIO/embedding）。"""

import asyncio

import pytest

from app.textbook_flow.nodes.split_text_and_store import split_text_and_store
from tests.textbook_flow.nodes.conftest import TEXTBOOKS_PDF


@pytest.mark.integration
def test_split_text_and_store_skips_when_done(writer_events) -> None:
    """ingestion_done=True 时幂等短路，不重复入库。"""
    state = {
        "textbook_exists": False,
        "ingestion_done": True,
    }

    out = asyncio.run(split_text_and_store(state, writer=writer_events.append))
    assert out.get("ingestion_done") is True


@pytest.mark.integration
def test_split_text_and_store_empty_dirs(writer_events) -> None:
    """无章节解析结果时优雅终止，不触发 Milvus/MinIO。"""
    state = {
        "textbook_exists": False,
        "ingestion_done": False,
        "extracted_dirs": [],
    }

    out = asyncio.run(split_text_and_store(state, writer=writer_events.append))
    assert out.get("ingestion_done") is False


@pytest.mark.integration
def test_split_text_and_store_full_pipeline(writer_events) -> None:
    """完整链路：对真实 mineru_split 章节目录切块并入库（需 Milvus/MinIO/embedding 就绪）。

    仅当真实产物齐全时才执行，避免空跑触发外部调用；入库后按注册表幂等跳过。
    """
    mineru_split = TEXTBOOKS_PDF / "mineru_split"
    if not mineru_split.is_dir():
        pytest.skip(f"缺少 mineru_split 目录: {mineru_split}")

    chapter_dirs = [str(d) for d in mineru_split.iterdir() if (d / "full.md").exists()]
    if not chapter_dirs:
        pytest.skip("mineru_split 下无含 full.md 的教材目录")

    state = {
        "textbook_exists": False,
        "textbook_path": str(TEXTBOOKS_PDF),
        "extracted_dirs": chapter_dirs,
    }

    out = asyncio.run(split_text_and_store(state, writer=writer_events.append))
    # 入库成功或已存在（注册表跳转）均视为完成
    assert out.get("ingestion_done") is True
