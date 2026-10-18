"""parse_to_md 节点集成测试：MinerU 章节 Markdown 解析（依赖 pdf_split 子目录）。"""

import asyncio

import pytest

from app.textbook_flow.nodes.parse_to_md import parse_to_md
from tests.textbook_flow.nodes.conftest import TEXTBOOKS_PDF


@pytest.mark.integration
def test_parse_to_md_resolves_extracted_dirs(writer_events) -> None:
    """应回填 extracted_dirs；已解析产物走幂等短路，缺失时触发真实 MinerU 调用。"""
    pdf_split = TEXTBOOKS_PDF / "pdf_split"
    if not pdf_split.is_dir() or not any(p.is_dir() for p in pdf_split.iterdir()):
        pytest.skip(f"缺少 pdf_split 章节目录: {pdf_split}")

    sub_pdf_paths = [str(p) for p in sorted(pdf_split.iterdir()) if p.is_dir()]
    state = {
        "textbook_exists": False,
        "textbook_path": str(TEXTBOOKS_PDF),
        "sub_pdf_paths": sub_pdf_paths,
    }

    out = asyncio.run(parse_to_md(state, writer=writer_events.append))
    assert "extracted_dirs" in out
    if out["extracted_dirs"]:
        for d in out["extracted_dirs"]:
            assert (d / "full.md").exists()
