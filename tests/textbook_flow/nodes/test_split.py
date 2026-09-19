"""split 节点集成测试：对真实教材执行章节切割（依赖 textbooks/pdf 及其 mineru_toc）。"""

import asyncio

import pytest

from app.textbook_flow.nodes.split import split
from tests.textbook_flow.nodes.conftest import TEXTBOOKS_PDF


def _pdfs_in(root) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(f.name for f in root.iterdir() if f.suffix == ".pdf" and not f.name.endswith("_toc.pdf"))


@pytest.mark.integration
def test_split_chapters_with_pre_injected_offsets(writer_events) -> None:
    """注入与教材数一致的 offsets，跳过人工确认直接切割。"""
    pdfs = _pdfs_in(TEXTBOOKS_PDF)
    if not pdfs:
        pytest.skip(f"缺少真实教材 PDF: {TEXTBOOKS_PDF}")

    offsets = [{"textbook_name": name, "offset": 10} for name in pdfs]
    state = {
        "textbook_exists": False,
        "textbook_path": str(TEXTBOOKS_PDF),
        "offsets": offsets,
    }

    out = asyncio.run(split(state, writer=writer_events.append))
    # 幂等/首次执行都会回填 sub_pdf_paths
    assert out.get("sub_pdf_paths") is not None
    # 已存在产物时走短路，否则完成切割后写入目录列表
    if out["sub_pdf_paths"]:
        for p in out["sub_pdf_paths"]:
            assert p.startswith(str(TEXTBOOKS_PDF / "pdf_split"))
