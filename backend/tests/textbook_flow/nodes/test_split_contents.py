"""split_contents 节点集成测试：MinerU 目录解析（真实目录已缓存改走短路，缺失时提示）。"""

import asyncio

import pytest

from app.textbook_flow.nodes.split_contents import split_contents
from tests.textbook_flow.nodes.conftest import TEXTBOOKS_PDF


@pytest.mark.integration
def test_split_contents_resolves_toc_dirs(writer_events) -> None:
    """应回填 extracted_contents_dirs（幂等短路复用已解析目录）。

    触发真实验证即会调用 MinerU API；若前端产物不存在则跳过（提示手动准备）。
    """
    if not TEXTBOOKS_PDF.is_dir():
        pytest.skip(f"缺少真实教材目录: {TEXTBOOKS_PDF}")

    state = {"textbook_exists": False, "textbook_path": str(TEXTBOOKS_PDF)}
    out = asyncio.run(split_contents(state, writer=writer_events.append))

    assert len(out.get("extracted_contents_dirs", [])) == len(
        [f for f in TEXTBOOKS_PDF.iterdir() if f.suffix == ".pdf"]
    )
    for d in out["extracted_contents_dirs"]:
        assert (d / "full.md").exists()
