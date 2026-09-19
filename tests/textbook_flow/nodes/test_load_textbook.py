"""load_textbook 节点集成测试：依赖真实 textbooks/pdf 目录。"""

import asyncio

import pytest

from app.textbook_flow.nodes.load_textbook import load_textbook
from tests.textbook_flow.nodes.conftest import TEXTBOOKS_PDF


@pytest.mark.integration
def test_load_textbook_scans_supported_files(writer_events) -> None:
    """真实目录下应识别出 pdf/doc/ppt 且写入 textbook_path。"""
    if not TEXTBOOKS_PDF.is_dir():
        pytest.skip(f"缺少真实教材目录: {TEXTBOOKS_PDF}")

    state = {"textbook_exists": False}
    out = asyncio.run(load_textbook(state, writer=writer_events.append))

    assert out["textbook_path"] == str(TEXTBOOKS_PDF)
    # 校验事件流含进度上报
    assert any(e.get("progress") == 0.1 for e in writer_events)


@pytest.mark.integration
def test_load_textbook_raises_without_files(tmp_path) -> None:
    """空目录应抛出 ValueError 引导人工检查。"""
    state = {"textbook_exists": False, "textbook_path": str(tmp_path)}

    with pytest.raises(ValueError, match="没有找到支持"):
        asyncio.run(load_textbook(state, writer=lambda _e: None))
