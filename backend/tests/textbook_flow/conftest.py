"""textbook_flow 各节点集成测试的共享夹具与路径定位。"""

from pathlib import Path

import pytest

# 项目根：tests/textbook_flow/ 上溯 2 级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXTBOOKS_PDF = PROJECT_ROOT / "textbooks" / "pdf"


def _writer() -> list[dict]:
    events: list[dict] = []
    return events


@pytest.fixture
def writer_events() -> list[dict]:
    return _writer()
