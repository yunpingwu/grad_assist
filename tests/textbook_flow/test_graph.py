"""textbook_flow graph 集成测试：整链烟雾（需全部外部依赖就绪，缺数据时短路跳过）。"""

import asyncio

import pytest

from app.textbook_flow.graph import build_graph
from app.textbook_flow.state import TextBookState
from tests.textbook_flow.nodes.conftest import TEXTBOOKS_PDF


@pytest.mark.integration
def test_graph_full_pipeline_runs() -> None:
    """执行整条摄入流水线：load_textbook → … → split_text_and_store。

    需 MinerU API / Milvus / MinIO / embedding 全部就绪；真实教材目录缺失时短路跳过。
    注：原 __main__ 的 mermaid 绘图依赖 IPython 环境，已从测试中剔除。
    """
    if not TEXTBOOKS_PDF.is_dir():
        pytest.skip(f"缺少真实教材目录: {TEXTBOOKS_PDF}")

    # 偏移量沿用原 __main__ 实测校准值（与教科书名需完全一致，缺失项不影响短路路径）
    names = sorted(f.stem for f in TEXTBOOKS_PDF.iterdir() if f.suffix == ".pdf")
    if not names:
        pytest.skip(f"无教材 PDF: {TEXTBOOKS_PDF}")

    state: TextBookState = {
        "textbook_exists": False,
        "offsets": [{"textbook_name": name, "offset": 10} for name in names],
    }

    graph = build_graph()
    final = asyncio.run(graph.ainvoke(state, config={"recursion_limit": 50}))
    # 链路末端（或已入库短路）均应产出 ingestion_done
    assert "ingestion_done" in final
