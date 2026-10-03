"""目录树解析与 toc SSE 事件的单元测试（不依赖真实教材与 MinerU 调用）。"""

import asyncio
from pathlib import Path

from pypdf import PdfWriter

from app.textbook_flow.nodes.split import split
from app.textbook_flow.nodes.split_contents import split_contents
from app.textbook_flow.nodes.toc_tree import build_toc_payload, extract_toc_tree, toc_trees

TOC_MD = """# 机器学习

## 目录

第1章 绪论 …… 1
1.1 绪论 …… 1
1.2 使用原因 …… 2
第2章 模型评估与选择 15
2.1 经验误差与过拟合 …… 15
2.2 评估方法 …… 18
3.1 噪声挂不到任何章
第16章 分布式处理、客户-服务器
和集群……426
"""


def _toc_text() -> str:
    return TOC_MD[TOC_MD.index("## 目录") :]


def test_extract_sections_with_dot_leaders() -> None:
    """有点线小节挂到对应章下，编号/标题/页码正确。"""
    chapters = extract_toc_tree(_toc_text())
    ch1 = next(c for c in chapters if c["num"] == "第1章")
    assert [s["num"] for s in ch1["sections"]] == ["1.1", "1.2"]
    assert ch1["sections"][0]["title"] == "绪论"
    assert ch1["sections"][0]["printed_page"] == 1


def test_extract_sections_without_dot_leaders() -> None:
    """无点线（空格直连页码）小节同样解析。"""
    chapters = extract_toc_tree(_toc_text())
    ch2 = next(c for c in chapters if c["num"] == "第2章")
    assert ch2["printed_page"] == 15
    assert [(s["num"], s["printed_page"]) for s in ch2["sections"]] == [("2.1", 15), ("2.2", 18)]


def test_orphan_and_mismatched_sections_dropped() -> None:
    """挂不到章的小节（编号前缀不匹配）被丢弃。"""
    chapters = extract_toc_tree(_toc_text())
    all_secs = [s["num"] for c in chapters for s in c["sections"]]
    assert "3.1" not in all_secs


def test_cross_line_chapter_has_empty_sections() -> None:
    """跨行标题章解析成功且 sections 初始为空列表。"""
    chapters = extract_toc_tree(_toc_text())
    ch16 = next(c for c in chapters if c["num"] == "第16章")
    assert ch16["printed_page"] == 426
    assert ch16["sections"] == []


def _make_pdf(path: Path) -> None:
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)


def _build_textbook_dir(root: Path) -> list[Path]:
    """构造最小教材目录：一本 PDF + 对应 mineru_toc/full.md。"""
    _make_pdf(root / "机器学习.pdf")
    toc_dir = root / "mineru_toc" / "机器学习"
    toc_dir.mkdir(parents=True)
    (toc_dir / "full.md").write_text(TOC_MD, encoding="utf-8")
    return [root / "机器学习.pdf"]


def test_toc_trees_reads_full_md(tmp_path: Path) -> None:
    """toc_trees 按 mineru_toc/{name}/full.md 的「## 目录」段解析各教材。"""
    pdfs = _build_textbook_dir(tmp_path)
    trees = toc_trees(tmp_path, pdfs)
    assert len(trees) == 1
    chapters = trees[0]
    assert [c["num"] for c in chapters] == ["第1章", "第2章", "第16章"]


def test_build_toc_payload_per_textbook(tmp_path: Path) -> None:
    """载荷形状与 toc SSE 事件一致：[{textbook, chapters 三层}]；无产物目录返回空。"""
    pdfs = _build_textbook_dir(tmp_path)
    payload = build_toc_payload(tmp_path)
    assert payload == [{"textbook": pdfs[0].stem, "chapters": toc_trees(tmp_path, pdfs)[0]}]
    assert build_toc_payload(tmp_path / "missing") == []


def test_split_reuse_path_has_no_toc_emit(tmp_path: Path, writer_events: list) -> None:
    """幂等复用短路：目录树不再由切割节点推送，恢复统一走 /resolve 的 info 帧。"""
    _build_textbook_dir(tmp_path)
    split_dir = tmp_path / "pdf_split" / "机器学习"
    split_dir.mkdir(parents=True)
    _make_pdf(split_dir / "第1章 绪论.pdf")  # 已切割完成的标志

    state = {"textbook_path": str(tmp_path), "offsets": [{"textbook_name": "机器学习", "offset": 10}]}
    out = asyncio.run(split(state, writer=writer_events.append))

    assert out["sub_pdf_paths"] == [str(split_dir)]
    assert [e for e in writer_events if e.get("type") == "toc"] == []


def test_split_contents_idempotent_path_emits_toc(tmp_path: Path, writer_events: list) -> None:
    """目录解析幂等复用短路：跳过 MinerU 调用，但目录树照常推送。"""
    _build_textbook_dir(tmp_path)  # mineru_toc/{name}/full.md 已齐 → 命中短路

    state = {"textbook_path": str(tmp_path), "task_id": "t1"}
    out = asyncio.run(split_contents(state, writer=writer_events.append))

    assert out["extracted_contents_dirs"] == [str(tmp_path / "mineru_toc" / "机器学习")]
    toc_events = [e for e in writer_events if e.get("type") == "toc"]
    assert len(toc_events) == 1
    assert toc_events[0]["textbook"] == "机器学习"
    assert toc_events[0]["chapters"][0]["num"] == "第1章"
