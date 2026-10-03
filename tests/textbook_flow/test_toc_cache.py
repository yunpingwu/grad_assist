"""toc_cache 单元测试：按 task_id 缓存目录树、miss 时从磁盘懒重建、终态删除。"""

import uuid
from pathlib import Path

from pypdf import PdfWriter

from app.textbook_flow.cache import toc_cache

TOC_MD = """# 机器学习

## 目录

第1章 绪论 …… 1
1.1 绪论 …… 1
第2章 模型评估与选择 15
2.1 经验误差与过拟合 …… 15
"""


def _make_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)


def _build_textbook_dir(root: Path) -> None:
    """最小教材目录：一本 PDF + mineru_toc/{stem}/full.md（树的磁盘来源）。"""
    _make_pdf(root / "机器学习.pdf")
    toc_dir = root / "mineru_toc" / "机器学习"
    toc_dir.mkdir(parents=True)
    (toc_dir / "full.md").write_text(TOC_MD, encoding="utf-8")


def _tid() -> str:
    return f"t-{uuid.uuid4().hex[:8]}"


def test_get_tocs_builds_from_disk(tmp_path: Path) -> None:
    """缓存 miss：从 mineru_toc/full.md 懒重建出 [{textbook, chapters}] 载荷。"""
    _build_textbook_dir(tmp_path)
    tocs = toc_cache.get_tocs(_tid(), tmp_path)
    assert [t["textbook"] for t in tocs] == ["机器学习"]
    assert tocs[0]["chapters"][0]["num"] == "第1章"
    assert tocs[0]["chapters"][0]["sections"][0]["num"] == "1.1"


def test_get_tocs_serves_cached_copy(tmp_path: Path) -> None:
    """命中缓存：磁盘产物已删仍能返回（重进恢复不依赖重解析）。"""
    task_id = _tid()
    _build_textbook_dir(tmp_path)
    first = toc_cache.get_tocs(task_id, tmp_path)
    (tmp_path / "mineru_toc" / "机器学习" / "full.md").unlink()
    assert toc_cache.get_tocs(task_id, tmp_path) == first


def test_empty_result_is_not_cached(tmp_path: Path) -> None:
    """目录未解析时返回空且不落缓存：解析完成后同 task_id 可拿到新树。"""
    task_id = _tid()
    assert toc_cache.get_tocs(task_id, tmp_path / "not-yet") == []
    _build_textbook_dir(tmp_path)
    assert [t["textbook"] for t in toc_cache.get_tocs(task_id, tmp_path)] == ["机器学习"]


def test_pop_tocs_deletes_on_terminal(tmp_path: Path) -> None:
    """完成即删：pop 之后缓存不再返回，下次凭磁盘重建。"""
    task_id = _tid()
    _build_textbook_dir(tmp_path)
    toc_cache.get_tocs(task_id, tmp_path)
    toc_cache.pop_tocs(task_id)
    (tmp_path / "mineru_toc").replace(tmp_path / "mineru_toc_moved")  # 断掉磁盘来源
    assert toc_cache.get_tocs(task_id, tmp_path) == []
