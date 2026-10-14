"""文件工具沙箱测试：临时实名目录内验证路径防线与读写编辑闭环。

通过 monkeypatch 替换 ``MATERIAL_ROOT`` 为 pytest 的 ``tmp_path``，
避免触碰真实 study/ 落盘目录。``_known_files`` 记录的是各测试独立的
tmp 路径，不跨测试互相污染。

两类失败分开验：路径越界是**真错误**（抛异常 → 图上官方中间件出说明文本），
文件不存在 / 没先读过 / old_string 不匹配是**可自愈的指引**（直接返回一句括号提示，不触发重试）。
"""

import asyncio

import pytest

from app.study_agent.tools import files


def _invoke(tool, **kwargs):
    """走 ``ainvoke``（模型侧同款入口）调用一次工具，返回工具正文。"""
    return asyncio.run(tool.ainvoke(kwargs))


def _bad_path_message(tmp_path, monkeypatch, bad: str) -> str:
    """写非法路径，返回报错说明（越界是真错误：抛异常，不是返回指引）。"""
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(
            files.write_file.ainvoke(
                {"filename": bad, "content": "x", "textbook_name": "测试教材", "task_id": "t1"}
            )
        )
    message = str(excinfo.value)
    assert "非法" in message or "沙箱" in message, message  # 拒因随平台落到不同分支
    return message


def test_write_file_rejects_escape_path(tmp_path, monkeypatch) -> None:
    """``..`` 逃逸：拒绝且不改任何文件。"""
    _bad_path_message(tmp_path, monkeypatch, "../逃逸.md")
    assert not (tmp_path / "逃逸.md").exists(), "越界路径竟然落盘了"


def test_write_file_rejects_nested_escape(tmp_path, monkeypatch) -> None:
    """子目录里再 ``..`` 逃逸：同样拒绝。"""
    _bad_path_message(tmp_path, monkeypatch, "子/../../逃逸.md")


def test_write_file_rejects_absolute_path(tmp_path, monkeypatch) -> None:
    """绝对路径：拒绝。"""
    _bad_path_message(tmp_path, monkeypatch, "/abs.md")


def test_append_file_missing_returns_hint(tmp_path, monkeypatch) -> None:
    """未先 write_file 就追加：返回指引（非异常，不触发重试）。"""
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)
    hint = _invoke(
        files.append_file, filename="没建过.md", content="x", textbook_name="测试教材", task_id="t1"
    )
    assert "先 write_file" in hint


def test_read_file_missing_returns_hint(tmp_path, monkeypatch) -> None:
    """读不存在的文件：返回指引（指向 list_files 定位已有文件）。"""
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)
    hint = _invoke(files.read_file, filename="没建过.md", textbook_name="测试教材", task_id="t1")
    assert "list_files" in hint


def test_files_read_write_edit_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)

    # 1) 写入 / 追加 / 编辑闭环
    _invoke(
        files.write_file,
        filename="第1章/总结.md",
        content="# 第1章\n正文A\n",
        textbook_name="测试教材",
        task_id="t1",
    )
    _invoke(files.append_file, filename="第1章/总结.md", content="正文B\n", textbook_name="测试教材", task_id="t1")
    edited = _invoke(
        files.edit_file,
        filename="第1章/总结.md",
        old_string="正文A",
        new_string="甲",
        textbook_name="测试教材",
        task_id="t1",
    )
    assert "已替换 1 处" in edited

    # 2) 未 read 也未 write 的既有文件：edit 返回指引，模型可自愈（先 read_file 再改）
    pre_dir = tmp_path / "测试教材" / "pre_existing"
    pre_dir.mkdir(parents=True, exist_ok=True)
    (pre_dir / "旧.md").write_text("旧内容\n", encoding="utf-8")
    hint = _invoke(
        files.edit_file,
        filename="旧.md",
        old_string="旧内容",
        new_string="新内容",
        textbook_name="测试教材",
        task_id="pre_existing",
    )
    assert "read_file" in hint
    page = _invoke(files.read_file, filename="旧.md", textbook_name="测试教材", task_id="pre_existing")
    assert "旧内容" in page
    edited = _invoke(
        files.edit_file,
        filename="旧.md",
        old_string="旧内容",
        new_string="新内容",
        textbook_name="测试教材",
        task_id="pre_existing",
    )
    assert "已替换 1 处" in edited

    # 3) read_file 分页与行数标头；read 后可编辑（免再读）
    page = _invoke(files.read_file, filename="第1章/总结.md", textbook_name="测试教材", task_id="t1")
    assert "共 3 行" in page
    edited = _invoke(
        files.edit_file,
        filename="第1章/总结.md",
        old_string="甲",
        new_string="正文A",
        textbook_name="测试教材",
        task_id="t1",
    )
    assert "已替换 1 处" in edited

    # 4) 不唯一匹配：返回指引（加长 old_string 或 replace_all）
    _invoke(
        files.append_file,
        filename="第1章/总结.md",
        content="正文A\n正文A\n",
        textbook_name="测试教材",
        task_id="t1",
    )
    hint = _invoke(
        files.edit_file,
        filename="第1章/总结.md",
        old_string="正文A",
        new_string="乙",
        textbook_name="测试教材",
        task_id="t1",
    )
    assert "replace_all" in hint

    # 5) replace_all 生效（正文A 现出现 3 处）
    edited = _invoke(
        files.edit_file,
        filename="第1章/总结.md",
        old_string="正文A",
        new_string="乙",
        replace_all=True,
        textbook_name="测试教材",
        task_id="t1",
    )
    assert "已替换 3 处" in edited

    # 6) list_files 可见已落盘文件，且无 .tmp 残留
    listing = _invoke(files.list_files, textbook_name="测试教材", task_id="t1")
    assert "第1章/总结.md" in listing
    leftovers = list((tmp_path / "测试教材" / "t1").rglob("*.tmp"))
    assert not leftovers


def test_edit_file_missing_old_string_returns_hint(tmp_path, monkeypatch) -> None:
    """old_string 找不到：返回指引（先 read_file 看清原文）。"""
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)
    _invoke(files.write_file, filename="a.md", content="内容\n", textbook_name="教材", task_id="t1")
    hint = _invoke(
        files.edit_file, filename="a.md", old_string="不存在", new_string="x", textbook_name="教材", task_id="t1"
    )
    assert "read_file" in hint
