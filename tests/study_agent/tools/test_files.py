"""文件工具沙箱测试：临时实名目录内验证路径防线与读写编辑闭环。

通过 monkeypatch 替换 ``MATERIAL_ROOT`` 为 pytest 的 ``tmp_path``，
避免触碰真实 study/ 落盘目录。``_known_files`` 记录的是各测试独立的
tmp 路径，不跨测试互相污染。
"""

import asyncio

import pytest

from app.study_agent.tools import files


@pytest.mark.parametrize(
    "bad",
    ["../逃逸.md", "子/../../逃逸.md", "/abs.md"],
)
def test_write_file_rejects_unsafe_path(tmp_path, monkeypatch, bad: str) -> None:
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(files.write_file.coroutine(
            filename=bad, content="x", textbook_name="测试教材", task_id="t1"
        ))


def test_files_read_write_edit_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(files, "MATERIAL_ROOT", tmp_path)

    # 1) 写入 / 追加 / 编辑闭环
    asyncio.run(files.write_file.coroutine(
        filename="第1章/总结.md", content="# 第1章\n正文A\n", textbook_name="测试教材", task_id="t1"
    ))
    asyncio.run(files.append_file.coroutine(
        filename="第1章/总结.md", content="正文B\n", textbook_name="测试教材", task_id="t1"
    ))
    edited = asyncio.run(files.edit_file.coroutine(
        filename="第1章/总结.md", old_string="正文A", new_string="甲", textbook_name="测试教材", task_id="t1"
    ))
    assert "已替换 1 处" in edited

    # 2) 未 read 也未 write 的既有文件：edit 返回引导提示（模型可自愈），不抛异常
    pre_dir = tmp_path / "测试教材" / "pre_existing"
    pre_dir.mkdir(parents=True, exist_ok=True)
    (pre_dir / "旧.md").write_text("旧内容\n", encoding="utf-8")
    fe = asyncio.run(files.edit_file.coroutine(
        filename="旧.md", old_string="旧内容", new_string="新内容",
        textbook_name="测试教材", task_id="pre_existing",
    ))
    assert "read_file" in fe
    page = asyncio.run(files.read_file.coroutine(
        filename="旧.md", textbook_name="测试教材", task_id="pre_existing"
    ))
    assert "旧内容" in page
    edited = asyncio.run(files.edit_file.coroutine(
        filename="旧.md", old_string="旧内容", new_string="新内容",
        textbook_name="测试教材", task_id="pre_existing",
    ))
    assert "已替换 1 处" in edited

    # 3) read_file 分页与行数标头；read 后可编辑（免再读）
    page = asyncio.run(files.read_file.coroutine(
        filename="第1章/总结.md", textbook_name="测试教材", task_id="t1"
    ))
    assert "共 3 行" in page
    edited = asyncio.run(files.edit_file.coroutine(
        filename="第1章/总结.md", old_string="甲", new_string="正文A",
        textbook_name="测试教材", task_id="t1",
    ))
    assert "已替换 1 处" in edited

    # 4) 不唯一匹配应报错
    asyncio.run(files.append_file.coroutine(
        filename="第1章/总结.md", content="正文A\n正文A\n", textbook_name="测试教材", task_id="t1"
    ))
    with pytest.raises(ValueError):
        asyncio.run(files.edit_file.coroutine(
            filename="第1章/总结.md", old_string="正文A", new_string="乙",
            textbook_name="测试教材", task_id="t1",
        ))

    # 5) replace_all 生效（正文A 现出现 3 处）
    edited = asyncio.run(files.edit_file.coroutine(
        filename="第1章/总结.md", old_string="正文A", new_string="乙",
        replace_all=True, textbook_name="测试教材", task_id="t1",
    ))
    assert "已替换 3 处" in edited

    # 6) list_files 可见已落盘文件，且无 .tmp 残留
    listing = asyncio.run(files.list_files.coroutine(textbook_name="测试教材", task_id="t1"))
    assert "第1章/总结.md" in listing
    leftovers = list((tmp_path / "测试教材" / "t1").rglob("*.tmp"))
    assert not leftovers
