"""文件工具：在 ``MATERIAL_ROOT/{教材名}/{task_id}/`` 沙箱内读写、修改文本文件。

拒绝绝对路径、``..`` 与 symlink 逃逸；模型可自愈的错误返回一句指引，真故障才抛异常。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.core import logger
from app.study_agent.safety import wrap_untrusted

# 文件落盘根目录：任务文件落在 {MATERIAL_ROOT}/{教材名}/{task_id}/ 下
MATERIAL_ROOT = Path(__file__).parents[3] / "study"

# 读文件每页行数（对齐 SWE-agent 文件查看器每次约 100 行的分页策略）
READ_PAGE_LINES = 100

# 本进程内「已知内容」的文件集合：read_file 读过 / write_file·append_file 刚写过
# （edit 前置校验；进程内单写者场景足够）
_known_files: set[Path] = set()

# 追加/编辑的每文件锁（同进程内串行化同一文件的并发改写）
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def _resolve(textbook_name: str | None, task_id: str | None, filename: str) -> Path:
    """解析沙箱内目标路径；文件名非法或越出任务目录时抛 ValueError。"""
    root = (MATERIAL_ROOT / (textbook_name or "") / (task_id or "")).resolve()
    if not filename or "\x00" in filename or Path(filename).is_absolute():
        raise ValueError(f"文件名非法（filename 只能用任务目录内的相对路径，如「第3章/总结.md」）: {filename!r}")
    path = root
    for part in Path(filename).parts:
        if part in ("", ".", ".."):
            raise ValueError(f"文件名非法（不能含「.」「..」等路径片段）: {filename!r}")
        path = (path / part).resolve(strict=False)
        if path != root and not path.is_relative_to(root):
            raise ValueError(f"路径越出任务沙箱（只能读写本任务目录内的文件）: {filename!r}")
    return path


def _lock_for(path: Path) -> threading.Lock:
    """取该文件的进程内互斥锁（同进程内串行化同一文件的并发改写）。"""
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


@tool
async def write_file(
    filename: str,
    content: str,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """创建或整体覆盖文本文件（自动建父目录），仅当用户要求保存资料时使用。"""
    path = _resolve(textbook_name, task_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 「临时文件 + os.replace」原子落盘（同目录保证同文件系统 rename）
    with _lock_for(path):
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)

    size = path.stat().st_size
    logger.info(f"write_file: {path}（{size} 字节）")
    _known_files.add(path)  # 内容为本 agent 所写，视为已知，edit 可免 read 直接改
    return f"已写入 {path.as_posix()}（{size} 字节）"


@tool
async def append_file(
    filename: str,
    content: str,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """向已创建的文本文件末尾追加内容，长文档分段写入时用。"""
    path = _resolve(textbook_name, task_id, filename)
    if not path.is_file():
        return f"（文件不存在：{filename}——先 write_file 建好这个文件，再 append_file 续写）"
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content)

    logger.info(f"append_file: {filename} +{len(content)} 字符")
    _known_files.add(path)  # 追加内容为本 agent 所写，同样视为已知
    return f"已向 {filename} 追加 {len(content)} 字符（现 {path.stat().st_size} 字节）"


@tool
async def read_file(
    filename: str,
    offset: int = 1,
    limit: int = READ_PAGE_LINES,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """按行范围分页读取沙箱内文件（写前检查、续写定位、edit 前必读）。"""
    path = _resolve(textbook_name, task_id, filename)
    if not path.is_file():
        return f"（文件不存在：{filename}——先用 list_files 看本任务已有哪些文件，或按用户要求新建）"

    _known_files.add(path)  # 记录已读，供 edit_file 前置校验
    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    if offset < 1 or offset > total:
        return f"（行号越界：{filename} 共 {total} 行，offset 应在 1~{total}）"

    page = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{offset + i:>5} | {line}" for i, line in enumerate(page))
    last = offset + len(page) - 1
    marker = f"[{filename} 共 {total} 行，显示 {offset}-{last} 行]"
    # marker 行里的 filename 也是模型给的，连同正文一起进边界，不留在边界外显得「可信」
    return wrap_untrusted(f"{marker}\n{numbered}", "文件")


@tool
async def edit_file(
    filename: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """对已读过的文件做精确文本替换；old_string 默认要求唯一命中，空 new_string 表示删除。"""
    path = _resolve(textbook_name, task_id, filename)
    if not path.is_file():
        return f"（文件不存在：{filename}——先 write_file 建好该文件，再编辑）"
    if path not in _known_files:
        return f"（编辑前需要先确认文件内容：请先 read_file 读取 {path.name}，再调用 edit_file）"

    with _lock_for(path):
        text = path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if not old_string or count == 0:
            return (
                f"（未找到可替换的原文：{old_string[:50]!r}——先 read_file 看清原文"
                "（含空白与缩进），再照抄为 old_string）"
            )
        if not replace_all and count > 1:
            return (
                f"（原文出现 {count} 次不唯一——加长 old_string 让它唯一，"
                "或置 replace_all=true 全部替换）"
            )
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        # 「临时文件 + os.replace」原子落盘
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, path)
        # 替换后上下文确认（用于模型自查改动是否符合预期）
        idx = new_text.find(new_string) if new_string else 0
        context = new_text[max(0, idx - 60) : idx + len(new_string) + 120]

    logger.info(f"edit_file: {filename} 替换 {count} 处")
    # 回显的上下文是文件正文（可能含从外部来源抄入的文本）：包边界；「已替换 N 处」是 app 自写
    return f"已替换 {count} 处\n上下文: …{wrap_untrusted(context, '文件')}…"


@tool
async def list_files(
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """查或改已有资料前先定位：列出本任务文件目录内已生成的全部文件。

    Returns:
        相对路径 + 字节数清单；目录或目录内无文件时返回「尚未生成任何文件」。
    """
    root = (MATERIAL_ROOT / (textbook_name or "") / (task_id or "")).resolve()
    if not root.is_dir():
        return "（尚未生成任何文件）"

    items = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    if not items:
        return "（尚未生成任何文件）"

    lines = [f"- {p.relative_to(root).as_posix()}（{p.stat().st_size} 字节）" for p in items]
    logger.info(f"list_files: {len(items)} 个文件")
    return "\n".join(lines)


