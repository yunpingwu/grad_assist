"""文件工具：在本任务专属的沙箱目录内落盘/读取/修改文本文件（推荐 Markdown）。

安全设计（对齐开源 agent 实践，context7 调研确认）：

- 沙箱根 `materials/{textbook_name}/{task_id}/`：目录级任务隔离；
- 路径防线：拒绝绝对路径、`..` 逃逸、symlink 逃逸（对齐 OpenCode read 工具策略）；
- 原子写：临时文件 + os.replace（对齐 DeepSeek Harness writeFileAtomic 思路）；
- 先读后改：edit_file 要求本进程内先 read_file 过该文件（对齐 OpenCode edit 约束）；
- edit_file 的 old_string 默认要求唯一命中（对齐 DeepSeek Harness edit 语义）。

路径约定：文件根目录为模块常量 ``MATERIAL_ROOT``（默认 ``./materials``），
各工具内部自行解析「沙箱根 = MATERIAL_ROOT / 教材名 / 任务 id」并校验路径，
不依赖配置注入；改动落盘根只需改本文件顶部常量。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.core import logger

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


@tool
async def write_file(
    filename: str,
    content: str,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """创建或整体覆盖一个文本文件（建议 Markdown，自动建父目录）——仅当用户要求把资料保存成文件时使用。

    Args:
        filename: 相对文件名，可含子目录，如「第3章/知识点总结.md」。
        content: 文件完整内容（UTF-8）。

    Returns:
        写入成功说明（含文件路径与字节数）。
    """
    # 解析沙箱根并逐级校验路径：拒绝绝对路径 / `..` 逃逸 / symlink 逃逸
    root = (MATERIAL_ROOT / (textbook_name or "") / (task_id or "")).resolve()
    if not filename or "\x00" in filename or Path(filename).is_absolute():
        raise ValueError(f"文件名非法（须为非空相对路径）: {filename!r}")
    path = root
    for part in Path(filename).parts:
        if part in ("", ".", ".."):
            raise ValueError(f"非法路径片段: {filename!r}")
        path = (path / part).resolve(strict=False)
        if path != root and not path.is_relative_to(root):
            raise ValueError(f"路径越出任务沙箱: {filename!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    # 「临时文件 + os.replace」原子落盘（同目录保证同文件系统 rename）
    with _locks.setdefault(path, threading.Lock()):
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)

    logger.info(f"write_file: {path}（{path.stat().st_size} 字节）")
    _known_files.add(path)  # 内容为本 agent 所写，视为已知，edit 可免 read 直接改
    return f"已写入 {path.as_posix()}（{path.stat().st_size} 字节）"


@tool
async def append_file(
    filename: str,
    content: str,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """续写已有资料时用：向文件末尾追加内容（长文档分段写入用，避免整文重写）。

    Args:
        filename: 相对文件名（必须已由 write_file 创建）。
        content: 追加的文本。

    Returns:
        追加结果（文件新大小）。
    """
    # 同 write_file 的路径解析与校验
    root = (MATERIAL_ROOT / (textbook_name or "") / (task_id or "")).resolve()
    if not filename or "\x00" in filename or Path(filename).is_absolute():
        raise ValueError(f"文件名非法（须为非空相对路径）: {filename!r}")
    path = root
    for part in Path(filename).parts:
        if part in ("", ".", ".."):
            raise ValueError(f"非法路径片段: {filename!r}")
        path = (path / part).resolve(strict=False)
        if path != root and not path.is_relative_to(root):
            raise ValueError(f"路径越出任务沙箱: {filename!r}")

    if not path.is_file():
        raise ValueError(f"文件不存在（请先 write_file 创建）: {filename}")
    with _locks.setdefault(path, threading.Lock()):
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
    """查看资料内容：按行范围分页读取沙箱内文件（写前检查、续写定位、edit 前必读）。

    Args:
        filename: 相对文件名。
        offset: 起始行号（从 1 开始，默认 1）。
        limit: 每页最大行数（默认 100）。

    Returns:
        带行号的页面内容与总行数。
    """
    # 同 write_file 的路径解析与校验
    root = (MATERIAL_ROOT / (textbook_name or "") / (task_id or "")).resolve()
    if not filename or "\x00" in filename or Path(filename).is_absolute():
        raise ValueError(f"文件名非法（须为非空相对路径）: {filename!r}")
    path = root
    for part in Path(filename).parts:
        if part in ("", ".", ".."):
            raise ValueError(f"非法路径片段: {filename!r}")
        path = (path / part).resolve(strict=False)
        if path != root and not path.is_relative_to(root):
            raise ValueError(f"路径越出任务沙箱: {filename!r}")

    if not path.is_file():
        raise ValueError(f"文件不存在: {filename}")

    _known_files.add(path)  # 记录已读，供 edit_file 前置校验
    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    if offset < 1 or offset > total:
        return f"（行号越界：{filename} 共 {total} 行，offset 应在 1~{total}）"

    page = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{offset + i:>5} | {line}" for i, line in enumerate(page))
    marker = f"[{filename} 共 {total} 行，显示 {offset}-{offset + len(page) - 1} 行]"
    return f"{marker}\n{numbered}"


@tool
async def edit_file(
    filename: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """修改已生成资料：对已知内容文件做精确文本替换（须先 read_file 读过，或本任务内 write_file 写过）。

    Args:
        filename: 相对文件名。
        old_string: 要被替换的原文，须与文件内容完全一致；默认要求文件中唯一出现。
        new_string: 替换后的文本，传空字符串表示删除。
        replace_all: True 时替换全部匹配（默认 False）。

    Returns:
        替换结果（命中次数与替换处上下文）；前置不满足时返回引导提示（供模型自愈，不抛异常）。
    """
    # 同 write_file 的路径解析与校验
    root = (MATERIAL_ROOT / (textbook_name or "") / (task_id or "")).resolve()
    if not filename or "\x00" in filename or Path(filename).is_absolute():
        raise ValueError(f"文件名非法（须为非空相对路径）: {filename!r}")
    path = root
    for part in Path(filename).parts:
        if part in ("", ".", ".."):
            raise ValueError(f"非法路径片段: {filename!r}")
        path = (path / part).resolve(strict=False)
        if path != root and not path.is_relative_to(root):
            raise ValueError(f"路径越出任务沙箱: {filename!r}")

    if not path.is_file():
        return f"工具提示：文件不存在（请先 write_file 创建该文件）: {filename}"
    if path not in _known_files:
        # 返回引导文本而非抛异常：错误会以工具结果回给模型，模型可自动 read_file 后重试（自愈循环）
        return (
            f"工具提示：编辑前需要先确认文件内容——请先调用 read_file 读取该文件"
            f"（或先在本任务内 write_file 写入它）后再编辑: {path.name}"
        )

    with _locks.setdefault(path, threading.Lock()):
        text = path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if not old_string or count == 0:
            raise ValueError(f"未找到可替换的原文: {old_string[:50]!r}")
        if not replace_all and count > 1:
            raise ValueError(f"原文出现 {count} 次不唯一，请加长上下文或置 replace_all=true")
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        # 「临时文件 + os.replace」原子落盘
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, path)

    # 替换后上下文确认（用于模型自查改动是否符合预期）
    idx = new_text.find(new_string) if new_string else 0
    context = new_text[max(0, idx - 60) : idx + len(new_string) + 120]
    logger.info(f"edit_file: {filename} 替换 {count} 处")
    return f"已替换 {count} 处\n上下文: …{context}…"


@tool
async def list_files(
    textbook_name: Annotated[str, InjectedState("textbook_name")] = None,
    task_id: Annotated[str, InjectedState("task_id")] = None,
) -> str:
    """查或改已有资料前先定位：列出本任务文件目录内已生成的全部文件。

    Returns:
        相对路径 + 字节数清单。
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


# 冒烟测试已迁移至 tests/study_agent/tools/test_files.py
