import asyncio
import json
import os
import re
import shutil
from pathlib import Path

from langgraph.types import StreamWriter, interrupt
from pypdf import PdfReader, PdfWriter

from app.core import log_node, logger
from app.textbook_flow.nodes.toc_tree import toc_trees
from app.textbook_flow.state import TextBookState


def _parse_content_list(file_path: Path, all_match: list[dict]) -> None:
    """解析单个 content_list.json，提取章节偏移信息。"""
    with open(file_path, encoding="utf-8") as fh:
        content_list = json.load(fh)
    match_list: list[dict] = []
    found_content = False
    content_index = 0
    first_chapter = None

    for i, item in enumerate(content_list):
        if not found_content and re.match(r"目\s*录", item.get("text", "")) and i < len(content_list) - 1:
            # 从下一项提取章节号（如 "第1章"、"第 3 章"、"第一篇"）
            next_text = content_list[i + 1].get("text", "")
            m = re.match(r"(第?\s*\S+\s*[章篇])", next_text.strip())
            if m:
                first_chapter = m.group(1)
            content_index = item["page_idx"]
            found_content = True
        if found_content and first_chapter and item["page_idx"] >= content_index:
            item_text = item.get("text", "").strip()
            # "第一篇" 结构：后续条目匹配 "第X章" 或 "第X篇"
            if "篇" in first_chapter:
                if re.match(r"第?\s*\S+\s*[章篇]", item_text):
                    match_list.append(item)
            elif item_text.startswith(first_chapter):
                match_list.append(item)
    if len(match_list) >= 2:
        all_match.append(match_list[1])
    else:
        all_match.append({"page_idx": 10, "text": ""}) # 表示没有找到合适的位置，使用兜底值做起始切割点


def get_pre_offset(extract_dirs: Path) -> list[dict]:
    """获取目录章节的起始页。

    优先在目录自身下找 content_list.json，兼容 mineru_toc 目录结构：
    - 顶层目录下直接放 content_list.json（当前结构）
    - 也兼容子目录嵌套的旧结构
    """
    all_match: list[dict] = []

    # 当前目录自身下的 content_list.json
    for f in extract_dirs.iterdir():
        if f.is_file() and f.name.endswith("content_list.json"):
            _parse_content_list(f, all_match)

    # 兼容旧结构：子目录下的 content_list.json
    for subdir in sorted(extract_dirs.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.is_file() and f.name.endswith("content_list.json"):
                _parse_content_list(f, all_match)

    return all_match


def split_chapter(
    textbook_path: str, all_match: list[dict], toc_trees: list[list[dict] | None]
) -> list[str] | None:
    """将教材按章节切割，保存到 textbooks/pdf/pdf_split/{教材名}/ 下

    章节页码由调用方预先解析（toc_trees）并传入，本函数只做切割。
    """
    textbook_path = Path(textbook_path)

    # 查找所有 PDF（排除 _toc.pdf）
    pdfs = sorted(f for f in textbook_path.iterdir() if f.suffix == ".pdf" and not f.name.endswith("_toc.pdf"))
    if not pdfs:
        logger.warning(f"未找到 PDF 文件: {textbook_path}")
        return

    output_root = textbook_path / "pdf_split"

    if len(all_match) != len(pdfs):
        raise ValueError(f"偏移量数量({len(all_match)})与PDF数量({len(pdfs)})不一致，无法按章节切割")

    sub_pdf_paths = []
    for i, pdf_path in enumerate(pdfs):
        textbook_name = pdf_path.stem
        chapter_output_dir = output_root / textbook_name
        sub_pdf_paths.append(str(chapter_output_dir))

        # 已完成则跳过：目录原子提交，存在且含章节 PDF 即视为完整
        if chapter_output_dir.is_dir() and any(p.suffix == ".pdf" for p in chapter_output_dir.iterdir()):
            logger.info(f"跳过（已存在）: {chapter_output_dir}")
            continue

        logger.info(f"处理教材 [{i + 1}/{len(pdfs)}]: {textbook_name}")
        chapters = toc_trees[i] if i < len(toc_trees) else None
        if not chapters:
            logger.warning(f"缺少目录解析结果: {pdf_path.name}")
            continue

        offset = all_match[i]["page_idx"] if i < len(all_match) else "N/A"
        logger.info(f"[{textbook_name}] 正则匹配到 {len(chapters)} 章，offset={offset}")
        for ch in chapters:
            logger.info(f"  {ch['num']} {ch['title']} → 印刷页码={ch['printed_page']}")

        if len(chapters) < 2:
            logger.warning(f"章节数不足 ({len(chapters)}): {pdf_path.name}")
            continue

        # 偏移值由 get_pre_offset 预先计算，直接取 all_match[i]
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        offset = all_match[i]["page_idx"] if i < len(all_match) else 0

        # 在 staging 目录内切割所有章节，全部完成后原子提交，整本教材要么完整要么不存在
        staging = output_root / f".{textbook_name}.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        for ci, ch in enumerate(chapters):
            start = ch["printed_page"] + offset - 1
            if ci < len(chapters) - 1:
                end = chapters[ci + 1]["printed_page"] + offset - 1
            else:
                end = total

            if start >= total or start >= end:
                continue

            writer = PdfWriter()
            for page_idx in range(start, min(end, total)):
                writer.add_page(reader.pages[page_idx])

            safe_title = re.sub(r'[\\/:*?"<>|]', "-", ch["title"])
            output_path = staging / f"{ch['num']} {safe_title}.pdf"
            with open(output_path, "wb") as f:
                writer.write(f)

        # 原子提交：替换旧的半成品目录
        if chapter_output_dir.exists():
            shutil.rmtree(chapter_output_dir)
        os.replace(staging, chapter_output_dir)
        logger.info(f"切割完成: {len(chapters)} 章 → {chapter_output_dir}")

    return sub_pdf_paths


@log_node
async def split(state: TextBookState, *, writer: StreamWriter) -> dict:
    """将整本教材按照章节分割"""
    textbook_path = Path(state.get("textbook_path"))
    output_root = textbook_path / "pdf_split"

    writer({"type": "message", "status": "running", "message": "开始按章节切割教材", "progress": 0.4})

    # 期望切割的教材全集（排除目录页 _toc.pdf）
    pdfs = sorted(
        f for f in textbook_path.iterdir() if f.suffix == ".pdf" and not f.name.endswith("_toc.pdf")
    )
    if not pdfs:
        logger.warning(f"未找到 PDF 文件: {textbook_path}")
        state["sub_pdf_paths"] = []
        return state

    # 幂等：仅当 “全部教材” 都已切割完成才短路（目录存在且含章节 PDF）
    if output_root.exists() and all(
        (output_root / pdf.stem).is_dir() and any(p.suffix == ".pdf" for p in (output_root / pdf.stem).iterdir())
        for pdf in pdfs
    ):
        sub_pdf_paths = [str(output_root / pdf.stem) for pdf in pdfs]
        state["sub_pdf_paths"] = sub_pdf_paths
        logger.info(f"pdf_split 已存在，跳过切割，共 {len(sub_pdf_paths)} 个教材目录")
        writer({"type": "message", "status": "running", "message": "章节切割结果已存在，直接复用", "progress": 0.55})
        return state

    extract_dirs_list = state.get("extracted_contents_dirs", [])
    # 遍历每个 MinerU 解析结果目录，获取各教材目录章节的起始页
    all_match: list[dict] = []
    for d in extract_dirs_list:
        all_match.extend(get_pre_offset(Path(d)))
    for match in all_match:
        logger.info(f"[{match['text'][:20]}] {match['page_idx']}")

    # 章节页码表：切割的数据源（展示用的 toc 事件由 split_contents 单点推送，此处不再 emit）
    trees = await asyncio.to_thread(toc_trees, textbook_path, pdfs)

    # 组装候选 offset（教材名 + 自动 page_idx），供人工校准
    candidates = [
        {"textbook_name": pdf.stem, "offset": all_match[i]["page_idx"] if i < len(all_match) else 0}
        for i, pdf in enumerate(pdfs)
    ]

    # 已注入且数量与教材数一致 → 直接采用，跳过人工确认（便于无前端直接跑通全图）
    pre_offsets = state.get("offsets") or []
    if pre_offsets and len(pre_offsets) == len(pdfs):
        confirmed = pre_offsets
        logger.info(f"已提供 {len(confirmed)} 个 offset，跳过人工确认直接采用")
    else:
        writer({"type": "ask_offset", "offsets": candidates, "message": "请确认章节页码偏移", "progress": 0.5})
        review = interrupt({"type": "offset_review", "offsets": candidates})
        confirmed = review.get("offsets", []) if isinstance(review, dict) else []

    # 用确认值覆盖 all_match 的 page_idx（split_chapter 只读该字段，内部无需变更）
    name_to_offset = {c["textbook_name"]: c["offset"] for c in confirmed}
    for i, pdf in enumerate(pdfs):
        if pdf.stem not in name_to_offset:
            continue
        if i >= len(all_match):
            all_match.append({"text": ""})
        all_match[i]["page_idx"] = name_to_offset[pdf.stem]

    state["offsets"] = confirmed
    # 按章节切割教材（pypdf 属 CPU/IO 密集，放线程池避免阻塞事件循环）
    sub_pdf_paths = await asyncio.to_thread(split_chapter, textbook_path, all_match, trees)
    state["sub_pdf_paths"] = sub_pdf_paths
    writer({"type": "message", "status": "running", "message": f"章节切割完成，共 {len(sub_pdf_paths)} 本教材", "progress": 0.55})

    return state


# 集成测试已迁移至 tests/textbook_flow/nodes/test_split.py
