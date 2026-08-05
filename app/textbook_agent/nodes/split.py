import json
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter
from app.core import log_node, logger
from app.textbook_agent.state import TextBookState
from app.utils import update_task


def _parse_content_list(file_path: Path, all_match: list[dict]) -> None:
    """解析单个 content_list.json，提取章节偏移信息。"""
    with open(file_path, "r", encoding="utf-8") as fh:
        content_list = json.load(fh)
    match_list: list[dict] = []
    found_content = False
    content_index = 0
    first_chapter = None

    for i, item in enumerate(content_list):
        if not found_content and re.match(r'目\s*录', item.get("text", "")) and i < len(content_list) - 1:
            # 从下一项提取章节号（如 "第1章"、"第 3 章"、"第一篇"）
            next_text = content_list[i + 1].get("text", "")
            m = re.match(r'(第?\s*\S+\s*[章篇])', next_text.strip())
            if m:
                first_chapter = m.group(1)
            content_index = item["page_idx"]
            found_content = True
        if found_content and first_chapter and item["page_idx"] >= content_index:
            item_text = item.get("text", "").strip()
            # "第一篇" 结构：后续条目匹配 "第X章" 或 "第X篇"
            if "篇" in first_chapter:
                if re.match(r'第?\s*\S+\s*[章篇]', item_text):
                    match_list.append(item)
            elif item_text.startswith(first_chapter):
                match_list.append(item)
    if len(match_list) >= 2:
        all_match.append(match_list[1])


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


def split_chapter(textbook_path: str, all_match: list[dict]):
    """将教材按章节切割，保存到 textbooks/pdf/pdf_split/{教材名}/ 下

    从 full.md 提取章节页码（效率高于 content_list.json）：
    - full.md: 每行一个章节，正则直读，1 步到位
    """
    textbook_path = Path(textbook_path)

    # 查找所有 PDF（排除 _toc.pdf）
    pdfs = sorted(
        f for f in textbook_path.iterdir()
        if f.suffix == ".pdf" and not f.name.endswith("_toc.pdf")
    )
    if not pdfs:
        logger.warning(f"未找到 PDF 文件: {textbook_path}")
        return

    # mineru_toc 目录已按教材名重命名，与 PDF 名一致，排序后一一对应
    mineru_toc = textbook_path / "mineru_toc"
    mineru_dirs = sorted(
        d for d in mineru_toc.iterdir()
        if d.is_dir() and (d / "full.md").exists()
    )

    output_root = textbook_path / "pdf_split"

    # 章节正则：兼容 …… / .... 分隔符，章节号中间可能有空格（如"第 3 章"）
    chapter_pattern = re.compile(
        r"^(第?\s*\d+\s*章)\s+(.+?)\s*[…….]{2,}\s*(\d+)\s*$",
        re.MULTILINE,
    )
    if len(all_match) != len(pdfs):
        logger.error(f"偏移量数量({len(all_match)})与PDF数量({len(pdfs)})不一致，终止切割")
        return

    sub_pdf_paths = []
    for i, pdf_path in enumerate(pdfs):
        textbook_name = pdf_path.stem
        chapter_output_dir = output_root / textbook_name
        sub_pdf_paths.append(str(chapter_output_dir))

        # 删除已有切割文件，便于重复测试
        if chapter_output_dir.exists():
            import shutil
            shutil.rmtree(chapter_output_dir)
            logger.info(f"已删除旧文件: {chapter_output_dir}")

        logger.info(f"处理教材 [{i+1}/{len(pdfs)}]: {textbook_name}")
        # 按名称匹配 mineru_toc 目录（排序后索引对应）
        if i >= len(mineru_dirs):
            logger.warning(f"缺少 MinerU 解析结果: {pdf_path.name}")
            continue

        full_md = mineru_dirs[i] / "full.md"

        # 读取 ## 目录 部分（兼容 "## 目 录" 中间带空格的情况）
        text = full_md.read_text(encoding="utf-8")
        toc_match = re.search(r"##\s*目\s*录", text)
        if not toc_match:
            logger.warning(f"未找到 '## 目录': {full_md}")
            continue

        toc_text = text[toc_match.start():]
        chapters = []
        for m in chapter_pattern.finditer(toc_text):
            num = re.sub(r'\s+', '', m.group(1))
            if not num.startswith('第'):
                num = '第' + num
            chapters.append({
                "num": num,
                "title": m.group(2).strip(),
                "printed_page": int(m.group(3)),
            })

        logger.info(f"[{textbook_name}] 正则匹配到 {len(chapters)} 章，offset={all_match[i]['page_idx'] if i < len(all_match) else 'N/A'}")
        for ch in chapters:
            logger.info(f"  {ch['num']} {ch['title']} → 印刷页码={ch['printed_page']}")

        if len(chapters) < 2:
            logger.warning(f"章节数不足 ({len(chapters)}): {pdf_path.name}")
            continue

        # 偏移值由 get_pre_offset 预先计算，直接取 all_match[i]
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        offset = all_match[i]["page_idx"] if i < len(all_match) else 0

        # 按章节页码切割 PDF
        chapter_output_dir.mkdir(parents=True, exist_ok=True)
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

            safe_title = re.sub(r'[\\/:*?"<>|]', '-', ch["title"])
            output_path = chapter_output_dir / f"{ch['num']} {safe_title}.pdf"
            with open(output_path, "wb") as f:
                writer.write(f)

        logger.info(f"切割完成: {len(chapters)} 章 → {chapter_output_dir}")

    return sub_pdf_paths


@log_node
def split(state: TextBookState):
    """将整本教材按照章节分割"""
    textbook_path = Path(state.get("textbook_path"))
    output_root = textbook_path / "pdf_split"

    task_id = state.get("task_id")
    if task_id:
        update_task(task_id=task_id, message="开始按章节切割教材", progress=0.4)

    # 幂等：如果 pdf_split 已有切割结果，直接复用
    if output_root.exists() and any(output_root.iterdir()):
        sub_pdf_paths = [
            str(d) for d in output_root.iterdir()
            if d.is_dir() and any(p.suffix == ".pdf" for p in d.iterdir())
        ]
        state["sub_pdf_paths"] = sub_pdf_paths
        logger.info(f"pdf_split 已存在，跳过切割，共 {len(sub_pdf_paths)} 个教材目录")
        if task_id:
            update_task(task_id=task_id, message="章节切割结果已存在，直接复用", progress=0.55)
        return state

    extract_dirs_list = state.get("extracted_contents_dirs", [])
    # 遍历每个 MinerU 解析结果目录，获取各教材目录章节的起始页
    all_match: list[dict] = []
    for d in extract_dirs_list:
        all_match.extend(get_pre_offset(Path(d)))
    for i, match in enumerate(all_match):
        logger.info(f"[{match['text'][:20]}] {match['page_idx']}")
    # 按章节切割教材
    sub_pdf_paths = split_chapter(textbook_path, all_match)
    state["sub_pdf_paths"] = sub_pdf_paths
    if task_id:
        update_task(task_id=task_id, message=f"章节切割完成，共 {len(sub_pdf_paths)} 本教材", progress=0.55)
    return state
    
# 单元测试
if __name__ == '__main__':
    from pathlib import Path

    textbook_path = Path("D:/PycharmProjects/grad_assist/textbooks/pdf")
    extract_dirs = textbook_path / "mineru_toc"

    state: TextBookState = {
        "textbook_exists": False,
        "textbook_path": str(textbook_path),
        "extracted_contents_dirs": [str(extract_dirs)],
    }

    split(state)
