from pathlib import Path
import shutil

from app.textbook_agent.core import log_node
from app.textbook_agent.core.logger import get_logger
from app.textbook_agent.nodes.split_contents import (
    mineru_download_and_extract,
    mineru_upload_and_poll,
)
from app.textbook_agent.state import TextBookState

logger = get_logger(__name__)


def collect_chapter_pdfs(sub_pdf_paths: list[str]) -> dict[str, list[Path]]:
    """从各教材目录中收集所有章节 PDF，按教材名分组"""
    grouped: dict[str, list[Path]] = {}
    for dir_str in sub_pdf_paths:
        d = Path(dir_str)
        if not d.is_dir():
            continue
        pdfs = sorted(p for p in d.iterdir() if p.suffix == ".pdf")
        if pdfs:
            grouped[d.name] = pdfs

    if not grouped:
        logger.error("未找到章节 PDF 文件")
    else:
        total = sum(len(v) for v in grouped.values())
        logger.info(f"共收集到 {total} 个章节 PDF，分布在 {len(grouped)} 本教材")
    return grouped


def mineru_parse_chapters(textbook_path: Path, grouped: dict[str, list[Path]]) -> list[str]:
    """按教材分组上传章节 PDF 到 MinerU，解压到 mineru_split/{教材名}/ 下"""
    all_dirs: list[str] = []

    for textbook_name, pdfs in grouped.items():
        output_dir = textbook_path / "mineru_split" / textbook_name
        full_zip_urls = mineru_upload_and_poll(pdfs, output_dir)
        names = [pdf.stem for pdf in pdfs]
        dirs = mineru_download_and_extract(full_zip_urls, output_dir, names=names)
        all_dirs.extend(dirs)

    return all_dirs


@log_node
def parse_to_md(state: TextBookState):
    """将分割后的各章节 PDF 用 MinerU 解析为 Markdown"""

    textbook_path = Path(state.get("textbook_path"))
    output_dir = textbook_path / "mineru_split"

    # 幂等：如果 mineru_split 已有解析结果，直接复用
    if output_dir.exists() and any(
        chapter_dir.is_dir() and (chapter_dir / "full.md").exists()
        for textbook_dir in output_dir.iterdir()
        if textbook_dir.is_dir()
        for chapter_dir in textbook_dir.iterdir()
    ):
        extracted_dirs: list[str] = []
        for textbook_dir in output_dir.iterdir():
            if not textbook_dir.is_dir():
                continue
            for chapter_dir in textbook_dir.iterdir():
                if chapter_dir.is_dir() and (chapter_dir / "full.md").exists():
                    extracted_dirs.append(str(chapter_dir))
        state["extracted_dirs"] = extracted_dirs
        logger.info(f"mineru_split 已存在，跳过解析，共 {len(extracted_dirs)} 个章节目录")
        return state

    sub_pdf_paths = state.get("sub_pdf_paths", [])

    # 按教材分组收集章节 PDF
    grouped = collect_chapter_pdfs(sub_pdf_paths)

    # 分组解析
    extracted_dirs = mineru_parse_chapters(textbook_path, grouped)
    state["extracted_dirs"] = extracted_dirs

    # 清理临时目录页切割产物
    toc_dir = textbook_path / "pdf_toc"
    if toc_dir.exists():
        shutil.rmtree(toc_dir)
        logger.info(f"已清理临时目录: {toc_dir}")

    return state

# 单元测试
if __name__ == '__main__':

    textbook_path = Path("D:/PycharmProjects/grad_assist/textbooks/pdf")
    sub_pdf_dirs = list((textbook_path / "pdf_split").iterdir())

    state: TextBookState = {
        "textbook_exists": False,
        "textbook_path": str(textbook_path),
        "sub_pdf_paths": [str(d) for d in sub_pdf_dirs if d.is_dir()],
        "messages": [],
        "original_question": "",
    }

    parse_to_md(state)
