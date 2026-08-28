import shutil
from pathlib import Path

from langgraph.types import StreamWriter

from app.core import log_node, logger
from app.textbook_flow.nodes.split_contents import (
    mineru_download_and_extract,
    mineru_upload_and_poll,
)
from app.textbook_flow.state import TextBookState


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


async def mineru_parse_chapters(textbook_path: Path, grouped: dict[str, list[Path]]) -> list[str]:
    """按教材分组上传章节 PDF 到 MinerU，解压到 mineru_split/{教材名}/ 下"""
    all_dirs: list[str] = []

    for textbook_name, pdfs in grouped.items():
        output_dir = textbook_path / "mineru_split" / textbook_name

        # 仅上传尚未解析（无 full.md）的章节，避免重复上传浪费 MinerU 调用
        pending_pdfs = []
        pending_names = []
        for pdf in pdfs:
            if (output_dir / pdf.stem / "full.md").exists():
                continue
            pending_pdfs.append(pdf)
            pending_names.append(pdf.stem)

        if pending_pdfs:
            full_zip_urls = await mineru_upload_and_poll(pending_pdfs, output_dir)
            await mineru_download_and_extract(full_zip_urls, output_dir, names=pending_names)

        # 全部章节目录（按 pdfs 顺序）
        all_dirs.extend(str(output_dir / pdf.stem) for pdf in pdfs)

    return all_dirs


@log_node
async def parse_to_md(state: TextBookState, *, writer: StreamWriter) -> dict:
    """将分割后的各章节 PDF 用 MinerU 解析为 Markdown"""

    textbook_path = Path(state.get("textbook_path"))
    output_dir = textbook_path / "mineru_split"

    writer({"type": "message", "status": "running", "message": "开始解析章节 Markdown（MinerU）", "progress": 0.55})

    sub_pdf_paths = state.get("sub_pdf_paths", [])

    # 按教材分组收集章节 PDF（期望解析的章节全集）
    grouped = collect_chapter_pdfs(sub_pdf_paths)
    if not grouped:
        state["extracted_dirs"] = []
        return state

    # 幂等：仅当 “全部章节” 的 full.md 都完整存在才短路，避免半成品被当成已完成
    if output_dir.exists() and all(
        (output_dir / textbook_name / pdf.stem / "full.md").exists()
        for textbook_name, pdfs in grouped.items()
        for pdf in pdfs
    ):
        extracted_dirs = [
            str(output_dir / textbook_name / pdf.stem)
            for textbook_name, pdfs in grouped.items()
            for pdf in pdfs
        ]
        state["extracted_dirs"] = extracted_dirs
        logger.info(f"mineru_split 已存在，跳过解析，共 {len(extracted_dirs)} 个章节目录")
        writer({"type": "message", "status": "running", "message": "章节解析结果已存在，直接复用", "progress": 0.8})
        return state

    # 分组解析（内部按章节跳过已完成项）
    extracted_dirs = await mineru_parse_chapters(textbook_path, grouped)
    state["extracted_dirs"] = extracted_dirs
    writer(
        {
            "type": "message",
            "status": "running",
            "message": f"章节解析完成，共 {len(extracted_dirs)} 个章节",
            "progress": 0.8,
        }
    )

    # 清理临时目录页切割产物
    toc_dir = textbook_path / "pdf_toc"
    if toc_dir.exists():
        shutil.rmtree(toc_dir)
        logger.info(f"已清理临时目录: {toc_dir}")

    return state


# 单元测试
if __name__ == "__main__":
    import asyncio

    def writer(chunk):
        print("event:", chunk)

    textbook_path = Path("D:/PycharmProjects/grad_assist/textbooks/pdf")
    sub_pdf_dirs = list((textbook_path / "pdf_split").iterdir())

    state: TextBookState = {
        "textbook_exists": False,
        "textbook_path": str(textbook_path),
        "sub_pdf_paths": [str(d) for d in sub_pdf_dirs if d.is_dir()],
    }

    asyncio.run(parse_to_md(state, writer=writer))
