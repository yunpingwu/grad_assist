import shutil
import time
import uuid
import zipfile
from pathlib import Path

import requests
from pypdf import PdfReader, PdfWriter

from app.config import mineru_config
from app.core import log_node, logger
from app.textbook_agent.state import TextBookState
from app.utils import update_task


def find_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        return [f for f in path.iterdir() if f.suffix == ".pdf"]
    return [path] if path.suffix == ".pdf" else []


# 步骤 1：切目录页
def cut_toc_pages(toc_dir: Path, pdfs: list[Path], toc_pages: int = 30) -> list[Path]:
    """切割教材的前30页（关注里面包含的目录）"""
    if toc_dir.exists():
        shutil.rmtree(toc_dir)
    toc_dir.mkdir(parents=True)

    toc_files = []
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        writer = PdfWriter()
        for i in range(min(toc_pages, len(reader.pages))):
            writer.add_page(reader.pages[i])
        toc = toc_dir / f"{pdf.stem}_toc.pdf"
        with open(toc, "wb") as f:
            writer.write(f)
        toc_files.append(toc)

    logger.info(f"切割出 {len(toc_files)} 个目录文件")
    return toc_files


# 步骤 2：MinerU 在线 API
def mineru_upload_and_poll(toc_files: list[Path], toc_dir: Path) -> list[str]:
    """利用mineru解析切割的目录pdf文件

    完整流程：
    1. POST /api/v4/file-urls/batch 获取预签名上传URL和batch_id
    2. PUT 上传每个文件到对应URL（系统自动提交解析任务）
    3. GET /api/v4/extract-results/batch/{batch_id} 轮询直到全部完成
    """
    token = mineru_config.token
    base_url = mineru_config.url

    # ========== Step 1: 获取上传URL ==========
    files_info = [{"name": f.name, "data_id": str(uuid.uuid4())[:8]} for f in toc_files]
    header = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    data = {"files": files_info, "model_version": "vlm"}

    logger.info(f"请求上传URL，共 {len(toc_files)} 个文件")
    resp = requests.post(f"{base_url}/api/v4/file-urls/batch", headers=header, json=data)
    resp.raise_for_status()
    result = resp.json()

    if result["code"] != 0:
        raise RuntimeError(f"获取上传URL失败: {result['msg']}")

    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"]
    logger.info(f"获取上传URL成功，batch_id: {batch_id}，URL数量: {len(urls)}")

    # ========== Step 2: 上传文件到预签名URL ==========
    for i, (file_path, upload_url) in enumerate(zip(toc_files, urls, strict=True)):
        with open(file_path, "rb") as f:
            put_resp = requests.put(upload_url, data=f)
            if put_resp.status_code == 200:
                logger.info(f"上传成功 [{i + 1}/{len(toc_files)}]: {file_path.name}")
            else:
                logger.error(f"上传失败 [{i + 1}/{len(toc_files)}]: {file_path.name}, HTTP {put_resp.status_code}")
                raise RuntimeError(f"上传失败: {file_path.name}, HTTP {put_resp.status_code}")

    logger.info("所有文件上传完成，开始轮询解析结果...")

    # ========== Step 3: 轮询解析结果 ==========
    poll_url = f"{base_url}/api/v4/extract-results/batch/{batch_id}"
    max_timeout = 600
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_timeout:
            raise TimeoutError(f"轮询超时 ({max_timeout}s)，batch_id={batch_id}")

        poll_resp = requests.get(poll_url, headers=header)
        poll_resp.raise_for_status()
        poll_result = poll_resp.json()

        if poll_result["code"] != 0:
            raise RuntimeError(f"查询解析结果失败: {poll_result['msg']}")

        extract_results = poll_result["data"]["extract_result"]
        all_done = True
        for r in extract_results:
            state = r["state"]
            file_name = r["file_name"]
            if state == "failed":
                logger.error(f"解析失败: {file_name}, 原因: {r.get('err_msg', '未知')}")
            elif state == "done":
                logger.info(f"解析完成: {file_name}, zip: {r.get('full_zip_url', 'N/A')}")
            else:
                all_done = False
                progress = r.get("extract_progress", {})
                if progress:
                    logger.info(
                        f"解析中: {file_name}, 状态: {state}, "
                        f"进度: {progress.get('extracted_pages', '?')}/{progress.get('total_pages', '?')}"
                    )
                else:
                    logger.info(f"解析中: {file_name}, 状态: {state}")

        if all_done:
            full_zip_urls = [r["full_zip_url"] for r in extract_results if r["state"] == "done"]
            logger.info(f"batch_id={batch_id} 全部解析完成，共 {len(full_zip_urls)} 个结果")
            return full_zip_urls

        time.sleep(3)


def mineru_download_and_extract(
    full_zip_urls: list[str], output_dir: Path, names: list[str] | None = None
) -> list[str]:
    """下载 MinerU 解析结果 zip 并解压到 output_dir/{name}/ 下，返回目录路径字符串列表

    names: 自定义目录名列表，长度与 full_zip_urls 一致。不传则用 URL 的 stem。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_dirs: list[str] = []
    for i, url in enumerate(full_zip_urls):
        zip_name = names[i] if names else Path(url).stem
        extract_dir = output_dir / zip_name

        # 已存在则跳过，保证重复运行幂等
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"跳过（已存在）: {extract_dir}")
            extracted_dirs.append(str(extract_dir))
            continue

        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"下载: {url}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        zip_path = extract_dir / f"{zip_name}.zip"
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"解压到: {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        zip_path.unlink()
        extracted_dirs.append(str(extract_dir))
        logger.info(f"完成: {extract_dir}")

    logger.info(f"全部下载解压完成，共 {len(extracted_dirs)} 个目录")
    return extracted_dirs


# 主节点
@log_node
def split_contents(state: TextBookState):
    """处理教材目录。

    Args:
        state: 当前工作流状态，含 textbook_path / task_id。

    Returns:
        更新后的状态（写入 extracted_contents_dirs）。
    """
    textbook_path = Path(state.get("textbook_path"))
    output_dir = textbook_path / "mineru_toc"

    task_id = state.get("task_id")
    if task_id:
        update_task(task_id=task_id, message="开始解析教材目录（MinerU）", progress=0.15)

    # 幂等：如果 mineru_toc 已有解析结果，直接复用
    if output_dir.exists() and any(d.is_dir() and (d / "full.md").exists() for d in output_dir.iterdir()):
        # 从已有目录重新收集
        extracted_dirs = [str(d) for d in output_dir.iterdir() if d.is_dir() and (d / "full.md").exists()]
        state["extracted_contents_dirs"] = extracted_dirs
        logger.info(f"mineru_toc 已存在，跳过解析，共 {len(extracted_dirs)} 个目录")
        if task_id:
            update_task(task_id=task_id, message="目录解析结果已存在，直接复用", progress=0.4)
        return state

    toc_dir = textbook_path / "pdf_toc"
    # 获取所有 PDF
    pdfs = find_pdfs(Path(textbook_path))
    if not pdfs:
        raise ValueError("没有找到 PDF 文件")
    # 切目录页
    toc_files = cut_toc_pages(toc_dir, pdfs, toc_pages=30)
    # 获取解析结果
    full_zip_urls = mineru_upload_and_poll(toc_files, toc_dir)
    # 下载并解压解析结果（用教材名命名目录）
    toc_names = [f.stem.replace("_toc", "") for f in toc_files]
    output_dir = toc_dir.parent / "mineru_toc"
    extracted_dirs = mineru_download_and_extract(full_zip_urls, output_dir, names=toc_names)

    state["extracted_contents_dirs"] = extracted_dirs
    if task_id:
        update_task(task_id=task_id, message=f"目录解析完成，共 {len(extracted_dirs)} 本教材", progress=0.4)
    return state


# 单元测试
if __name__ == "__main__":
    state: TextBookState = {
        "textbook_exists": False,
        "textbook_path": "D:/PycharmProjects/grad_assist/textbooks/pdf/pdf_toc",
    }
    split_contents(state)
