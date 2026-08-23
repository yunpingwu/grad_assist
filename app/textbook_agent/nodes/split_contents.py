import asyncio
import os
import shutil
import uuid
import zipfile
from pathlib import Path

import httpx
from langgraph.types import StreamWriter
from pypdf import PdfReader, PdfWriter

from app.config import mineru_config
from app.core import log_node, logger
from app.textbook_agent.state import TextBookState


def find_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.suffix == ".pdf")
    return [path] if path.suffix == ".pdf" else []


# 步骤 1：切目录页
def cut_toc_pages(toc_dir: Path, pdfs: list[Path], toc_pages: int = 30) -> list[Path]:
    """切割教材的前30页（关注里面包含的目录）"""
    toc_dir.mkdir(parents=True, exist_ok=True)

    toc_files = []
    for pdf in pdfs:
        toc = toc_dir / f"{pdf.stem}_toc.pdf"

        # 已存在则跳过：目录页为确定性产物，存在即视为完整生成
        if toc.exists():
            toc_files.append(toc)
            continue

        reader = PdfReader(str(pdf))
        writer = PdfWriter()
        for i in range(min(toc_pages, len(reader.pages))):
            writer.add_page(reader.pages[i])

        # 先写临时文件，再原子替换，避免中断残留半个 PDF
        tmp = toc_dir / f"{pdf.stem}_toc.pdf.tmp"
        with open(tmp, "wb") as f:
            writer.write(f)
        os.replace(tmp, toc)
        toc_files.append(toc)

    logger.info(f"切割出 {len(toc_files)} 个目录文件")
    return toc_files


# 步骤 2：MinerU 在线 API（异步：上传 + 轮询）
async def mineru_upload_and_poll(
    toc_files: list[Path], toc_dir: Path, max_retries: int = 2
) -> list[str | None]:
    """利用 mineru 解析 pdf 文件，对解析失败的文件进行有限次重试。

    完整流程：
    1. POST /api/v4/file-urls/batch 获取预签名上传URL和batch_id
    2. PUT 上传每个文件到对应URL（系统自动提交解析任务）
    3. GET /api/v4/extract-results/batch/{batch_id} 轮询直到全部完成

    对 state=failed 的文件最多重试 max_retries 次，仍失败的以 None 占位返回，
    由下载环节跳过（既不 raise 中断整批，也不因丢项导致下载错位）。

    Returns:
        与 toc_files 一一对应的 full_zip_url 列表，失败项为 None。
    """
    token = mineru_config.token
    base_url = mineru_config.url
    header = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    url_map: dict[str, str] = {}  # 文件名 -> full_zip_url（成功）
    remaining = list(toc_files)

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(max_retries + 1):
            if not remaining:
                break

            # ========== Step 1: 获取上传URL ==========
            files_info = [{"name": f.name, "data_id": str(uuid.uuid4())[:8]} for f in remaining]
            data = {"files": files_info, "model_version": "vlm"}
            logger.info(f"请求上传URL，共 {len(remaining)} 个文件（第 {attempt + 1} 轮）")
            resp = await client.post(f"{base_url}/api/v4/file-urls/batch", headers=header, json=data)
            resp.raise_for_status()
            result = resp.json()
            if result["code"] != 0:
                raise RuntimeError(f"获取上传URL失败: {result['msg']}")

            batch_id = result["data"]["batch_id"]
            urls = result["data"]["file_urls"]

            # ========== Step 2: 上传文件到预签名URL ==========
            for i, (file_path, upload_url) in enumerate(zip(remaining, urls, strict=True)):
                with open(file_path, "rb") as f:
                    put_resp = await client.put(upload_url, content=f.read())
                if put_resp.status_code != 200:
                    raise RuntimeError(f"上传失败: {file_path.name}, HTTP {put_resp.status_code}")

            # ========== Step 3: 轮询解析结果 ==========
            poll_url = f"{base_url}/api/v4/extract-results/batch/{batch_id}"
            start_time = asyncio.get_running_loop().time()
            while True:
                if asyncio.get_running_loop().time() - start_time > 600:
                    raise TimeoutError(f"轮询超时，batch_id={batch_id}")
                poll_resp = await client.get(poll_url, headers=header)
                poll_resp.raise_for_status()
                poll_result = poll_resp.json()
                if poll_result["code"] != 0:
                    raise RuntimeError(f"查询解析结果失败: {poll_result['msg']}")

                extract_results = poll_result["data"]["extract_result"]
                running = [r for r in extract_results if r["state"] not in ("done", "failed")]
                if running:
                    for r in running:
                        progress = r.get("extract_progress", {})
                        if progress:
                            logger.info(
                                f"解析中: {r['file_name']}, 状态: {r['state']}, "
                                f"进度: {progress.get('extracted_pages', '?')}/{progress.get('total_pages', '?')}"
                            )
                        else:
                            logger.info(f"解析中: {r['file_name']}, 状态: {r['state']}")
                    await asyncio.sleep(3)
                    continue

                # 本轮批次全部结束，分拣成功/失败
                failed_names = []
                for r in extract_results:
                    if r["state"] == "done":
                        url_map[r["file_name"]] = r["full_zip_url"]
                        logger.info(f"解析完成: {r['file_name']}")
                    elif r["state"] == "failed":
                        failed_names.append(r["file_name"])
                        logger.error(f"解析失败: {r['file_name']}, 原因: {r.get('err_msg', '未知')}")
                break

            if not failed_names:
                break
            failed_set = set(failed_names)
            remaining = [f for f in toc_files if f.name in failed_set]
            logger.warning(f"有 {len(failed_names)} 个解析失败，将重试（剩余次数 {max_retries - attempt}）")

    still_failed = [f.name for f in toc_files if f.name not in url_map]
    if still_failed:
        logger.error(f"解析仍失败（已重试 {max_retries} 次），跳过: {still_failed}")

    # 按 toc_files 顺序返回，失败项为 None
    return [url_map.get(f.name) for f in toc_files]


async def mineru_download_and_extract(
    full_zip_urls: list[str | None], output_dir: Path, names: list[str] | None = None
) -> list[str]:
    """下载 MinerU 解析结果 zip 并解压到 output_dir/{name}/ 下，返回目录路径字符串列表

    names: 自定义目录名列表，长度与 full_zip_urls 一致。不传则用 URL 的 stem。
    full_zip_urls 中可能含 None（解析失败上传环节占位），对应项跳过不下载。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_dirs: list[str] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i, url in enumerate(full_zip_urls):
            if url is None:
                continue
            zip_name = names[i] if names else Path(url).stem
            extract_dir = output_dir / zip_name

            # 已完成标志：关键产物 full.md 存在才跳过（半成品目录不满足，会被重做）
            if (extract_dir / "full.md").exists():
                logger.info(f"跳过（已存在）: {extract_dir}")
                extracted_dirs.append(str(extract_dir))
                continue

            # 临时路径：zip 与解压产物都先落在 staging，避免中断留下半成品
            zip_tmp = output_dir / f".{zip_name}.zip.part"
            staging = output_dir / f".{zip_name}.tmp"

            # 清理上次中断可能残留的半成品
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            if zip_tmp.exists():
                zip_tmp.unlink()

            logger.info(f"下载: {url}")
            resp = await client.get(url)
            resp.raise_for_status()

            # 先写入 .part 文件，避免在最终目录内留下不完整的 zip
            zip_tmp.write_bytes(resp.content)

            logger.info(f"解压到临时目录: {staging}")
            with zipfile.ZipFile(zip_tmp, "r") as zf:
                zf.extractall(staging)

            # 校验关键产物完整后再原子提交
            if not (staging / "full.md").exists():
                raise RuntimeError(f"解压结果缺少 full.md: {zip_name}")

            # 原子提交：整本教科书目录作为一个单元出现，要么完整要么不存在
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            os.replace(staging, extract_dir)

            zip_tmp.unlink()
            extracted_dirs.append(str(extract_dir))
            logger.info(f"完成: {extract_dir}")

    logger.info(f"全部下载解压完成，共 {len(extracted_dirs)} 个目录")
    return extracted_dirs


# 主节点
@log_node
async def split_contents(state: TextBookState, *, writer: StreamWriter) -> dict:
    """处理教材目录。

    Args:
        state: 当前工作流状态，含 textbook_path / task_id。

    Returns:
        更新后的状态（写入 extracted_contents_dirs）。
    """
    textbook_path = Path(state.get("textbook_path"))
    output_dir = textbook_path / "mineru_toc"

    writer({"type": "message", "status": "running", "message": "开始解析教材目录（MinerU）", "progress": 0.15})

    # 获取所有 PDF（期望解析的教材全集）
    pdfs = find_pdfs(Path(textbook_path))
    if not pdfs:
        raise ValueError("没有找到 PDF 文件")

    # 幂等：只有当 “全部教材” 的 full.md 都完整存在时才短路，避免把半成品当成已完成
    if output_dir.exists() and all((output_dir / pdf.stem / "full.md").exists() for pdf in pdfs):
        extracted_dirs = [str(output_dir / pdf.stem) for pdf in pdfs]
        state["extracted_contents_dirs"] = extracted_dirs
        logger.info(f"mineru_toc 已存在，跳过解析，共 {len(extracted_dirs)} 个目录")
        writer({"type": "message", "status": "running", "message": "目录解析结果已存在，直接复用", "progress": 0.4})
        return state

    toc_dir = textbook_path / "pdf_toc"
    # 切目录页（内部按教材名跳过已完成项）
    toc_files = cut_toc_pages(toc_dir, pdfs, toc_pages=30)

    # 仅上传尚未解析（无 full.md）的教材，避免重复上传浪费 MinerU 调用
    pending = []
    toc_names = []
    for toc_file in toc_files:
        name = toc_file.stem.replace("_toc", "")
        if (output_dir / name / "full.md").exists():
            continue
        pending.append(toc_file)
        toc_names.append(name)

    if pending:
        full_zip_urls = await mineru_upload_and_poll(pending, toc_dir)
        await mineru_download_and_extract(full_zip_urls, output_dir, names=toc_names)

    # 此时全部教材 full.md 就绪，按 pdfs 顺序重建目录列表，对齐 split 的 sorted 顺序
    extracted_dirs = [str(output_dir / toc_file.stem.replace("_toc", "")) for toc_file in toc_files]

    state["extracted_contents_dirs"] = extracted_dirs
    writer({"type": "message","status": "running","message": f"目录解析完成，共 {len(extracted_dirs)} 本教材","progress": 0.4,})
    return state


# 单元测试
if __name__ == "__main__":
    import asyncio

    def writer(chunk):
        print("event:", chunk)

    state: TextBookState = {"textbook_exists": False}
    asyncio.run(split_contents(state, writer=writer))
