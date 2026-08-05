import json
import re
import uuid
from pathlib import Path

from app.clients import milvus_client
from app.core import log_node, logger
from app.textbook_agent.state import TextBookState
from app.utils import generate_embeddings
from app.utils import (
    get_collection_by_name,
    next_collection_name,
    register_textbook,
)
from app.utils import upload_and_map
from app.utils import update_task

# 最小块字符数
MIN_CHUNK_SIZE = 500
# 最大块字符数
CHUNK_SIZE = 2000
# 块重叠字符数
OVERLAP = 200
# 被合并最大字符数
MERGE_TARGET = 1500

# 匹配 Markdown 图片语法: ![...](images/xxx.jpg)
_IMAGE_PATTERN = re.compile(r'!\[.*?\]\((images/.*?)\)')
# 匹配围栏代码块: ```lang\n...\n```
_CODE_PATTERN = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
# 按 ## 小节标题分割
_SECTION_SPLIT = re.compile(r'(?=^## )', re.MULTILINE)


def _extract_images_and_code(text: str) -> tuple[str, list[dict], list[dict]]:
    """从文本中提取图片引用和代码块，返回(清理后文本, 图片列表, 代码列表)"""
    images = [{"path": m.group(1)} for m in _IMAGE_PATTERN.finditer(text)]
    text = _IMAGE_PATTERN.sub('', text)

    codes = [{"language": m.group(1) or "text", "code": m.group(2).strip()}
             for m in _CODE_PATTERN.finditer(text)]
    text = _CODE_PATTERN.sub('', text)

    return text.strip(), images, codes


def _char_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """按字符数硬切，带 overlap"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _merge_small_chunks(chunks: list[dict]) -> list[dict]:
    """合并同一 section 内过小的 chunk（< 500 字符）。
    
    合并后若仍 < 1500 字符，继续向后贪心合并，避免碎片浪费。
    """
    if not chunks:
        return chunks

    # 按 (chapter) 分组
    groups: list[list[dict]] = []
    current_group = [chunks[0]]
    for c in chunks[1:]:
        last = current_group[-1]
        if c["chapter"] == last["chapter"]:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)

    def _combine(a: dict, b: dict) -> dict:
        return {
            **a,
            "content": a["content"] + "\n" + b["content"],
            "images": a["images"] + b["images"],
            "codes": a["codes"] + b["codes"],
        }

    merged = []
    for group in groups:
        i = 0
        while i < len(group):
            cur = group[i]
            if (i + 1 < len(group)
                    and len(cur["content"]) < MIN_CHUNK_SIZE
                    and len(cur["content"]) + len(group[i + 1]["content"]) <= CHUNK_SIZE):
                # 合并到 group[i]，删除 group[i+1]
                group[i] = _combine(cur, group[i + 1])
                del group[i + 1]
                # 合并后若仍较小（< 1500），i 不变继续贪心合并下一个
                if len(group[i]["content"]) >= MERGE_TARGET:
                    i += 1
            else:
                i += 1
        merged.extend(group)

    _renumber_chunks(merged)
    return merged


def _renumber_chunks(chunks: list[dict]) -> None:
    """按 (chapter, section) 分组，重新给 chunk_index / total_chunks 编号"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        key = (c["chapter"], c["section"])
        groups.setdefault(key, []).append(c)

    for group in groups.values():
        total = len(group)
        for idx, c in enumerate(group):
            c["total_chunks"] = total
            c["chunk_index"] = idx


# ==================== 文本切割 ====================

def chunk_textbook(textbook_name: str, chapter_dirs: list[str]) -> list[dict]:
    """对一本教材的所有章节做文本切割。

    策略：
    - 按 ## 小节标题自然分割
    - 小节 ≤ 2000 字符：直接作为一个 chunk
    - 小节 > 2000 字符：按字符数硬切（overlap=200）
    - 公式保留在文本中不单独处理
    - 图片引用提取后上传到 MinIO，并携带 object_name / url 存入 images 字段
    - 代码块提取后存入 codes 字段

    返回每个 chunk 包含:
        content, textbook_name, chapter, section,
        chunk_index, total_chunks, images, codes
    """
    all_chunks: list[dict] = []

    for ch_dir_str in sorted(chapter_dirs):
        ch_dir = Path(ch_dir_str)
        md_path = ch_dir / "full.md"
        if not md_path.exists():
            logger.warning(f"跳过，缺少 full.md: {ch_dir}")
            continue

        full_text = md_path.read_text(encoding="utf-8")
        chapter_title = ch_dir.name
        first_line = full_text.split("\n", 1)[0].strip()
        if first_line.startswith("# "):
            chapter_title = first_line[2:]

        # 上传本章 markdown 引用的图片，构建 相对路径 → {object_name, url} 映射
        rel_to_image = upload_and_map(
            chapter_dir=ch_dir,
            textbook_name=textbook_name,
            chapter=ch_dir.name,
            rel_paths=set(_IMAGE_PATTERN.findall(full_text)),
        )

        # ---- 按 ## 拆分成 section ----
        sections = _SECTION_SPLIT.split(full_text)
        current_section_title = ""

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # 提取 ## 标题
            header_match = re.match(r'^## (.+)', sec)
            if header_match:
                current_section_title = header_match.group(1).strip()

            # 提取图片和代码（公式保持原样）
            clean_text, images, codes = _extract_images_and_code(sec)
            if not clean_text:
                continue

            # 为图片引用补充 MinIO object_name / url（未上传成功的保持仅有 path）
            for img in images:
                info = rel_to_image.get(img["path"])
                if info:
                    img.update(info)

            # ---- 切分 ----
            if len(clean_text) <= CHUNK_SIZE:
                chunk = {
                    "content": f"# {chapter_title} > ## {current_section_title}\n{clean_text}",
                    "textbook_name": textbook_name,
                    "chapter": chapter_title,
                    "section": current_section_title,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "images": images,
                    "codes": codes,
                }
                all_chunks.append(chunk)
            else:
                sub_texts = _char_split(clean_text)
                total = len(sub_texts)
                for idx, sub in enumerate(sub_texts):
                    chunk = {
                        "content": f"# {chapter_title} > ## {current_section_title}\n{sub}",
                        "textbook_name": textbook_name,
                        "chapter": chapter_title,
                        "section": current_section_title,
                        "chunk_index": idx,
                        "total_chunks": total,
                        "images": images if idx == 0 else [],
                        "codes": codes if idx == 0 else [],
                    }
                    all_chunks.append(chunk)

    all_chunks = _merge_small_chunks(all_chunks)
    logger.info(f"[{textbook_name}] 切割完成: {len(chapter_dirs)} 章 → {len(all_chunks)} 个 chunk")
    return all_chunks


# ==================== 向量化 + 存储 ====================

# 每批 embedding 的最大文本数，避免 OOM
_EMBED_BATCH_SIZE = 64

def embed_and_store(textbook_name: str, chunks: list[dict]) -> bool:
    """将切割好的 chunks 向量化后存入 Milvus。

    策略：
    - BGE-M3 生成 dense + sparse 双向量（混合检索）
    - 每本教材独立 collection（tb_XX），并登记到注册表
    - 文本块以 block_type="text" 入库
    - 代码块以 block_type="code" 单独入库，有独立 embedding
    - 图片引用存入 metadata_json，不单独建向量

    幂等性：
    - 注册表中已登记该教材 → 直接跳过，不重复向量化

    Returns:
        True 表示入库成功（或已入库跳过）
    """
    if not chunks:
        logger.warning(f"[{textbook_name}] chunks 为空，跳过入库")
        return False

    milvus_client.connect_milvus()

    # 幂等：注册表中已存在该教材，说明已完成入库，跳过
    existing = get_collection_by_name(textbook_name)
    if existing:
        logger.info(f"[{textbook_name}] 已入库（collection={existing}），跳过重复摄入")
        return True

    # 分配独立 collection 并确保索引就绪
    collection_name = next_collection_name()
    milvus_client.create_collection(collection_name)
    if not milvus_client.collection_exists(collection_name):
        logger.error(f"集合 {collection_name} 不存在，入库失败")
        return False
    milvus_client.create_indexes(collection_name)

    # 收集所有待入库的文本
    entries: list[dict] = []  # {text, block_type, chapter, section, ...}
    for c in chunks:
        # 文本块
        entries.append({
            "text": c["content"],
            "block_type": "text",
            "textbook_name": c["textbook_name"],
            "chapter": c["chapter"],
            "section": c["section"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "metadata_json": json.dumps({"images": c.get("images", [])}, ensure_ascii=False),
        })
        # 代码块
        for code in c.get("codes", []):
            code_text = f"```{code['language']}\n{code['code']}\n```"
            entries.append({
                "text": code_text,
                "block_type": "code",
                "textbook_name": c["textbook_name"],
                "chapter": c["chapter"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                "metadata_json": json.dumps({"language": code["language"]}, ensure_ascii=False),
            })

    # 分批生成 embedding
    total_inserted = 0
    for batch_start in range(0, len(entries), _EMBED_BATCH_SIZE):
        batch_entries = entries[batch_start:batch_start + _EMBED_BATCH_SIZE]
        texts = [e["text"] for e in batch_entries]

        emb = generate_embeddings(texts)
        dense_vecs = emb["dense"]
        sparse_vecs = emb["sparse"]

        # 组装插入行
        rows: list[dict] = []
        for i, entry in enumerate(batch_entries):
            chunk_id = f"{textbook_name}_{batch_start + i}_{uuid.uuid4().hex[:8]}"
            rows.append({
                "id": chunk_id,
                "text": entry["text"],
                "embedding": dense_vecs[i],
                "sparse_embedding": sparse_vecs[i] if i < len(sparse_vecs) else {},
                "block_type": entry["block_type"],
                "textbook_name": entry["textbook_name"],
                "chapter": entry["chapter"],
                "section": entry["section"],
                "chunk_index": entry["chunk_index"],
                "total_chunks": entry["total_chunks"],
                "metadata_json": entry["metadata_json"],
            })

        inserted = milvus_client.batch_insert(collection_name, rows)
        total_inserted += inserted

    # 入库成功后登记到注册表，供检索时按教材名定位
    register_textbook(textbook_name, collection_name, total_inserted)
    logger.info(f"[{textbook_name}] 入库完成: {total_inserted} 行（collection={collection_name}）")
    return total_inserted > 0


# ==================== 主节点 ====================

@log_node
def split_text_and_store(state: TextBookState) -> dict:
    """遍历所有教材的章节解析结果，逐本切割 → 向量化入库"""

    # 幂等：如果已完成摄入，直接跳过
    if state.get("ingestion_done"):
        logger.info("ingestion_done 已为 True，跳过重复摄入")
        return state

    task_id = state.get("task_id")
    if task_id:
        update_task(task_id=task_id, message="开始切块与向量化入库", progress=0.8)

    extracted_dirs = state.get("extracted_dirs", [])
    if not extracted_dirs:
        logger.warning("extracted_dirs 为空，无章节解析结果可处理")
        state["ingestion_done"] = False
        if task_id:
            update_task(task_id=task_id, message="无章节解析结果，任务终止", progress=1.0)
        return state

    # 按教材名分组: mineru_split/{教材名}/{章节名}/
    grouped: dict[str, list[str]] = {}
    for d in extracted_dirs:
        p = Path(d)
        grouped.setdefault(p.parent.name, []).append(d)

    logger.info(f"共 {len(grouped)} 本教材待处理")

    total_chunks = 0
    all_chunk_contents: list[str] = []
    total_textbooks = len(grouped)
    for idx, (textbook_name, chapter_dirs) in enumerate(grouped.items()):
        chunks = chunk_textbook(textbook_name, chapter_dirs)
        if not chunks:
            logger.warning(f"[{textbook_name}] 切割结果为空，跳过")
            continue

        if embed_and_store(textbook_name, chunks):
            total_chunks += len(chunks)
            all_chunk_contents.extend(c["content"] for c in chunks)
            if task_id:
                progress = 0.8 + 0.2 * (idx + 1) / total_textbooks
                update_task(
                    task_id=task_id,
                    message=f"[{textbook_name}] 入库完成（{len(chunks)} chunks）",
                    progress=progress,
                )
        else:
            logger.error(f"[{textbook_name}] 入库失败")

    state["ingestion_done"] = total_chunks > 0
    logger.info(f"全部完成: {total_chunks} 个 chunk 已入库")
    if task_id:
        update_task(task_id=task_id, message=f"全部完成：{total_chunks} 个 chunk 已入库", progress=1.0)
    return state


# ==================== 单元测试 ====================
if __name__ == '__main__':
    textbook_path = Path("D:/Projects/grad_assist/textbooks/pdf")
    split_dirs = list((textbook_path / "mineru_split").iterdir())

    state: TextBookState = {
        "textbook_exists": False,
        "textbook_path": str(textbook_path),
        # 仅测试 C 语言教材
        "extracted_dirs": [str(d) for d in (textbook_path / "mineru_split" / "C语言程序设计（第五版）_(谭浩强)_(z-library.sk,_1lib.sk,_z-lib.sk)").iterdir() if d.is_dir()],
    }

    split_text_and_store(state)
