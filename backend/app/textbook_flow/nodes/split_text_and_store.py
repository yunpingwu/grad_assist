import asyncio
import json
import re
from pathlib import Path

from langgraph.types import StreamWriter

from app.clients import milvus_client
from app.clients.minio_client import upload_and_map
from app.core import log_node, logger
from app.ml import agenerate_embeddings
from app.retrieval.store import deterministic_collection_name, get_collection_by_name, register_textbook
from app.textbook_flow.chunk_id import build_chunk_id
from app.textbook_flow.state import TextBookState

# 最小块字符数
MIN_CHUNK_SIZE = 500
# 最大块字符数
CHUNK_SIZE = 2000
# 块重叠字符数
OVERLAP = 200
# 被合并最大字符数
MERGE_TARGET = 1500

# 匹配 Markdown 图片语法: ![alt](images/xxx.jpg),捕获 (alt, 相对路径)
_IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((images/.*?)\)")
# 匹配围栏代码块: ```lang\n...\n```
_CODE_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
# 按 ## 小节标题分割
_SECTION_SPLIT = re.compile(r"(?=^## )", re.MULTILINE)


def _extract_images_and_code(text: str) -> tuple[str, list[dict], list[dict]]:
    """提取图片与代码块，返回 (清理后文本, 图片列表, 代码列表)。

    图片行原位替换为「【图: 简介】」并记录 offset，供硬切时归属到正确子块；
    代码块从正文删除，紧随其后的「> 代码说明」行保留。
    """
    images = [{"path": m.group(2), "description": m.group(1).strip()} for m in _IMAGE_PATTERN.finditer(text)]

    def _replace_image(m: re.Match) -> str:
        alt = m.group(1).strip()
        return f"【图: {alt}】" if alt else ""  # 图片本地缺失(富化时原样保留)则整行删除

    text = _IMAGE_PATTERN.sub(_replace_image, text)

    codes = [{"language": m.group(1) or "text", "code": m.group(2).strip()} for m in _CODE_PATTERN.finditer(text)]
    text = _CODE_PATTERN.sub("", text)

    clean_text = text.strip()

    # 原位替换使 images 与「【图: 简介】」标记按顺序一一对应(富化副本 alt 恒非空)；
    # 但代码删除/strip 会改变字符偏移，故在最终的 clean_text 上重新定位每个标记，回填
    # offset 供硬切分支把图片精确归属到图标记实际所在的子块。
    mark_matches = list(re.finditer(r"【图: .*?】", clean_text))
    if len(mark_matches) == len(images):
        for img, m in zip(images, mark_matches, strict=True):
            img["offset"] = m.start()

    return clean_text, images, codes


def _char_split_with_offsets(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[tuple[str, int]]:
    """按字符数硬切，带 overlap；返回每个子块的 (text, start)。

    start 为子块起始字符偏移，供硬切分支把图片精确归属到图标记所在子块。
    """
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        chunks.append((text[start : start + chunk_size], start))
        start += chunk_size - overlap
    return chunks


def _assign_images_to_blocks(images: list[dict], sub_texts: list[tuple[str, int]]) -> list[list[dict]]:
    """把图片按 offset 精确归属到各子块，返回与 sub_texts 等长的列表。

    图片的 offset 即图标记在 clean_text 中的字符偏移。子块区间为
    [start, start+len(text))；相邻子块因 overlap 有重叠区，图标记落入重叠区时归属
    第一个(更靠前)包含它的子块，避免跨 hit 重复回绑。images 已按出现顺序升序。
    """
    assigned: list[list[dict]] = [[] for _ in sub_texts]
    img_idx = 0
    for bi, (sub, sub_start) in enumerate(sub_texts):
        sub_end = sub_start + len(sub)
        while img_idx < len(images) and sub_start <= images[img_idx].get("offset", 0) < sub_end:
            assigned[bi].append(images[img_idx])
            img_idx += 1
    return assigned


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

    merged = []
    for group in groups:
        i = 0
        while i < len(group):
            cur = group[i]
            if (
                i + 1 < len(group)
                and len(cur["content"]) < MIN_CHUNK_SIZE
                and len(cur["content"]) + len(group[i + 1]["content"]) <= CHUNK_SIZE
            ):
                # 合并到 group[i]，删除 group[i+1]
                nxt = group[i + 1]
                group[i] = {
                    **cur,
                    "content": cur["content"] + "\n" + nxt["content"],
                    "images": cur["images"] + nxt["images"],
                    "codes": cur["codes"] + nxt["codes"],
                }
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
    """对一本教材的所有章节做文本切割，返回 chunk 列表。

    读取富化副本 ``full_captioned.md``（缺副本的章节跳过），按 ``##`` 小节切分；
    超长小节按字符硬切，图片与代码块分别存入 ``images`` / ``codes`` 字段。
    """
    all_chunks: list[dict] = []

    for ch_dir_str in sorted(chapter_dirs):
        ch_dir = Path(ch_dir_str)
        md_path = ch_dir / "full_captioned.md"
        if not md_path.exists():
            logger.warning(f"跳过,缺少富化副本 full_captioned.md: {ch_dir}")
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
            rel_paths={rel for _, rel in _IMAGE_PATTERN.findall(full_text)},
        )

        # ---- 按 ## 拆分成 section ----
        sections = _SECTION_SPLIT.split(full_text)
        current_section_title = ""

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # 提取 ## 标题
            header_match = re.match(r"^## (.+)", sec)
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
                content = f"# {chapter_title} > ## {current_section_title}\n{clean_text}"
                chunk = {
                    "content": content,
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
                sub_texts = _char_split_with_offsets(clean_text)
                total = len(sub_texts)
                # 图片按图标记 offset 精确归属到所在子块，保证「图标记文本」与「url 元数据」同块
                images_by_block = _assign_images_to_blocks(images, sub_texts)
                for idx, (sub, _sub_start) in enumerate(sub_texts):
                    chunk = {
                        "content": f"# {chapter_title} > ## {current_section_title}\n{sub}",
                        "textbook_name": textbook_name,
                        "chapter": chapter_title,
                        "section": current_section_title,
                        "chunk_index": idx,
                        "total_chunks": total,
                        "images": images_by_block[idx],
                        "codes": codes if idx == 0 else [],
                    }
                    all_chunks.append(chunk)

    all_chunks = _merge_small_chunks(all_chunks)
    logger.info(f"[{textbook_name}] 切割完成: {len(chapter_dirs)} 章 → {len(all_chunks)} 个 chunk")
    return all_chunks


# ==================== 向量化 + 存储 ====================

# 每批 embedding 的最大文本数，避免 OOM
_EMBED_BATCH_SIZE = 64


async def embed_and_store(textbook_name: str, chunks: list[dict]) -> bool:
    """把 chunks 向量化后写入该教材的独立 collection，并登记注册表。

    已登记的教材直接跳过（幂等）；文本块与代码块分别以 text / code 类型入库。
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

    # 分配独立 collection：由教材名确定性生成，同一教材重跑复用同一集合
    collection_name = deterministic_collection_name(textbook_name)
    if milvus_client.collection_exists(collection_name):
        # 已存在（上次中断残留的半库）→ 删除后重建再重入，避免孤儿 collection；
        # 服务端 2.4.10 不支持 truncate RPC，用 drop + create 等效清空（schema/索引下方统一重建）
        milvus_client.drop_collection(collection_name)
    milvus_client.create_collection(collection_name)

    # 索引就绪：无索引定义则建索引（内部含加载），否则幂等加载
    if milvus_client.list_indexes(collection_name):
        milvus_client.load_collection(collection_name)
    else:
        milvus_client.create_indexes(collection_name)

    # 收集所有待入库的文本
    entries: list[dict] = []  # {text, block_type, chapter, section, ...}
    for c in chunks:
        # 文本块:metadata 瘦身,仅保留 {url, description}(path/object_name 入库后无消费)
        images_meta = [
            {"url": img["url"], "description": img.get("description", "")}
            for img in c.get("images", [])
            if img.get("url")
        ]
        entries.append(
            {
                "text": c["content"],
                "block_type": "text",
                "textbook_name": c["textbook_name"],
                "chapter": c["chapter"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                "metadata_json": json.dumps({"images": images_meta}, ensure_ascii=False),
            }
        )
        # 代码块:保留独立记录,metadata 清空(language 零消费,围栏内已携带)
        for code in c.get("codes", []):
            code_text = f"```{code['language']}\n{code['code']}\n```"
            entries.append(
                {
                    "text": code_text,
                    "block_type": "code",
                    "textbook_name": c["textbook_name"],
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "metadata_json": "{}",
                }
            )

    # 分批生成 embedding
    total_inserted = 0
    for batch_start in range(0, len(entries), _EMBED_BATCH_SIZE):
        batch_entries = entries[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [e["text"] for e in batch_entries]

        # BGE-M3 编码走动态攒批器（内部 to_thread 前向），避免阻塞事件循环
        emb = await agenerate_embeddings(texts)
        dense_vecs = emb["dense"]
        sparse_vecs = emb["sparse"]

        # 组装插入行
        rows: list[dict] = []
        for i, entry in enumerate(batch_entries):
            chunk_id = build_chunk_id(
                textbook_name=textbook_name,
                textbook_version="v1",
                chapter=entry["chapter"],
                section=entry["section"],
                block_type=entry["block_type"],
                chunk_index=entry["chunk_index"],
                text=entry["text"],
            )
            rows.append(
                {
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
                }
            )

        inserted = milvus_client.batch_insert(collection_name, rows)
        total_inserted += inserted

    # 入库成功后登记到注册表，供检索时按教材名定位
    register_textbook(textbook_name, collection_name, total_inserted)
    logger.info(f"[{textbook_name}] 入库完成: {total_inserted} 行（collection={collection_name}）")
    return total_inserted > 0


# ==================== 主节点 ====================


@log_node
async def split_text_and_store(state: TextBookState, *, writer: StreamWriter) -> dict:
    """遍历所有教材的章节解析结果，逐本切割 → 向量化入库"""

    # 幂等：如果已完成摄入，直接跳过
    if state.get("ingestion_done"):
        logger.info("ingestion_done 已为 True，跳过重复摄入")
        return state

    writer({"type": "message", "status": "running", "message": "开始切块与向量化入库", "progress": 0.85})

    extracted_dirs = state.get("extracted_dirs", [])
    if not extracted_dirs:
        logger.warning("extracted_dirs 为空，无章节解析结果可处理")
        state["ingestion_done"] = False
        writer({"type": "message", "status": "running", "message": "无章节解析结果，任务终止", "progress": 1.0})
        return state

    # 按教材名分组: mineru_split/{教材名}/{章节名}/
    grouped: dict[str, list[str]] = {}
    for d in extracted_dirs:
        p = Path(d)
        grouped.setdefault(p.parent.name, []).append(d)

    logger.info(f"共 {len(grouped)} 本教材待处理")

    total_chunks = 0
    total_textbooks = len(grouped)
    for idx, (textbook_name, chapter_dirs) in enumerate(grouped.items()):
        # 切块含文件读写与 MinIO 上传，放线程池避免阻塞事件循环
        chunks = await asyncio.to_thread(chunk_textbook, textbook_name, chapter_dirs)
        if not chunks:
            logger.warning(f"[{textbook_name}] 切割结果为空，跳过")
            continue

        if await embed_and_store(textbook_name, chunks):
            total_chunks += len(chunks)
            progress = 0.85 + 0.15 * (idx + 1) / total_textbooks
            writer({"type": "message","status": "running","message": f"[{textbook_name}] 入库完成（{len(chunks)} chunks）","progress": progress})
        else:
            logger.error(f"[{textbook_name}] 入库失败")

    state["ingestion_done"] = total_chunks > 0
    logger.info(f"全部完成: {total_chunks} 个 chunk 已入库")
    writer({"type": "message","status": "running","message": f"全部完成：{total_chunks} 个 chunk 已入库","progress": 1.0,})
    return state


