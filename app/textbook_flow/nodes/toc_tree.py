"""目录树解析（摄入流水线各节点与 toc 缓存共用）。

三层结构（教材 → 章 → 小节）从 mineru_toc/{name}/full.md 正则解析，
纯展示用：只喂前端目录树面板，不参与正文切割的页码计算。
分发方式：split_contents 解析完成后逐本推 SSE toc 事件（首条流实时展示）；
断线重进由 /resolve 的 info 帧经 toc_cache.get_tocs 快照带回，各下游节点不再重复推送。
"""

import re
from pathlib import Path

from app.core import logger


def extract_toc_tree(toc_text: str) -> list[dict]:
    """从目录文本解析三层目录树：章 + 小节（编号 / 标题 / 印刷页码）。

    兼容三种目录格式：
    1. 单行点线引导：``第1章 绪论 …… 3``
    2. 单行无点线（空格直连页码）：``第13章 半监督学习 293``
    3. 标题跨行（页码落在下一行）：``第16章 分布式处理、客户-服务器`` + 下一行 ``和集群……426``

    小节行（``1.1 绪论 …… 3`` / ``1.2.3 …``）挂到编号匹配的当前章下；
    仅用于前端目录树展示，不参与正文切割。解析完成后按印刷页码升序排序，
    消除 MinerU 双栏混排导致的目录原始顺序错乱。
    """
    start_re = re.compile(r"^\s*(第?\s*\d+\s*章)\s+(.*)$")
    page_re = re.compile(r"([…….]{2,}\s*|\s+)(\d+)\s*$")
    section_re = re.compile(r"^\s*\d+\.\d+")  # 小节行，跨行合并标题时遇到即停
    section_head_re = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?)\s+(.*)$")  # 小节标题行

    chapters: list[dict] = []
    lines = toc_text.splitlines()
    i = 0
    while i < len(lines):
        m = start_re.match(lines[i])
        if not m:
            # 章行之外尝试小节行：必须能取到页码且挂在已解析的章下
            sm = section_head_re.match(lines[i])
            if sm and chapters:
                spm = page_re.search(sm.group(2))
                if spm:
                    cur = chapters[-1]
                    chapter_digits = re.sub(r"\D", "", cur["num"])
                    sec_num = sm.group(1)
                    # 编号前缀须匹配当前章（如 1.2 挂 第1章），否则视为混排噪声丢弃
                    if sec_num.startswith(f"{chapter_digits}."):
                        sec_title = sm.group(2)[: spm.start()].strip()
                        cur["sections"].append(
                            {"num": sec_num, "title": sec_title, "printed_page": int(spm.group(2))}
                        )
                i += 1
                continue
            i += 1
            continue

        num = re.sub(r"\s+", "", m.group(1))
        if not num.startswith("第"):
            num = "第" + num

        title_text = m.group(2).strip()
        pm = page_re.search(title_text)
        page = None

        if pm:
            # 首行已含页码：剥离点线/空格及其后的页码
            title = title_text[: pm.start()].strip()
            page = int(pm.group(2))
        else:
            # 标题可能跨行：向后合并延续行直到取到页码（最多 3 行）
            parts = [title_text]
            found = False
            j = i + 1
            while j < len(lines) and j <= i + 3:
                line = lines[j].strip()
                if not line or start_re.match(line) or section_re.match(line):
                    break
                pm2 = page_re.search(line)
                if pm2:
                    parts.append(line[: pm2.start()].strip())
                    page = int(pm2.group(2))
                    found = True
                    break
                parts.append(line)
                j += 1
            title = "".join(parts).strip()
            if not found:
                # 无页码（如在线章节/附录），参与不到正文切割，跳过
                i += 1
                continue

        chapters.append({"num": num, "title": title, "printed_page": page, "sections": []})
        i += 1

    chapters.sort(key=lambda c: c["printed_page"])
    for ch in chapters:
        ch["sections"].sort(key=lambda s: s["printed_page"])
    return chapters


def toc_trees(textbook_path: Path, pdfs: list[Path]) -> list[list[dict] | None]:
    """按 pdfs 顺序解析各教材的目录树（mineru_toc/{name}/full.md）。

    mineru_toc 目录已按教材名重命名，与 PDF 名一致，排序后索引一一对应；
    目录缺失或解析结果不足的教材以 None 占位。
    """
    mineru_toc = textbook_path / "mineru_toc"
    if not mineru_toc.is_dir():
        logger.warning(f"mineru_toc 不存在，目录树不可用: {mineru_toc}")
        return [None] * len(pdfs)
    mineru_dirs = sorted(d for d in mineru_toc.iterdir() if d.is_dir() and (d / "full.md").exists())

    trees: list[list[dict] | None] = []
    for i, pdf_path in enumerate(pdfs):
        if i >= len(mineru_dirs):
            logger.warning(f"缺少 MinerU 解析结果: {pdf_path.name}")
            trees.append(None)
            continue
        full_md = mineru_dirs[i] / "full.md"
        text = full_md.read_text(encoding="utf-8")
        toc_match = re.search(r"##\s*目\s*录", text)
        if not toc_match:
            logger.warning(f"未找到 '## 目录': {full_md}")
        toc_text = text[toc_match.start() :] if toc_match else text
        chapters = extract_toc_tree(toc_text)
        sec_count = sum(len(c["sections"]) for c in chapters)
        logger.info(f"[{pdf_path.stem}] 目录解析到 {len(chapters)} 章 {sec_count} 小节")
        trees.append(chapters)
    return trees


def build_toc_payload(textbook_path: Path) -> list[dict]:
    """按教材目录现状解析全部教材的目录树载荷（与 SSE toc 事件字段同构）。

    返回 [{"textbook": 教材名, "chapters": [三层章表]}]；
    目录或解析产物缺失时返回空列表（无 mineru_toc 属正常中间态，仅记日志）。
    """
    textbook_path = Path(textbook_path)
    if not textbook_path.is_dir():
        return []
    pdfs = sorted(
        f for f in textbook_path.iterdir() if f.suffix == ".pdf" and not f.name.endswith("_toc.pdf")
    )
    if not pdfs:
        return []
    return [
        {"textbook": pdf_path.stem, "chapters": chapters}
        for pdf_path, chapters in zip(pdfs, toc_trees(textbook_path, pdfs), strict=True)
        if chapters
    ]
