"""教材章节富化节点:为图片/代码生成简介,写入 full.md 的副本 full_captioned.md。

富化产物供切块入库使用:
- 图片行 alt = 视觉模型生成的简介(切块时原位嵌入正文参与检索);
- 代码块后插入「> 代码说明: …」行(保留在正文,供语义类问题感知代码存在)。
"""

import base64
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

from app.clients.llm import get_llm_client
from app.config import llm_config
from app.core import load_prompt, log_node, logger
from app.textbook_agent.state import TextBookState

# 匹配 Markdown 图片语法: ![alt](images/xxx.jpg),捕获 (alt, 相对路径)
_IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((images/.*?)\)")
# 匹配围栏代码块: ```lang\n...\n```
_CODE_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
# 简介/说明长度上限(兜底截断,防超长)
_CAPTION_LIMIT = 200
# 视觉模型输入图片:最长边压缩上限与 JPEG 质量。
# 768px 实测与 1024px 简介质量相当(表格/结构/宽图均无信息丢失),
# 但图片 token 花费省 ~44%;耗时瓶颈在服务端生成,与分辨率关系不大。
_IMG_MAX_DIM = 768
_IMG_JPEG_QUALITY = 85
# LLM 并发数。视觉并发上限实测:32 并发无 429,64 并发触发 Throttling.Concurrency;
# 单次视觉约 15s,32 并发 ≈ 2.1 req/s ≈ 128 req/min,远低于 RPM 30000。
_LLM_WORKERS = 32
# 生成简介时取匹配位置之前的上下文长度
_IMG_CONTEXT_CHARS = 300
_CODE_CONTEXT_CHARS = 200


def get_model() -> tuple[Any, Any]:
    """返回 (文本模型, 视觉模型),分别用于代码块与图片的富化。"""
    return (
        get_llm_client(llm_config.model),
        get_llm_client(llm_config.visual_model),
    )


def _image_to_base64_data_url(img_path: Path) -> str:
    """读取图片,压缩(最长边 ≤ {_IMG_MAX_DIM}px、统一转 JPEG)后返回 base64 data URI。

    视觉模型请求体是 base64 内嵌图片:原图直传会放大传输与模型处理延迟,
    教材配图多为照片/截图,统一缩放转 JPEG 在保真与体积间取平衡。
    """
    with Image.open(img_path) as im:
        if max(im.size) > _IMG_MAX_DIM:
            im.thumbnail((_IMG_MAX_DIM, _IMG_MAX_DIM))
        if im.mode != "RGB":
            im = im.convert("RGB")  # RGBA/P 等转 RGB,JPEG 不支持透明通道
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=_IMG_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _caption_image(model: Any, prompt: str, chapter_dir: Path, m: re.Match, raw_text: str) -> str:
    """为单个图片生成简介,返回替换后的图片行(alt=简介);失败用文件名兜底。"""
    rel = m.group(2)
    img_path = chapter_dir / rel
    if not img_path.is_file():
        return m.group(0)  # 本地缺失,原样保留

    context = raw_text[max(0, m.start() - _IMG_CONTEXT_CHARS) : m.start()].strip()
    try:
        data_url = _image_to_base64_data_url(img_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt.format(context=context)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        alt = _call_llm(model, message)
    except Exception as exc:
        logger.warning(f"图片简介生成失败({img_path.name}): {exc}")
        alt = ""
    return f"![{alt or rel}]({rel})"


def _caption_code(model: Any, prompt: str, m: re.Match, raw_text: str) -> str:
    """为单个代码块生成说明,返回代码块(成功后追加说明行);失败仅返回代码块。"""
    lang, code = m.group(1), m.group(2)
    context = raw_text[max(0, m.start() - _CODE_CONTEXT_CHARS) : m.start()].strip()
    block = f"```{lang}\n{code}\n```"
    try:
        messages = ChatPromptTemplate.from_template(prompt).format_messages(context=context, code=code)
        desc = _call_llm(model, messages)
    except Exception as exc:
        logger.warning(f"代码说明生成失败: {exc}")
        desc = ""
    return f"{block}\n> 代码说明: {desc}" if desc else block


def _call_llm(model: Any, messages: Any) -> str:
    """统一的 LLM 调用(同步 invoke),返回文本内容;失败返回空串。"""
    try:
        # BaseChatModel.invoke 要求 list / PromptValue / str,兼容单个消息
        if not isinstance(messages, list):
            messages = [messages]
        resp = model.invoke(messages)
        return (resp.content or "").strip()[:_CAPTION_LIMIT]
    except Exception as exc:
        logger.warning(f"LLM 调用失败: {exc}")
        return ""


def _rebuild_text(
    raw_text: str,
    tasks: list[tuple[str, re.Match, int, int]],
    results: list[str],
) -> str:
    """按位置切片重组:原文段落与处理结果交替拼接。"""
    parts: list[str] = []
    cursor = 0
    for (_, _, start, end), res in zip(tasks, results, strict=True):
        parts.append(raw_text[cursor:start])  # 保留区间前的原文
        parts.append(res)  # 填入处理结果
        cursor = end
    parts.append(raw_text[cursor:])  # 尾部剩余原文
    return "".join(parts)


def code_and_image_caption(extracted_dirs) -> None:
    """遍历章节目录,为图片/代码生成简介,写入 full.md 的副本 full_captioned.md。

    流程:
    1. 收集: 跳过已富化章节(幂等),复制副本,收集各章任务(图片/代码按位置排序);
    2. 执行: 所有章节的所有任务扁平化,统一线程池并发调 LLM(跨章并行);
    3. 重组: 按章节分组,位置切片重组后写回各自副本。
    单个任务失败仅降级(图片 alt 用文件名、代码块不插说明行),不阻断整章。
    """
    text_model, visual_model = get_model()
    image_prompt = load_prompt("image_introduction")
    code_prompt = load_prompt("code_introduction")

    # 1. 收集各章计划(复制副本 + 任务列表)
    plans: list[dict] = []  # {dir, raw_text, tasks}
    for chapter_dir in extracted_dirs:
        chapter_dir = Path(chapter_dir)
        md_path = chapter_dir / "full.md"
        if not md_path.exists():
            logger.warning(f"跳过,缺少 full.md: {chapter_dir}")
            continue

        captioned_path = chapter_dir / "full_captioned.md"
        if captioned_path.exists():
            logger.info(f"副本已存在,跳过富化: {captioned_path}")
            continue

        # 1. 收集本计划(副本由处理完成后原子生成,见步骤 3——避免中断留下半成品副本)
        raw_text = md_path.read_text(encoding="utf-8")

        # 收集所有待处理匹配(图片+代码,位置可能交错),按 start 排序保证切片正确
        tasks: list[tuple[str, re.Match, int, int]] = []  # (kind, match, start, end)
        for m in _IMAGE_PATTERN.finditer(raw_text):
            tasks.append(("image", m, m.start(), m.end()))
        for m in _CODE_PATTERN.finditer(raw_text):
            tasks.append(("code", m, m.start(), m.end()))
        tasks.sort(key=lambda t: t[2])

        plans.append({"dir": chapter_dir, "raw_text": raw_text, "tasks": tasks})

    if not plans:
        logger.info("无待富化章节")
        return

    # 执行前总览:让长时间执行期间也有可见进度(此前收集阶段后即静默)
    total_tasks = sum(len(p["tasks"]) for p in plans)
    logger.info(f"开始富化: {len(plans)} 章 / {total_tasks} 个任务,并发 {_LLM_WORKERS}")
    for ci, plan in enumerate(plans):
        n_img = sum(1 for t in plan["tasks"] if t[0] == "image")
        logger.info(f"  [{ci + 1}/{len(plans)}] 提交: {plan['dir'].name}(图 {n_img} / 码 {len(plan['tasks']) - n_img})")

    # 2. 扁平化所有章节任务,统一并发(跨章并行;上下文取各自 raw_text,天然无污染)
    flat: list[tuple[int, tuple]] = [(ci, task) for ci, plan in enumerate(plans) for task in plan["tasks"]]

    def _run(item: tuple[int, tuple]) -> str:
        ci, task = item
        plan = plans[ci]
        kind, m, _, _ = task
        if kind == "image":
            return _caption_image(visual_model, image_prompt, plan["dir"], m, plan["raw_text"])
        return _caption_code(text_model, code_prompt, m, plan["raw_text"])

    with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as ex:
        results = list(ex.map(_run, flat))  # ex.map 保序,flat[i] ↔ results[i]

    # 3. 按章节分组重组,写回各自副本
    per_chapter: list[list[str]] = [[] for _ in plans]
    for (ci, _), res in zip(flat, results, strict=True):
        per_chapter[ci].append(res)

    for ci, plan in enumerate(plans):
        text = _rebuild_text(plan["raw_text"], plan["tasks"], per_chapter[ci])
        captioned_path = plan["dir"] / "full_captioned.md"
        # 原子写:先写临时文件再 rename,进程中断也不会留下未处理的半成品副本
        tmp_path = plan["dir"] / "full_captioned.md.tmp"
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(captioned_path)
        logger.info(f"富化完成: {captioned_path}")


@log_node
def enrich_md(state: TextBookState):
    """富化节点:为每章生成带图片简介/代码说明的 full_captioned.md 副本。"""
    extracted_dirs = state.get("extracted_dirs") or []
    if not extracted_dirs:
        logger.warning("extracted_dirs 为空,跳过富化")
        return state

    code_and_image_caption(extracted_dirs)
    return state
