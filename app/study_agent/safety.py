"""外部内容边界（prompt 注入防护）：把不可信正文包进带随机串的数据边界。

背景：教材正文（PDF/MinerU 解析）、网页摘要、文件正文都会进模型上下文，其中若出现
「忽略以上指令…」这类文字，模型可能把它当指令执行。防线分两层：
本模块负责结构（明确的数据边界 + 内容清洗），提示词层负责语义
（``study_chat.prompt`` 底线：边界内一律当资料，不当指令）。

边界为什么带随机串：固定分隔符（如 ``<data>…</data>``）可被正文伪造闭合标签、把后续系统
文本「圈」进自己名下；随机串在包裹时才生成，正文作者不可能预知，伪造闭合即失效。每次包裹
新生成一个并冻结进文本——同一轮的重发必须字节相同，否则前缀缓存每轮全断。

清洗的取舍：只删 Unicode Cf 类字符与正文里伪造的边界写法；
不做 NFKC——教材里的全角括号、代码缩进要保真，规范化会把它们改形。
"""

from __future__ import annotations

import re
import secrets
import unicodedata

from app.core import load_prompt

# 正文里伪造的边界写法（含中英文括号变体）：中和为「（外部资料…」，杜绝视觉混淆
_FAKE_BOUNDARY_RE = re.compile(r"[〔【\[（(]\s*/?\s*外部资料")
# 展示侧要去掉的边界标记行（模型侧不经过这里）：连同标记行前后的分隔换行一起吃掉
_DISPLAY_MARKER_RE = re.compile(r"\n?^〔/?外部资料·[^〕]*〕\n?", re.MULTILINE)
# 输出侧留痕的敏感形态：系统提示词被诱导复述 / 配置串漏出
_LEAK_PATTERNS = (
    re.compile(r"api[_\s\-]?key", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
)
# 连续片段命中长度：系统提示词里任意 ≥ 该长度的连续文本出现在回答里即视为复述
_LEAK_SPAN = 24


def scrub_untrusted(text: str) -> str:
    """清洗进入边界前的正文：删 Cf 控制符、中和伪造的边界前缀（其余内容保真）。"""
    cleaned = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _FAKE_BOUNDARY_RE.sub("（外部资料", cleaned)


def wrap_untrusted(text: str, kind: str) -> str:
    """把不可信正文包进 ``〔外部资料·<类型> <随机串>〕`` … ``〔/外部资料·<类型> <随机串>〕``。

    空白文本原样返回（空边界对会给模型「这里有内容」的错误暗示）。
    调用点只在「模型可见的 content」上用；artifact / 落盘 / 前端回显继续用干净原文。

    Args:
        text: 不可信正文（教材片段、网页摘要、文件内容等）。
        kind: 中文类型名（教材 / 网页 / 章节 / 文件），随边界一同给模型看。
    """
    body = scrub_untrusted(text)
    if not body.strip():
        return body
    nonce = secrets.token_hex(3)
    return f"〔外部资料·{kind} {nonce}〕\n{body}\n〔/外部资料·{kind} {nonce}〕"


def strip_boundary(text: str) -> str:
    """展示侧剥掉边界标记行（正文原样保留）——模型侧的防护标记不该漏给用户看。

    仅用于前端展示路径（SSE 工具事件 / /history 回显）；模型侧 content 永远带边界。
    """
    return _DISPLAY_MARKER_RE.sub("", text)


def detect_prompt_leak(answer: str, system_prompt: str | None = None) -> bool:
    """输出侧留痕：答案是否疑似复述了系统提示词、或漏出配置形态（只检测，不拦截）。

    流式回答逐 token 已发给前端、事后无法撤回，所以这层只能把「模型没守住」变成
    可观测（metrics 留痕 + 告警），不能当闸门。system_prompt 缺省取静态壳。

    Args:
        answer: 本轮完整回答文本。
        system_prompt: 比对基准；缺省为 study_chat 静态壳。
    """
    if not answer:
        return False
    if any(pattern.search(answer) for pattern in _LEAK_PATTERNS):
        return True
    prompt = system_prompt if system_prompt is not None else load_prompt("study_chat")
    for i in range(max(len(answer) - _LEAK_SPAN, 0) + 1):
        if answer[i : i + _LEAK_SPAN] in prompt:
            return True
    return False
