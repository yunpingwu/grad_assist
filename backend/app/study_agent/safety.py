"""外部内容边界：把教材/网页/文件正文包进随机串数据边界，并清洗伪造边界的写法。

边界串每次包裹时新生成并冻结进文本：正文无法预知，伪造闭合即失效；同一轮重发
必须字节一致，否则前缀缓存断裂。
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
    """把不可信正文包进带随机串的边界；空白文本原样返回。

    随机串在包裹时新生成并冻结进正文（同轮重发必须字节一致）；只用于模型可见的 content，
    artifact / 落盘 / 前端回显继续用干净原文。
    """
    body = scrub_untrusted(text)
    if not body.strip():
        return body
    nonce = secrets.token_hex(3)
    return f"〔外部资料·{kind} {nonce}〕\n{body}\n〔/外部资料·{kind} {nonce}〕"


def strip_boundary(text: str) -> str:
    """展示侧剥掉边界标记行；仅用于 SSE 工具事件与 /history 回显。"""
    return _DISPLAY_MARKER_RE.sub("", text)


def detect_prompt_leak(answer: str, system_prompt: str | None = None) -> bool:
    """检测回答是否疑似复述系统提示词或漏出配置形态；只留痕告警，不拦截。"""
    if not answer:
        return False
    if any(pattern.search(answer) for pattern in _LEAK_PATTERNS):
        return True
    prompt = system_prompt if system_prompt is not None else load_prompt("study_chat")
    for i in range(max(len(answer) - _LEAK_SPAN, 0) + 1):
        if answer[i : i + _LEAK_SPAN] in prompt:
            return True
    return False
