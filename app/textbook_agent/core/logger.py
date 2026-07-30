"""封装 logging，提供美观的彩色日志输出。"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# ── 颜色定义 ──────────────────────────────────────────────
class _Color:
    """ANSI 转义码，Windows 10+ 终端 / 现代终端均支持。"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


# 日志级别 → (标签, 前景色, 背景色)
_LEVEL_STYLE = {
    logging.DEBUG:    ("DEBUG",    _Color.CYAN,   None),
    logging.INFO:     ("INFO ",    _Color.GREEN,  None),
    logging.WARNING:  ("WARN ",    _Color.YELLOW, _Color.BG_YELLOW),
    logging.ERROR:    ("ERROR",    _Color.RED,    _Color.BG_RED),
    logging.CRITICAL: ("CRIT ",    _Color.WHITE,  _Color.BG_RED),
}


class _ColorfulFormatter(logging.Formatter):
    """带颜色和图标的美观日志格式器。"""

    # 各级别图标
    _ICONS = {
        logging.DEBUG:    "🔍",
        logging.INFO:     "✅",
        logging.WARNING:  "⚠️ ",
        logging.ERROR:    "❌",
        logging.CRITICAL: "💥",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag, fg, bg = _LEVEL_STYLE.get(record.levelno, ("?????", _Color.WHITE, None))
        icon = self._ICONS.get(record.levelno, "•")

        # 构建彩色标签
        label = f"{fg}{_Color.BOLD}{tag}{_Color.RESET}"
        if bg:
            label = f"{bg}{_Color.BOLD}{tag}{_Color.RESET}"

        # 时间戳用暗色
        ts = f"{_Color.DIM}{self.formatTime(record, self.datefmt)}{_Color.RESET}"

        # logger 名称用青色
        name = f"{_Color.CYAN}{record.name}{_Color.RESET}"

        # 消息正文
        msg = record.getMessage()

        line = f"{icon} {ts} │ {label} │ {name} │ {msg}"

        # 异常信息追加
        if record.exc_info and record.exc_info[0] is not None:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ── 公共接口 ──────────────────────────────────────────────

def get_logger(
    name: str = "grad_assist",
    level: Optional[int] = None,
) -> logging.Logger:
    """获取美观日志 logger。

    Args:
        name:  logger 名称，通常用模块名，如 "grad_assist.nodes.retrieve"。
        level: 日志级别，默认读取环境变量 LOG_LEVEL，否则 INFO。

    Returns:
        配置好 Handler 和 Formatter 的 Logger 实例。
    """
    import os

    _logger = logging.getLogger(name)

    # 避免重复添加 handler（getLogger 同名返回同一对象）
    if _logger.handlers:
        return _logger

    effective_level = level or getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    _logger.setLevel(effective_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(effective_level)
    handler.setFormatter(_ColorfulFormatter(datefmt="%H:%M:%S"))
    _logger.addHandler(handler)

    # 关闭向上传播，避免子 logger（如 grad_assist.node）被父 logger 重复打印
    _logger.propagate = False

    return _logger


# 模块级默认实例，支持 logger.info(...) 直接调用
logger: logging.Logger = get_logger()
