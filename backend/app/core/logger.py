"""封装 loguru，提供统一日志入口。"""

from __future__ import annotations

import os
import sys

from loguru import logger as _logger

# 移除 loguru 默认 handler，统一接管输出格式
_logger.remove()

# 日志格式：时间 │ 级别 │ 文件:行号 │ 消息（无颜色图标，深浅终端均清晰）
# {file}:{line} 由 loguru 从调用栈动态解析，自动显示调用日志处的脚本名与行号
_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> │ "
    "<level>{level: <8}</level> │ "
    "<cyan>{file}:{line}</cyan> │ "
    "{message}"
)

# 日志级别，默认读取环境变量 LOG_LEVEL，否则 INFO
_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_logger.add(
    sys.stdout,
    format=_LOG_FORMAT,
    level=_LEVEL,
    colorize=True,
    enqueue=True,  # 线程安全，避免多线程交错
)

# 模块级默认实例，支持 logger.info(...) 直接调用
logger = _logger
