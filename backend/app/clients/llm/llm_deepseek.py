"""DeepSeek 官方平台口径：思考开关与结构化输出方式收在本文件。"""

from __future__ import annotations

from typing import Any

from app.config import deepseek_config

NAME = "deepseek"
API_KEY_ENV = "DEEPSEEK_API_KEY"
STRUCTURED_OUTPUT_METHOD = "function_calling"


def client_kwargs(model: str | None, enable_thinking: bool | None) -> dict[str, Any]:
    """返回 ``init_chat_model`` 的平台侧入参；思考开关缺省跟 ``DEEPSEEK_ENABLE_THINKING``。"""
    thinking = deepseek_config.enable_thinking if enable_thinking is None else enable_thinking
    extra_body: dict[str, Any] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    if thinking:
        extra_body["reasoning_effort"] = deepseek_config.reasoning_effort
    return {
        "model": model or deepseek_config.model,
        "api_key": deepseek_config.api_key,
        "base_url": deepseek_config.base_url,
        "extra_body": extra_body,
    }
