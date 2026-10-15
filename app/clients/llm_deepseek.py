"""DeepSeek 官方平台口径：模型/密钥/地址取 ``DeepSeekSettings``，与百炼不同的两处都关在本文件里。

- **思考模式**：平台侧默认开启（effort 默认 high），要关必须显式 ``thinking={"type": "disabled"}``；
  开启时另带 ``reasoning_effort``（low/high/max）；
- **结构化输出**：目前不支持 ``response_format=json_schema``（实测 400：This response_format
  type is unavailable now），走 ``function_calling``。

契约（``llm.py`` 的门面按这四项消费，新增供应商照此实现）：
``NAME`` / ``API_KEY_ENV`` / ``STRUCTURED_OUTPUT_METHOD`` / ``client_kwargs``。
"""

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
