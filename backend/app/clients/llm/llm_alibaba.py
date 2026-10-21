"""阿里云百炼平台口径：思考开关与结构化输出方式收在本文件。"""

from __future__ import annotations

from typing import Any

from app.config import llm_config

NAME = "alibaba"
API_KEY_ENV = "ALIBABA_API_KEY"
STRUCTURED_OUTPUT_METHOD = "json_schema"


def client_kwargs(model: str | None, enable_thinking: bool | None) -> dict[str, Any]:
    """返回 ``init_chat_model`` 的平台侧入参（model/api_key/base_url，必要时 extra_body）。

    ``enable_thinking`` 缺省为 True——百炼侧维持历史行为（默认开思考）；省时省 token 的
    调用点（intent/hyde）显式传 False。
    """
    thinking = True if enable_thinking is None else enable_thinking
    kwargs: dict[str, Any] = {
        "model": model or llm_config.model,
        "api_key": llm_config.api_key,
        "base_url": llm_config.base_url,
    }
    if not thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return kwargs
