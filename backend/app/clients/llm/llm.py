"""LLM 客户端门面：供应商注册与装配，并统一结构化输出入口。

供应商差异收在 ``llm_<供应商>.py``；本文件另含 reasoning_content 的解析补丁。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai.chat_models import base as _lcb

from app.clients.llm import llm_alibaba, llm_deepseek
from app.config import llm_config

# 供应商注册表：新增一家 = 加一个 llm_<名>.py 模块并登记在这里
_PROVIDERS = {module.NAME: module for module in (llm_alibaba, llm_deepseek)}

# ── 补丁：保留流式响应里的 reasoning_content ──────────────────────────
_ORIG_CONVERT_DELTA = _lcb._convert_delta_to_message_chunk
_ORIG_CONVERT_DICT = _lcb._convert_dict_to_message


def _convert_delta_with_reasoning(_dict: Any, default_class: type) -> Any:
    """包装 delta→chunk 转换，把 reasoning_content 并入 additional_kwargs。"""
    chunk = _ORIG_CONVERT_DELTA(_dict, default_class)
    reasoning = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
    if reasoning and getattr(chunk, "additional_kwargs", None) is not None:
        chunk = chunk.model_copy(
            update={"additional_kwargs": {**chunk.additional_kwargs, "reasoning_content": reasoning}}
        )
    return chunk


def _convert_dict_with_reasoning(_dict: Any) -> Any:
    """包装非流式消息转换，把 reasoning_content 挂到 additional_kwargs。"""
    msg = _ORIG_CONVERT_DICT(_dict)
    reasoning = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
    if reasoning and getattr(msg, "additional_kwargs", None) is not None:
        msg = msg.model_copy(
            update={"additional_kwargs": {**msg.additional_kwargs, "reasoning_content": reasoning}}
        )
    return msg


_lcb._convert_delta_to_message_chunk = _convert_delta_with_reasoning
_lcb._convert_dict_to_message = _convert_dict_with_reasoning


@lru_cache(maxsize=128)
def get_llm_client(
    model: str | None = None,
    *,
    max_retries: int = 2,
    timeout: int = 120,
    enable_thinking: bool | None = None,
    provider: str | None = None,
) -> Any:
    """获取裸 Chat 模型（非 Agent）；参数相同则复用缓存实例。

    provider 缺省取 ``llm_config.provider``，enable_thinking 缺省按供应商默认。
    """
    module = _provider_module(provider)
    kwargs = module.client_kwargs(model, enable_thinking)
    if not (kwargs.get("api_key") or "").strip():
        # 不拦的话报错来自 OpenAI SDK（Missing credentials），看不出是缺哪个平台的 key
        raise ValueError(f"未配置 {module.NAME} 平台的 API Key（.env 里的 {module.API_KEY_ENV}）")
    return init_chat_model(
        model_provider="openai",
        temperature=llm_config.temperature,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )


def bind_structured_output(client: Any, schema: Any, *, provider: str | None = None) -> Any:
    """按供应商选择结构化输出方式并绑定到 Chat 模型。"""
    module = _provider_module(provider)
    return client.with_structured_output(schema, method=module.STRUCTURED_OUTPUT_METHOD)


def _provider_module(provider: str | None) -> Any:
    """按名称取供应商模块（``llm_alibaba`` / ``llm_deepseek``）；未知名称给出可用清单。"""
    name = (provider or llm_config.provider or "alibaba").lower()
    module = _PROVIDERS.get(name)
    if module is None:
        raise ValueError(f"未知的 LLM 供应商: {name!r}（可用：{' / '.join(_PROVIDERS)}）")
    return module


