"""LLM 客户端门面：按供应商装配 Chat 模型；两家的写法差异各自关在 ``llm_<供应商>.py`` 里。

- ``llm_alibaba.py``：阿里云百炼（默认，``LLM_PROVIDER=alibaba``）；
- ``llm_deepseek.py``：DeepSeek 官方平台。

本文件只做三件事：供应商注册与分发、公共装配（``init_chat_model`` 的共用参数）、
以及 langchain-openai 的 reasoning_content 补丁。供应商模块契约（新增一家照此实现）：
``NAME``（LLM_PROVIDER 取值）/ ``API_KEY_ENV``（缺 key 时提示的变量名）/
``STRUCTURED_OUTPUT_METHOD``（结构化输出的实现方式）/ ``client_kwargs(model, enable_thinking)``。

补丁说明：langchain-openai 1.4.x 的响应对 ``reasoning_content``（DeepSeek 等
推理模型的思考字段）不做解析，流式 chunk 会把它直接丢弃。这里对
``_convert_delta_to_message_chunk`` 做无侵入包装，把它并入
``additional_kwargs["reasoning_content"]``，供上层按 thought 推送（两家平台共用）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai.chat_models import base as _lcb

from app.clients import llm_alibaba, llm_deepseek
from app.config import llm_config

# 供应商注册表：新增一家 = 加一个 llm_<名>.py 模块并登记在这里
_PROVIDERS = {module.NAME: module for module in (llm_alibaba, llm_deepseek)}

# ── 补丁：保留流式响应里的 reasoning_content ──────────────────────────
_ORIG_CONVERT_DELTA = _lcb._convert_delta_to_message_chunk
_ORIG_CONVERT_DICT = _lcb._convert_dict_to_message


def _convert_delta_with_reasoning(_dict: Any, default_class: type) -> Any:
    """包装 delta→chunk 转换：把 DeepSeek 的 reasoning_content 并入 additional_kwargs。

    Args:
        _dict: 原始 delta（含 reasoning_content 字段）。
        default_class: 默认 chunk 类。

    Returns:
        原逻辑产物；若原始 delta 带 reasoning_content 则额外挂上该字段。
    """
    chunk = _ORIG_CONVERT_DELTA(_dict, default_class)
    reasoning = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
    if reasoning and getattr(chunk, "additional_kwargs", None) is not None:
        chunk = chunk.model_copy(
            update={"additional_kwargs": {**chunk.additional_kwargs, "reasoning_content": reasoning}}
        )
    return chunk


def _convert_dict_with_reasoning(_dict: Any) -> Any:
    """包装整段(非流式)消息转换：同样的丢失问题出现在 ``_convert_dict_to_message``。

    Args:
        _dict: 原始 assistant 消息字典（含 reasoning_content 字段）。

    Returns:
        原逻辑产物；assistant 消息若带 reasoning_content 则额外挂到 additional_kwargs。
    """
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
    """获取裸 Chat 模型（非 Agent）。

    Args:
        model: 模型名；缺省用所选供应商配置里的模型。
        max_retries: 网络错误 / 429 / 5xx 的自动重试次数。
        timeout: 单次请求超时秒数。
        enable_thinking: 是否开启思考模式；缺省按供应商默认（各自的开关写法与默认值
            见 ``llm_<供应商>.py`` 的模块 docstring）。
        provider: 供应商；缺省取 ``llm_config.provider``。

    Returns:
        配置好的 Chat 模型，支持 ``ainvoke`` / ``bind_tools``。
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
    """给 Chat 模型绑结构化输出，按供应商挑可用的实现方式（单一口径，别在各调用点各写一份）。

    两家的可用方式见各自模块的 ``STRUCTURED_OUTPUT_METHOD``（官方平台不支持 json_schema）。

    Args:
        client: ``get_llm_client`` 返回的 Chat 模型。
        schema: 目标结构（pydantic 模型或 TypedDict）。
        provider: 供应商；缺省取 ``llm_config.provider``。
    """
    module = _provider_module(provider)
    return client.with_structured_output(schema, method=module.STRUCTURED_OUTPUT_METHOD)


def _provider_module(provider: str | None) -> Any:
    """按名称取供应商模块（``llm_alibaba`` / ``llm_deepseek``）；未知名称给出可用清单。"""
    name = (provider or llm_config.provider or "alibaba").lower()
    module = _PROVIDERS.get(name)
    if module is None:
        raise ValueError(f"未知的 LLM 供应商: {name!r}（可用：{' / '.join(_PROVIDERS)}）")
    return module


# 集成测试已迁移至 tests/clients/test_llm.py
