"""LLM 客户端：初始化并返回可用的 Chat 模型实例。

模型 / 密钥 / 地址 / 温度统一从 ``app.config.llm_config`` 读取，
不再在客户端内重复 ``load_dotenv()`` / ``os.getenv``。

补丁说明：langchain-openai 1.4.x 的响应对 ``reasoning_content``（DeepSeek 等
推理模型的思考字段）不做解析，流式 chunk 会把它直接丢弃。这里对
``_convert_delta_to_message_chunk`` 做无侵入包装，把它并入
``additional_kwargs["reasoning_content"]``，供上层按 thought 推送。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai.chat_models import base as _lcb

from app.config import llm_config

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
    enable_thinking: bool = True,
) -> Any:
    """获取裸 Chat 模型（非 Agent）。

    Args:
        model: 模型名，缺省用 llm_config.model。
        max_retries: 网络错误 / 429 / 5xx 的自动重试次数。
        timeout: 单次请求超时秒数。
        enable_thinking: 是否开启思考模式。百炼混合思考型模型（如 deepseek-v4-flash）
            通过 extra_body={"enable_thinking": False} 关闭思考，减少时延与推理 token 消耗。

    Returns:
        配置好的 Chat 模型，支持 ``ainvoke`` / ``bind_tools``。
    """
    kwargs: dict[str, Any] = {}
    if not enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return init_chat_model(
        model=model or llm_config.model,
        model_provider="openai",
        temperature=llm_config.temperature,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )


# 单元测试
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        llm = get_llm_client()
        resp = await llm.ainvoke("你好，你是谁?")
        print(resp.content)

    asyncio.run(main())
