"""LLM 客户端：初始化并返回可用的 Chat 模型实例。

模型 / 密钥 / 地址 / 温度统一从 ``app.config.llm_config`` 读取，
不再在客户端内重复 ``load_dotenv()`` / ``os.getenv``。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model

from app.config import llm_config


@lru_cache(maxsize=128)
def get_llm_client(model: str | None = None, *, max_retries: int = 2, timeout: int = 120) -> Any:
    """获取裸 Chat 模型（非 Agent）。

    Args:
        model: 模型名，缺省用 llm_config.model。
        max_retries: 网络错误 / 429 / 5xx 的自动重试次数（透传 ChatOpenAI）。
        timeout: 单次请求超时秒数（透传 ChatOpenAI；SDK 默认 600s 过长,
                 配合重试会放大故障静默时间）。

    Returns:
        配置好的 Chat 模型，支持 ``ainvoke`` / ``bind_tools``。
    """
    return init_chat_model(
        model=model or llm_config.model,
        model_provider="openai",
        temperature=llm_config.temperature,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        max_retries=max_retries,
        timeout=timeout,
    )


# 单元测试
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        llm = get_llm_client()
        resp = await llm.ainvoke("你好，你是谁?")
        print(resp.content)

    asyncio.run(main())
