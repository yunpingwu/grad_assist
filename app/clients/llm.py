"""LLM 客户端：初始化并返回可用的 Chat 模型实例。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


def get_llm_client(model: str | None = None, *, max_retries: int = 2, timeout: int = 120) -> Any:
    """获取裸 Chat 模型（非 Agent）。

    Args:
        model: 模型名，缺省用环境变量 MODEL。
        max_retries: 网络错误 / 429 / 5xx 的自动重试次数（透传 ChatOpenAI）。
        timeout: 单次请求超时秒数（透传 ChatOpenAI；SDK 默认 600s 过长,
                 配合重试会放大故障静默时间）。

    Returns:
        配置好的 Chat 模型，支持 ``ainvoke`` / ``bind_tools``。
    """
    load_dotenv()

    return init_chat_model(
        model=model or os.getenv("MODEL"),
        model_provider="openai",
        temperature=0.1,
        api_key=os.getenv("ALIBABA_API_KEY"),
        base_url=os.getenv("ALIBABA_BASE_URL"),
        max_retries=max_retries,
        timeout=timeout,
    )


# 单元测试
if __name__ == '__main__':
    import asyncio

    async def main() -> None:
        llm = get_llm_client()
        resp = await llm.ainvoke("你好，你是谁?")
        print(resp.content)

    asyncio.run(main())
