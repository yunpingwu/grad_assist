"""llm 客户端集成测试：真实调用依赖 ALIBABA_API_KEY，无凭据时自动跳过。

纯构造（get_llm_client 返回值结构）不联网，始终可跑；
真实 ainvoke 需要 LLM 服务的密钥与网络。
"""

import asyncio

import pytest

from app.clients import llm
from app.clients.llm import get_llm_client
from app.config import llm_config

_has_api_key = bool((llm_config.api_key or "").strip())


def test_get_llm_client_returns_invokable() -> None:
    """构造校验：返回对象应支持 ainvoke / bind_tools（不联网）。"""
    client = get_llm_client(model="deepseek-v4-flash", max_retries=1, timeout=5)
    assert hasattr(client, "ainvoke")
    assert hasattr(client, "bind_tools")


def test_get_llm_client_reasoning_switch_is_ignored_at_build() -> None:
    """enable_thinking 只是透传 extra_body，不应影响构建成功。"""
    client = get_llm_client(model="deepseek-v4-flash", enable_thinking=False)
    assert hasattr(client, "ainvoke")


@pytest.mark.integration
@pytest.mark.skipif(not _has_api_key, reason="未配置 ALIBABA_API_KEY")
class TestLlmRealCall:
    """凭据型集成测试：仅当配置了 API Key 时运行。"""

    def test_ainvoke_returns_reply(self) -> None:
        async def _run() -> str:
            client = get_llm_client(model="deepseek-v4-flash", max_retries=1, timeout=60)
            resp = await client.ainvoke("你好，你是谁?")
            return resp.content

        content = asyncio.run(_run())
        assert content and content.strip()
        llm.get_llm_client.cache_clear()
