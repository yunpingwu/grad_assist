"""llm 客户端集成测试：真实调用依赖 ALIBABA_API_KEY，无凭据时自动跳过。

纯构造（get_llm_client 返回值结构）不联网，始终可跑；
真实 ainvoke 需要 LLM 服务的密钥与网络。
"""

import asyncio

import pytest

from app.clients import llm
from app.clients.llm import bind_structured_output, get_llm_client
from app.config import deepseek_config, llm_config

_has_api_key = bool((llm_config.api_key or "").strip())


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """get_llm_client 带 lru_cache：用例里可能 monkeypatch 配置，前后都清，避免拿到上例构造的实例。"""
    get_llm_client.cache_clear()
    yield
    get_llm_client.cache_clear()


def test_get_llm_client_returns_invokable() -> None:
    """构造校验：返回对象应支持 ainvoke / bind_tools（不联网）。"""
    client = get_llm_client(model="deepseek-v4-flash", max_retries=1, timeout=5)
    assert hasattr(client, "ainvoke")
    assert hasattr(client, "bind_tools")


def test_get_llm_client_reasoning_switch_is_ignored_at_build() -> None:
    """enable_thinking 只是透传 extra_body，不应影响构建成功。"""
    client = get_llm_client(model="deepseek-v4-flash", enable_thinking=False)
    assert hasattr(client, "ainvoke")


def test_provider_switches_model_base_url_and_thinking_wiring(monkeypatch) -> None:
    """供应商切换：模型/base_url 取对应平台配置，思考开关按两家各自的口径落进 extra_body。"""
    monkeypatch.setattr(deepseek_config, "api_key", "sk-test-dummy")
    # 钉住默认档：不钉的话断言会随 .env 的 DEEPSEEK_ENABLE_THINKING 飘
    monkeypatch.setattr(deepseek_config, "enable_thinking", False)

    deep = get_llm_client(provider="deepseek", timeout=5)
    assert deep.model_name == deepseek_config.model
    assert deep.openai_api_base == deepseek_config.base_url
    assert deep.extra_body == {"thinking": {"type": "disabled"}}, "官方平台关思考要显式 thinking.type=disabled"

    deep_think = get_llm_client(provider="deepseek", enable_thinking=True, timeout=6)
    assert deep_think.extra_body["thinking"] == {"type": "enabled"}
    assert deep_think.extra_body["reasoning_effort"] == deepseek_config.reasoning_effort

    ali = get_llm_client(provider="alibaba", enable_thinking=False, timeout=7)
    assert ali.extra_body == {"enable_thinking": False}, "百炼的开关写法与官方平台不同"


def test_bind_structured_output_picks_method_per_provider() -> None:
    """结构化输出方式随供应商挑：官方平台走 function_calling（json_schema 实测 400）。"""

    class _RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def with_structured_output(self, _schema, **kwargs):
            self.calls.append(kwargs)
            return self

    client = _RecordingClient()
    bind_structured_output(client, {"type": "object"}, provider="deepseek")
    bind_structured_output(client, {"type": "object"}, provider="alibaba")
    assert client.calls == [{"method": "function_calling"}, {"method": "json_schema"}]


def test_unknown_provider_and_missing_key_raise_clear_errors(monkeypatch) -> None:
    """选错供应商 / 没配 key 都要是看得懂的中文报错，而不是 OpenAI SDK 的 Missing credentials。"""
    with pytest.raises(ValueError, match="未知的 LLM 供应商"):
        get_llm_client(provider="openai")

    monkeypatch.setattr(deepseek_config, "api_key", "")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_llm_client(provider="deepseek")


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
