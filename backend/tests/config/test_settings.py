"""config Settings 聚合配置测试：实例化并校验各域非敏感字段可读。"""

from app.config.settings import Settings


def test_settings_loads_all_domains() -> None:
    s = Settings()
    # 仅断言非敏感字段的存在性与类型，避免依赖 .env 具体值
    assert isinstance(s.llm.model, str) and s.llm.model
    assert s.llm.provider in ("alibaba", "deepseek")
    assert isinstance(s.deepseek.model, str) and s.deepseek.model
    assert isinstance(s.deepseek.base_url, str) and s.deepseek.base_url
    assert isinstance(s.deepseek.enable_thinking, bool)
    assert s.deepseek.reasoning_effort in ("low", "high", "max")
    assert isinstance(s.embedding.dim, int) and isinstance(s.embedding.device, str)
    assert isinstance(s.milvus.uri, str) and s.milvus.uri
    assert isinstance(s.minio.endpoint, str) and isinstance(s.minio.secure, bool)
    assert isinstance(s.mongo.db, str) and s.mongo.db
    assert isinstance(s.web_search.search_count, int) and s.web_search.search_count > 0
    assert isinstance(s.rerank.top_k, int) and s.rerank.top_k > 0
    assert 0.0 <= s.rerank.fusion_alpha <= 1.0
    assert isinstance(s.budget.enabled, bool)
    assert isinstance(s.budget.limit, float) and s.budget.limit > 0
    assert 0.0 < s.budget.cache_weight <= 1.0
    assert s.budget.output_weight > 0
