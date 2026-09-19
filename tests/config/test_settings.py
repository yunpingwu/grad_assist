"""config Settings 聚合配置测试：实例化并校验各域非敏感字段可读。"""

from app.config.settings import Settings


def test_settings_loads_all_domains() -> None:
    s = Settings()
    # 仅断言非敏感字段的存在性与类型，避免依赖 .env 具体值
    assert isinstance(s.llm.model, str) and s.llm.model
    assert isinstance(s.embedding.dim, int) and isinstance(s.embedding.device, str)
    assert isinstance(s.milvus.uri, str) and s.milvus.uri
    assert isinstance(s.minio.endpoint, str) and isinstance(s.minio.secure, bool)
    assert isinstance(s.mongo.db, str) and s.mongo.db
    assert isinstance(s.web_search.search_count, int) and s.web_search.search_count > 0
    assert isinstance(s.rerank.top_k, int) and s.rerank.top_k > 0
    assert 0.0 <= s.rerank.fusion_alpha <= 1.0
