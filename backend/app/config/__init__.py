"""统一配置入口：实例化一次 Settings，并扁平导出各域配置对象。"""

from app.config.settings import Settings

settings = Settings()

# 扁平导出各域配置对象（llm_config / milvus_config 等）
llm_config = settings.llm
deepseek_config = settings.deepseek
embedding_config = settings.embedding
milvus_config = settings.milvus
mineru_config = settings.mineru
minio_config = settings.minio
postgres_config = settings.postgres
web_search_config = settings.web_search
rerank_config = settings.rerank
budget_config = settings.budget
trace_config = settings.trace


def current_context_window() -> int:
    """当前供应商配的模型窗口（token）。

    两家平台各配各的（``MODEL_CONTEXT_WINDOW`` / ``DEEPSEEK_CONTEXT_WINDOW``），
    消费方是随窗口走的护栏：会话压缩的触发线、轮内工具结果预算（见 ``study_agent/``）。
    """
    return deepseek_config.context_window if llm_config.provider == "deepseek" else llm_config.context_window


__all__ = [
    "Settings",
    "settings",
    "llm_config",
    "deepseek_config",
    "embedding_config",
    "milvus_config",
    "mineru_config",
    "minio_config",
    "postgres_config",
    "web_search_config",
    "rerank_config",
    "budget_config",
    "trace_config",
    "current_context_window",
]
