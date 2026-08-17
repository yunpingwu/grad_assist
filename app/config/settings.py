"""统一配置中心：基于 pydantic-settings 的 BaseSettings 聚合。

取代原先 7 个「dataclass + os.getenv + load_dotenv」散落配置模块：

- 各域配置类继承 BaseSettings，通过 ``validation_alias`` 映射到既有的扁平环境变量名，
  保持 ``.env`` / ``.env.example`` 不变；
- 环境变量仅在模块顶部 ``load_dotenv()`` 一次，各子类从 ``os.environ`` 读取；
- 字段声明即文档，自带类型校验（``"1024"`` → int、``"false"`` → bool），
  默认值内聚在字段定义上，不再散落在各文件的 ``os.getenv`` 调用里。

用法::

    from app.config import llm_config, settings
    llm_config.model   # 读 MODEL
    settings.milvus.uri
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 全项目仅此一处加载 .env；子 Settings 类从 os.environ 读取
load_dotenv()


class LLMSettings(BaseSettings):
    """大语言模型配置（阿里云百炼 OpenAI 兼容接口）。"""

    model_config = SettingsConfigDict(extra="ignore")

    model: str = Field(default="deepseek-v4-flash", validation_alias="MODEL")
    visual_model: str = Field(default="", validation_alias="VISUAL_MODEL")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="ALIBABA_BASE_URL",
    )
    api_key: str = Field(default="", validation_alias="ALIBABA_API_KEY")
    temperature: float = 0.1


class EmbeddingSettings(BaseSettings):
    """嵌入模型配置（FlagEmbedding BGE-M3）。"""

    model_config = SettingsConfigDict(extra="ignore")

    # 本地模型路径（优先使用），缺省为空字符串
    model_path: str = Field(default="", validation_alias="EMBEDDING_MODEL_PATH")
    # HuggingFace ID（本地路径不存在时使用）
    model_name: str = Field(default="BAAI/bge-m3", validation_alias="EMBEDDING_MODEL")
    device: str = Field(default="cpu", validation_alias="EMBEDDING_DEVICE")
    dim: int = Field(default=1024, validation_alias="EMBEDDING_DIM")


class MilvusSettings(BaseSettings):
    """Milvus 向量数据库配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    uri: str = Field(default="http://localhost:19530", validation_alias="MILVUS_URI")
    token: str = Field(default="", validation_alias="MILVUS_TOKEN")


class MineruSettings(BaseSettings):
    """MinerU 文档解析服务配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    url: str = Field(default="", validation_alias="MINERU_BASE_URL")
    token: str = Field(default="", validation_alias="MINERU_API_KEY")


class MinioSettings(BaseSettings):
    """MinIO 对象存储配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    endpoint: str = Field(default="localhost:9000", validation_alias="MINIO_ENDPOINT")
    access_key: str = Field(default="admin", validation_alias="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="", validation_alias="MINIO_SECRET_KEY")
    bucket: str = Field(default="textbook-assets", validation_alias="MINIO_BUCKET")
    secure: bool = Field(default=False, validation_alias="MINIO_SECURE")


class MongoSettings(BaseSettings):
    """MongoDB 配置（会话/对话历史存储）。"""

    model_config = SettingsConfigDict(extra="ignore")

    uri: str = Field(default="mongodb://localhost:27017", validation_alias="MONGO_URI")
    host: str = Field(default="localhost", validation_alias="MONGO_HOST")
    db: str = Field(default="grad_assist", validation_alias="MONGO_DB")
    chat_collection: str = Field(default="chat_sessions", validation_alias="MONGO_CHAT_COLLECTION")
    access_key: str = Field(default="", validation_alias="MONGO_ACCESS_KEY")
    secret_key: str = Field(default="", validation_alias="MONGO_SECRET_KEY")


class WebSearchSettings(BaseSettings):
    """百炼 WebSearch MCP 联网搜索配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    mcp_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
        validation_alias="WEB_SEARCH_MCP_URL",
    )
    tool_name: str = Field(default="bailian_web_search", validation_alias="WEB_SEARCH_TOOL")
    search_count: int = Field(default=3, validation_alias="WEB_SEARCH_COUNT")


class RerankSettings(BaseSettings):
    """重排序（Rerank）模型配置（FlagEmbedding BGE-Reranker）。

    与 embedding 同源：优先加载本地模型路径，缺省回退 HuggingFace ID。
    """

    model_config = SettingsConfigDict(extra="ignore")

    # 本地模型路径（优先使用），缺省为空字符串
    model_path: str = Field(default="", validation_alias="RERANKER_MODEL_PATH")
    # HuggingFace ID（本地路径不存在时使用）
    model_name: str = Field(default="BAAI/bge-reranker-large", validation_alias="RERANKER_MODEL")
    device: str = Field(default="cpu", validation_alias="RERANKER_DEVICE")
    top_k: int = Field(default=5, validation_alias="RERANKER_TOP_K")


class Settings(BaseModel):
    """顶层配置聚合，实例化一次后按域访问。

    Attributes:
        llm: 大语言模型配置。
        embedding: 嵌入模型配置。
        milvus: Milvus 配置。
        mineru: MinerU 配置。
        minio: MinIO 配置。
        mongo: MongoDB 配置。
        web_search: 联网搜索配置。
        rerank: 重排序模型配置。
    """

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    mineru: MineruSettings = Field(default_factory=MineruSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


# 单元测试
if __name__ == "__main__":
    s = Settings()
    # 只打印非敏感字段，敏感信息（api_key / token / secret_key）不输出
    print("model:", s.llm.model)
    print("visual_model:", s.llm.visual_model or "(未配置)")
    print("milvus.uri:", s.milvus.uri)
    print("minio.endpoint:", s.minio.endpoint, "secure:", s.minio.secure)
    print("embedding.dim:", s.embedding.dim, "device:", s.embedding.device)
    print("mongo.db:", s.mongo.db)
    print("web_search.search_count:", s.web_search.search_count)
    print("device:", s.rerank.device, "top_k:", s.rerank.top_k)
