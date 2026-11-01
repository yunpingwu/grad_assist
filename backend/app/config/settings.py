"""统一配置中心：pydantic-settings 聚合，环境变量名经 ``validation_alias`` 映射。

settings 在 import 时实例化一次；运行时改值请 monkeypatch 域对象属性。
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 全项目仅此一处加载 .env；子 Settings 类从 os.environ 读取
load_dotenv()


class LLMSettings(BaseSettings):
    """大语言模型配置（默认走阿里云百炼 OpenAI 兼容接口，可切 DeepSeek 官方平台）。"""

    model_config = SettingsConfigDict(extra="ignore")

    # 供应商选择：alibaba=阿里云百炼（默认）；deepseek=DeepSeek 官方平台（见 DeepSeekSettings）
    provider: Literal["alibaba", "deepseek"] = Field(default="alibaba", validation_alias="LLM_PROVIDER")
    model: str = Field(default="deepseek-v4-flash", validation_alias="MODEL")
    visual_model: str = Field(default="", validation_alias="VISUAL_MODEL")
    # 模型输入窗口（token）：会话压缩的触发线（窗口 × 50%）按它算
    # （见 agent/compaction.py，取值入口 current_context_window）。
    # 按所用模型的真实窗口填——填小了过早压缩、填大了窗口溢出前不压缩，都不报错，只是行为悄悄变。
    context_window: int = Field(default=500_000, validation_alias="MODEL_CONTEXT_WINDOW")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="ALIBABA_BASE_URL",
    )
    api_key: str = Field(default="", validation_alias="ALIBABA_API_KEY")
    temperature: float = 0.1


class DeepSeekSettings(BaseSettings):
    """DeepSeek 官方平台配置（``LLM_PROVIDER=deepseek`` 时启用）。"""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    base_url: str = Field(default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")
    model: str = Field(default="deepseek-flash", validation_alias="DEEPSEEK_MODEL")
    # 默认关思考：agent 一轮十几到几十次调用，思考是主要耗时与成本项；需要时再开
    enable_thinking: bool = Field(default=False, validation_alias="DEEPSEEK_ENABLE_THINKING")
    # 思考强度（官方取值 low/high/max，默认 high；关闭思考走 enable_thinking）
    reasoning_effort: Literal["low", "high", "max"] = Field(default="high", validation_alias="DEEPSEEK_REASONING_EFFORT")
    # 模型输入窗口（token）：两家平台的窗口各配各的（含义见 LLMSettings.context_window）
    context_window: int = Field(default=500_000, validation_alias="DEEPSEEK_CONTEXT_WINDOW")


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


class PostgresSettings(BaseSettings):
    """PostgreSQL 配置（LangGraph checkpoint 持久化）。

    对话与摄入各用一个 schema（默认 study / textbook），由连接 search_path 隔离；
    两套图的 thread_id 因此不会互相覆盖。
    """

    model_config = SettingsConfigDict(extra="ignore")

    uri: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/grad_assist",
        validation_alias="POSTGRES_URI",
    )
    study_schema: str = Field(default="study", validation_alias="POSTGRES_STUDY_SCHEMA")
    textbook_schema: str = Field(default="textbook", validation_alias="POSTGRES_TEXTBOOK_SCHEMA")
    pool_min_size: int = Field(default=1, validation_alias="POSTGRES_POOL_MIN_SIZE")
    pool_max_size: int = Field(default=5, validation_alias="POSTGRES_POOL_MAX_SIZE")


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
    # 深度路径每路召回的候选条数（RRF 融合前），给融合/精排更大候选空间
    candidate_pool: int = Field(default=20, validation_alias="RERANKER_CANDIDATE_POOL")
    # 加权融合中精排分的权重：final = alpha*rerank + (1-alpha)*minmax(rrf)
    # alpha=1 等价纯精排；alpha=0 等价 RRF 融合序（不实际调用精排）
    fusion_alpha: float = Field(default=0.6, validation_alias="RERANKER_FUSION_ALPHA")


class BudgetSettings(BaseSettings):
    """会话成本预算护栏配置（per-session 加权 token 上限，口径见 app/trace/llm_usage.weighted_cost）。"""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=True, validation_alias="BUDGET_ENABLED")
    # 会话累计加权 token 上限；默认 100w：≈30 个中位轮、≈¥2-3 量级——护栏兜的是失控会话，
    # 且超限只是降级（禁 deep/停追问/收尾说明）不是中断。标定口径见 logs/<日期>/metrics.log 的 turn_cost
    limit: float = Field(default=1_000_000, validation_alias="BUDGET_LIMIT")
    # 缓存命中 token 的折扣权重（与 tool_history「命中按 20% 计费折算」的既有口径一致）
    cache_weight: float = Field(default=0.2, validation_alias="BUDGET_CACHE_WEIGHT")
    # 输出 token 相对未命中输入 token 的权重（DeepSeek 系输出单价约为输入的数倍）
    output_weight: float = Field(default=4.0, validation_alias="BUDGET_OUTPUT_WEIGHT")


class TraceSettings(BaseSettings):
    """观测落盘保真度（steps 时间线里 message/工具结果的正文留多少）。

    三档：digest 只留长度与指纹（最省，判「两次调用是否同前缀」够用）；preview 另留首尾
    摘要（排障主用档）；full 连正文一起落（日志大 10~20 倍，只在复现具体问题时开）。
    单个 agent 调用的输入有一两万 token，全量落盘会让单条记录膨胀到数百 KB。
    """

    model_config = SettingsConfigDict(extra="ignore")

    payload: Literal["digest", "preview", "full"] = Field(default="preview", validation_alias="TRACE_PAYLOAD")
    preview_head: int = Field(default=120, validation_alias="TRACE_PREVIEW_HEAD")
    preview_tail: int = Field(default=40, validation_alias="TRACE_PREVIEW_TAIL")


class Settings(BaseModel):
    """顶层配置聚合：实例化一次后按域访问。"""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    mineru: MineruSettings = Field(default_factory=MineruSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)
