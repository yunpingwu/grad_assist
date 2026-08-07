import os

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

load_dotenv()

# 嵌入模型配置
@dataclass
class EmbeddingConfig:
    model_path: str    # 本地模型路径
    model_name: str    # HuggingFace ID（本地路径不存在时使用）
    device: str        # cpu 或 cuda
    dim: int           # 向量维度


embedding_config = EmbeddingConfig(
    model_path=os.getenv("EMBEDDING_MODEL_PATH", ""),
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    device=os.getenv("EMBEDDING_DEVICE", "cpu"),
    dim=int(os.getenv("EMBEDDING_DIM", "1024")),
)
