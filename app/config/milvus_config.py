import os

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

load_dotenv()

# Milvus 配置
@dataclass
class MilvusConfig:
    uri: str
    token: str


milvus_config = MilvusConfig(
    uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
    token=os.getenv("MILVUS_TOKEN", ""),
)
