import os

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

load_dotenv()

# Minio 配置
@dataclass
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool


minio_config = MinioConfig(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    bucket=os.getenv("MINIO_BUCKET", "textbook-assets"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
)
