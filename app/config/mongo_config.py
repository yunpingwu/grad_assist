import os

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

load_dotenv()

# MongoDB 配置
@dataclass
class MongoConfig:
    uri: str
    db: str
    chat_collection: str
    access_key: str
    secret_key: str


mongo_config = MongoConfig(
    uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    db=os.getenv("MONGO_DB", "grad_assist"),
    chat_collection=os.getenv("MONGO_CHAT_COLLECTION", "chat_sessions"),
    access_key=os.getenv("MONGO_ACCESS_KEY", ""),
    secret_key=os.getenv("MONGO_SECRET_KEY", ""),
)
