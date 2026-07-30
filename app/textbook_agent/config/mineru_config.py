import os

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

load_dotenv()

# Mineru配置
@dataclass
class MineruConfig:
    url: str
    token: str

mineru_config = MineruConfig(
    url=os.getenv("MINERU_BASE_URL"),
    token=os.getenv("MINERU_API_KEY"),
)