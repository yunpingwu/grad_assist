import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# LLM 配置
@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float

llm_config = LLMConfig(
    model=os.getenv("MODEL"),
    base_url=os.getenv("ALIBABA_BASE_URL"),
    api_key=os.getenv("ALIBABA_API_KEY"),
    temperature=0.7,
)