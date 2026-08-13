import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# 百炼 WebSearch MCP 配置
@dataclass
class WebSearchConfig:
    mcp_url: str
    tool_name: str
    search_count: int


web_search_config = WebSearchConfig(
    mcp_url=os.getenv("WEB_SEARCH_MCP_URL", "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"),
    tool_name=os.getenv("WEB_SEARCH_TOOL", "bailian_web_search"),
    search_count=int(os.getenv("WEB_SEARCH_COUNT", "3")),
)
