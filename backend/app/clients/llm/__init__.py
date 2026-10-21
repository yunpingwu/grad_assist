"""LLM 客户端包：对外只暴露门面（原 ``app.clients.llm`` 模块的导入路径保持不变）。"""

from app.clients.llm.llm import bind_structured_output, get_llm_client

__all__ = ["bind_structured_output", "get_llm_client"]
