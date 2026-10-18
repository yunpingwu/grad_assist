"""教材知识学习 Agent 的工具集：汇总导出，供图构建一次绑定。

这里是**唯一**的工具清单来源：图构建直接引用 AGENT_TOOLS，避免"两份清单各自漂移"。
工具失败的兜底与限流不在这里——它们是官方的 ToolRetryMiddleware / ToolCallLimitMiddleware，装在图侧（见 graph.py）。
"""

from app.study_agent.tools.chapters import list_chapters
from app.study_agent.tools.clarify import ask_clarification
from app.study_agent.tools.files import (
    append_file,
    edit_file,
    list_files,
    read_file,
    write_file,
)
from app.study_agent.tools.search import search_textbook
from app.study_agent.tools.web import search_web

# 绑定给 create_agent 的完整工具列表（检索类 + 文件类 + 澄清类）
AGENT_TOOLS = [
    search_textbook,
    list_chapters,
    search_web,
    ask_clarification,
    write_file,
    append_file,
    read_file,
    edit_file,
    list_files,
]

__all__ = [
    "AGENT_TOOLS",
    "search_textbook",
    "list_chapters",
    "search_web",
    "ask_clarification",
    "write_file",
    "append_file",
    "read_file",
    "edit_file",
    "list_files",
]
