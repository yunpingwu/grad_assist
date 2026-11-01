"""教材摄入的应用编排层：上传落盘、断点续跑与 SSE 事件流。"""

from app.retrieval.store import list_textbooks as list_textbooks  # 显式再导出：路由经服务层取教材列表
from app.services.textbook.lifecycle import get_graph, init_graph, setup_checkpointer
from app.services.textbook.resolve import (
    MissingTextbookPath,
    ResolvePlan,
    TextbookPathNotFound,
    prepare_resolve,
    resolve_events,
)
from app.services.textbook.upload import save_uploads

__all__ = [
    "MissingTextbookPath",
    "ResolvePlan",
    "TextbookPathNotFound",
    "get_graph",
    "init_graph",
    "list_textbooks",
    "prepare_resolve",
    "resolve_events",
    "save_uploads",
    "setup_checkpointer",
]
