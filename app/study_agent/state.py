"""复习资料生成 Agent 的状态定义。

单层 ReAct 方案：状态即 create_agent（LangChain Agent）的输入输出，
附加任务信息与收尾产物字段，风格与 query_functions/QueryState 保持一致。
"""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class Material(BaseModel):
    """一份已落盘的复习资料。

    Attributes:
        title: 资料标题（文件首个一级标题，缺省回退文件名）。
        filename: 文件名，如 第3章-知识点总结.md。
        rel_path: 相对沙箱根 materials/{textbook_name}/{task_id}/ 的路径。
        size: 字节数。
    """

    title: str
    filename: str
    rel_path: str
    size: int


class StudyState(TypedDict):
    """复习资料生成任务各节点间传递的状态。"""

    # 匿名设备身份（前端 X-User-Id 请求头传入），用于多用户任务隔离
    user_id: NotRequired[str]
    # 任务 ID（thread 后缀，断点续跑用）
    task_id: str
    # 教材名（检索过滤键 + 落盘目录名）
    textbook_name: str
    # 用户的要求（缺省有默认指令）
    requirement: str
    # 多轮对话（create_agent 依赖；add_messages 每轮追加而非覆盖）
    messages: Annotated[list[AnyMessage], add_messages]
    # 落盘清单（收尾后写回）
    materials: NotRequired[list[Material]]
    # 是否完成（收尾后置 True）
    finished: NotRequired[bool]
