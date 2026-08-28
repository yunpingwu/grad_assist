"""教材知识学习 Agent 图：create_agent 单层 ReAct（方案 A：单层循环 + 护栏）。

不再手写 StateGraph 包装——create_agent 本身即编译好的图对象
（CompiledStateGraph），内部已编排 model ⇄ tools 循环与终止条件；
收尾（扫描落盘、登记清单）为图外的业务函数，见 service/API 层。
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware, SummarizationMiddleware

from app.clients.llm import get_llm_client
from app.core import load_prompt
from app.study_agent.state import StudyState
from app.study_agent.tools import (
    append_file,
    edit_file,
    list_chapters,
    list_files,
    read_file,
    search_textbook,
    search_web,
    write_file,
)


def build_graph(checkpointer=None):
    """构建教材知识学习 Agent（create_agent 单层 ReAct）。

    Args:
        checkpointer: 状态持久化器（断点续跑），建议传 MongoDBSaver。

    Returns:
        编译后的 CompiledStateGraph，可直接 stream / invoke。
    """
    return create_agent(
        model=get_llm_client(),
        tools=[search_textbook, list_chapters, search_web,
               write_file, append_file, read_file, edit_file, list_files],
        system_prompt=load_prompt("study_agent"),
        state_schema=StudyState,  # 需含 messages(add_messages)
        checkpointer=checkpointer,
        middleware=[
            ModelCallLimitMiddleware(run_limit=20),  # 护栏1：步数上限（到顶 end 收尾）
            SummarizationMiddleware(model=get_llm_client()),  # 护栏2：自动上下文压缩
            HumanInTheLoopMiddleware(interrupt_on={"write_file": True}),  # 可选：写文件前人工确认
        ],
    )
