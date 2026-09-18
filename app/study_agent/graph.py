"""教材知识学习 Agent 图：create_agent 单层 ReAct（方案 A：单层循环 + 护栏）。

不再手写 StateGraph 包装——create_agent 本身即编译好的图对象
（CompiledStateGraph），内部已编排 model ⇄ tools 循环与终止条件；
收尾（扫描落盘、登记清单）为图外的业务函数，见 service/API 层。

动态提示词与前置 query 理解通过 langchain.agents 的 middleware 机制接入：
- ``@before_agent``：作为入口节点，每轮对话只执行一次，做意图识别 + 问题重写；
- ``@dynamic_prompt``：每次 model 调用前按 state（含意图）动态渲染 system prompt。
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    before_agent,
    dynamic_prompt,
)
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.clients.llm import get_llm_client
from app.core import astage, get_metrics, load_prompt, logger
from app.study_agent.query_functions.intent_detection import detect_intent
from app.study_agent.query_functions.rewrite_query import format_questions, rewrite
from app.study_agent.state import StudyState
from app.study_agent.tools import (
    append_file,
    ask_clarification,
    edit_file,
    list_chapters,
    list_files,
    read_file,
    search_textbook,
    search_web,
    write_file,
)

# 意图 → 行为块文件的映射（白名单，防越界读文件）
_INTENT_BLOCKS = frozenset({"explain", "generate", "quiz", "plan", "chat"})
_DEFAULT_INTENT = "explain"


@dynamic_prompt
def study_system_prompt(request: ModelRequest) -> str:
    """按当前 state（含意图）渲染 system prompt 正文。

    拼装逻辑内聚在本函数：意图未识别/越界回退默认「讲解」块；
    提示词文件统一走 ``load_prompt``（自带 LRU 缓存读盘，避免重复 IO）；
    骨架模板为小型静态文本，每次调用直接解析，开销可忽略（与 rewrite_query 的处理一致）。
    """
    state = request.state
    requirement = state.get("requirement") or "回答教材相关问题，或按需整理成学习资料。"
    intent = state.get("intent") or _DEFAULT_INTENT
    intent_name = intent if intent in _INTENT_BLOCKS else _DEFAULT_INTENT
    return PromptTemplate.from_template(load_prompt("study_chat")).format(
        textbook_name=state.get("textbook_name", ""),
        requirement=requirement,
        intent_block=load_prompt(f"intent/{intent_name}"),
    )


@before_agent(state_schema=StudyState)
async def understand_query(state: StudyState, runtime: Runtime) -> dict:
    """前置「query 理解」（before_agent 入口节点）：意图识别 + 问题重写合流。

    每轮对话只执行一次（resume 不重跑），产物写入 state.intent / state.rewritten_query，
    供 dynamic_prompt（选行为块）与 search_textbook（取检索问句）消费。

    Args:
        state: 当前图状态（含 messages 完整历史与新 query）。
        runtime: 运行时上下文。

    Returns:
        state 更新字典（intent / rewritten_query）。
    """
    requirement = state.get("requirement") or ""
    textbook_name = state.get("textbook_name") or ""
    history = format_questions(state.get("messages") or [])

    # 首轮通常没有指代或省略，原问题已经是独立检索问句；此时跳过一次重写 LLM。
    # 只有存在历史用户问题时才做重写，解决“它/这个/上面提到的”等跨轮消歧。
    # before_agent 可以直接按 state 动态选择路径；resume 续跑不会重复进入这里。
    if history.strip():
        async def _rewrite_with_metric() -> str:
            async with astage("query_rewrite_ms"):
                return await rewrite(requirement, textbook_name, history)

        # 多轮场景下，意图识别与问题重写相互独立，并行执行。
        intent_result, rewritten = await asyncio.gather(
            detect_intent(requirement),
            _rewrite_with_metric(),
        )
        rewrite_mode = "llm"
    else:
        # 单轮清晰问题直接保留原文，避免一次无收益的重写调用。
        intent_result = await detect_intent(requirement)
        rewritten = requirement.strip()
        rewrite_mode = "skipped_no_history"
    metrics = get_metrics()
    if metrics is not None:
        metrics.intent = intent_result.intent
        metrics.intent_source = intent_result.source
        metrics.rewrite_mode = rewrite_mode
    logger.info(
        f"query 理解: intent={intent_result.intent}"
        f"({intent_result.source}, {intent_result.confidence:.2f}), "
        f"rewrite_mode={rewrite_mode} → rewrite={rewritten!r}"
    )
    return {
        "intent": intent_result.intent,
        "rewritten_query": rewritten,
    }


def build_graph(checkpointer=None):
    """构建教材知识学习 Agent（create_agent 单层 ReAct）。

    Args:
        checkpointer: 状态持久化器（断点续跑），传 MongoDBSaver。

    Returns:
        编译后的 CompiledStateGraph，可直接 stream / invoke。
    """
    return create_agent(
        model=get_llm_client(),
        tools=[search_textbook, list_chapters, search_web,
               ask_clarification,
               write_file, append_file, read_file, edit_file, list_files],
        middleware=[
            understand_query,  # before_agent：前置 query 理解（意图 + 重写），每轮一次
            study_system_prompt,  # dynamic_prompt：按意图动态渲染 system prompt
            ModelCallLimitMiddleware(run_limit=20),  # 护栏1：步数上限（到顶 end 收尾）
            SummarizationMiddleware(
                model=get_llm_client(),
                trigger=("tokens", 32000),  # 护栏2：单次上下文超 3.2w token 才触发摘要（控成本+延迟，非防爆，窗口 1M）
                keep=("messages", 5),        # 触发后保留最近 5 条消息不摘要
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"write_file": True, "edit_file": True, "append_file": True}
            ),  # 护栏：所有写盘动作写前人工确认
        ],
        state_schema=StudyState,  # 需含 messages(add_messages)
        checkpointer=checkpointer,
    )
