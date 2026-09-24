"""教材知识学习 Agent 图：create_agent 单层 ReAct（方案 A：单层循环 + 护栏）。

不再手写 StateGraph 包装——create_agent 本身即编译好的图对象
（CompiledStateGraph），内部已编排 model ⇄ tools 循环与终止条件；
收尾（扫描落盘、登记清单）为图外的业务函数，见 service/API 层。

动态提示词与前置 query 理解通过 langchain.agents 的 middleware 机制接入：
- ``@before_agent``：作为入口节点，每轮对话只执行一次，做意图识别（失败回退默认行为块，不阻断本轮）；
- ``@dynamic_prompt``：每次 model 调用前按 state（含意图）动态渲染 system prompt。

检索问句不再前置重写：模型在 ReAct 循环里看得见完整对话（历史用户问题 + 回答），
由它自己组织自包含的 ``search_textbook(query=...)``；缺省回退本轮原问题（requirement）。
跨轮消歧所需的历史改喂给意图识别的 LLM 兜底层（那里才是真的缺上下文判不准的地方）。
"""

from __future__ import annotations

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
from app.core import get_metrics, load_prompt, logger
from app.study_agent.query_functions.context_stub import stub_old_search_results
from app.study_agent.query_functions.intent_detection import detect_intent, format_history
from app.study_agent.state import StudyState
from app.study_agent.tools import (
    append_file,
    ask_clarification,
    edit_file,
    list_chapters,
    list_files,
    read_chunk,
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
    骨架模板为小型静态文本，每次调用直接解析，开销可忽略。
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
    """前置「query 理解」（before_agent 入口节点）：三档漏斗意图识别。

    每轮对话只执行一次（resume 不重跑），产物写入 state.intent，供 dynamic_prompt
    选行为块。对话历史一并交给它——兜底层靠历史消解跨轮指代与省略。

    整节点包在 try 里：这一步只是「选哪个行为块」的增强，判不出还有 explain 兜着，
    不值得为它让本轮对话直接失败，故异常一律告警后回退默认块（``logger.warning`` 保证
    线上可见，不做静默吞掉）。``except Exception`` 不拦 BaseException，
    取消信号（asyncio.CancelledError）照常向上传播。

    Args:
        state: 当前图状态（含 messages 完整历史与新 query）。
        runtime: 运行时上下文。

    Returns:
        state 更新字典（intent）；识别链路异常时为默认意图。
    """
    try:
        requirement = state.get("requirement") or ""
        history = format_history(state.get("messages") or [])
        intent_result = await detect_intent(requirement, history)
        metrics = get_metrics()
        if metrics is not None:
            metrics.intent = intent_result.intent
            metrics.intent_source = intent_result.source
        logger.info(
            f"query 理解: intent={intent_result.intent}"
            f"({intent_result.source}, {intent_result.confidence:.2f}), "
            f"多轮={bool(history.strip())}"
        )
        return {"intent": intent_result.intent}
    except Exception as exc:
        logger.warning(f"前置 query 理解失败，回退默认行为块 {_DEFAULT_INTENT}: {exc!r}")
        return {"intent": _DEFAULT_INTENT}


def build_graph(checkpointer=None):
    """构建教材知识学习 Agent（create_agent 单层 ReAct）。

    Args:
        checkpointer: 状态持久化器（断点续跑），传 MongoDBSaver。

    Returns:
        编译后的 CompiledStateGraph，可直接 stream / invoke。
    """
    return create_agent(
        model=get_llm_client(),
        tools=[search_textbook, read_chunk, list_chapters, search_web,
               ask_clarification,
               write_file, append_file, read_file, edit_file, list_files],
        middleware=[
            understand_query,  # before_agent：前置 query 理解（意图识别），每轮一次
            stub_old_search_results,  # wrap_model_call：历史轮检索结果存根化（只改请求视图，不动 state）
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
