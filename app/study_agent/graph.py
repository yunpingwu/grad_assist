"""教材知识学习 Agent 图：create_agent 单层 ReAct（方案 A：单层循环 + 护栏）。

不再手写 StateGraph 包装——create_agent 本身即编译好的图对象
（CompiledStateGraph），内部已编排 model ⇄ tools 循环与终止条件；
收尾（扫描落盘、登记清单）为图外的业务函数，见 service/API 层。

动态提示词与前置 query 理解通过 langchain.agents 的 middleware 机制接入：
- ``@before_agent``：作为入口节点，每轮对话只执行一次，依次做意图识别（失败回退默认行为块，
  不阻断本轮）、把「本轮指令」落进 state、把往轮检索原文回收成一句占位并写回 state；
- ``@dynamic_prompt``：渲染 system prompt，**只装全局静态壳**（教材名/意图/任务目标一律不进，为的是前缀缓存）。

「本轮指令」（教材名 + 意图行为块 + 意图预判 + 任务目标）在轮首作为 SystemMessage 追加到
历史之后一次，轮内各次调用共用同一条——每轮变动的内容留在 system prompt 里会切断历史前缀，
每次调用重贴则会在轮内切断（详见 ``record_turn_instruction``）。

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
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    before_agent,
    dynamic_prompt,
)
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.clients.llm import get_llm_client
from app.config import budget_config
from app.core import load_prompt, logger
from app.study_agent.middleware import SessionBudgetMiddleware
from app.study_agent.query_functions.intent_detection import detect_intent, format_history
from app.study_agent.query_functions.tool_history import demote_used_tool_results
from app.study_agent.state import StudyState
from app.study_agent.tools import AGENT_TOOLS
from app.trace import get_metrics

# 意图 → 行为块文件的映射（白名单，防越界读文件）
_INTENT_BLOCKS = frozenset({"explain", "generate", "quiz", "plan", "chat"})
_DEFAULT_INTENT = "explain"
# 本轮指令消息的来源标记：与摘要消息同款机制，供历史过滤与同轮去重识别
_TURN_INSTRUCTION_SOURCE = "turn_instruction"


@dynamic_prompt
def study_system_prompt(request: ModelRequest) -> str:
    """渲染 system prompt 正文：只留全局静态壳，本轮变动的一律不进。

    前缀缓存按「从第一条 token 起完全相同」匹配，教材名/意图/任务目标换个会话或换轮就变，
    留在壳里就把其后的整段历史划出公共前缀；它们改由 ``record_turn_instruction`` 落到历史之后。
    """
    return load_prompt("study_chat")


def _render_turn_instruction(state: dict) -> str:
    """渲染「本轮指令」正文：教材名 + 意图行为块 + 意图预判 + 任务目标。

    意图未识别/越界回退默认「讲解」块；提示词统一走 ``load_prompt``（自带 LRU 缓存读盘），
    骨架模板为小型文本，每次调用直接解析，开销可忽略。
    """
    requirement = state.get("requirement") or "回答教材相关问题，或按需整理成学习资料。"
    intent = state.get("intent") or _DEFAULT_INTENT
    intent_name = intent if intent in _INTENT_BLOCKS else _DEFAULT_INTENT
    conf, src = state.get("intent_confidence"), state.get("intent_source") or "?"
    if isinstance(conf, (int, float)):
        intent_note = f"前置识别判定本轮意图为「{intent_name}」（置信度 {conf:.2f}，来源 {src}）。"
        if conf < 0.6:
            intent_note += "该判定置信度偏低：与用户原话对不上时以原话为准；产出方向确实不明，可先用 ask_clarification 反问一次。"
    else:
        intent_note = f"前置识别未产出置信度（降级路径），按「{intent_name}」块行事；拿不准用户要什么时可先用 ask_clarification 反问一次。"
    return PromptTemplate.from_template(load_prompt("study_turn")).format(
        textbook_name=state.get("textbook_name", ""),
        requirement=requirement,
        intent_block=load_prompt(f"intent/{intent_name}"),
        intent_note=intent_note,
    )


@before_agent(state_schema=StudyState)
async def record_turn_instruction(state: StudyState, runtime: Runtime) -> dict:
    """把「本轮指令」作为 SystemMessage 落进 state，**一轮只落一次**。

    放尾部而不是 system prompt 里，是为了让每轮变动的教材名/意图/任务目标不切断历史前缀；
    落进 state 而不是每次模型调用重贴，是为了轮内前缀单调增长——重贴时第 2 次起消息变成
    ``[历史, AI1, tool1, 指令]``，与第 1 次存下的 ``[历史, 指令]`` 在那个位置分叉。
    内容一致即视为本轮已落定（resume 回同一线程时不重复追加）。
    """
    content = _render_turn_instruction(state)
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if (
            last.type == "system"
            and (last.additional_kwargs or {}).get("lc_source") == _TURN_INSTRUCTION_SOURCE
            and str(last.content) == content
        ):
            return {}
    return {
        "messages": [
            SystemMessage(content=content, additional_kwargs={"lc_source": _TURN_INSTRUCTION_SOURCE})
        ]
    }


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
            # 置信度一并留痕：判错时要靠它区分「阈值/例句池检索」问题，消融实验也读这个字段
            metrics.intent_confidence = intent_result.confidence
        logger.info(
            f"query 理解: intent={intent_result.intent}"
            f"({intent_result.source}, {intent_result.confidence:.2f}), "
            f"多轮={bool(history.strip())}"
        )
        # 置信度与来源一并入 state：system prompt 的「意图预判」段据此提示模型
        # 低置信时自行斟酌 ask_clarification（路由判不准 ≠ 必反问，最终裁量在模型）
        return {
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
            "intent_source": intent_result.source,
        }
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
    # 护栏：会话成本预算（BUDGET_ENABLED=false 时整体不装）。必须排在列表最后：
    # after_model 逆注册序执行，最后注册的先跑——被 HITL interrupt 掐断的调用也能先折叠成本
    budget_guard = [SessionBudgetMiddleware()] if budget_config.enabled else []
    return create_agent(
        model=get_llm_client(),
        tools=list(AGENT_TOOLS),  # 工具清单唯一来源（tools/__init__.py），避免两处各自漂移
        middleware=[
            understand_query,  # before_agent：前置 query 理解（意图识别），每轮一次
            record_turn_instruction,  # before_agent：本轮指令落 state 一次（须在意图之后）
            demote_used_tool_results,  # before_agent：往轮检索原文回收成一句占位，写回 state
            study_system_prompt,  # dynamic_prompt：只渲染全局静态壳（本轮变量不进壳）
            ModelCallLimitMiddleware(run_limit=20),  # 护栏：步数上限（到顶 end 收尾）
            # 护栏：检索是最贵也最不稳的两个工具，按轮封顶——超额的调用在模型侧就被拦下
            ToolCallLimitMiddleware(tool_name="search_textbook", run_limit=4),
            ToolCallLimitMiddleware(tool_name="search_web", run_limit=2),
            # 护栏：工具抛异常 → 先原地重试一次，耗尽后回官方说明文本（模型据此改参数或收尾）
            ToolRetryMiddleware(max_retries=1, on_failure="continue"),
            SummarizationMiddleware(
                # 打标只给摘要这次调用：create_agent 的 model 参数只收 BaseChatModel，
                # 所以 agent 主体调用保持未打标——观测侧把无 trace_role 的调用一律当 agent
                model=get_llm_client().with_config(tags=["trace_role:summary"]),
                # 护栏：6.4w 摘要
                trigger=("tokens", 64000),
                keep=("messages", 3),        # 触发后保留最近 3 条消息不摘要
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"write_file": True, "edit_file": True, "append_file": True}
            ),  # 护栏：所有写盘动作写前人工确认
            *budget_guard,  # 护栏：per-session 成本上限（装配位置理由见 app/study_agent/middleware.py）
        ],
        state_schema=StudyState,  # 需含 messages(add_messages)
        checkpointer=checkpointer,
    )
