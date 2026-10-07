"""study_agent 图装配测试：前置 query 理解节点的产物与 system prompt 渲染（桩掉意图识别，只看编排）。"""

import asyncio
import dataclasses

from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.study_agent import graph
from app.study_agent.entity.intent import IntentResult
from app.trace import RequestMetrics


def test_summarization_trigger_above_online_p90(monkeypatch) -> None:
    """摘要触发阈值钉在 6.4w：线上 p90 单轮 4.8w，不该撞上它。

    摘要一触发就把历史换成「摘要+最近 5 条」的新形态，跨轮前缀当场作废（2026-10-06
    探针实见 t2 命中塌回 3072），还要多付一次 7w token 量级的摘要调用。
    trim 预算不在这里断言：窗口 1M 装得下触发时的全量消息，中间件默认的 4000 裁剪
    对本图不构成约束（曾按「必须放宽 trim」立断言，2026-10-07 复核为不需要，作废）。
    """
    captured: dict = {}
    monkeypatch.setattr(graph, "create_agent", lambda **kw: captured.update(kw) or "stub_graph")
    assert graph.build_graph() == "stub_graph"
    mw = next(m for m in captured["middleware"] if isinstance(m, SummarizationMiddleware))
    assert mw.trigger == ("tokens", 64000), (
        "阈值提到 6.4w：摘要一触发就把历史换成「摘要+最近 5 条」的新形态，跨轮前缀当场作废"
        "（2026-10-06 探针实见 t2 命中塌回 3072），线上 p90 单轮 4.8w 不该撞上它"
    )


def test_summarization_call_is_tagged_for_trace(monkeypatch) -> None:
    """摘要那次 LLM 调用要带 ``trace_role:summary``，与 agent 首轮区分开。

    摘要一触发就是 7w token 量级的调用（2026-10-06 实见），不带标签时它在观测里与
    agent 调用混在一起，前后对比会把摘要的账记到提示词重排上。
    """
    captured: dict = {}
    monkeypatch.setattr(graph, "create_agent", lambda **kw: captured.update(kw) or "stub_graph")
    graph.build_graph()
    mw = next(m for m in captured["middleware"] if isinstance(m, SummarizationMiddleware))
    assert "trace_role:summary" in mw.model.config["tags"]


def _render_system_prompt(state: dict) -> str:
    """走 dynamic_prompt 中间件渲染一次 system prompt，返回注入后的文本。"""
    kwargs = {f.name: None for f in dataclasses.fields(ModelRequest)}
    kwargs.update(messages=[], state=state)
    request = ModelRequest(**kwargs)
    captured: dict[str, str] = {}

    async def _handler(req: ModelRequest) -> ModelResponse:
        captured["prompt"] = req.system_prompt
        return ModelResponse(result=[])

    asyncio.run(graph.study_system_prompt.awrap_model_call(request, _handler))
    return captured["prompt"]


def test_understand_query_passes_history_and_returns_intent_only(monkeypatch) -> None:
    """节点把「当前问句 + 历史」交给意图识别，产物只有意图三元组（结果+置信度+来源，重写产物已下线）。"""
    seen: list[tuple[str, str]] = []

    async def _fake_detect(text: str, history: str = "") -> IntentResult:
        seen.append((text, history))
        return IntentResult("quiz", 0.95, "keyword")

    monkeypatch.setattr(graph, "detect_intent", _fake_detect)
    record = RequestMetrics(request_id="t1")
    monkeypatch.setattr(graph, "get_metrics", lambda: record)
    state = {
        "task_id": "t1",
        "textbook_name": "C语言程序设计",
        "requirement": "那再来十道",
        "messages": [
            HumanMessage(content="帮我出五道选择题"),
            AIMessage(content="1. ……"),
            HumanMessage(content="那再来十道"),
        ],
    }

    update = asyncio.run(graph.understand_query.abefore_agent(state, None))

    assert update == {"intent": "quiz", "intent_confidence": 0.95, "intent_source": "keyword"}, f"不应再产出 rewritten_query: {update}"
    assert seen == [("那再来十道", "用户: 帮我出五道选择题")], "历史未随当前问句一起送进意图识别"
    assert (record.intent, record.intent_source, record.intent_confidence) == (
        "quiz", "keyword", 0.95
    ), "意图三元组未写进请求指标，线上无法复盘判错是阈值还是例句池问题"


def test_understand_query_first_turn_has_empty_history(monkeypatch) -> None:
    """首轮无历史：兜底层拿到空串，由它自己标注首次提问。"""
    seen: list[tuple[str, str]] = []

    async def _fake_detect(text: str, history: str = "") -> IntentResult:
        seen.append((text, history))
        return IntentResult("explain", 0.6, "embedding")

    monkeypatch.setattr(graph, "detect_intent", _fake_detect)
    monkeypatch.setattr(graph, "get_metrics", lambda: None)
    state = {
        "task_id": "t2",
        "textbook_name": "C语言程序设计",
        "requirement": "什么是指针?",
        "messages": [HumanMessage(content="什么是指针?")],
    }

    asyncio.run(graph.understand_query.abefore_agent(state, None))

    assert seen == [("什么是指针?", "")]


def test_understand_query_swallows_intent_failure_and_uses_default(monkeypatch) -> None:
    """前置理解整段失败不外抛：回退默认「讲解」块并留 warning 日志，主链路照常跑。"""
    warnings: list[str] = []

    async def _boom(text: str, history: str = "") -> IntentResult:
        raise RuntimeError("意图识别炸了")

    monkeypatch.setattr(graph, "detect_intent", _boom)
    monkeypatch.setattr(graph, "get_metrics", lambda: None)
    monkeypatch.setattr(graph.logger, "warning", lambda msg: warnings.append(str(msg)))
    state = {
        "task_id": "t3",
        "textbook_name": "C语言程序设计",
        "requirement": "什么是指针?",
        "messages": [HumanMessage(content="什么是指针?")],
    }

    update = asyncio.run(graph.understand_query.abefore_agent(state, None))

    assert update == {"intent": "explain"}, f"失败时未回退默认行为块: {update}"
    assert warnings, "静默回退会让线上故障不可见，必须留告警"


def test_understand_query_swallows_history_formatting_failure(monkeypatch) -> None:
    """兜底覆盖整节点而非只包住识别调用：取历史这一步抛异常同样回退默认块。"""
    seen: list[tuple[str, str]] = []

    async def _fake_detect(text: str, history: str = "") -> IntentResult:
        seen.append((text, history))
        return IntentResult("quiz", 0.9, "keyword")

    def _boom(_messages):
        raise AttributeError("消息体畸形")

    monkeypatch.setattr(graph, "format_history", _boom)
    monkeypatch.setattr(graph, "detect_intent", _fake_detect)
    monkeypatch.setattr(graph, "get_metrics", lambda: None)
    monkeypatch.setattr(graph.logger, "warning", lambda _msg: None)
    state = {
        "task_id": "t4",
        "textbook_name": "C语言程序设计",
        "requirement": "再来十道",
        "messages": [HumanMessage(content="再来十道")],
    }

    update = asyncio.run(graph.understand_query.abefore_agent(state, None))

    assert update == {"intent": "explain"}, "整节点异常也必须走默认块，不能外抛"
    assert seen == [], "识别依赖历史产物，历史失败时不应带着半截状态继续识别"


def _render_turn(state: dict) -> str:
    """渲染「本轮指令」正文（方案 A：动态字段已移出 system prompt）。"""
    return graph._render_turn_instruction(state)


def test_turn_instruction_shows_low_confidence_hint() -> None:
    """置信度透传：低置信判定的「意图预判」段提示以原话为准、可用 ask_clarification 反问。"""
    prompt = _render_turn(
        {"requirement": "q", "textbook_name": "T", "intent": "explain", "intent_confidence": 0.42, "intent_source": "llm"}
    )
    assert "## 意图预判" in prompt
    section = prompt.split("## 意图预判")[1].split("## 任务目标")[0]
    assert "置信度 0.42" in section and "llm" in section, section
    assert "ask_clarification" in section, "低置信必须提示模型可自行反问的出口"
    assert "置信度偏低" in section


def test_turn_instruction_high_confidence_skips_clarify_hint() -> None:
    """高置信判定只报意图来源，不加「可反问」暗示，避免模型见句就想澄清。"""
    prompt = _render_turn(
        {"requirement": "q", "textbook_name": "T", "intent": "quiz", "intent_confidence": 0.95, "intent_source": "keyword"}
    )
    section = prompt.split("## 意图预判")[1].split("## 任务目标")[0]
    assert "置信度 0.95" in section and "keyword" in section, section
    assert "ask_clarification" not in section


def test_turn_instruction_degrades_without_confidence() -> None:
    """降级路径（节点失败/旧 state 无置信度字段）也要渲染出兜底说明，不留空段。"""
    prompt = _render_turn({"requirement": "q", "textbook_name": "T"})
    section = prompt.split("## 意图预判")[1].split("## 任务目标")[0]
    assert "未产出置信度" in section and "explain" in section, section


def test_system_prompt_is_turn_invariant() -> None:
    """system prompt 只留全局静态壳：换教材、换意图、换问句，渲染结果必须逐字相同。

    前缀缓存按「从第一条 token 起完全相同」匹配，壳里只要还有一个每轮变的字段，
    它后面的整段历史就被划出公共前缀（2026-10-06 探针实见：跨轮命中钉死 3072，
    t3 的 9.7k token 历史 0 复用）。本轮变动改由 ``append_turn_instruction`` 放尾部。
    """
    states = [
        {"requirement": "什么是指针?", "textbook_name": "T1", "intent": "explain", "intent_confidence": 0.42, "intent_source": "llm"},
        {"requirement": "再来十道题", "textbook_name": "C语言程序设计", "intent": "quiz", "intent_confidence": 0.95, "intent_source": "keyword"},
        {"requirement": "帮我整理成笔记", "textbook_name": "数据结构", "intent": "generate"},
    ]
    rendered = [_render_system_prompt(s) for s in states]
    assert all(r == rendered[0] for r in rendered), "system prompt 随本轮 state 变了，跨轮前缀必丢"
    assert "{" not in rendered[0], "模板占位符没被替换干净"
    for state in states:
        assert state["requirement"] not in rendered[0], "任务目标漏进静态壳"
        assert state["textbook_name"] not in rendered[0], "教材名漏进静态壳"


def test_turn_instruction_recorded_into_state_once() -> None:
    """本轮指令在轮首落进 state 一次：之后各次调用都沿用同一条，位置不再漂移。

    旧写法是每次模型调用往尾部重贴，于是第 2 次起消息变成 ``[历史, AI1, tool1, 指令]``，
    与第 1 次存下的 ``[历史, 指令]`` 前缀当场分叉（planA64k 实见 t1 later=[3072, 14336]，
    指令进壳时是 [3584, 8960, 18432] 单调上行）。
    """
    state = {
        "requirement": "本轮问题",
        "textbook_name": "T",
        "intent": "explain",
        "intent_confidence": 0.9,
        "intent_source": "keyword",
        "messages": [HumanMessage(content="上一轮问题"), AIMessage(content="上一轮回答"), HumanMessage(content="本轮问题")],
    }
    update = asyncio.run(graph.record_turn_instruction.abefore_agent(state, None))
    added = update["messages"]
    assert len(added) == 1 and isinstance(added[0], SystemMessage), added
    content = added[0].content
    assert "## 任务目标" in content and "本轮问题" in content
    assert "当前教材：《T》" in content and "置信度 0.90" in content
    assert added[0].additional_kwargs.get("lc_source") == "turn_instruction", "需可识别，供过滤与去重"


def test_turn_instruction_not_appended_twice_in_one_turn() -> None:
    """同一轮重入（resume 回同一线程）不重复追加：内容一致即视为已落定。"""
    state = {
        "requirement": "本轮问题",
        "textbook_name": "T",
        "intent": "explain",
        "intent_confidence": 0.9,
        "intent_source": "keyword",
        "messages": [HumanMessage(content="本轮问题")],
    }
    first = asyncio.run(graph.record_turn_instruction.abefore_agent(state, None))["messages"][0]
    reentry = {**state, "messages": [*state["messages"], first]}
    assert asyncio.run(graph.record_turn_instruction.abefore_agent(reentry, None)) in (None, {})


def test_turn_instruction_hook_is_before_agent_not_wrap_model_call(monkeypatch) -> None:
    """指令必须走 ``@before_agent``（一轮一次），不能再挂 ``wrap_model_call``（每调用一次）。

    每调用重贴就是轮内前缀分叉的成因；同时 ``before_agent`` 排在 ``understand_query`` 之后，
    才拿得到本轮意图去选行为块。
    """
    captured: dict = {}
    monkeypatch.setattr(graph, "create_agent", lambda **kw: captured.update(kw) or "stub_graph")
    graph.build_graph()
    chain = captured["middleware"]
    assert not hasattr(graph, "append_turn_instruction"), "尾部重贴的旧挂点应已删除"
    assert chain.index(graph.record_turn_instruction) > chain.index(graph.understand_query), "指令早于意图，行为块会选错"
    assert graph.record_turn_instruction in chain and graph.stub_old_search_results in chain
