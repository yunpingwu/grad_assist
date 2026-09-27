"""study_agent 图装配测试：前置 query 理解节点的产物与 system prompt 渲染（桩掉意图识别，只看编排）。"""

import asyncio
import dataclasses

from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.study_agent import graph
from app.study_agent.entity.intent import IntentResult


def test_summarization_trim_limit_not_default(monkeypatch) -> None:
    """摘要中间件必须显式放宽 trim 预算：langchain 默认 4000 会被生成文档任务饿死。

    整篇文档写进单条 write_file 消息后，待摘要消息按 strategy=last + start_on=human
    裁到 4000 token 会剩空列表，中间件于是把「Previous conversation was too long to
    summarize.」当摘要注入对话，模型原样转给用户（2026-09-27 联调实见）。
    """
    captured: dict = {}
    monkeypatch.setattr(graph, "create_agent", lambda **kw: captured.update(kw) or "stub_graph")
    assert graph.build_graph() == "stub_graph"
    mw = next(m for m in captured["middleware"] if isinstance(m, SummarizationMiddleware))
    assert mw.trim_tokens_to_summarize is None, "摘要输入不应裁剪：触发时全量约 32k，模型窗口 1M 装得下"


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
    monkeypatch.setattr(graph, "get_metrics", lambda: None)
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


def test_system_prompt_shows_low_confidence_hint() -> None:
    """置信度透传：低置信判定的「意图预判」段提示以原话为准、可用 ask_clarification 反问。"""
    prompt = _render_system_prompt(
        {"requirement": "q", "textbook_name": "T", "intent": "explain", "intent_confidence": 0.42, "intent_source": "llm"}
    )
    assert "# 意图预判" in prompt
    section = prompt.split("# 意图预判")[1].split("# 底线")[0]
    assert "置信度 0.42" in section and "llm" in section, section
    assert "ask_clarification" in section, "低置信必须提示模型可自行反问的出口"
    assert "置信度偏低" in section


def test_system_prompt_high_confidence_skips_clarify_hint() -> None:
    """高置信判定只报意图来源，不加「可反问」暗示，避免模型见句就想澄清。"""
    prompt = _render_system_prompt(
        {"requirement": "q", "textbook_name": "T", "intent": "quiz", "intent_confidence": 0.95, "intent_source": "keyword"}
    )
    section = prompt.split("# 意图预判")[1].split("# 底线")[0]
    assert "置信度 0.95" in section and "keyword" in section, section
    assert "ask_clarification" not in section


def test_system_prompt_degrades_without_confidence() -> None:
    """降级路径（节点失败/旧 state 无置信度字段）也要渲染出兜底说明，不留空段。"""
    prompt = _render_system_prompt({"requirement": "q", "textbook_name": "T"})
    section = prompt.split("# 意图预判")[1].split("# 底线")[0]
    assert "未产出置信度" in section and "explain" in section, section
