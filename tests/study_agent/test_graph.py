"""study_agent 图装配测试：前置 query 理解节点的产物（桩掉意图识别，只看编排）。"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.study_agent import graph
from app.study_agent.entity.intent import IntentResult


def test_understand_query_passes_history_and_returns_intent_only(monkeypatch) -> None:
    """节点把「当前问句 + 历史」交给意图识别，产物只有 intent（重写产物已下线）。"""
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

    assert update == {"intent": "quiz"}, f"不应再产出 rewritten_query: {update}"
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
