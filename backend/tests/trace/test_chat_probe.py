"""chat_probe 纯函数测试：问题集校验、会话筛选、命中率统计与曲线。"""

import asyncio
import json

import httpx
import pytest

from app.trace.scripts import chat_probe as probe

_SHIPPED = probe._DEFAULT_QUESTIONS


def _write(tmp_path, payload: dict):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _record(prompt: int, cached: int, *, query: str = "问题", cost: float = 1.0, compacted: dict | None = None) -> dict:
    """造一条与本仓库 metrics.log 同形的请求记录（只留统计用得到的键）。"""
    return {"query": query, "prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": 10,
            "turn_cost": cost, "steps": [], "compacted": compacted}


# ── 问题集 ──────────────────────────────────────────────


def test_shipped_question_set_is_valid() -> None:
    """随仓库发布的问题集本身要能过校验：10 段会话、每段 10 题（改题时别把结构写坏）。"""
    conversations = probe.load_conversations(_SHIPPED)

    assert len(conversations) == 10, "问题集应有 10 段会话"
    assert all(len(c["questions"]) >= 10 for c in conversations), "每段会话至少 10 题"
    assert max(len(c["questions"]) for c in conversations) >= 50, (
        "至少要有一段 50 轮的长会话：命中率随轮数上升，短会话量不出长会话的水平"
    )
    assert all(c["textbook"] for c in conversations), "每段会话都要指定教材子串"
    names = [c["name"] for c in conversations]
    assert len(set(names)) == len(names), "会话名要唯一，否则 --only 子串匹配到谁说不准"


def test_load_rejects_wrong_schema_and_missing_fields(tmp_path) -> None:
    """schema 不对、或某段会话缺字段，都要在开跑前报清楚——跑起来才发现等于白等一小时。"""
    with pytest.raises(SystemExit, match="schema"):
        probe.load_conversations(_write(tmp_path, {"schema": "other/v1", "conversations": []}))

    with pytest.raises(SystemExit, match="缺字段"):
        probe.load_conversations(_write(tmp_path, {"schema": "chat_probe_questions/v1",
                                                   "conversations": [{"name": "x", "questions": ["q"]}]}))

    with pytest.raises(SystemExit, match="没有任何会话"):
        probe.load_conversations(_write(tmp_path, {"schema": "chat_probe_questions/v1", "conversations": []}))


def test_select_by_index_and_substring(tmp_path) -> None:
    """--only 支持序号与名字子串：序号是给「复现第 3 段」用的，子串是给「跑所有图相关的」用的。"""
    conversations = probe.load_conversations(_SHIPPED)

    by_index = probe.select_conversations(conversations, "1,3")
    assert [i for i, _ in by_index] == [1, 3]

    by_name = probe.select_conversations(conversations, "数据结构")
    assert [c[1]["name"] for c in by_name] == ["数据结构·树与二叉树", "数据结构·图与最短路径"]

    with pytest.raises(SystemExit, match="没匹配到"):
        probe.select_conversations(conversations, "不存在的名字")


# ── 统计口径 ────────────────────────────────────────────


def test_session_stats_splits_first_turn_from_steady_state() -> None:
    """首轮没有跨轮缓存，稳态（第 2 轮起）才是治理效果——两个口径必须分开算。

    续跑（写盘确认，query 为空）不算「轮」，但它的 token 与成本照算（钱是实打实花掉的）。
    """
    records = [
        _record(1000, 100, query="第一问", cost=2.0),          # 首轮：只有静态壳命中
        _record(4000, 3200, query="第二问", cost=5.0),          # 稳态
        _record(5000, 4000, query="", cost=6.0),              # 续跑（写盘确认，无 query）
    ]

    stat = probe.session_stats(records)

    assert stat["turns"] == 2 and stat["resumes"] == 1
    assert stat["total_prompt"] == 10000 and stat["total_cached"] == 7300
    assert stat["hit"] == pytest.approx(0.73)
    assert stat["first_hit"] == pytest.approx(0.1)
    assert stat["steady_hit"] == pytest.approx(3200 / 4000)
    assert stat["peak_prompt"] == 5000, "峰值输入取单条记录的最大输入，用于看有没有顶窗口"
    assert stat["cost"] == pytest.approx(13.0)


def test_session_stats_collects_compaction_events() -> None:
    """压缩事件从记录里捞出来（compacted 字段），报告据此说清发生在哪一轮。"""
    info = {"trigger_tokens": 25000, "trigger_source": "reported", "dropped_msgs": 12, "dropped_tokens": 9000, "kept_msgs": 4}
    records = [_record(20000, 1000, query="问一"), _record(26000, 5000, query="问二", compacted=info)]

    stat = probe.session_stats(records)

    assert stat["compactions"] == [info]


def test_turn_hit_curve_averages_by_position() -> None:
    """曲线按轮位平均，且只有跑到该轮位的会话参与——短会话不拉低后面的轮次。"""
    stats = {
        1: probe.session_stats([_record(1000, 100, query="a1"), _record(1000, 900, query="a2")]),
        2: probe.session_stats([_record(1000, 200, query="b1")]),  # 只有一轮，不该出现在第 2 位上
    }

    curve = probe.turn_hit_curve(stats)

    assert [(pos, n) for pos, _, n in curve] == [(1, 2), (2, 1)]
    assert curve[0][1] == pytest.approx((0.1 + 0.2) / 2)
    assert curve[1][1] == pytest.approx(0.9)


def test_session_records_filters_by_session_and_cutoff() -> None:
    """切记录要同时卡会话与开跑时刻：同 user 的旧会话与并发跑的另一段都不能混进来。"""
    records = [
        {"session_id": "s1", "ts": "2026-10-16T10:00:00.000+08:00", "query": "旧的"},
        {"session_id": "s1", "ts": "2026-10-16T11:00:00.000+08:00", "query": "本轮一"},
        {"session_id": "s1", "ts": "2026-10-16T11:00:05.000+08:00", "query": "本轮二"},
        {"session_id": "s2", "ts": "2026-10-16T11:00:01.000+08:00", "query": "另一段会话"},
    ]

    picked = probe.session_records(records, "s1", "2026-10-16T11:00:00.000+08:00")

    assert [r["query"] for r in picked] == ["本轮一", "本轮二"]


def test_report_renders_without_compaction() -> None:
    """无压缩事件的报告要给出「怎么才能看到压缩」的可执行指引，而不是留白。"""
    stats = probe.session_stats([_record(1000, 100, query="问一"), _record(2000, 1500, query="问二")])

    report = probe.build_report([(1, "某会话", stats)], timeline=False)

    assert "本轮没有触发会话压缩" in report and "DEEPSEEK_CONTEXT_WINDOW" in report
    assert "跨会话命中率曲线" in report and "会话汇总" in report


def test_report_flags_missing_conversations() -> None:
    """跑挂一段也要出报告，并在开头点明缺了几段——二十来分钟的探针不能一份数据都不留。"""
    stats = probe.session_stats([_record(1000, 100, query="问一")])

    report = probe.build_report([(1, "某会话", stats)], timeline=False, total=10)

    assert "计划 10 段会话，实际完成 1 段" in report
    assert probe.build_report([], timeline=False) == "没有可汇总的会话：全部被跳过或没跑成（控制台里有各段的 [!] 原因）。"


# ── 教材名解析（探针跑挂过的那个坑）─────────────────────────


def _list_client(names: list[str]) -> httpx.AsyncClient:
    """假 /list：返回给定教材名（httpx.MockTransport，与仓库里假 SSE 同一手法）。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"textbook_name": n} for n in names]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_resolve_textbook_picks_first_match_and_warns_on_ambiguity(monkeypatch) -> None:
    """子串撞上多本时取 /list 的第一本并告警——库里有两本「计算机组成原理」，静默挑一本会看不出跑的是哪本。"""
    warnings: list[str] = []
    monkeypatch.setattr(probe.logger, "warning", lambda msg: warnings.append(str(msg)))
    names = ["2023计算机组成原理考研复习指导 (王道论坛)", "计算机组成原理_第6版_(白中英,_戴志涛)"]

    resolved = asyncio.run(probe._resolve_textbook(_list_client(names), "http://x", "计算机组成原理"))

    assert resolved == names[0]
    assert warnings and "匹配到 2 本" in warnings[0]


def test_client_does_not_trust_system_proxy() -> None:
    """探针客户端必须关掉代理信任：Windows 的系统代理会把 127.0.0.1 的请求截走（真机踩过 502）。"""
    client = probe.build_client(timeout=5.0)
    try:
        assert client.trust_env is False
    finally:
        asyncio.run(client.aclose())


def test_resolve_textbook_missing_raises_with_available_names() -> None:
    """匹配不到要抛可处理的异常（而不是 SystemExit 带走整轮），错误里带上现有教材名。"""
    with pytest.raises(LookupError, match="C语言程序设计"):
        asyncio.run(probe._resolve_textbook(_list_client(["C语言程序设计（第五版）"]), "http://x", "C程序设计"))


def test_shipped_question_set_textbooks_are_resolvable() -> None:
    """问题集的教材子串必须能唯一地落到真实库的命名上。

    真机踩过：库里是「C语言程序设计（第五版）」，问题集里写的是「C程序设计」（那是 app/eval
    数据集里的旧名字），一段会话因此在 /list 匹配不到——这里把「子串必须出现在某个真实教材名里」
    钉住，库里的名字改了也能立刻发现。
    """
    registered = [
        "2023计算机组成原理考研复习指导 (王道论坛)",
        "C语言程序设计（第五版）_(谭浩强)",
        "操作系统：精髓与设计原理_第8版_(斯托林斯)",
        "数据库系统概论(第6版)_(王珊,杜小勇,陈红)",
        "数据结构 (陈越、何钦铭、徐镜春、魏宝刚、杨枨编)",
        "计算机组成原理_第6版_(白中英,_戴志涛)",
        "计算机网络（第8版）_(谢希仁)",
    ]

    for conv in probe.load_conversations(_SHIPPED):
        assert any(conv["textbook"] in name for name in registered), f"{conv['name']} 的教材子串 {conv['textbook']!r} 在库里找不到"
