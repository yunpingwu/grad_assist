"""intent.run_eval 装配与聚合逻辑单测：三来源加载、默认口径、组过滤截断、多数票（不触碰真实检测）。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.eval.intent.run_eval import build_cases, filter_cases, load_extra, load_multi_anchor, load_single_anchor, resolve_majority


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_single_anchor_maps_all_to_explain(tmp_path: Path) -> None:
    book = tmp_path / "教材A"
    _write(book / "qa_set.json", {"textbook": "教材A", "questions": [{"id": "Q1", "question": "什么是指针"}, {"question": "TCP握手机制"}]})
    cases = load_single_anchor(tmp_path)
    assert len(cases) == 2
    assert all(c["expected"] == ["explain"] and c["group"] == "anchor_single" and c["history"] == "" for c in cases)
    assert cases[1]["id"] == "S-2", "缺 id 用序号兜底"


def test_load_multi_anchor_builds_history_prefix(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "qa_set.json",
        {
            "conversations": [
                {
                    "turns": [
                        {"id": "T1", "question": "什么是外键", "turn_role": "initial"},
                        {"id": "T2", "question": "那主键呢", "turn_role": "followup", "anaphora": "ellipsis"},
                    ]
                }
            ]
        },
    )
    cases = load_multi_anchor(path)
    assert cases[0]["group"] == "anchor_multi_initial" and cases[0]["history"] == ""
    assert cases[1]["group"] == "anchor_multi_follow"
    assert cases[1]["history"] == "用户: 什么是外键", "与线上 format_history 同构"
    assert cases[1]["anaphora"] == "ellipsis"


def test_load_extra_prefixes_ids(tmp_path: Path) -> None:
    one = {"id": "EX-QUIZ-01", "group": "extra_quiz", "query": "出题", "history": "", "expected": ["quiz"]}
    path = _write(tmp_path / "extra.json", {"cases": [one]})
    cases = load_extra(path)
    assert cases[0]["id"] == "E-EX-QUIZ-01"


def test_build_cases_defaults_to_extra_only(tmp_path: Path) -> None:
    extra = _write(tmp_path / "extra.json", {"cases": [{"id": "A", "group": "extra_quiz", "query": "q", "history": "", "expected": ["quiz"]}]})
    single_root = tmp_path / "eval"
    _write(single_root / "教材A" / "qa_set.json", {"textbook": "教材A", "questions": [{"id": "Q1", "question": "什么是外键"}]})
    args = SimpleNamespace(extra=extra, single=single_root, multi=None, include_anchors=False)
    cases = build_cases(args)
    assert [c["group"] for c in cases] == ["extra_quiz"], "默认口径只跑补充集"
    args.multi = _write(tmp_path / "multi.json", {"conversations": [{"turns": [{"id": "T1", "question": "q1", "turn_role": "initial"}]}]})
    args.include_anchors = True
    groups = {c["group"] for c in build_cases(args)}
    assert groups == {"extra_quiz", "anchor_single", "anchor_multi_initial"}


def test_filter_cases_group_whitelist_and_per_group_limit() -> None:
    cases = [
        {"group": "a", "id": 1},
        {"group": "a", "id": 2},
        {"group": "b", "id": 3},
        {"group": "c", "id": 4},
    ]
    assert [c["id"] for c in filter_cases(cases, "a,c", 0)] == [1, 2, 4]
    assert [c["id"] for c in filter_cases(cases, "", 1)] == [1, 3, 4], "空组名全收，按组截断"
    assert filter_cases(cases, "zzz", 0) == []


def _run_row(intent: str, ms: float = 10.0) -> dict:
    return {"intent": intent, "source": "llm", "confidence": 0.8, "ms": ms}


def test_resolve_majority_majority_and_instability() -> None:
    case = {"id": "x", "query": "q", "expected": ["quiz"]}
    row = resolve_majority(case, [_run_row("quiz"), _run_row("explain"), _run_row("quiz")])
    assert row["intent"] == "quiz" and row["unstable"] is True
    stable = resolve_majority(case, [_run_row("quiz", 5.0), _run_row("quiz", 15.0)])
    assert stable["intent"] == "quiz" and stable["unstable"] is False
    assert stable["ms"] == 10.0, "ms 取中位数"


def test_resolve_majority_tie_prefers_first_seen() -> None:
    case = {"id": "x", "query": "q", "expected": ["chat"]}
    row = resolve_majority(case, [_run_row("explain"), _run_row("chat")])
    assert row["intent"] == "explain", "平局取最先出现的预测，结果可复现"
