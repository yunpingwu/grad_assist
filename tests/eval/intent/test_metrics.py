"""intent.metrics 纯函数单测：正确判定、分组/逐类统计、弃权率与时延分位数。"""

from __future__ import annotations

from app.eval.intent import metrics as im


def _row(gold: list[str], intent: str, *, group: str = "g", source: str = "llm", ms: float | None = None) -> dict:
    return {"expected": gold, "intent": intent, "group": group, "source": source, "confidence": 0.9, "ms": ms}


def test_is_correct_multi_acceptable() -> None:
    assert im.is_correct(_row(["explain", "generate"], "generate"))
    assert not im.is_correct(_row(["explain"], "generate"))


def test_accuracy_empty_and_basic() -> None:
    assert im.accuracy([]) == 0.0
    rows = [_row(["quiz"], "quiz"), _row(["quiz"], "explain"), _row(["chat"], "chat")]
    assert abs(im.accuracy(rows) - 2 / 3) < 1e-9


def test_confusion_matrix_uses_first_gold_and_keeps_order() -> None:
    rows = [
        _row(["explain", "generate"], "generate"),
        _row(["explain"], "unclear"),
        _row(["quiz"], "quiz"),
    ]
    matrix = im.confusion_matrix(rows)
    assert matrix["explain"] == {"generate": 1, "unclear": 1}  # 双值行记第一金标
    assert matrix["quiz"] == {"quiz": 1}
    assert list(matrix) == ["explain", "quiz"], "输出按 LABELS 顺序"


def test_group_accuracy_counts() -> None:
    rows = [
        _row(["quiz"], "quiz", group="a"),
        _row(["quiz"], "explain", group="a"),
        _row(["chat"], "chat", group="b"),
    ]
    out = im.group_accuracy(rows, "group")
    assert out["a"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert out["b"]["accuracy"] == 1.0


def test_per_class_stats_boundary_row_counts_in_both_golds() -> None:
    rows = [
        _row(["explain", "generate"], "generate"),
        _row(["explain"], "explain"),
    ]
    stats = im.per_class_stats(rows)
    # explain 金标 2 行只中 1 → recall 0.5；generate 金标 1 行中 1 → precision 1.0
    assert stats["explain"]["gold_n"] == 2 and stats["explain"]["recall"] == 0.5
    assert stats["generate"]["gold_n"] == 1 and stats["generate"]["precision"] == 1.0


def test_abstention_rate_only_counts_wrong_unclear() -> None:
    rows = [
        _row(["explain"], "unclear"),  # 错误弃权
        _row(["unclear"], "unclear"),  # 金标就是 unclear，不计入分母也不算弃权
        _row(["explain"], "explain"),
    ]
    assert abs(im.abstention_rate(rows) - 1 / 2) < 1e-9


def test_latency_percentiles_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    out = im.latency_percentiles(values)
    assert out == {"p50": 30.0, "p95": 50.0}
    assert im.latency_percentiles([]) == {"p50": 0.0, "p95": 0.0}


def test_evaluate_aggregates_all_sections() -> None:
    rows = [_row(["quiz"], "quiz", group="a", source="keyword", ms=5.0), _row(["chat"], "explain", group="b", source="llm", ms=900.0)]
    summary = im.evaluate(rows)
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert set(summary["by_group"]) == {"a", "b"}
    assert set(summary["by_source"]) == {"keyword", "llm"}
    assert summary["confusion"]["chat"] == {"explain": 1}
    assert summary["latency_ms"]["p50"] == 5.0
