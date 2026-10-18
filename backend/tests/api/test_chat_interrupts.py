"""写盘确认中断的恢复计数：决定数必须与挂起的工具调用数一一对应。

真机 2026-10-15：模型一轮内发出两个 edit_file（同一个中断里挂起两起写盘），而恢复时按
graph task 数（恒为 1）发决定 → HITL 中间件抛 "Number of human decisions (1) does not
match number of hanging tool calls (2)"，线程卡死，下一轮模型调用跟着 400。
"""

from types import SimpleNamespace

from app.services import study as svc


def _snapshot(*values: dict) -> SimpleNamespace:
    """造一个带 interrupts 的快照：一个 task 下挂若干中断（与真机 snapshot 同形）。"""
    return SimpleNamespace(
        tasks=[SimpleNamespace(interrupts=[SimpleNamespace(value=v) for v in values])]
    )


def test_counts_every_hanging_tool_call_in_one_interrupt() -> None:
    """一次中断里挂两起写盘 → 决定数必须是 2（数 task 数是 1，会让中间件直接抛错）。"""
    snap = _snapshot({"action_requests": [{}, {}], "review_configs": [{}, {}]})
    assert svc._hanging_write_count(snap) == 2


def test_falls_back_to_one_without_action_requests() -> None:
    """澄清中断（无 action_requests）与无快照都兜底 1，不因计数把恢复路径打断。"""
    assert svc._hanging_write_count(_snapshot({"type": "clarify", "questions": []})) == 1
    assert svc._hanging_write_count(SimpleNamespace(tasks=[])) == 1
