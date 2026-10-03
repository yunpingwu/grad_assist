"""test_resolve_reconnect.py — /resolve 断线重连语义的单元测试（假图，不触达 Mongo）。"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from langgraph.types import Command
from pypdf import PdfWriter

from app.api import textbook_service as svc
from app.textbook_flow.cache import toc_cache

CANDIDATES = [{"textbook_name": "机器学习", "offset": 10}]


class _Interrupt:
    def __init__(self, value: dict) -> None:
        self.value = value


class _Task:
    def __init__(self, interrupts: list[_Interrupt]) -> None:
        self.interrupts = interrupts


class _Snap:
    def __init__(self, values: dict, tasks: list[_Task], nxt) -> None:
        self.values = values
        self.tasks = tasks
        self.next = nxt


def _offset_snap() -> _Snap:
    """停在 offset_review 中断的快照。"""
    snap = _Snap({"textbook_path": "/tmp/x", "ingestion_done": False}, [], None)
    snap.tasks = [_Task([_Interrupt({"type": "offset_review", "offsets": CANDIDATES})])]
    snap.next = ["split"]
    return snap


class _FakeGraph:
    """记录 astream 入参并按脚本依次吐出快照与事件。"""

    def __init__(self, snapshots: list[_Snap], events: list[dict]) -> None:
        self.snapshots = snapshots
        self.events = events
        self.inputs: list = []

    async def aget_state(self, config) -> _Snap:
        return self.snapshots.pop(0)

    async def astream(self, run_input, *, config, stream_mode, durability):
        self.inputs.append(run_input)
        for ev in self.events:
            yield ev


async def _frames(resp) -> list[dict]:
    out = []
    async for chunk in resp.body_iterator:
        for line in chunk.split("\n\n"):
            line = line.strip()
            if line.startswith("data:"):
                out.append(json.loads(line[len("data:") :].strip()))
    return out


def test_reconnect_at_offset_without_offsets_replays_ask_offset(monkeypatch) -> None:
    """停在偏移确认、不带 offsets 的重连：重放 ask_offset 供前端重弹面板，不发 done。"""
    # 前快照停在偏移中断；重放后仍停在偏移中断
    fake = _FakeGraph([_offset_snap(), _offset_snap()], [{"type": "ask_offset", "offsets": CANDIDATES}])
    monkeypatch.setattr(svc, "textbook_graph", fake)

    async def run():
        # 直调端点函数不走 FastAPI 参数解析，Body 默认值需显式传 None
        resp = await svc.resolve_textbooks(task_id="t1", user_id="u1", offsets=None)
        return await _frames(resp)

    frames = asyncio.run(run())

    assert fake.inputs == [None]  # 不带 Command(resume)，astream(None) 重放中断节点
    assert frames[0]["type"] == "info" and frames[0]["resumed"] is True
    assert frames[1]["type"] == "ask_offset"
    assert frames[1]["offsets"] == CANDIDATES
    assert all(f["type"] != "done" for f in frames)


def test_reconnect_at_offset_with_offsets_resumes(monkeypatch) -> None:
    """带 offsets 的续跑：以 Command(resume=...) 推进图，完成后发 done。"""
    done_snap = _Snap({"textbook_path": "/tmp/x", "ingestion_done": True}, [], None)
    fake = _FakeGraph([_offset_snap(), done_snap], [{"type": "message", "message": "切割完成"}])
    monkeypatch.setattr(svc, "textbook_graph", fake)

    async def run():
        resp = await svc.resolve_textbooks(task_id="t1", user_id="u1", offsets=CANDIDATES)
        return await _frames(resp)

    frames = asyncio.run(run())

    assert isinstance(fake.inputs[0], Command)
    assert fake.inputs[0].resume == {"offsets": CANDIDATES}
    assert frames[-1]["type"] == "done"


def test_reconnect_resumable_without_path_ok(monkeypatch) -> None:
    """停在节点边界（非偏移中断）的重连：不带 textbook_path 也能续跑。"""
    snap = _Snap({"textbook_path": "/tmp/x", "ingestion_done": False}, [], ["parse_to_md"])
    done_snap = _Snap({"textbook_path": "/tmp/x", "ingestion_done": True}, [], None)
    fake = _FakeGraph([snap, done_snap], [{"type": "message", "message": "继续解析"}])
    monkeypatch.setattr(svc, "textbook_graph", fake)

    async def run():
        resp = await svc.resolve_textbooks(task_id="t1", user_id="u1")
        return await _frames(resp)

    frames = asyncio.run(run())

    assert fake.inputs == [None]
    assert frames[0]["type"] == "info" and frames[0]["resumed"] is True
    assert frames[-1]["type"] == "done"


def test_unknown_task_without_path_still_400(monkeypatch) -> None:
    """无 task 状态又缺 textbook_path：维持原 400 校验。"""
    empty = _Snap({}, [], None)
    fake = _FakeGraph([empty], [])
    monkeypatch.setattr(svc, "textbook_graph", fake)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(svc.resolve_textbooks(task_id="t1", user_id="u1"))
    assert exc.value.status_code == 400


# ---------- 目录树缓存分发：info 帧带 tocs、终态即删 ----------

_TOC_MD = """# 机器学习

## 目录

第1章 绪论 …… 1
1.1 绪论 …… 1
第2章 模型评估与选择 15
"""


def _textbook_dir_with_toc(root: Path) -> None:
    """构造含 mineru_toc 产物的教材目录（info 帧懒重建的数据源）。"""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(root / "机器学习.pdf", "wb") as fh:
        writer.write(fh)
    toc_dir = root / "mineru_toc" / "机器学习"
    toc_dir.mkdir(parents=True)
    (toc_dir / "full.md").write_text(_TOC_MD, encoding="utf-8")


def test_reconnect_info_frame_carries_tocs_and_done_pops_cache(monkeypatch, tmp_path) -> None:
    """断线重进：info 帧凭 task_id 从缓存/磁盘带回目录树；done 发出前缓存删除。"""
    _textbook_dir_with_toc(tmp_path)
    task_id = "toc-task-1"
    offset_snap = _Snap({"textbook_path": str(tmp_path), "ingestion_done": False}, [], ["split"])
    offset_snap.tasks = [_Task([_Interrupt({"type": "offset_review", "offsets": CANDIDATES})])]
    # 续跑至完成的流：带 offsets 提交后一路到 done
    done_snap = _Snap({"textbook_path": str(tmp_path), "ingestion_done": True}, [], None)
    fake = _FakeGraph([offset_snap, done_snap], [{"type": "message", "message": "入库完成"}])
    monkeypatch.setattr(svc, "textbook_graph", fake)

    async def run():
        resp = await svc.resolve_textbooks(task_id=task_id, user_id="u1", offsets=CANDIDATES)
        return await _frames(resp)

    frames = asyncio.run(run())

    info = frames[0]
    assert info["type"] == "info"
    assert [t["textbook"] for t in info["tocs"]] == ["机器学习"]
    assert info["tocs"][0]["chapters"][0]["sections"][0]["num"] == "1.1"
    assert frames[-1]["type"] == "done"
    # 完成即删：终态后缓存不再持有该任务
    assert toc_cache._toc_cache.get(task_id) is None


def test_first_run_info_without_toc_keeps_sse_path(monkeypatch, tmp_path) -> None:
    """首次解析：目录尚未产出，info 不带 tocs 键；树由 split_contents 的实时 toc 事件到达。"""
    fake = _FakeGraph([_Snap({}, [], None), _Snap({"ingestion_done": True}, [], None)], [])
    monkeypatch.setattr(svc, "textbook_graph", fake)

    async def run():
        resp = await svc.resolve_textbooks(textbook_path=str(tmp_path), user_id="u1", offsets=None)
        return await _frames(resp)

    frames = asyncio.run(run())
    assert frames[0]["type"] == "info" and "tocs" not in frames[0]
