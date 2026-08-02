"""
任务状态工具

为摄入等长任务提供 任务ID + 状态/进度/事件 的集中管理，
供上层 FastAPI 通过 SSE 向前端实时推送进度。

使用方式：
1. 上层编排调用 create_task() 创建任务，拿到 task_id 放入 state["task_id"]；
2. 摄入节点内调用 update_task(task_id, ...) 上报状态/进度（同步/异步均可调用）；
3. FastAPI 的 SSE 端点通过 subscribe(task_id) 订阅事件流，逐条下发前端。
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, AsyncGenerator

from app.textbook_agent.core.logger import get_logger

logger = get_logger(__name__)

# 任务状态常量
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

# SSE 心跳间隔（秒），无新事件时保活连接
_HEARTBEAT_INTERVAL = 15.0

# 任务存储：task_id → 任务记录（含事件日志）
_lock = threading.Lock()
_tasks: dict[str, dict] = {}
# 唤醒订阅者的 asyncio 事件（跨线程 set 安全，Python 3.10+）
_events: dict[str, asyncio.Event] = {}


def create_task() -> str:
    """创建任务并返回 task_id。

    Returns:
        新任务的 task_id（uuid hex）。
    """
    task_id = uuid.uuid4().hex
    with _lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "status": STATUS_PENDING,
            "progress": 0.0,
            "message": "",
            "events": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    logger.info(f"任务已创建: {task_id}")
    return task_id


def update_task(
    task_id: str,
    status: str | None = None,
    message: str | None = None,
    progress: float | None = None,
    data: dict | None = None,
) -> None:
    """更新任务状态并追加一条事件（线程安全，同步/异步均可调用）。

    Args:
        task_id: 任务 ID。
        status: 新状态（pending/running/succeeded/failed），None 表示不变。
        message: 状态消息，如 "解析中 3/10"。
        progress: 进度 0~1，None 表示不变。
        data: 附加结构化数据，随事件下发。
    """
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            logger.warning(f"任务不存在，忽略更新: {task_id}")
            return

        if status:
            task["status"] = status
        if progress is not None:
            task["progress"] = round(progress, 4)
        if message:
            task["message"] = message
        task["updated_at"] = time.time()
        task["events"].append({
            "ts": time.time(),
            "status": status or task["status"],
            "message": message or "",
            "progress": task["progress"],
            "data": data or {},
        })

    # 唤醒等待该任务的 SSE 订阅者
    event = _events.get(task_id)
    if event is not None:
        event.set()


def get_task(task_id: str) -> dict | None:
    """获取任务快照。

    Args:
        task_id: 任务 ID。

    Returns:
        任务记录（events 为事件列表副本），任务不存在返回 None。
    """
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return None
        return {k: (list(v) if k == "events" else v) for k, v in task.items()}


async def subscribe(task_id: str) -> AsyncGenerator[dict, None]:
    """订阅任务事件流（SSE 用）。

    - 先下发当前状态快照（event=status）；
    - 之后实时下发新增事件（event=message）；
    - 无新事件时定时下发心跳保活（event=heartbeat）；
    - 任务结束时下发终止事件（event=done）并退出。

    Yields:
        {"event": "status"|"message"|"heartbeat"|"done"|"error", "data": {...}}
    """
    last_index = 0
    while True:
        with _lock:
            task = _tasks.get(task_id)
            if task is None:
                yield {"event": "error", "data": {"message": f"任务不存在: {task_id}"}}
                return
            status = task["status"]
            progress = task["progress"]
            message = task["message"]
            events = list(task["events"])

        # 首帧下发状态快照
        if last_index == 0:
            yield {
                "event": "status",
                "data": {"status": status, "progress": progress, "message": message},
            }

        # 下发增量事件
        while last_index < len(events):
            event = events[last_index]
            last_index += 1
            yield {"event": "message", "data": event}

        # 终态：结束订阅
        if status in (STATUS_SUCCEEDED, STATUS_FAILED):
            yield {"event": "done", "data": {"status": status}}
            return

        # 等待新事件，超时则发心跳
        try:
            await _wait_for_update(task_id)
        except asyncio.TimeoutError:
            yield {"event": "heartbeat", "data": {}}


async def _wait_for_update(task_id: str) -> None:
    """等待任务出现新事件，超时抛 TimeoutError。"""
    event = _events.get(task_id)
    if event is None:
        event = asyncio.Event()
        _events[task_id] = event
    event.clear()
    await asyncio.wait_for(event.wait(), timeout=_HEARTBEAT_INTERVAL)


# 单元测试
if __name__ == '__main__':
    import asyncio


    async def demo() -> None:
        task_id = create_task()
        update_task(task_id, status=STATUS_RUNNING, message="开始解析", progress=0.1)

        async def emit() -> None:
            await asyncio.sleep(0.2)
            update_task(task_id, message="解析中 1/3", progress=0.4)
            await asyncio.sleep(0.2)
            update_task(task_id, status=STATUS_SUCCEEDED, message="完成", progress=1.0)

        asyncio.get_running_loop().create_task(emit())

        async for event in subscribe(task_id):
            print(event)


    asyncio.run(demo())
