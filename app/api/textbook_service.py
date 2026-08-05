import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from app.core import logger
from app.textbook_agent.graph import build_graph
from app.textbook_agent.state import TextBookState
from app.utils import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    create_task,
    get_task,
    subscribe,
    update_task,
)

# 教材根目录（本文件位于 app/api/ 下，项目根为 parents[2]）。
# 每次上传在根目录下新建独立子目录（pdf-{特征值}），与测试数据 textbooks/pdf 隔离。
TEXTBOOK_ROOT = Path(__file__).resolve().parents[2] / "textbooks"


def _new_textbook_dir() -> Path:
    """在 textbooks/ 下创建本次上传的独立目录（pdf-{时间戳}-{随机}）。

    每次上传都是干净目录，天然规避旧解析产物的幂等误判；
    向量化入库后该目录可整体删除（当前保留，便于验证解析效果）。
    """
    feature = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    textbook_dir = TEXTBOOK_ROOT / f"pdf-{feature}"
    textbook_dir.mkdir(parents=True, exist_ok=False)
    return textbook_dir

# 创建 FastAPI 实例
app = FastAPI(
    title="Textbook Agent",
    description="一个将教材向量化后存储入向量数据库的langgraph流程",
    version="0.1.0",
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_pipeline(graph, state: TextBookState, task_id: str) -> None:
    """后台线程执行摄入流水线，结束后更新任务终态。

    Args:
        graph: 编译后的 LangGraph 图。
        state: 流水线初始状态（含 task_id）。
        task_id: 任务 ID，用于终态更新。
    """
    try:
        final_state = graph.invoke(state)
        if final_state.get("ingestion_done"):
            update_task(task_id=task_id, status=STATUS_SUCCEEDED, message="解析完成", progress=1.0)
        else:
            update_task(task_id=task_id, status=STATUS_FAILED, message="解析失败：未完成入库", progress=1.0)
    except Exception as exc:
        logger.error(f"任务 {task_id} 执行异常: {exc}")
        update_task(task_id=task_id, status=STATUS_FAILED, message=f"解析失败: {exc}", progress=1.0)


async def _sse_stream(task_id: str):
    """将 subscribe 的事件 dict 格式化为 SSE 文本流。"""
    async for event in subscribe(task_id):
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


@app.post("/upload", summary="上传教材", description="支持批量上传，接受格式当前为pdf")
async def upload_textbooks(files: List[UploadFile] = File(...)):
    """上传教材：每次上传在 textbooks/ 下创建独立目录 pdf-{特征值}/ 并保存文件。

    Args:
        files: 批量上传的教材 PDF 文件。

    Returns:
        textbook_path: 本次上传的独立目录，供 /resolve 使用。
    """
    textbook_dir = _new_textbook_dir()
    # 保存文件
    saved_files = []
    for file in files:
        with open(textbook_dir / file.filename, "wb") as f:
            f.write(await file.read())
        saved_files.append(file.filename)

    return JSONResponse(
        {
            "textbook_path": str(textbook_dir),
            "message": f"上传成功：{len(saved_files)} 个文件",
        }
    )


@app.post("/resolve", summary="解析教材", description="解析教材，将教材向量化并保存入向量数据库")
async def resolve_textbooks(textbook_path: str):
    """解析教材：在 /upload 返回的独立目录下执行摄入流水线，创建任务后立即返回 task_id。

    Args:
        textbook_path: /upload 返回的教材目录路径（必填）。

    Returns:
        任务 ID，前端可凭此轮询 /status 或订阅 /events。
    """
    if not textbook_path:
        raise HTTPException(status_code=400, detail="缺少 textbook_path，请先调用 /upload 上传教材")
    textbook_dir = Path(textbook_path)
    if not textbook_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"教材目录不存在: {textbook_path}")

    # 创建任务
    task_id = create_task()
    update_task(task_id=task_id, status=STATUS_RUNNING, message="开始解析", progress=0.1)
    # 构建流水线，放入 state["task_id"] 供节点上报进度
    state: TextBookState = {
        "textbook_path": textbook_path,
        "task_id": task_id,
    }
    graph = build_graph()
    # 后台线程执行同步流水线，避免阻塞事件循环
    threading.Thread(target=_run_pipeline, args=(graph, state, task_id), daemon=True).start()

    return JSONResponse({"task_id": task_id, "message": "任务已创建"})


@app.get("/events", summary="订阅任务进度", description="SSE 实时推送任务进度，前端按 task_id 订阅")
async def task_events(task_id: str):
    """订阅任务进度事件流（SSE）。

    Args:
        task_id: 任务 ID。

    Returns:
        text/event-stream 流：status 快照 → message 增量 → heartbeat 保活 → done 终态。
    """
    return StreamingResponse(
        _sse_stream(task_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status", summary="获取任务状态", description="获取任务状态")
async def get_task_status(task_id: str):
    """获取任务状态"""
    current_task = get_task(task_id)
    return JSONResponse(current_task)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
