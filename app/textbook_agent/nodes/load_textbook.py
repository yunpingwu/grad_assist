from pathlib import Path

from langgraph.types import StreamWriter

from app.core import log_node, logger
from app.textbook_agent.state import TextBookState


@log_node
async def load_textbook(state: TextBookState, *, writer: StreamWriter) -> dict:
    """检查教材文件是否存在，判断类型后导向不同节点"""
    writer({"type": "message", "status": "running", "message": "校验教材文件", "progress": 0.05})

    # 获取教材路径
    raw_path = state.get("textbook_path")
    if raw_path:
        textbook_path = Path(raw_path)
    else:
        textbook_path = Path(__file__).resolve().parents[3] / "textbooks" / "pdf"

    # 获取所有支持的教材
    supported = {".pdf", ".doc", ".ppt"}
    files = []
    for file in textbook_path.iterdir():
        if file.suffix in supported:
            files.append(file)

    # 检测教材数量
    if len(files) == 0:
        raise ValueError("没有找到支持的类型的教材")

    state["textbook_path"] = textbook_path
    logger.info(f"load_textbook:成功加载 {len(files)} 个教材")
    writer(
        {
            "type": "message",
            "status": "running",
            "message": f"教材校验通过，共 {len(files)} 个文件",
            "progress": 0.1,
        }
    )
    return state


# 单元测试
if __name__ == "__main__":
    import asyncio

    def writer(chunk):
        print("event:", chunk)

    state: TextBookState = {"textbook_exists": False}
    asyncio.run(load_textbook(state, writer=writer))
