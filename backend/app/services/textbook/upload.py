"""教材上传落盘：每次上传创建独立目录。"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

# 教材根目录（本文件位于 app/services/textbook/ 下，项目根为 parents[3]）。
# 每次上传在根目录下新建独立子目录（pdf-{特征值}），与测试数据 textbooks/pdf 隔离。
TEXTBOOK_ROOT = Path(__file__).resolve().parents[3] / "textbooks"


def _new_textbook_dir() -> Path:
    """在 textbooks/ 下创建本次上传的独立目录（pdf-{时间戳}-{随机}）。

    每次上传都是干净目录，天然规避旧解析产物的幂等误判；
    向量化入库后该目录可整体删除（当前保留，便于验证解析效果）。
    """
    feature = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    textbook_dir = TEXTBOOK_ROOT / f"pdf-{feature}"
    textbook_dir.mkdir(parents=True, exist_ok=False)
    return textbook_dir


async def save_uploads(files: list[UploadFile]) -> tuple[Path, int]:
    """保存上传文件到独立目录，返回 ``(目录, 文件数)``。"""
    textbook_dir = _new_textbook_dir()
    saved = 0
    for file in files:
        with open(textbook_dir / file.filename, "wb") as f:
            f.write(await file.read())
        saved += 1
    return textbook_dir, saved
