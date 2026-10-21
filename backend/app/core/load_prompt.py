"""加载 prompts 目录下的提示词文件。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# prompts 根目录：app/prompts/
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

# 提示词文件统一后缀
PROMPT_SUFFIX = ".prompt"


@lru_cache(maxsize=128)
def load_prompt(name: str, prompts_dir: Path | None = None) -> str:
    """读取 ``app/prompts/{name}.prompt``；带 lru_cache，开发期改完需重启进程。"""
    base = prompts_dir or PROMPTS_DIR
    path = base / f"{name}{PROMPT_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {path}")
    return path.read_text(encoding="utf-8").strip()
