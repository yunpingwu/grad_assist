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
    """加载提示词文件 ``app/prompts/{name}.prompt`` 的内容（带缓存）。

    提示词文件只在开发期变动，缓存后高频节点无需重复读磁盘。

    Args:
        name: 提示词名称，不带后缀，如 ``"rewrite_query"``。
        prompts_dir: 提示词根目录，缺省用 ``app/prompts/``（便于测试注入）。

    Returns:
        提示词文件内容（去除首尾空白）。

    Raises:
        FileNotFoundError: 找不到对应的提示词文件。
    """
    base = prompts_dir or PROMPTS_DIR
    path = base / f"{name}{PROMPT_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {path}")
    return path.read_text(encoding="utf-8").strip()
