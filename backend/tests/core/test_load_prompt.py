"""load_prompt 提示词加载函数测试（tmp_path 隔离，不触碰真实 prompts 目录）。"""

import pytest

from app.core.load_prompt import load_prompt


def test_load_prompt_hit(tmp_path) -> None:
    load_prompt.cache_clear()
    (tmp_path / "test.prompt").write_text("测试提示词", encoding="utf-8")
    assert load_prompt("test", prompts_dir=tmp_path) == "测试提示词"


def test_load_prompt_missing(tmp_path) -> None:
    load_prompt.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_prompt("missing", prompts_dir=tmp_path)
