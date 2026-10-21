"""意图识别相关实体：LLM 结构化输出判定与对外返回结果。"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class _IntentDecision(BaseModel):
    """LLM 结构化输出的意图判定。"""

    intent: str = Field(description="意图枚举值")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0~1")


@dataclass(frozen=True)
class IntentResult:
    """意图识别结果：意图、置信度与来源（keyword/embedding/llm/fallback）。"""

    intent: str
    confidence: float
    source: str
