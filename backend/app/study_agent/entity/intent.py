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
    """意图识别结果。

    Attributes:
        intent: 意图枚举值（explain/generate/quiz/plan/chat/unclear）。
        confidence: 置信度 0~1。
        source: 判定来源（keyword/embedding/llm/fallback），便于观测与回归。
    """

    intent: str
    confidence: float
    source: str
