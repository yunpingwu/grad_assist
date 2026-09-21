"""澄清问题工具：任务描述不明确、缺关键信息时，由模型向用户反问（结构化），人工作答后回灌继续执行。

设计要点：
- 一次工具调用 = 一轮所有澄清问题（最多 ``MAX_CLARIFY_QUESTIONS`` 个），规避并行 interrupt；
- 工具入参（ClarifyQuestion）即模型产物：选项型(choice)带建议选项+对应解决方法，开放型(open)让用户自定义；
- 工具返回值（list[ClarifyAnswer]）即模型续跑拿到的结果：每个问题 {id, answer}（选中的 method 或自定义文本）；
- 人工作答经 ``Command(resume={"answers": [...]})`` 续跑；resume 数据不合法时在工具内再次 interrupt
  提示重填，直到通过；服务端可在新问题撞上澄清挂起时发 ``Command(resume={"cancelled": True})`` 自动取消。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

# 单次澄清请求的问题数上限（模型超出直接报错，让它自行收敛重试）
MAX_CLARIFY_QUESTIONS = 3

# 取消澄清的 resume 标记：前端/服务端在"新问题撞上澄清挂起"时发送 → 工具返回空答案，模型自行兜底
CANCEL_MARKER = "cancelled"


class ClarifyOption(BaseModel):
    """澄清问题的建议选项：label 展示给用户，method 为选中后回灌模型的解决方法正文。"""

    label: str
    method: str


class ClarifyQuestion(BaseModel):
    """模型产出的澄清问题：选项型带建议选项，开放型允许自定义回答。"""

    id: str = Field(description="唯一标识，如 q1")
    question: str = Field(description="澄清问题的文本")
    type: Literal["choice", "open"] = "choice"
    options: list[ClarifyOption] = Field(default_factory=list, description="choice 模式：建议选项（含对应解决方法）")
    hint: str | None = Field(default=None, description="open 模式：给用户的提示/示例")


class ClarifyAnswer(BaseModel):
    """工具返回值：人工确认后每个问题的最终答案。"""

    id: str
    answer: str


def _normalize_answers(resume: Any, questions: list[ClarifyQuestion]) -> list[ClarifyAnswer] | str:
    """校验并归一化 resume 数据；合法返回答案列表，非法返回可在前端展示的错误信息。

    Args:
        resume: Command(resume=...) 传入的数据（服务端已按请求体重构）。
        questions: 本工具当前这轮提出的问题列表。

    Returns:
        list[ClarifyAnswer] 表示校验通过；str 表示错误信息（触发再次 interrupt 提示重填）。
    """
    if not isinstance(resume, dict):
        return "澄清回应格式无效：需为 {'answers': [{'id': ..., 'answer': ...}]}"
    # 取消标记：用户放弃澄清（服务端在收到新问题时发出）
    if resume.get(CANCEL_MARKER):
        return [ClarifyAnswer(id=q.id, answer="") for q in questions]
    raw = resume.get("answers")
    if not isinstance(raw, list):
        return "澄清回应缺少 answers 列表"
    raw_by_id: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        answer = item.get("answer")
        raw_by_id[str(item["id"])] = "" if answer is None else str(answer)
    if not raw_by_id:
        return "澄清回应缺少有效的 answer 条目"
    missing = [q.id for q in questions if q.id not in raw_by_id]
    if missing:
        return f"以下问题未作答，请补充：{', '.join(missing)}"
    return [ClarifyAnswer(id=q.id, answer=raw_by_id[q.id]) for q in questions]


@tool
def ask_clarification(questions: list[ClarifyQuestion]) -> list[ClarifyAnswer]:
    """任务描述不明确、缺少决定产出方向的关键信息时，向用户提出澄清问题，等人工回答后继续。

    用法：
    - 一次调用连问 1-3 个问题，一个问题不需澄清就别问；能合理默认的就别问；
    - 每个问题给 2-3 个建议选项（label 展示、method 为选中后的对应做法），或 type=open 让用户自定义回答；
    - 需要澄清时只调用本工具这一个，不要与其它工具并行。

    Args:
        questions: 澄清问题列表，1-3 个。

    Returns:
        人工确认后每个问题的最终答案（选中的 method 或自定义文本），按 {id, answer} 返回供继续执行原任务。
    """
    if not 1 <= len(questions) <= MAX_CLARIFY_QUESTIONS:
        raise ValueError(f"一次最多提出 {MAX_CLARIFY_QUESTIONS} 个澄清问题，请精简后再调用")

    # 输入归一化：choice 无选项时降级为 open，避免前端空卡片
    normalized: list[ClarifyQuestion] = []
    for q in questions:
        if q.type == "choice" and not q.options:
            q = q.model_copy(update={"type": "open"})
        normalized.append(q)

    payload: dict[str, Any] = {
        "type": "clarify",
        "questions": [q.model_dump(mode="json") for q in normalized],
        "max_questions": MAX_CLARIFY_QUESTIONS,
        "error": "",
    }

    answers: list[ClarifyAnswer] | None = None
    while answers is None:
        resume = interrupt(payload)
        checked = _normalize_answers(resume, normalized)
        if isinstance(checked, str):
            payload["error"] = checked  # 携带错误再次 interrupt，前端提示重填
            continue
        answers = checked
    return answers
