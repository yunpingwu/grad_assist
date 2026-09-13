"""意图识别：三档漏斗（关键词 → embedding 语义路由 → LLM 兜底），query 理解前置步骤。

与问题重写同属「query 理解」：在 agent 工具循环之前（pre_model_hook）每个 query 只
执行一次，产出带置信度的意图（IntentResult），供 system prompt 动态选择行为/输出块。

三档设计（按延迟/成本递增，命中即停）：
1. 关键词快车道：只对「行为类」强信号词做保守判定，歧义/未命中不硬判；
2. embedding 语义路由：query 与各意图描述向量比对（复用 BGE-M3），高相似直接定、
   低相似判「领域外」，中等置信交下一档；
3. LLM 结构化兜底：最终判定，白名单校验防越界；失败则回退安全默认「讲解」。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.clients.llm import get_llm_client
from app.core import load_prompt, logger
from app.utils import agenerate_embeddings

# 意图枚举顺序（embedding 相似度结果按此下标对齐，保序）
INTENT_ORDER = ("explain", "generate", "quiz", "plan", "chat")

# 各意图的语义描述：embedding 路由把 query 与这些描述做相似度比对
INTENT_DESCRIPTIONS: dict[str, str] = {
    "explain": "讲解概念、答疑、解释原理、归纳对比、总结知识点",
    "generate": "生成学习资料、整理成文档、考点总结、复习笔记、大纲、存成文件",
    "quiz": "出题、自测题、练习题、习题集、测试卷、选择题、简答题",
    "plan": "复习计划、学习计划、学习路径、如何安排学习、备考规划",
    "chat": "寒暄、问候、感谢、道别、日常闲聊、常识性短答",
}

# 低置信度兜底：无法判定的意图（由消费端回退通用行为 + 引导澄清）
UNCLEAR = "unclear"
VALID_INTENTS = frozenset({*INTENT_ORDER, UNCLEAR})

# embedding 语义路由的相似度阈值（BGE-M3 稠密向量已 L2 归一化，点积即 cosine）。
# 经验值，建议用真实 query 日志按 precision/recall 校准。
_EMBED_HIGH = 0.5  # 达到即高置信直接定
_EMBED_LOW = 0.3  # 低于即判「领域外/无法判」；介于两者之间交 LLM 兜底

# 关键词快车道的置信度（规则命中视为高置信，留少量余量）
_KEYWORD_CONF = 0.95

# 关键词规则：只覆盖「行为类」强信号词（generate/quiz/plan）。
# 刻意不做 explain/chat——寒暄词（如「谢谢」）常混在真实问题里易误判，
# 这两类交给 embedding/LLM 用语义判定更稳。
_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("generate", ("生成", "整理成", "做成", "考点总结", "复习资料", "学习资料", "大纲", "存成文件", "写成文档")),
    ("quiz", ("出题", "习题", "练习题", "自测题", "测试题", "试卷", "卷子", "题库", "选择题", "简答题")),
    ("plan", ("复习计划", "学习计划", "学习路径", "备考", "怎么学", "如何学", "学习安排")),
)


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


def _cosine(a: list[float], b: list[float]) -> float:
    """向量相似度：dense 已 L2 归一化，点积即余弦相似度。"""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _classify_by_keyword(text: str) -> IntentResult | None:
    """关键词快车道：仅当命中集中在单一意图时直接定，否则返回 None 交下一档。

    Args:
        text: 用户 query 原文。

    Returns:
        唯一命中且高置信时返回结果；无命中或多个意图并列（歧义）返回 None。
    """
    # 布尔命中：只关心「是否命中任一强信号词」，避免包含关系（如「练习题」同时算
    # 「练习题」「习题」）造成的计数虚高
    scores = {intent: 0 for intent in INTENT_ORDER}
    for intent, keywords in _KEYWORD_RULES:
        scores[intent] = int(any(kw in text for kw in keywords))

    best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return None  # 未命中任何强信号词
    if list(scores.values()).count(best_score) > 1:
        return None  # 多意图并列（如「生成练习题」），歧义交下一档
    return IntentResult(intent=best_intent, confidence=_KEYWORD_CONF, source="keyword")


async def _classify_by_embedding(text: str) -> IntentResult | None:
    """embedding 语义路由：query 与各意图描述向量比对（复用 BGE-M3）。

    Args:
        text: 用户 query 原文。

    Returns:
        高置信或「领域外」时返回结果；中等置信返回 None 交 LLM 兜底。
    """
    descriptions = [INTENT_DESCRIPTIONS[i] for i in INTENT_ORDER]
    emb = await agenerate_embeddings([text, *descriptions])
    query_vec = emb["dense"][0]
    sims = [_cosine(query_vec, emb["dense"][1 + i]) for i in range(len(INTENT_ORDER))]
    best_idx = max(range(len(sims)), key=sims.__getitem__)
    best_sim = float(sims[best_idx])

    if best_sim >= _EMBED_HIGH:
        return IntentResult(INTENT_ORDER[best_idx], best_sim, "embedding")
    if best_sim < _EMBED_LOW:
        return IntentResult(UNCLEAR, best_sim, "embedding")  # 与所有意图都低相似
    return None  # 中等置信，交 LLM 兜底


class _IntentDecision(BaseModel):
    """LLM 结构化输出的意图判定。"""

    intent: str = Field(description="意图枚举值")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0~1")


async def _classify_by_llm(text: str) -> IntentResult:
    """LLM 结构化分类兜底（对 embedding 中等置信的查询做最终判定）。"""
    llm = get_llm_client().with_structured_output(_IntentDecision)
    prompt = PromptTemplate.from_template(load_prompt("intent_classify"))
    decision = await llm.ainvoke(prompt.format(query=text))
    # 白名单校验：模型可能输出枚举外的值，收敛到 unclear（防注入/越界）
    intent = decision.intent if decision.intent in VALID_INTENTS else UNCLEAR
    return IntentResult(
        intent=intent,
        confidence=max(0.0, min(1.0, decision.confidence)),
        source="llm",
    )


async def detect_intent(text: str) -> IntentResult:
    """三档漏斗识别用户意图（query 理解前置步骤，每个 query 只调一次）。

    Args:
        text: 用户原始 query（原文，非重写后——意图要看真实诉求，重写版会丢失意图信息）。

    Returns:
        带置信度与来源的意图结果；source=fallback 表示降级，intent 为安全兜底。
    """
    query = (text or "").strip()
    if not query:
        return IntentResult(UNCLEAR, 0.0, "fallback")

    # 1) 关键词快车道
    if (result := _classify_by_keyword(query)) is not None:
        return result

    # 2) embedding 语义路由（失败不阻断，降级 LLM）
    try:
        if (result := await _classify_by_embedding(query)) is not None:
            return result
    except Exception as exc:
        logger.warning(f"意图识别 embedding 路由失败，降级 LLM: {exc}")

    # 3) LLM 兜底（再失败则回退安全默认「讲解」，不阻断主流程）
    try:
        return await _classify_by_llm(query)
    except Exception as exc:
        logger.warning(f"意图识别 LLM 兜底失败，回退 explain: {exc}")
        return IntentResult("explain", 0.0, "fallback")


# 冒烟测试：桩掉 embedding 与 LLM，只验证三档编排与关键词层
if __name__ == "__main__":
    import asyncio

    async def _fake_embeddings(texts: list[str]) -> dict:
        # query=[1.0]、其余描述=[0.4] → 每个 cosine=0.4，落在中等区间（交 LLM 兜底）
        n = len(texts)
        return {"dense": [[1.0] if i == 0 else [0.4] for i in range(n)], "sparse": [{0: 1.0} for _ in range(n)]}

    async def _fake_llm(text: str) -> IntentResult:
        return IntentResult("generate", 0.8, "llm")

    # 覆盖模块级绑定（与 search.py 冒烟测试同一模式），只验证编排
    agenerate_embeddings = _fake_embeddings
    _classify_by_llm = _fake_llm

    async def _run() -> None:
        # 关键词唯一命中 → 直接定 quiz
        r1 = _classify_by_keyword("帮我出几道选择题")
        assert r1 is not None and r1.intent == "quiz" and r1.source == "keyword", r1

        # 关键词多意图并列（生成 + 练习题）→ 歧义，交下一档
        r2 = _classify_by_keyword("帮我生成第三章的练习题")
        assert r2 is None, r2

        # 空输入 → fallback unclear
        r3 = await detect_intent("")
        assert r3.intent == "unclear" and r3.source == "fallback", r3

        # 关键词未命中 → embedding 中等置信 → 交 LLM 兜底
        r4 = await detect_intent("随便讲讲这本书")
        assert r4.intent == "generate" and r4.source == "llm", r4

        print("intent 三档漏斗冒烟测试通过")

    asyncio.run(_run())