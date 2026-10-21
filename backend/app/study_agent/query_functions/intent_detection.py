"""意图识别：关键词 → embedding 例句池 → LLM 三级漏斗，每个 query 只执行一次。

按延迟/成本递增、命中即停；历史只参与第三档，用于多轮指代与省略消歧。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AnyMessage
from langchain_core.prompts import PromptTemplate

from app.clients.llm import bind_structured_output, get_llm_client
from app.core import load_prompt, logger
from app.ml import agenerate_embeddings
from app.study_agent.entity.intent import IntentResult, _IntentDecision
from app.study_agent.message_sources import is_summary_message
from app.trace import astage

# 兜底层历史段的空值占位：显式标注首轮，避免模型把空白当成「问题被截断」
_NO_HISTORY = "（无历史问题，本次为首次提问）"

# 意图枚举顺序（embedding 相似度结果按此下标对齐，保序）
INTENT_ORDER = ("explain", "generate", "quiz", "plan", "chat")

# 低置信度兜底：无法判定的意图。消费端只把它当作「保守回退通用行为块」（见 graph 的
# _INTENT_BLOCKS 白名单），是否向用户反问由模型在 ReAct 循环里自行决定（ask_clarification）——
# 路由层的「判不准」不等于用户需要被澄清，两者刻意不打通。
UNCLEAR = "unclear"
VALID_INTENTS = frozenset({*INTENT_ORDER, UNCLEAR})

# 各意图的例句池：query 与池内例句算 cosine、取 top-3 均值作为该意图得分。
# 池里不放 unclear（拒识靠 _EMBED_LOW），避开关键词已拦截的措辞，且不得与评测集重复。
INTENT_EXAMPLES: dict[str, tuple[str, ...]] = {
    "explain": (
        "指针到底是什么，它到底有什么用",
        "为什么浮点数判断相等不能用等号",
        "规范和范式这两个词我总搞混",
        "这段代码为什么会报段错误",
        "快排的划分思想给我捋一捋",
        "没看懂，能不能说人话版",
    ),
    "generate": (
        "把这章的重点公式收拢到一个文件里",
        "最近作业里的差错在哪，汇总一下给我",
        "这一讲的重点给我弄成一页纸",
        "把刚才讲的东西存成md发我",
        "帮我把各章难点汇总成对照表",
    ),
    "quiz": (
        "看看我是真懂了还是假懂了",
        "给我来几道填空",
        "下午就考试了先来两道手感一下",
        "面我几个链表的问题",
        "让我自己动手敲几道",
    ),
    "plan": (
        "接下来二十天的复习帮我拉开",
        "数学和408并行来得及吗",
        "给我整个每周学习周表",
        "零基础先啃哪本",
        "每天通勤路上该看点什么",
    ),
    "chat": (
        "嘿，在忙吗",
        "烦死了不想学了",
        "你是真人还是机器人",
        "撤了，明天再聊",
        "中午吃啥好",
    ),
}

# 关键词规则：只覆盖「行为类」强信号词（generate/quiz/plan）。
# 刻意不做 explain/chat——寒暄词（如「谢谢」）常混在真实问题里易误判，
# 这两类交给 embedding/LLM 用语义判定更稳。
_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("generate", ("生成", "整理成", "做成", "考点总结", "复习资料", "学习资料", "大纲", "存成文件", "写成文档")),
    ("quiz", ("出题", "习题", "练习题", "自测题", "测试题", "试卷", "卷子", "题库", "选择题", "简答题")),
    ("plan", ("复习计划", "学习计划", "学习路径", "备考", "怎么学", "如何学", "学习安排")),
)


# embedding 语义路由的直判判据（例句池 top-3 均值；BGE-M3 稠密向量已 L2 归一化，点积即 cosine）。
_EMBED_HIGH = 0.55  # top1 达到且与次冠池差 ≥ _EMBED_MARGIN 才高置信直判
_EMBED_MARGIN = 0.05  # top1 与 top2 的最小领先量，防两池并列时的强行定夺
_EMBED_LOW = 0.3  # 单轮低于即判「领域外」；多轮短句易落在此处，下沉 LLM 层用历史消歧

# 关键词快车道的置信度（规则命中视为高置信，留少量余量）
_KEYWORD_CONF = 0.95

# 兜底 LLM 的时延收口：意图判错只影响行为块选择（explain 是安全默认），不值得让整轮
# 对话等满客户端默认的 120s×(1+retries)。分类输出仅两个字段，正常几百毫秒即回，
# 故收紧超时并关闭自动重试——超时即按异常处理，回退 explain 继续跑。
_LLM_TIMEOUT_S = 8
_LLM_MAX_RETRIES = 0

# ── 意图例句向量的预计算缓存 ─────────────────────────────────
# 例句池是常量，进程内懒加载编码一次；之后的路由只编码 query 单条。
_precomputed_example_vectors: list[list[list[float]]] | None = None
_example_vectors_lock = asyncio.Lock()


async def _get_intent_example_vectors() -> list[list[list[float]]]:
    """获取各意图例句池的稠密向量（外层与 INTENT_ORDER 对齐，内层与池内例句对齐）。

    进程内懒计算一次（lock 保证并发首个请求只触发一次编码），随后回填进程缓存。
    """
    global _precomputed_example_vectors
    if _precomputed_example_vectors is not None:
        return _precomputed_example_vectors
    async with _example_vectors_lock:
        if _precomputed_example_vectors is not None:
            return _precomputed_example_vectors
        texts = [q for intent in INTENT_ORDER for q in INTENT_EXAMPLES[intent]]
        dense = (await agenerate_embeddings(texts))["dense"]
        sizes = [len(INTENT_EXAMPLES[i]) for i in INTENT_ORDER]
        pools, cursor = [], 0
        for size in sizes:
            pools.append(dense[cursor : cursor + size])
            cursor += size
        _precomputed_example_vectors = pools
        return _precomputed_example_vectors


# 意图得分取池内相似度前几名做均值：3 能覆盖住「多句式池」的主流问法，
# 又不至于把池中弱相关例句也平均进来
_POOL_TOP_K = 3


def _pool_scores(query_vec: list[float], pools: list[list[list[float]]]) -> list[float]:
    """逐意图算 query 与例句池相似度的 top-K 均值（与 pools/INTENT_ORDER 对齐）。

    dense 已 L2 归一化，点积即余弦相似度（内联，免去独立辅助函数）。池内例句不足
    K 条时按实际条数取均值。
    """
    scores = []
    for pool in pools:
        sims = sorted((sum(x * y for x, y in zip(query_vec, v, strict=True)) for v in pool), reverse=True)
        top = sims[:_POOL_TOP_K]
        scores.append(sum(top) / len(top))
    return scores


def format_history(messages: list[AnyMessage]) -> str:
    """把历史消息转成「用户: …」纯文本；压缩回顾会被排除。"""
    questions = [f"用户: {m.content}" for m in messages[:-1] if m.type == "human" and not is_summary_message(m)]
    return "\n".join(questions)


def _classify_by_keyword(text: str) -> IntentResult | None:
    """关键词快车道：唯一命中时直判，无命中或歧义返回 None。"""
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


async def _classify_by_embedding(text: str, multi_turn: bool = False, scores: list[float] | None = None) -> IntentResult | None:
    """embedding 语义路由：池内 top-3 均值；高置信直判，中等置信交 LLM。

    多轮场景下低相似不判死，下沉到 LLM 用历史消歧。
    """
    if scores is None:
        pools = await _get_intent_example_vectors()
        emb = await agenerate_embeddings([text])
        scores = _pool_scores(emb["dense"][0], pools)
    ranked = sorted(scores, reverse=True)
    best_idx = max(range(len(scores)), key=scores.__getitem__)
    top1, top2 = ranked[0], ranked[1]

    if top1 >= _EMBED_HIGH and top1 - top2 >= _EMBED_MARGIN:
        return IntentResult(INTENT_ORDER[best_idx], float(top1), "embedding")
    if top1 < _EMBED_LOW and not multi_turn:
        return IntentResult(UNCLEAR, float(top1), "embedding")  # 与所有例句池都低相似
    return None  # 中等置信 / 过线但没拉开 / 多轮低相似，交 LLM 兜底


async def _classify_by_llm(text: str, history: str = "") -> IntentResult:
    """LLM 结构化兜底：按「历史 + 最新问题」判定，输出越界时收敛为 unclear。"""
    # 结构化输出方式按供应商挑（官方平台不支持 json_schema，见 bind_structured_output 注释）
    llm = bind_structured_output(
        get_llm_client(enable_thinking=False, timeout=_LLM_TIMEOUT_S, max_retries=_LLM_MAX_RETRIES),
        _IntentDecision,
    )
    prompt = PromptTemplate.from_template(load_prompt("intent_classify"))
    decision = await llm.ainvoke(
        prompt.format(questions_history=(history or "").strip() or _NO_HISTORY, query=text),
        config={"tags": ["trace_role:intent"]},
    )
    # 白名单校验：模型可能输出枚举外的值，收敛到 unclear（防注入/越界）
    intent = decision.intent if decision.intent in VALID_INTENTS else UNCLEAR
    return IntentResult(
        intent=intent,
        confidence=max(0.0, min(1.0, decision.confidence)),
        source="llm",
    )


async def detect_intent(text: str, history: str = "") -> IntentResult:
    """三档漏斗识别意图；source=fallback 表示降级到安全默认。"""
    async with astage("intent_ms"):
        query = (text or "").strip()
        if not query:
            return IntentResult(UNCLEAR, 0.0, "fallback")
        multi_turn = bool((history or "").strip())

        # 1) 关键词快车道 + 例句池 top1 核验：命中不直返，先看池冠是否同意。
        #    强信号字词面巧合（「生成树」含「生成」）黑名单写不完，语义核验兜住；
        #    核验要编码（GPU 单条几十毫秒），对多数本就下沉的 query 是零边际成本。
        kw = _classify_by_keyword(query)
        pool_scores: list[float] | None = None
        if kw is not None:
            try:
                emb_pools = await _get_intent_example_vectors()
                emb = await agenerate_embeddings([query])
                pool_scores = _pool_scores(emb["dense"][0], emb_pools)
            except Exception as exc:
                logger.warning(f"关键词池核验失败，信任关键词档: {exc}")
                return kw
            pool_top1 = INTENT_ORDER[max(range(len(pool_scores)), key=pool_scores.__getitem__)]
            if pool_top1 == kw.intent:
                return kw
            logger.info(f"关键词命中被池核验降级: 「{query[:30]}」 {kw.intent}≠池冠 {pool_top1}，视同未命中续走漏斗")

        # 2) embedding 语义路由（失败不阻断，降级 LLM；核验编过码则直接复用分数）
        try:
            if (result := await _classify_by_embedding(query, multi_turn, scores=pool_scores)) is not None:
                return result
        except Exception as exc:
            logger.warning(f"意图识别 embedding 路由失败，降级 LLM: {exc}")

        # 3) LLM 兜底（带上历史消歧；再失败则回退安全默认「讲解」，不阻断主流程）
        try:
            return await _classify_by_llm(query, history)
        except Exception as exc:
            logger.warning(f"意图识别 LLM 兜底失败，回退 explain: {exc}")
            return IntentResult("explain", 0.0, "fallback")


