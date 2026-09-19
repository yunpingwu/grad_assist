"""查询重写工具函数：供 search_textbook 消歧多轮指代/省略。

从 query_functions 收编而来：仅保留纯函数 rewrite / format_questions（原节点封装
rewrite_query 已随 query_functions 图一并移除）。
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.clients.llm import get_llm_client
from app.core import load_prompt


def format_questions(messages: list[AnyMessage]) -> str:
    """将 LangChain 消息列表转为 str 供 LLM 输入。

    Args:
        messages: 对话消息列表（取除最新一条外的历史用户问题）。

    Returns:
        纯文本的历史用户问题；无历史返回空字符串。
    """
    questions = []
    for m in messages[:-1]:
        if m.type == "human":
            questions.append(f"用户: {m.content}")
    return "\n".join(questions)


async def rewrite(original_query: str, textbook_name: str = "", questions_history: str = "") -> str:
    """将原始问题重写为适合检索的独立问题。

    Args:
        original_query: 用户原始问题。
        textbook_name: 教材名，用于限定改写上下文，缺省为空。
        questions_history: 最近几轮对话历史（纯文本），缺省为空（单轮）。

    Returns:
        改写后的问题（LLM 输出，已去空白）。
    """
    # 加载模板（lru_cache 缓存，不带后缀）
    template = load_prompt("rewrite_query")
    prompt = ChatPromptTemplate.from_template(template)
    # 重写任务用非思考模式：省时延省推理 token（百炼混合思考型模型经 extra_body 关闭思考）
    llm = get_llm_client(enable_thinking=False)
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke(
        {
            "original_query": original_query,
            "textbook_name": textbook_name,
            "questions_history": questions_history or "（无历史问题，本次为首次提问）",
        }
    )
    return output.strip()


# 冒烟测试已迁移至 tests/study_agent/query_functions/test_rewrite_query.py
