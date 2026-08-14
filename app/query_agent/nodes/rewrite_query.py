from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter

from app.clients.llm import get_llm_client
from app.core import load_prompt, log_node, logger
from app.query_agent.state import QueryState


def format_questions(messages: list[AnyMessage]) -> str:
    """将 LangChain 消息列表转为str供 LLM 输入。

    Args:

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
    llm = get_llm_client()
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke(
        {
            "original_query": original_query,
            "textbook_name": textbook_name,
            "questions_history": questions_history or "（无历史问题，本次为首次提问）",
        }
    )
    return output.strip()


@log_node
async def rewrite_query(state: QueryState, *, writer: StreamWriter) -> dict:
    """将原始问题重写为更适合检索的查询。

    多轮对话的入口：先加载最近几轮历史（缺省 session_id 时生成并写回 state），
    供 rewrite 消歧；writer 推送阶段提示。
    """
    writer({"type": "stage", "stage": "rewrite", "message": "正在改写问题…"})
    original_query = state.get("original_query")
    textbook_name = state.get("textbook_name", "")
    session_id = state.get("session_id")
    messages = state.get("messages", [])

    questions_history = format_questions(messages)
    logger.info(f"会话 {session_id}: 加载历史问题 {len(questions_history)} 字符")

    rewritten_query = await rewrite(original_query, textbook_name, questions_history)
    logger.info(f"重写问题：{original_query} → {rewritten_query}")
    state["rewritten_query"] = rewritten_query

    return {"rewritten_query": rewritten_query}


# 单元测试
if __name__ == "__main__":
    import asyncio

    def writer(chunk):
        pass

    test_state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "如何使用它?",
        "messages": [
            HumanMessage(content="什么是指针?"),  # 第1轮 问
            AIMessage(content="指针是一种…"),  # 第1轮 答
        ],
    }
    assert asyncio.run(rewrite_query(test_state, writer=writer)) != {}
