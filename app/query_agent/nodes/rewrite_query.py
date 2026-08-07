import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter

from app.clients.llm import get_llm_client
from app.core import load_prompt, logger, log_node
from app.query_agent.state import QueryState
from app.utils.chat_util import load_chat_history


def load_history(session_id: str | None) -> tuple[str, str]:
    """加载最近几轮对话历史，供问题重写消歧。

    Args:
        session_id: 会话 ID（MongoDB 文档 _id），缺省时生成新会话。

    Returns:
        (session_id, chat_history)：session_id 缺省时返回新生成的 ID，
        调用方需写回 state，供后续 save_history 复用；chat_history 为纯文本，
        无会话/读取失败时为空字符串（降级为单轮问答）。
    """
    session_id = session_id or uuid.uuid4().hex
    return session_id, load_chat_history(session_id)


async def rewrite(original_query: str, textbook_name: str = "", chat_history: str = "") -> str:
    """将原始问题重写为适合检索的独立问题。

    Args:
        original_query: 用户原始问题。
        textbook_name: 教材名，用于限定改写上下文，缺省为空。
        chat_history: 最近几轮对话历史（纯文本），缺省为空（单轮）。

    Returns:
        改写后的问题（LLM 输出，已去空白）。
    """
    # 加载模板（lru_cache 缓存，不带后缀）
    template = load_prompt("rewrite_query")
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm_client()
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke({
        "original_query": original_query,
        "textbook_name": textbook_name,
        "chat_history": chat_history or "（无历史对话，本次为首次提问）",
    })
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

    session_id, chat_history = load_history(state.get("session_id"))
    logger.info(f"会话 {session_id}: 加载历史 {len(chat_history)} 字符")

    rewritten_query = await rewrite(original_query, textbook_name, chat_history)
    logger.info(f"重写问题：{original_query} → {rewritten_query}")
    state["session_id"] = session_id
    state["chat_history"] = chat_history
    state["rewritten_query"] = rewritten_query

    return state


# 单元测试
if __name__ == '__main__':
    import asyncio

    test_state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "如何使用指针?",
        "chat_history": "",
    }
    asyncio.run(rewrite_query(test_state))
