import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter

from app.clients.llm import get_llm_client
from app.core import log_node, logger, load_prompt
from app.query_agent.state import QueryState
from app.utils.chat_util import append_turn


def save_history(state: QueryState, answer: str) -> None:
    """将本轮 user/assistant 消息持久化到 MongoDB。

    存储是旁路：失败由 append_turn 内部降级（仅告警），不阻断问答主链路。
    附带存档改写问题与候选图片，供前端历史回显。
    """
    append_turn(
        session_id=state.get("session_id"),
        textbook_name=state.get("textbook_name", ""),
        user_msg=state.get("original_query", ""),
        assistant_msg=answer,
        rewritten_query=state.get("rewritten_query"),
        images=collect_image_options(state.get("merged_chunks", []) or []),
    )


def collect_image_options(chunks: list[dict]) -> list[dict]:
    """从召回片段中汇总候选图片（解析 metadata_json）。

    Args:
        chunks: RRF 融合后的召回 hit 列表（entity 含 metadata_json）。

    Returns:
        候选图片列表，每项 {index, url, description}。
    """
    images: list[dict] = []
    for hit in chunks:
        entity = hit.get("entity") or hit  # pymilvus 命中结构兼容
        meta_raw = entity.get("metadata_json")
        if not meta_raw:
            continue
        try:
            meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for img in meta.get("images", []) or []:
            url = img.get("url")
            if not url:
                continue
            images.append({
                "index": len(images) + 1,
                "url": url,
                "description": img.get("path", "教材图片"),
            })
    return images


async def ask_llm(chunks: list[dict], original_query: str, *, writer: StreamWriter) -> str:
    """向 LLM 提问，基于召回片段流式生成答案。

    Args:
        chunks: 参考片段。
        original_query: 用户原始问题。
        writer: 流式 writer，逐 token 推送 {"type": "token", "content": …}。

    Returns:
        LLM 生成的完整答案文本。
    """
    # 组装上下文
    context_parts = []
    for i, hit in enumerate(chunks, start=1):
        entity = hit.get("entity") or hit
        text = (entity.get("text") or "").strip()
        if text:
            context_parts.append(f"[片段{i}] {text}")
    context = "\n\n".join(context_parts) or "（无召回片段）"
    # 调用 llm 流式生成回答
    template = load_prompt("generate_answer")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm_client() | StrOutputParser()
    parts: list[str] = []
    async for token in chain.astream({
        "context": context,
        "original_query": original_query,
    }):
        parts.append(token)
        writer({"type": "token", "content": token})
    return "".join(parts)


@log_node
async def generate_answer(state: QueryState, *, writer: StreamWriter) -> dict:
    """依据召回片段流式生成答案，图片引用由主 LLM 决定。

    Args:
        state: 含 merged_chunks / original_query / textbook_name。
        writer: 流式 writer：stage 提示 + token 逐字输出 + done 收尾。

    Returns:
        写回 answer（文本，可能含 markdown 图片引用）。
    """
    writer({"type": "stage", "stage": "generate", "message": "正在生成回答…"})
    original_query = state.get("original_query", "")
    chunks = state.get("merged_chunks", []) or []
    # 只取融合后 TOP5 片段，避免上下文过大、干扰回答
    chunks = chunks[:5]
    answer = await ask_llm(chunks, original_query, writer=writer)
    logger.info(f"答案: {answer[:120]!r}")
    # 多轮对话的出口：把本轮问答持久化（旁路，失败不阻断）
    save_history(state, answer)
    # 流结束：通知前端收尾并回传 session_id（custom 流里不含 answer）
    writer({"type": "done", "session_id": state.get("session_id")})
    # 只返回本节点写入的字段
    return {"answer": answer}


# 单元测试
if __name__ == '__main__':
    import asyncio

    test_chunks = [
        {
            "id": "c1",
            "distance": 0.8,
            "entity": {
                "text": "# 第3章 栈 > ## 栈的基本操作\n栈是一种后进先出的数据结构。",
                "chapter": "第3章 栈",
                "section": "栈的基本操作",
                "metadata_json": json.dumps({
                    "images": [{"path": "images/stack.png", "url": "http://minio/stack.png"}]
                }),
            },
        }
    ]
    test_state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "什么是栈的先进后出特性？",
        "rewritten_query": "什么是栈的先进后出特性？",
        "merged_chunks": test_chunks,
    }
    result = asyncio.run(generate_answer(test_state))
    print(f"答案: {result.get('answer', '')[:120]!r}")
