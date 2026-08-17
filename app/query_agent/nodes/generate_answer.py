import json
import re

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter

from app.clients.llm import get_llm_client
from app.core import load_prompt, log_node, logger
from app.query_agent.state import QueryState


def _extract_image_urls(answer: str) -> list[dict]:
    """从 LLM 回答中提取实际引用的图片 URL（markdown 引用 + 裸 URL，限图片扩展名）。

    - 只取图片扩展名（jpg/jpeg/png/gif/webp/bmp）结尾的 URL，防止把网络搜索来源链接等
      普通链接误存为图片；
    - 去重并编号，供历史回显。

    Args:
        answer: LLM 生成的答案文本（可能含 ![图注](url) 或裸 URL）。

    Returns:
        形如 [{index, url}] 的列表；无图片引用返回空列表。
    """
    urls: list[str] = []
    for m in re.finditer(r"https?://[^\s)\]}]+\b", answer):
        url = m.group(0).rstrip(",.;")
        if re.search(r"\.(jpg|jpeg|png|gif|webp|bmp)(\?|$)", url, re.IGNORECASE) and url not in urls:
            urls.append(url)
    return [{"index": i + 1, "url": u} for i, u in enumerate(urls)]


def build_image_candidates(chunks: list[dict]) -> str:
    """汇总召回片段的图片候选,拼成 markdown 片段注入 LLM context。

    格式(每张图一行,含可复制的引用格式,供 LLM 决定是否插入回答):
        - 简介: <简介>
          引用: ![<图注>](<url>)
    """
    lines: list[str] = []
    for hit in chunks:
        entity = hit.get("entity") or hit
        meta_raw = entity.get("metadata_json")
        if not meta_raw:
            continue
        try:
            meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for img in meta.get("images", []) or []:
            url = img.get("url")
            desc = (img.get("description") or "").strip()
            if not url or not desc:
                continue
            lines.append(f"- 简介: {desc}\n  引用: ![{desc}]({url})")
    return "\n".join(lines)


async def ask_llm(
    chunks: list[dict],
    original_query: str,
    *,
    writer: StreamWriter,
    web_chunks: list[dict] | None = None,
) -> str:
    """向 LLM 提问，基于召回片段（及可选的联网结果）流式生成答案。

    Args:
        chunks: 参考片段。
        original_query: 用户原始问题。
        writer: 流式 writer，逐 token 推送 {"type": "token", "content": …}。
        web_chunks: 联网搜索结果 [{title, url, content}]，拼入【网络搜索】区块。

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
    # 图片候选:附上简介与可复制的引用格式,LLM 据此决定是否在回答中插入图片引用
    image_candidates = build_image_candidates(chunks)
    if image_candidates:
        context += f"\n\n【图片候选】\n{image_candidates}"
    # 联网结果:作为补充资料拼入上下文,明确标注来源,LLM 需注明引用
    web_parts = [
        f"- [{c.get('title', '')}]({c.get('url', '')}): {c.get('content', '')}"
        for c in (web_chunks or [])
        if c.get("content")
    ]
    if web_parts:
        context += (
            "\n\n【网络搜索】(以下为联网检索的外部资料,可能与教材表述不同;"
            "仅作补充,引用时注明来源,教材未覆盖时可参考)\n" + "\n".join(web_parts)
        )
    # 调用 llm 流式生成回答
    template = load_prompt("generate_answer")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm_client() | StrOutputParser()
    parts: list[str] = []
    async for token in chain.astream(
        {
            "context": context,
            "original_query": original_query,
        }
    ):
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
    # 优先使用 rerank 精排后的 TOP-K；未启用/精排失败时回退 RRF 融合结果
    chunks = state.get("reranked_chunks") or state.get("merged_chunks", []) or []
    # 只取 TOP5 片段，避免上下文过大、干扰回答（rerank 已截断，此处为兜底）
    chunks = chunks[:5]
    answer = await ask_llm(chunks, original_query, writer=writer, web_chunks=state.get("web_chunks"))
    logger.info(f"答案: {answer[:120]!r}")
    # 流结束：通知前端收尾并回传 session_id（custom 流里不含 answer）
    writer({"type": "done", "session_id": state.get("session_id")})
    # 只返回本节点写入的字段
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# 冒烟测试：桩掉 LLM 生成，验证片段优先级/截断与 answer/messages 写回
if __name__ == "__main__":
    import asyncio

    async def _fake_ask(
        chunks: list[dict],
        original_query: str,
        *,
        writer: StreamWriter,
        web_chunks: list[dict] | None = None,
    ) -> str:
        assert len(chunks) <= 5, f"chunks 应截断到 TOP5，实际 {len(chunks)}"
        return f"模拟答案（{len(chunks)} 个片段，{len(web_chunks or [])} 条联网）"

    ask_llm = _fake_ask  # 覆盖真实 LLM 流式生成

    def _hit(doc_id: str, text: str) -> dict:
        return {"id": doc_id, "distance": 0.8, "entity": {"text": text}}

    def writer(chunk):
        print(f"  [writer] {chunk}")

    test_state: QueryState = {
        "session_id": "test",
        "textbook_name": "C语言程序设计",
        "original_query": "什么是栈的先进后出特性？",
        # reranked_chunks 优先：即使 merged_chunks 更多，也应走精排结果
        "reranked_chunks": [_hit("r1", "精排片段1"), _hit("r2", "精排片段2")],
        "merged_chunks": [_hit(f"m{i}", f"融合片段{i}") for i in range(7)],
        "web_chunks": [{"title": "t", "url": "u", "content": "c"}],
    }
    result = asyncio.run(generate_answer(test_state, writer=writer))
    ans = result["answer"]
    assert "2 个片段" in ans and "1 条联网" in ans, f"reranked_chunks 未优先: {ans!r}"
    assert len(result["messages"]) == 1 and result["messages"][0].type == "ai", "应写回 AIMessage"
    assert result["messages"][0].content == ans, "messages 内容应与 answer 一致"

    # 回退路径：无 reranked_chunks 时取 merged_chunks 并截断到 TOP5
    fallback_state = {k: v for k, v in test_state.items() if k != "reranked_chunks"}
    ans2 = asyncio.run(generate_answer(fallback_state, writer=writer))["answer"]
    assert "5 个片段" in ans2, f"merged_chunks 未截断到 5: {ans2!r}"
    print(f"答案(精排优先): {ans}")
    print(f"答案(回退截断): {ans2}")
    print("generate_answer 测试通过")
