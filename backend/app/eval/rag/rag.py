"""被测系统接缝：检索 + 生成，四个指标的输入都从这里出。

配置：``hyb_50``（快速路径）与 ``rerank``（深度，对齐 search_textbook）；
题集在 ``datasets/``，答案与上下文按配置落在 ``samples/<config>/``。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.clients import llm as llm_facade
from app.clients import milvus_client
from app.config import llm_config, rerank_config
from app.core import logger
from app.core.load_prompt import load_prompt
from app.core.safety import wrap_untrusted
from app.ml.embedding import agenerate_embeddings
from app.retrieval.dense import search_by_vectors
from app.retrieval.fusion import rrf_merge
from app.retrieval.hyde import hyde_doc_generate
from app.retrieval.rerank import arerank_chunks_weighted
from app.retrieval.store import get_collection_by_name, hybrid_search

PACKAGE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PACKAGE_DIR / "datasets"
SAMPLES_DIR = PACKAGE_DIR / "samples"
PROMPTS_DIR = PACKAGE_DIR / "prompts"

# 支持的检索配置（= samples/ 下的子目录名）
CONFIGS = ("hyb_50", "rerank")

TOP_K = 5
HYB50_WEIGHTS = (0.5, 0.5)
# 生成并发（每题一次 LLM 调用）；Milvus / HyDE 也走各自的并发路径
CONCURRENCY = 8

_OUTPUT_FIELDS = ["text", "chapter", "section", "block_type"]


@dataclass(frozen=True)
class Hit:
    """一次检索命中的一条 chunk。"""

    id: str
    text: str
    chapter: str = ""


def _hits(raw: list[dict]) -> list[Hit]:
    """Milvus 命中（可能被融合/精排包了一层）→ Hit 列表。"""
    out: list[Hit] = []
    for hit in raw:
        entity = hit.get("entity") or hit
        out.append(
            Hit(
                id=str(hit.get("id")),
                text=str(entity.get("text") or ""),
                chapter=str(entity.get("chapter") or ""),
            )
        )
    return out


async def retrieve(question: str, textbook: str, config: str = "hyb_50") -> list[Hit]:
    """按配置（``hyb_50`` / ``rerank``）跑一次检索，返回 top-k 命中。"""
    if config not in CONFIGS:
        raise ValueError(f"未知配置: {config!r}（可用：{' / '.join(CONFIGS)}）")
    collection = get_collection_by_name(textbook)
    if not collection:
        raise ValueError(f"教材未在 Milvus 注册: {textbook}")

    if config == "hyb_50":
        embeddings = await agenerate_embeddings([question])
        raw = hybrid_search(
            embeddings["dense"][0],
            embeddings["sparse"][0],
            collection,
            limit=TOP_K,
            weights=HYB50_WEIGHTS,
        )
    else:
        raw = await _retrieve_deep(question, textbook)
    if not raw:
        raise ValueError(f"检索无结果: {question[:40]}")
    return _hits(raw)


async def _retrieve_deep(question: str, textbook: str) -> list[dict]:
    """深度路径；精排失败直接抛出，不静默降级（评测要的是真实配置的分数）。"""
    hyde_doc = await hyde_doc_generate(question)
    # 一次批量算 query 与 hyde 的向量，省一次模型前向（与生产同口径）
    embeddings = await agenerate_embeddings([question, hyde_doc + question])
    pool = rerank_config.candidate_pool
    embedding_chunks = await search_by_vectors(
        textbook, embeddings["dense"][0], embeddings["sparse"][0], None, limit=pool
    )
    hyde_chunks = await search_by_vectors(
        textbook, embeddings["dense"][1], embeddings["sparse"][1], None, limit=pool
    )
    merged = await rrf_merge(embedding_chunks, hyde_chunks)
    return await arerank_chunks_weighted(
        question, merged, TOP_K, rerank_config.fusion_alpha
    )


def fetch_contexts(textbook: str, chunk_ids: list[str]) -> dict[str, str]:
    """按 chunk id 从 Milvus 取回正文；缺 id 时抛 ValueError（语料重灌可立即发现）。"""
    collection = get_collection_by_name(textbook)
    if not collection:
        raise ValueError(f"教材未在 Milvus 注册: {textbook}")
    rows = milvus_client.get_client().query(
        collection_name=collection,
        filter=f"id in {json.dumps(chunk_ids, ensure_ascii=False)}",
        output_fields=_OUTPUT_FIELDS,
    )
    found = {str(row["id"]): str(row.get("text") or "") for row in rows}
    missing = [cid for cid in chunk_ids if cid not in found]
    if missing:
        raise ValueError(f"{textbook}: {len(missing)} 个 chunk id 在库里查不到，例如 {missing[:2]}")
    return found


def build_answer_prompt(question: str, contexts: list[str]) -> str:
    """拼生成 prompt：每条上下文各自 `wrap_untrusted` 包边界后再拼接。

    边界串每次随机生成，所以只进 prompt；落盘的样本只有 chunk id，因此可复现。
    """
    wrapped = "\n\n---\n\n".join(wrap_untrusted(text, "教材") for text in contexts)
    template = load_prompt("answer", prompts_dir=PROMPTS_DIR)
    return template.format(contexts=wrapped, question=question)


async def answer(question: str, contexts: list[str], *, client=None) -> str:
    """单轮生成：给定问题与**这一轮**的上下文，返回系统回答。

    单轮无历史是有意的——多轮会把检索结果之外的信息带进答案，Faithfulness 会判成幻觉。
    """
    client = client or llm_facade.get_llm_client(enable_thinking=False)
    result = await client.ainvoke(build_answer_prompt(question, contexts))
    content = getattr(result, "content", result)
    return content if isinstance(content, str) else str(content)


# ────────────────────────── 样本生成 ──────────────────────────


def prompt_fingerprint() -> str:
    """冻结 prompt 的内容短哈希，钉住「这批答案是用哪版 prompt 生成的」。"""
    raw = (PROMPTS_DIR / "answer.prompt").read_bytes()
    return f"answer.prompt@{hashlib.sha256(raw).hexdigest()[:12]}"


def samples_path(config: str, key: str) -> Path:
    return SAMPLES_DIR / config / f"{key}.json"


def load_samples(config: str, key: str) -> dict[str, dict]:
    """加载某配置的样本，返回 ``{题号: {"retrieved_chunk_ids": [...], "response": "..."}}``。"""
    path = samples_path(config, key)
    if not path.exists():
        raise SystemExit(f"缺少样本 {path}，先跑 `python -m app.eval.rag.rag {config} {key}`")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["samples"]}


async def _fill_one(item: dict, textbook: str, config: str, sem: asyncio.Semaphore, client) -> dict:
    async with sem:
        hits = await retrieve(item["question"], textbook, config)
        text = await answer(item["question"], [hit.text for hit in hits], client=client)
    return {"id": item["id"], "retrieved_chunk_ids": [hit.id for hit in hits], "response": text}


async def fill_responses(key: str, config: str, *, force: bool, concurrency: int = CONCURRENCY) -> dict:
    """给一份题集补齐某配置的样本，写到 `samples/<config>/<key>.json`。"""
    dataset = json.loads((DATASETS_DIR / f"{key}.json").read_text(encoding="utf-8"))
    path = samples_path(config, key)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    done = {} if force else {s["id"]: s for s in existing.get("samples", []) if s.get("response")}
    todo = [q for q in dataset["questions"] if q["id"] not in done]
    if not todo:
        logger.info(f"{key}[{config}]: 全部已有样本，跳过（重跑加 --force）")
        return {"key": key, "config": config, "generated": 0}

    logger.info(f"{key}[{config}]: 待生成 {len(todo)}/{len(dataset['questions'])} 题")
    sem = asyncio.Semaphore(concurrency)
    client = llm_facade.get_llm_client(enable_thinking=False)
    outcomes = await asyncio.gather(
        *[_fill_one(q, dataset["textbook"], config, sem, client) for q in todo],
        return_exceptions=True,
    )

    failures = [(q["id"], r) for q, r in zip(todo, outcomes, strict=True) if isinstance(r, BaseException)]
    if failures:
        detail = "; ".join(f"{i}: {e}" for i, e in failures[:5])
        logger.error(f"{key}[{config}]: {len(failures)} 题失败，**不写盘**（修好后重跑续上）：{detail}")
        return {"key": key, "config": config, "generated": 0, "failed": len(failures)}

    merged = {**done, **{s["id"]: s for s in outcomes}}  # type: ignore[union-attr]
    payload = {
        "schema": 1,
        "dataset": key,
        "collection": dataset["collection"],
        "config": config,
        "top_k": TOP_K,
        "generation": {
            "provider": llm_config.provider,
            "model": str(getattr(client, "model_name", "") or getattr(client, "model", "")),
            "temperature": llm_config.temperature,
            "prompt": prompt_fingerprint(),
            "rerank_fusion_alpha": rerank_config.fusion_alpha if config == "rerank" else None,
            "candidate_pool": rerank_config.candidate_pool if config == "rerank" else None,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "samples": [merged[q["id"]] for q in dataset["questions"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"{key}[{config}]: 写入 {len(todo)} 条，共 {len(payload['samples'])} 条")
    return {"key": key, "config": config, "generated": len(todo)}


async def _main(keys: list[str], config: str, *, force: bool) -> None:
    keys = keys or sorted(p.stem for p in DATASETS_DIR.glob("*.json"))
    for key in keys:
        print(json.dumps(await fill_responses(key, config, force=force), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="给评测集生成某配置的样本（检索 → 单轮生成）")
    parser.add_argument("config", choices=CONFIGS, help="检索配置")
    parser.add_argument("keys", nargs="*", help="数据集短键（c/co/db/ds/net/os），缺省全部")
    parser.add_argument("--force", action="store_true", help="覆盖已有样本重跑")
    args = parser.parse_args()
    asyncio.run(_main(args.keys, args.config, force=args.force))
    sys.exit(0)
