"""ragas 四指标评测：Context Precision/Recall + Faithfulness/Answer Relevancy。

并发由信号量控制；judge 默认取 ``LLM_PROVIDER``，建议与生成侧用不同模型。
用法：``uv run python -m app.eval.rag.evals [数据集] [--limit N]``。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from app.clients.llm import llm_alibaba, llm_deepseek
from app.config import llm_config
from app.core import logger
from app.eval.rag import rag
from app.ml.batch_manager.embedder import agenerate_embeddings
from app.ml.embedding_util import generate_embeddings

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONCURRENCY = 8
# judging 长上下文时输出很长（逐句/逐条 verdict），必须给足，否则 IncompleteOutputException
JUDGE_MAX_TOKENS = 8192
# 供应商模块注册表（与 app/clients/llm.py 的 _PROVIDERS 同源，用于复用思考开关口径）
_PROVIDER_MODULES = {"alibaba": llm_alibaba, "deepseek": llm_deepseek}
# AnswerRelevancy 反生成问题的条数（每多 1 条多一次 LLM 调用）
ANSWER_RELEVANCY_STRICTNESS = 3


class BgeM3Embeddings(BaseRagasEmbedding):
    """把项目 BGE-M3 接到 ragas embeddings 接口；dense 向量已 L2 归一化。"""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return generate_embeddings([text])["dense"][0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return (await agenerate_embeddings([text]))["dense"][0]

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        """一次批量编码（基类是逐条 gather，这里用攒批器一次前向更划算）。"""
        return (await agenerate_embeddings(list(texts)))["dense"]


def _judge_llm(provider: str):
    """按供应商构造 ragas 的 InstructorLLM（async client，两家均走 OpenAI 兼容端点）。

    ``extra_body`` 必须经 ``llm_factory`` 转发关思考（DeepSeek 默认开启），
    ``max_tokens`` 给足，否则长上下文 judging 会截断。
    """
    module = _PROVIDER_MODULES[provider]
    kwargs = module.client_kwargs(None, False)
    if not kwargs.get("api_key"):
        raise SystemExit(f"缺少 {provider} 的 API Key（.env 里的 {module.API_KEY_ENV}）")
    client = AsyncOpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"])
    llm = llm_factory(
        kwargs["model"],
        client=client,
        max_tokens=JUDGE_MAX_TOKENS,
        extra_body=kwargs.get("extra_body"),
    )
    return llm, kwargs["model"]


async def _score(job, **fields):
    """跑一个指标的一条样本，失败转 NaN（不中断整轮）。"""
    try:
        result = await job.ascore(**fields)
        return float(result.value)
    except Exception as exc:  # 单条失败不拖垮整轮，但会记进 nan 计数
        logger.warning(f"指标失败 {getattr(job, 'name', '?')}: {type(exc).__name__}: {exc}")
        return math.nan


def _samples_meta(config: str, key: str) -> dict:
    """读样本文件的 generation 块（生成模型、时间），报告里用来标明这批答案的来历。"""
    path = rag.samples_path(config, key)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("generation") or {}


def _mean(values: list[float]) -> float | None:
    """跳过 NaN 求均值；全 NaN 返回 None（别让 nan 混进汇总）。"""
    ok = [v for v in values if not math.isnan(v)]
    return statistics.fmean(ok) if ok else None


async def evaluate_dataset(key: str, config: str, *, provider: str, limit: int | None, concurrency: int) -> dict:
    """跑一份数据集在某配置下的四个指标，返回逐题结果与均值。"""
    dataset = json.loads((rag.DATASETS_DIR / f"{key}.json").read_text(encoding="utf-8"))
    samples = rag.load_samples(config, key)
    pairs = [(q, samples[q["id"]]) for q in dataset["questions"] if q["id"] in samples]
    if not pairs:
        raise SystemExit(f"{key}[{config}]: 样本为空")
    if limit:
        pairs = pairs[:limit]

    logger.info(f"{key}[{config}]: {len(pairs)} 题 × 4 指标")
    judge, judge_model = _judge_llm(provider)
    metrics = {
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=BgeM3Embeddings(), strictness=ANSWER_RELEVANCY_STRICTNESS
        ),
    }

    # 上下文按落盘的 id 从 Milvus 取回（只存 id 不存正文，见 rag.fetch_contexts）
    all_ids = sorted({cid for _, s in pairs for cid in s["retrieved_chunk_ids"]})
    contexts = rag.fetch_contexts(dataset["textbook"], all_ids) if all_ids else {}

    sem = asyncio.Semaphore(concurrency)

    async def one(question: dict, sample: dict) -> dict:
        # 关键：上下文与回答必须来自**同一个配置的同一轮**——这才是 Faithfulness 的判据
        ctx = [contexts[cid] for cid in sample["retrieved_chunk_ids"]]
        fields = {
            "user_input": question["question"],
            "retrieved_contexts": ctx,
            "reference": question["reference_answer"],
        }
        async with sem:
            cp, cr, fa, ar = await asyncio.gather(
                _score(metrics["context_precision"], **fields),
                _score(metrics["context_recall"], **fields),
                _score(
                    metrics["faithfulness"],
                    user_input=fields["user_input"],
                    response=sample["response"],
                    retrieved_contexts=ctx,
                ),
                _score(
                    metrics["answer_relevancy"],
                    user_input=fields["user_input"],
                    response=sample["response"],
                ),
            )
        return {"id": question["id"], "category": question["category"],
                "context_precision": cp, "context_recall": cr,
                "faithfulness": fa, "answer_relevancy": ar}

    rows = []
    for done, coro in enumerate(asyncio.as_completed([one(q, s) for q, s in pairs]), 1):
        rows.append(await coro)
        if done % 10 == 0 or done == len(pairs):
            logger.info(f"  {key}[{config}]: {done}/{len(pairs)}")

    names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    summary = {n: _mean([r[n] for r in rows]) for n in names}
    summary["nan_counts"] = {n: sum(1 for r in rows if math.isnan(r[n])) for n in names}
    return {
        "dataset": key,
        "config": config,
        "n": len(rows),
        "judge_provider": provider,
        "judge_model": judge_model,
        "generation_model": _samples_meta(config, key).get("model"),
        "sample_generated_at": _samples_meta(config, key).get("generated_at"),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "per_question": rows,
    }


def _print_table(results: list[dict], config: str) -> None:
    names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    width = max(len(r["dataset"]) for r in results) + 2
    print(f"\n============ RAG 四指标评测（配置 {config}）============\n")
    print("数据集".ljust(width) + "".join(n[:18].rjust(20) for n in names) + "      n")
    print("-" * (width + 20 * len(names) + 7))
    for r in results:
        cells = ""
        for n in names:
            v = r["summary"][n]
            cells += ("—" if v is None else f"{v:.4f}").rjust(20)
        print(r["dataset"].ljust(width) + cells + f"{r['n']:>7}")
    print("-" * (width + 20 * len(names) + 7))
    overall = {n: _mean([r["summary"][n] for r in results if r["summary"][n] is not None]) for n in names}
    print(
        "加权平均".ljust(width)
        + "".join(("—" if overall[n] is None else f"{overall[n]:.4f}").rjust(20) for n in names)
    )
    print(f"\n评审模型: {results[0]['judge_model']} | 生成模型: {results[0]['generation_model']}")
    nan = {n: sum(r["summary"]["nan_counts"][n] for r in results) for n in names}
    if any(nan.values()):
        print(f"[!] 失败(计为 NaN)条数: {nan}")   # 不用 emoji：Windows 控制台是 GBK，U+26A0 编不出来


async def main(keys: list[str], config: str, *, provider: str, limit: int | None, concurrency: int) -> None:
    keys = keys or sorted(p.stem for p in rag.DATASETS_DIR.glob("*.json"))
    out_dir = RESULTS_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key in keys:
        result = await evaluate_dataset(
            key, config, provider=provider, limit=limit, concurrency=concurrency
        )
        results.append(result)
        (out_dir / f"{key}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(f"{key}[{config}]: 已写入 results/{config}/{key}.json")
    # 总表从目录里**所有**结果文件重建：分批跑（比如只补 db/ds/net/os）不会把
    # 先前算好的 c/co 从总表与屏幕上抹掉
    merged = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.stem != "summary":
            merged[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    (out_dir / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "per_question"} for r in merged.values()],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print_table(list(merged.values()), config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 四指标评测")
    parser.add_argument("config", choices=rag.CONFIGS, help="检索配置（决定用哪套样本）")
    parser.add_argument("keys", nargs="*", help="数据集短键（c/co/db/ds/net/os），缺省全部")
    parser.add_argument("--judge-provider", default=llm_config.provider, choices=["alibaba", "deepseek"])
    parser.add_argument("--limit", type=int, default=None, help="每份数据集只跑前 N 题（打通用）")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    asyncio.run(
        main(args.keys, args.config, provider=args.judge_provider, limit=args.limit, concurrency=args.concurrency)
    )
    sys.exit(0)
