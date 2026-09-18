"""多用户并发、多轮会话评测。

默认发现 ``data/eval/*/qa_set.json`` 下的 6 个评测集，共 300 题。
每本教材拆成多个虚拟用户；同一用户的题目串行发送并复用 session_id，
不同用户之间并发执行，用来观察会话隔离、长上下文和服务并发稳定性。

示例::

    python -m app.eval.run_multi_user_eval \
        --users-per-book 5 --turns-per-user 10 --concurrency 10

注意：现有题目大多是独立问题，因此本脚本主要测试并发和会话复用。
若要专门测试“它/这个/上面内容”等指代消歧，应另建带 follow-up 的对话集。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from app.eval.dataset import load_dataset
from app.eval.metrics import aggregate_metrics, compute_metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "data" / "eval"
    parser = argparse.ArgumentParser(description="多用户并发、多轮复用 session_id 的 Agent 评测")
    parser.add_argument("--dataset-root", type=Path, default=root, help="包含多个教材 qa_set.json 的目录")
    parser.add_argument("--url", default="http://127.0.0.1:8000/study/chat", help="studyservice chat SSE 地址")
    parser.add_argument("--users-per-book", type=int, default=5, help="每本教材拆分的虚拟用户数，默认 5")
    parser.add_argument("--turns-per-user", type=int, default=10, help="每个用户最多发送的题数，默认 10；0 表示该用户拿到全部分配题")
    parser.add_argument("--concurrency", type=int, default=10, help="同时运行的用户会话数，默认 10")
    parser.add_argument("--timeout", type=float, default=300.0, help="单轮请求超时秒数，默认 300")
    parser.add_argument("--request-retries", type=int, default=3, help="502/503/504/429 重试次数，默认 3")
    parser.add_argument("--limit-per-book", type=int, default=0, help="每本教材最多取多少题，0 表示全部")
    parser.add_argument("--seed", type=int, default=20260918, help="随机打乱种子；不传 --shuffle 时仅用于结果元数据")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False, help="是否在分配用户前打乱每本教材题目")
    parser.add_argument("--output", type=Path, default=None, help="结果 JSON 路径")
    parser.add_argument("--user-prefix", default="multi-eval", help="X-User-Id 前缀")
    args = parser.parse_args(argv)
    if args.users_per_book < 1:
        parser.error("--users-per-book 必须大于 0")
    if args.turns_per_user < 0:
        parser.error("--turns-per-user 不能为负数")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于 0")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.request_retries < 0:
        parser.error("--request-retries 不能为负数")
    if args.limit_per_book < 0:
        parser.error("--limit-per-book 不能为负数")
    return args


def _discover_datasets(root: Path) -> list[Path]:
    paths = sorted(root.glob("*/qa_set.json"))
    if not paths:
        raise FileNotFoundError(f"未找到评测集: {root / '*/qa_set.json'}")
    return paths


def _build_user_sessions(args: argparse.Namespace) -> list[dict[str, Any]]:
    """按教材分配用户，保持每个用户的题目顺序以形成可复用会话。"""
    rng = random.Random(args.seed)
    sessions: list[dict[str, Any]] = []
    for book_index, path in enumerate(_discover_datasets(args.dataset_root), start=1):
        data = load_dataset(path)
        questions = list(data["questions"])
        if args.limit_per_book:
            questions = questions[: args.limit_per_book]
        if args.shuffle:
            rng.shuffle(questions)

        turns_per_user = args.turns_per_user or math.ceil(len(questions) / args.users_per_book)
        for user_index in range(args.users_per_book):
            start = user_index * turns_per_user
            assigned = questions[start : start + turns_per_user]
            if not assigned:
                continue
            user_id = f"{args.user_prefix}-b{book_index:02d}-u{user_index + 1:02d}"
            sessions.append(
                {
                    "user_id": user_id,
                    "session_id": f"session-{user_id}-{uuid.uuid4().hex[:8]}",
                    "textbook": data["textbook"],
                    "dataset": str(path),
                    "questions": assigned,
                }
            )
    return sessions


async def _request_turn(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    session: dict[str, Any],
    question: dict[str, Any],
    turn_index: int,
) -> dict[str, Any]:
    """发送一轮 SSE 请求；同一 session 的调用由上层串行调度。"""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "user_id": session["user_id"],
        "session_id": session["session_id"],
        "textbook": session["textbook"],
        "turn_index": turn_index,
        "query_id": question.get("id", ""),
        "question": question.get("question", ""),
        "gold_chunk_ids": list(dict.fromkeys(question.get("gold_chunk_ids", []))),
        "status": "error",
        "answer_content": "",
        "retrieved_chunk_ids": [],
        "tool_names": [],
        "request_ids": [],
    }
    payload = {
        "textbook_name": session["textbook"],
        "query": result["question"],
        "session_id": session["session_id"],
    }
    retry_statuses = {429, 502, 503, 504}

    try:
        for attempt in range(args.request_retries + 1):
            try:
                async with semaphore:
                    async with client.stream(
                        "POST",
                        args.url,
                        json=payload,
                        headers={"X-User-Id": session["user_id"]},
                    ) as response:
                        if response.status_code in retry_statuses and attempt < args.request_retries:
                            await response.aread()
                            await asyncio.sleep(min(2**attempt, 10))
                            continue
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            event = json.loads(raw)
                            event_type = event.get("type", "unknown")
                            if event_type == "token":
                                result["answer_content"] += str(event.get("content", ""))
                            elif event_type == "tool":
                                name = event.get("name")
                                if name and name not in result["tool_names"]:
                                    result["tool_names"].append(name)
                            elif event_type == "done":
                                result["status"] = "done"
                                result["request_ids"].append(event.get("request_id"))
                                result["request_id"] = event.get("request_id")
                                result["retrieved_chunk_ids"] = list(event.get("retrieved_chunk_ids") or [])
                                result["answer_content"] = event.get("answer_content") or result["answer_content"]
                            elif event_type == "ask_clarify":
                                result["status"] = "ask_clarify"
                                result["clarify_questions"] = event.get("questions") or []
                            elif event_type == "ask_confirm":
                                result["status"] = "ask_confirm"
                            elif event_type == "error":
                                result["error"] = event.get("message", "studyservice 返回错误")
                        break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in retry_statuses or attempt >= args.request_retries:
                    raise
                await asyncio.sleep(min(2**attempt, 10))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    result["retrieval_metrics"] = compute_metrics(set(result["gold_chunk_ids"]), result["retrieved_chunk_ids"])
    if not result["request_ids"]:
        result.pop("request_ids")
    return result


async def _run_session(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    """一个虚拟用户的多轮请求严格串行，确保 session 上下文顺序正确。"""
    rows: list[dict[str, Any]] = []
    for turn_index, question in enumerate(session["questions"], start=1):
        row = await _request_turn(client, semaphore, args, session, question, turn_index)
        rows.append(row)
    return rows


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    sessions = _build_user_sessions(args)
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 30.0))
    semaphore = asyncio.Semaphore(args.concurrency)
    # 本机评测必须直连，避免系统代理将 127.0.0.1 请求转发后产生 502。
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        session_rows = await asyncio.gather(
            *(_run_session(client, semaphore, args, session) for session in sessions)
        )
    rows = [row for group in session_rows for row in group]
    metric_rows = [
        {"query_id": row["query_id"], **row["retrieval_metrics"]}
        for row in rows
        if row["status"] == "done"
    ]
    status = Counter(row["status"] for row in rows)
    by_user: dict[str, dict[str, Any]] = {}
    for session in sessions:
        user_rows = [row for row in rows if row["user_id"] == session["user_id"]]
        by_user[session["user_id"]] = {
            "session_id": session["session_id"],
            "textbook": session["textbook"],
            "turns": len(user_rows),
            "succeeded": sum(row["status"] == "done" for row in user_rows),
            "avg_latency_ms": round(sum(row["latency_ms"] for row in user_rows) / len(user_rows), 2),
        }
    return {
        "endpoint": args.url,
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "shuffle": args.shuffle,
        "users": len(sessions),
        "turns": len(rows),
        "concurrency": args.concurrency,
        "status_counts": dict(status),
        "succeeded": status.get("done", 0),
        "sessions": by_user,
        "per_turn": rows,
        "retrieval_summary": aggregate_metrics(metric_rows),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = asyncio.run(_run(args))
    output = args.output or args.dataset_root / "multi_user_agent_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"完成: {payload['succeeded']}/{payload['turns']} 轮，"
        f"{payload['users']} 个用户会话，结果已写入 {output}"
    )
    print(json.dumps(payload["status_counts"], ensure_ascii=False))
    return 0 if payload["succeeded"] == payload["turns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
