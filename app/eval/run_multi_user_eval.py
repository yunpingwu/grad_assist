"""多用户并发、多轮会话评测。

支持两种数据集形态：

1. 扁平题集（默认 ``data/eval/*/qa_set.json``）：每本教材拆成多个虚拟用户，
   同一用户的题目串行发送并复用随机生成的 session_id。
2. 对话集（含 ``conversations`` 键的 qa_set.json，如
   ``data/eval_multi_turn/qa_set.json``）：直接使用数据集中的 user_id、
   session_key 与逐轮 turn_role/anaphora，session_id 固定为
   ``session-{user_id}-{session_key}``，用于指代消歧与会话隔离评测。

示例::

    # 扁平题集
    python -m app.eval.run_multi_user_eval \\
        --users-per-book 5 --turns-per-user 10 --concurrency 10

    # 多轮对话集（200 轮 / 50 会话 / 25 虚拟用户）
    python -m app.eval.run_multi_user_eval \\
        --dataset-root data/eval_multi_turn --concurrency 10
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
    paths = sorted({root / "qa_set.json", *root.glob("*/qa_set.json")})
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"未找到评测集: {root / '*/qa_set.json'}")
    return paths


def _sessions_from_conversations(
    path: Path,
    raw: dict[str, Any],
    limit_per_book: int,
) -> list[dict[str, Any]]:
    """对话集：一个 (user_id, session_key) 即一个会话，session_id 稳定可复现。"""
    sessions: list[dict[str, Any]] = []
    remaining: dict[Any, float] = {}
    for conv in raw["conversations"]:
        book_key = conv.get("book_key")
        left = remaining.get(book_key, limit_per_book or math.inf)
        if left <= 0:
            continue
        turns = list(conv["turns"])
        if len(turns) > left:
            turns = turns[: int(left)]
        remaining[book_key] = left - len(turns)
        if not turns:
            continue
        user_id = conv["user_id"]
        session_key = conv["session_key"]
        book = raw.get("books", {}).get(conv.get("book_key"), {})
        sessions.append(
            {
                "user_id": user_id,
                "session_id": f"session-{user_id}-{session_key}",
                "textbook": book.get("textbook", raw.get("textbook", "")),
                "dataset": str(path),
                "topic": conv.get("topic", ""),
                "questions": turns,
            }
        )
    return sessions


def _build_user_sessions(args: argparse.Namespace) -> list[dict[str, Any]]:
    """按教材分配用户，保持每个用户的题目顺序以形成可复用会话。"""
    rng = random.Random(args.seed)
    sessions: list[dict[str, Any]] = []
    for book_index, path in enumerate(_discover_datasets(args.dataset_root), start=1):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "conversations" in raw:
            sessions.extend(_sessions_from_conversations(path, raw, args.limit_per_book))
            continue

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
    """发送一轮 SSE 请求；同一 session 的调用由上层串行调度。

    计时口径（三件事分开测，别把排队当成 Agent 慢）：
    - ``queue_ms``：进入本函数到首次抢到并发槽位的等待，只记第一次排队；
    - ``service_ms``：请求真正发出到 SSE 收尾的耗时，即 Agent 端到端服务时间
      （重试的退避与二次排队落在这一段，不再回摊到 ``queue_ms``）；
    - ``first_token_ms``：请求发出到收到第一个 ``token`` 事件（用户可见正文）的等待；
    - ``latency_ms``：``queue_ms + service_ms`` 的总时长，仅作向后兼容保留。
    """
    t_enter = time.perf_counter()
    t_send: float | None = None
    t_first_token: float | None = None
    result: dict[str, Any] = {
        "user_id": session["user_id"],
        "session_id": session["session_id"],
        "textbook": session["textbook"],
        "turn_index": turn_index,
        "query_id": question.get("id", ""),
        "question": question.get("question", ""),
        "turn_role": question.get("turn_role"),
        "anaphora": question.get("anaphora"),
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
                    if t_send is None:
                        t_send = time.perf_counter()
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
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
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

    t_end = time.perf_counter()
    t_send = t_send if t_send is not None else t_end
    result["queue_ms"] = round((t_send - t_enter) * 1000, 2)
    result["service_ms"] = round((t_end - t_send) * 1000, 2)
    result["first_token_ms"] = (
        round((t_first_token - t_send) * 1000, 2) if t_first_token is not None else None
    )
    result["latency_ms"] = round((t_end - t_enter) * 1000, 2)
    result["retrieval_metrics"] = compute_metrics(set(result["gold_chunk_ids"]), result["retrieved_chunk_ids"])
    if not result["request_ids"]:
        result.pop("request_ids")
    return result


def _percentile(values: list[float], ratio: float) -> float:
    """最近秩法分位数（p50 → 升序第 ``int(0.5*(n-1))`` 个），与人工复盘口径一致。"""
    ordered = sorted(values)
    return ordered[int(ratio * (len(ordered) - 1))]


def _latency_stats(values: list[float]) -> dict[str, float]:
    """一组耗时的 p50 / p90 / max / 均值；空集合返回零值便于报告横向比较。"""
    if not values:
        return {"turns": 0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "turns": len(values),
        "p50": round(_percentile(values, 0.5), 2),
        "p90": round(_percentile(values, 0.9), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
    }


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """分列汇总服务耗时、排队等待与首 token 延迟。

    ``service_ms`` 才是 Agent 单轮真实耗时；``queue_ms`` 反映 ``--concurrency``
    下的排队程度，两者相加等于向后兼容保留的 ``latency_ms``。
    """
    return {
        "turns": len(rows),
        "service_ms": _latency_stats([float(row["service_ms"]) for row in rows]),
        "queue_ms": _latency_stats([float(row["queue_ms"]) for row in rows]),
        "first_token_ms": _latency_stats(
            [float(row["first_token_ms"]) for row in rows if row.get("first_token_ms") is not None]
        ),
    }


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


def _role_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按轮次角色分组统计成功率与检索指标（开场/独立轮 vs 指代追问轮）。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("turn_role") is None:
            continue
        key = "followup" if row["turn_role"] == "followup" else "opening"
        groups.setdefault(key, []).append(row)
    summary: dict[str, Any] = {}
    for key, group in groups.items():
        done = [row for row in group if row["status"] == "done"]
        summary[key] = {
            "turns": len(group),
            "succeeded": len(done),
            "success_rate": round(len(done) / len(group), 4),
            "retrieval": aggregate_metrics(
                [{"query_id": row["query_id"], **row["retrieval_metrics"]} for row in done]
            ),
        }
    return summary


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    t_wall_start = time.perf_counter()
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
    by_session: dict[str, dict[str, Any]] = {}
    for session, group in zip(sessions, session_rows, strict=True):
        entry = {
            "user_id": session["user_id"],
            "session_id": session["session_id"],
            "textbook": session["textbook"],
            "turns": len(group),
            "succeeded": sum(row["status"] == "done" for row in group),
            "avg_service_ms": round(sum(row["service_ms"] for row in group) / len(group), 2),
            "avg_queue_ms": round(sum(row["queue_ms"] for row in group) / len(group), 2),
            "avg_latency_ms": round(sum(row["latency_ms"] for row in group) / len(group), 2),
        }
        if session.get("topic"):
            entry["topic"] = session["topic"]
        by_session[session["session_id"]] = entry
    return {
        "endpoint": args.url,
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "shuffle": args.shuffle,
        "users": len({session["user_id"] for session in sessions}),
        "sessions": len(sessions),
        "turns": len(rows),
        "concurrency": args.concurrency,
        "wall_ms": round((time.perf_counter() - t_wall_start) * 1000, 2),
        "status_counts": dict(status),
        "succeeded": status.get("done", 0),
        "session_results": by_session,
        "per_turn": rows,
        "retrieval_summary": aggregate_metrics(metric_rows),
        "by_turn_role": _role_summary(rows),
        "latency_summary": _latency_summary(rows),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = asyncio.run(_run(args))
    output = args.output or args.dataset_root / "multi_user_agent_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latency = payload["latency_summary"]
    print(
        f"完成: {payload['succeeded']}/{payload['turns']} 轮，"
        f"{payload['users']} 个用户 / {payload['sessions']} 个会话，"
        f"墙钟 {payload['wall_ms'] / 1000 / 60:.1f} 分钟，结果已写入 {output}"
    )
    print(
        f"单轮服务耗时 p50 {latency['service_ms']['p50'] / 1000:.1f}s / "
        f"p90 {latency['service_ms']['p90'] / 1000:.1f}s / "
        f"max {latency['service_ms']['max'] / 1000:.1f}s；"
        f"首 token p50 {latency['first_token_ms']['p50'] / 1000:.1f}s；"
        f"排队 p50 {latency['queue_ms']['p50'] / 1000:.1f}s（并发 {payload['concurrency']}）"
    )
    print(json.dumps(payload["status_counts"], ensure_ascii=False))
    return 0 if payload["succeeded"] == payload["turns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
