"""通过真实 studyservice Agent 流程并发评测 C 语言问答集。

用法（项目根目录）：

    .venv\\Scripts\\python.exe -m app.eval.run_agent_eval
    .venv\\Scripts\\python.exe -m app.eval.run_agent_eval \
        "data/eval/C语言程序设计（第五版）_(谭浩强)/qa_set.json" \
        --url http://127.0.0.1:8000/study/chat --concurrency 4

服务需要先启动。脚本消费 ``/study/chat`` 的 SSE 流，保存每题最终回答、
实际检索到的有序 chunk ID，以及基于评测集 gold_chunk_ids 的检索指标。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.eval.dataset import DEFAULT_DATASET_PATH, load_dataset
from app.eval.metrics import aggregate_metrics, compute_metrics

_CONSOLE_LOCK: asyncio.Lock | None = None

_DEFAULT_C_DATASET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "eval"
    / "C语言程序设计（第五版）_(谭浩强)"
    / "qa_set.json"
)


def _default_dataset_path() -> Path:
    """优先使用当前仓库中的 C 语言题集，不存在时回退到通用默认路径。"""
    candidates = [
        _DEFAULT_C_DATASET,
        DEFAULT_DATASET_PATH,
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并发调用 studyservice，评测真实 Agent 的 C 语言问答结果")
    parser.add_argument("dataset", nargs="?", type=Path, default=_default_dataset_path(), help="评测集 JSON 路径")
    parser.add_argument("--url", default="http://127.0.0.1:8000/study/chat", help="studyservice chat SSE 地址")
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数，默认 4")
    parser.add_argument("--timeout", type=float, default=300.0, help="单请求超时秒数，默认 300")
    parser.add_argument("--request-retries", type=int, default=3, help="临时 HTTP 错误重试次数，默认 3")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0 表示全部")
    parser.add_argument("--output", type=Path, default=None, help="结果 JSON 路径，默认写入评测集目录")
    parser.add_argument("--user-id", default="agent-eval", help="X-User-Id 请求头")
    parser.add_argument(
        "--interactive-clarify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="遇到 ask_clarify 时在控制台输入答案并续跑（默认开启）",
    )
    parser.add_argument("--max-clarify-rounds", type=int, default=3, help="单题最多澄清轮数，默认 3")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于 0")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.request_retries < 0:
        parser.error("--request-retries 不能为负数")
    if args.limit < 0:
        parser.error("--limit 不能为负数")
    if args.max_clarify_rounds < 1:
        parser.error("--max-clarify-rounds 必须大于 0")
    return args


def _merge_unique(target: list[str], values: list[str]) -> None:
    """按出现顺序合并字符串列表并去重。"""
    for value in values:
        if value and value not in target:
            target.append(value)


async def _prompt_text(prompt: str) -> str:
    """在线程池中读取 stdin，避免阻塞 asyncio 事件循环。"""
    try:
        return await asyncio.to_thread(input, prompt)
    except EOFError as exc:
        raise RuntimeError("标准输入已关闭，无法继续回答澄清问题") from exc


async def _collect_clarify_answers(questions: list[dict[str, Any]], query_id: str) -> list[dict[str, str]]:
    """串行收集一轮澄清答案；choice 支持输入序号、标签或自由文本。"""
    global _CONSOLE_LOCK
    if _CONSOLE_LOCK is None:
        _CONSOLE_LOCK = asyncio.Lock()

    async with _CONSOLE_LOCK:
        print(f"\n[{query_id}] Agent 需要澄清，请在控制台回答：", flush=True)
        answers: list[dict[str, str]] = []
        for index, question in enumerate(questions, start=1):
            question_id = str(question.get("id") or f"q{index}")
            text = question.get("question") or "请补充信息"
            options = question.get("options") or []
            if options:
                print(f"{index}. {text}", flush=True)
                for option_index, option in enumerate(options, start=1):
                    print(f"   {option_index}) {option.get('label', '')}", flush=True)
                while True:
                    raw = (await _prompt_text("请选择序号/标签，或直接输入答案: ")).strip()
                    if not raw:
                        print("请输入一个选项或答案。", flush=True)
                        continue
                    if raw.isdigit() and 1 <= int(raw) <= len(options):
                        answer = str(options[int(raw) - 1].get("method") or options[int(raw) - 1].get("label", ""))
                        break
                    matched = next(
                        (option for option in options if raw.casefold() == str(option.get("label", "")).casefold()),
                        None,
                    )
                    if matched is not None:
                        answer = str(matched.get("method") or matched.get("label", ""))
                        break
                    answer = raw
                    break
            else:
                hint = question.get("hint") or ""
                suffix = f"（提示：{hint}）" if hint else ""
                answer = await _prompt_text(f"{index}. {text}{suffix}\n请输入答案: ")
            answers.append({"id": question_id, "answer": answer})
        return answers


async def _request_leg(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    payload: dict[str, Any],
    user_id: str,
    request_retries: int = 3,
) -> dict[str, Any]:
    """请求一段 SSE（初次提问或一次澄清续跑）。"""
    leg: dict[str, Any] = {
        "status": "error",
        "answer_content": "",
        "retrieved_chunk_ids": [],
        "tool_names": [],
    }
    retry_statuses = {429, 502, 503, 504}
    for attempt in range(request_retries + 1):
        try:
            async with semaphore:
                async with client.stream("POST", url, json=payload, headers={"X-User-Id": user_id}) as response:
                    if response.status_code in retry_statuses and attempt < request_retries:
                        await response.aread()
                        delay = min(2**attempt, 10)
                        print(
                            f"请求返回 HTTP {response.status_code}，第 {attempt + 1}/{request_retries} 次重试，等待 {delay}s",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
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
                            leg["answer_content"] += str(event.get("content", ""))
                        elif event_type == "tool":
                            name = event.get("name")
                            if name and name not in leg["tool_names"]:
                                leg["tool_names"].append(name)
                        elif event_type == "done":
                            leg["status"] = "done"
                            leg["request_id"] = event.get("request_id")
                            leg["retrieved_chunk_ids"] = list(event.get("retrieved_chunk_ids") or [])
                            leg["answer_content"] = event.get("answer_content") or leg["answer_content"]
                        elif event_type == "ask_confirm":
                            leg["status"] = "ask_confirm"
                        elif event_type == "ask_clarify":
                            leg["status"] = "ask_clarify"
                            leg["clarify_questions"] = list(event.get("questions") or [])
                        elif event_type == "error":
                            leg["error"] = event.get("message", "studyservice 返回错误")
                    return leg
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code not in retry_statuses or attempt >= request_retries:
                raise
            delay = min(2**attempt, 10)
            print(
                f"请求返回 HTTP {status_code}，第 {attempt + 1}/{request_retries} 次重试，等待 {delay}s",
                flush=True,
            )
            await asyncio.sleep(delay)
    return leg


async def _run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    textbook: str,
    question: dict[str, Any],
    user_id: str,
    interactive_clarify: bool = True,
    max_clarify_rounds: int = 3,
    request_retries: int = 3,
) -> dict[str, Any]:
    """调用一题并消费 SSE；遇到澄清时可在控制台回答后续跑。"""
    query_id = str(question.get("id") or uuid.uuid4().hex)
    session_id = f"eval-{query_id}-{uuid.uuid4().hex[:12]}"
    result: dict[str, Any] = {
        "query_id": query_id,
        "question": question.get("question", ""),
        "reference_answer": question.get("reference_answer", ""),
        "answer_points": question.get("answer_points", []),
        "gold_chunk_ids": list(dict.fromkeys(question.get("gold_chunk_ids", []))),
        "session_id": session_id,
        "status": "error",
        "retrieved_chunk_ids": [],
        "answer_content": "",
        "tool_names": [],
        "request_ids": [],
        "clarify_rounds": 0,
    }
    payload = {"textbook_name": textbook, "query": result["question"], "session_id": session_id}

    try:
        for _ in range(max_clarify_rounds + 1):
            leg = await _request_leg(client, semaphore, url, payload, user_id, request_retries)
            for name in leg["tool_names"]:
                if name not in result["tool_names"]:
                    result["tool_names"].append(name)
            _merge_unique(result["retrieved_chunk_ids"], leg["retrieved_chunk_ids"])
            if leg.get("request_id"):
                result["request_ids"].append(leg["request_id"])
                result["request_id"] = leg["request_id"]
            if leg.get("answer_content"):
                result["answer_content"] += leg["answer_content"]

            if leg["status"] == "done":
                result["status"] = "done"
                break
            if leg["status"] != "ask_clarify":
                result["status"] = leg["status"]
                if leg.get("error"):
                    result["error"] = leg["error"]
                break

            result["status"] = "ask_clarify"
            result["clarify_questions"] = leg.get("clarify_questions", [])
            if not interactive_clarify:
                break
            if result["clarify_rounds"] >= max_clarify_rounds:
                result["error"] = f"超过最大澄清轮数（{max_clarify_rounds}）"
                break
            answers = await _collect_clarify_answers(result["clarify_questions"], query_id)
            result["clarify_rounds"] += 1
            payload = {
                "textbook_name": textbook,
                "query": "",
                "session_id": session_id,
                "clarify_answers": answers,
            }
    except Exception as exc:
        # 单题失败写入结果并继续其它并发题目，便于定位偶发超时/模型错误。
        result["error"] = f"{type(exc).__name__}: {exc}"

    if not result["request_ids"]:
        result.pop("request_ids")
    result["retrieval_metrics"] = compute_metrics(set(result["gold_chunk_ids"]), result["retrieved_chunk_ids"])
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_dataset(args.dataset)
    questions = data["questions"][: args.limit or None]
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 30.0))
    semaphore = asyncio.Semaphore(args.concurrency)
    # trust_env=False：本机启用了系统代理(127.0.0.1:7892)，默认 trust_env=True 会把
    # 对 127.0.0.1:8000 的请求也转发给代理，导致 502 Bad Gateway；本地评测必须直连。
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        tasks = [
            _run_one(
                client,
                semaphore,
                args.url,
                data["textbook"],
                question,
                args.user_id,
                interactive_clarify=args.interactive_clarify,
                max_clarify_rounds=args.max_clarify_rounds,
                request_retries=args.request_retries,
            )
            for question in questions
        ]
        rows = await asyncio.gather(*tasks)

    metric_rows = [
        {"query_id": row["query_id"], **row["retrieval_metrics"]}
        for row in rows
        if row["status"] == "done"
    ]
    return {
        "textbook": data["textbook"],
        "dataset": str(args.dataset),
        "endpoint": args.url,
        "concurrency": args.concurrency,
        "total": len(rows),
        "succeeded": sum(row["status"] == "done" for row in rows),
        "per_query": rows,
        "retrieval_summary": aggregate_metrics(metric_rows),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dataset.exists():
        print(f"评测集不存在: {args.dataset}", file=sys.stderr)
        return 2
    payload = asyncio.run(_run(args))
    output = args.output or args.dataset.parent / "agent_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"完成: {payload['succeeded']}/{payload['total']} 题，"
        f"结果已写入 {output}"
    )
    print(json.dumps(payload["retrieval_summary"], ensure_ascii=False))
    return 0 if payload["succeeded"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
