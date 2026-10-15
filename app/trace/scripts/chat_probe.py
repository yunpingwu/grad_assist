"""十轮连续对话探针：对已启动的服务跑一轮真机多轮对话，再把 steps 时间线打成人读摘要。

比 cache_probe 简单得多——不问「命中涨没涨」，只干两件事：发一轮固定对话、把本 session 的
落盘记录按时序打出来（LLM 步与工具步穿插），用来肉眼检查观测实体在真实链路上的形状。
问题集按默认教材（计算机组成原理·Cache 主题）写；换教材时改 QUESTIONS 或调小 --turns。

用法（服务需已启动）::

    uv run python -m app.trace.scripts.chat_probe                 # 默认 10 轮，每 3 轮一次 deep
    uv run python -m app.trace.scripts.chat_probe --turns 4 --deep-every 2 --report out.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
_DECODER = json.JSONDecoder()

# 固定对话：围绕一个主题递进（追问→举例→对比→出题→整理成文件），覆盖纯问答、深度检索与写盘确认
QUESTIONS = [
    "你好，我想系统复习一下 Cache 这部分的考点",
    "Cache 和主存之间的三种地址映射方式分别是什么？先给我一个整体对比",
    "直接映射具体怎么定位主存块？用一个小例子说明",
    "那 4 路组相联和直接映射比，优势到底在哪？",
    "组相联里组满了要替换，常用的替换算法有哪些，各自的取舍是什么",
    "结合上面说的，出 3 道选择题考考我",
    "第 2 题的选项 C 为什么不对？",
    "把这几轮关于 Cache 映射与替换的要点整理成一份复习提纲，存成文件",
    "刚才的提纲里再补一节：写策略（写直达与写回）的区别",
    "最后帮我用一句话总结今天复习的主线",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多轮连续对话跑一遍，把 steps 时间线打成人读摘要")
    parser.add_argument("--url", default="http://127.0.0.1:8000/study/chat")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="用于 /list 解析教材全名")
    parser.add_argument("--user", default="chatprobe", help="X-User-Id（隔离用）")
    parser.add_argument("--textbook", default="计算机组成原理", help="教材名子串（从 /list 匹配全名）")
    parser.add_argument("--turns", type=int, default=10, help="跑前 N 轮（问题集见 QUESTIONS）")
    parser.add_argument("--deep-every", type=int, default=3, help="每 N 轮走一次深度检索；0 表示全不 deep")
    parser.add_argument("--session", default=None, help="复用已有会话 ID；缺省新建")
    parser.add_argument("--sleep", type=float, default=1.0, help="轮间隔秒数")
    parser.add_argument("--timeout", type=float, default=300.0, help="单轮 SSE 读流超时秒数")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--report", type=Path, default=None, help="把摘要另存一份到该文件（UTF-8）")
    args = parser.parse_args(argv)
    args.turns = max(1, min(args.turns, len(QUESTIONS)))
    return args


async def _resolve_textbook(client: httpx.AsyncClient, base: str, match: str) -> str:
    """从 /list 里按子串找教材全名（教材名带出版社/来源后缀，手敲不现实）。"""
    resp = await client.get(f"{base}/list", params={"page": 1, "page_size": 100})
    resp.raise_for_status()
    items = resp.json().get("items", [])
    hit = next((item["textbook_name"] for item in items if match in item["textbook_name"]), None)
    if hit is None:
        names = "\n".join(f"  - {item['textbook_name']}" for item in items)
        raise SystemExit(f"教材库里没有匹配 {match!r} 的教材，现有：\n{names}")
    return hit


async def _send_turn(client: httpx.AsyncClient, args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """发一轮并读完 SSE：只收摘要里要用的字段（状态、request_id、工具数、首字延迟、澄清问题）。"""
    out: dict[str, Any] = {"status": "no_done", "request_id": None, "tools": 0, "first_token_ms": None,
                           "answer_chars": 0, "questions": [], "error": None}
    t0 = time.perf_counter()
    async with client.stream("POST", args.url, json=payload, headers={"X-User-Id": args.user}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "token":
                out["answer_chars"] += len(event.get("content") or "")
                if out["first_token_ms"] is None:
                    out["first_token_ms"] = round((time.perf_counter() - t0) * 1000)
            elif etype == "tool":
                out["tools"] += 1
            elif etype == "ask_confirm":
                out["status"] = "ask_confirm"
            elif etype == "ask_clarify":
                out["status"] = "ask_clarify"
                out["questions"] = event.get("questions") or []
            elif etype == "done":
                out["status"] = "done"
                out["request_id"] = event.get("request_id")
            elif etype == "error":
                out["status"] = "error"
                out["error"] = event.get("message")
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


async def _ask_with_resume(client: httpx.AsyncClient, args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """发一轮；撞上写盘确认自动放行、撞上澄清反问按首个建议选项作答（续跑不占轮次）。"""
    out = await _send_turn(client, args, payload)
    for _ in range(2):  # 同一轮的续跑最多两跳（确认为一跳，澄清+确认两跳封顶）
        if out["status"] == "ask_confirm":
            print("        ↳ 撞上写盘确认，自动放行续跑")
            out = await _send_turn(client, args, {"textbook_name": payload["textbook_name"],
                                                  "query": "", "session_id": payload["session_id"],
                                                  "decision": "approve"})
        elif out["status"] == "ask_clarify":
            print(f"        ↳ 撞上澄清反问（{len(out['questions'])} 问），按首个建议选项作答")
            answers = []
            for q in out["questions"]:
                options = q.get("options") or []
                answers.append({"id": q.get("id"), "answer": (options[0].get("method") if options else "你来决定，按最合理的默认走")})
            out = await _send_turn(client, args, {"textbook_name": payload["textbook_name"],
                                                  "query": "", "session_id": payload["session_id"],
                                                  "clarify_answers": answers})
        else:
            break
    return out


def _read_records(logs_dir: Path) -> list[dict[str, Any]]:
    """读回全部落盘记录：``logs/<日期>/metrics.log`` 与历史 ``logs/metrics.log``（缩进 JSON 块拼接，只能流式解）。"""
    records: list[dict[str, Any]] = []
    for path in [logs_dir / "metrics.log", *sorted(logs_dir.glob("*/metrics.log"))]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        index = 0
        while index < len(text):
            while index < len(text) and text[index] in " \n\r\t":
                index += 1
            if index >= len(text):
                break
            obj, index = _DECODER.raw_decode(text, index)
            records.append(obj)
    return records


def _steps_lines(record: dict[str, Any]) -> list[str]:
    """把一条记录的 steps 时间线打成人读几行：LLM 步一行、工具步两行（含 meta 摘要）。"""
    lines: list[str] = []
    for step in record.get("steps") or []:
        seq = step.get("seq")
        if step.get("kind") == "llm":
            tokens = step.get("tokens") or {}
            bits = [f"{seq:>3} llm ", f"role={step.get('role') or 'agent':<7}"]
            bits.append(f"in={tokens.get('in')} cached={tokens.get('cached')} out={tokens.get('out')}")
            if step.get("parent_id"):
                bits.append(f"parent={step['parent_id'][:8]}…")
            if step.get("first_ms") is not None:
                bits.append(f"首chunk={step['first_ms']:.0f}ms")
            bits.append(f"共{step.get('ms')}ms")
            if step.get("status") != "ok":
                bits.append(f"[{step.get('status')}:{step.get('error')}]")
            lines.append("  " + "  ".join(bits))
            for call in (step.get("output") or {}).get("tool_calls") or []:
                flag = "" if call.get("resolved") else "  ← 未执行（被拦/中断）"
                # 参数截断显示：write_file 的正文动辄几千字，全文进摘要会把时间线淹掉
                args = json.dumps(call.get("args"), ensure_ascii=False)
                lines.append(f"        → {call.get('name')}({args[:90]}{'…' if len(args) > 90 else ''}){flag}")
        else:
            meta = step.get("meta") or {}
            phases = meta.get("phases")
            summary = {k: v for k, v in meta.items() if k != "phases"}
            if isinstance(summary.get("chunk_ids"), list):
                summary["chunk_ids"] = f"×{len(summary['chunk_ids'])}"
            meta_text = json.dumps(summary, ensure_ascii=False)[:140] if summary else ""
            lines.append(f"  {seq:>3} tool {step.get('name')}  {step.get('ms')}ms  [{step.get('status')}]  {meta_text}")
            if phases:
                lines.append(f"        phases: {json.dumps(phases, ensure_ascii=False)}")
    return lines


def _print_report(records: list[dict[str, Any]], session_id: str) -> str:
    """打两份视图：先一张跨轮总表，再逐轮的 steps 时间线全文。返回全文（供 --report 落文件）。"""
    out: list[str] = []
    out.append(f"=== 会话 {session_id} · {len(records)} 条记录 · 落盘于 logs/<日期>/metrics.log ===")
    out.append("")
    header = f"{'轮':>2}  {'LLM':>3} {'工具':>4} {'输入':>8} {'命中':>8} {'输出':>7} {'命中率':>6} {'成本':>8} {'总耗时':>7}  问题"
    out.append(header)
    totals = {"in": 0, "cached": 0, "out": 0, "cost": 0.0, "ms": 0.0}
    for i, record in enumerate(records, 1):
        in_tok = record.get("prompt_tokens", 0)
        cached = record.get("cached_tokens", 0)
        out_tok = record.get("completion_tokens", 0)
        llm_steps = [s for s in record.get("steps") or [] if s.get("kind") == "llm"]
        tool_steps = [s for s in record.get("steps") or [] if s.get("kind") == "tool"]
        total_ms = (record.get("stage_ms") or {}).get("total_ms") or 0.0
        ratio = f"{cached / in_tok:.0%}" if in_tok else "-"
        # 空 query 的记录是写盘确认/澄清的续跑请求（与上一轮同一线程），标出来免得读成新问题
        query = (record.get("query") or "（续跑·无新问题）")[:34]
        error = f"  [error={record['error_type']}]" if record.get("error_type") else ""
        cost = record.get("turn_cost", 0)
        out.append(
            f"{i:>2}  {len(llm_steps):>3} {len(tool_steps):>4} {in_tok:>8} {cached:>8} {out_tok:>7} "
            f"{ratio:>6} {cost:>8.0f} {total_ms:>6.0f}ms  {query}{error}"
        )
        totals["in"] += in_tok
        totals["cached"] += cached
        totals["out"] += out_tok
        totals["cost"] += record.get("turn_cost", 0.0)
        totals["ms"] += total_ms
    ratio = f"{totals['cached'] / totals['in']:.0%}" if totals["in"] else "-"
    out.append(
        f"{'合计':>2}      {'':>4} {totals['in']:>8} {totals['cached']:>8} {totals['out']:>7} "
        f"{ratio:>6} {totals['cost']:>8.0f} {totals['ms']:>6.0f}ms"
    )
    out.append("")
    for i, record in enumerate(records, 1):
        out.append(f"[轮 {i}] {record.get('query')}")
        out.extend(_steps_lines(record) or ["  （无 steps —— 观测没挂上？）"])
        out.append("")
    return "\n".join(out)


async def main() -> None:
    args = _parse_args()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        textbook = await _resolve_textbook(client, args.base, args.textbook)
        session_id = args.session or f"chatprobe-{uuid.uuid4().hex[:8]}"
        print(f"教材：{textbook}\n会话：{session_id}\n开始跑 {args.turns} 轮（每 {args.deep_every} 轮一次 deep）…\n")

        cutoff = datetime.now().astimezone().isoformat(timespec="milliseconds")
        for i, question in enumerate(QUESTIONS[: args.turns], 1):
            deep = args.deep_every > 0 and i % args.deep_every == 0
            out = await _ask_with_resume(client, args, {
                "textbook_name": textbook, "query": question, "deep": deep, "session_id": session_id,
            })
            first = f"{out['first_token_ms'] / 1000:.1f}s" if out["first_token_ms"] else "-"
            print(f"  轮 {i:>2} [{'deep' if deep else 'fast'}] {out['status']:<12} "
                  f"工具×{out['tools']}  首字 {first}  总 {out['ms'] / 1000:.1f}s  {question[:24]}")
            if out["status"] == "error":
                print(f"        ⚠ {out['error']}")
            await asyncio.sleep(args.sleep)

    # 按会话 + 开跑时刻切出本轮的记录，时序即请求完成序。
    # 不按 request_id 对：被写盘确认拦下的请求不发 done 事件（拿不到 id），而那正是要看的 resolved=false 那条
    records = [
        r for r in _read_records(args.logs_dir)
        if r.get("session_id") == session_id and str(r.get("ts", "")) >= cutoff
    ]
    records.sort(key=lambda r: str(r.get("ts", "")))

    report = _print_report(records, session_id)
    print("\n" + report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
        print(f"\n摘要已另存：{args.report}")


if __name__ == "__main__":
    asyncio.run(main())
