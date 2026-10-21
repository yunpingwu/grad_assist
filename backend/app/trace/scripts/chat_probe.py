"""多会话对话探针：按问题集跑多轮对话，汇总缓存命中、压缩事件与输入峰值。

服务需已启动；想看压缩事件需用小窗口启动服务（触发线 = 窗口 × 50%）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from app.core import logger

ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_QUESTIONS = _SCRIPT_DIR / "probe_questions.json"
_DECODER = json.JSONDecoder()


# ── 问题集 ───────────────────────────────────────────────


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """读问题集（schema=chat_probe_questions/v1），校验每段会话至少有 1 题。

    问题集里的字段缺一个都跑不起来（教材子串错只会等到 /list 匹配时才报），不如在这里一次说清。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "chat_probe_questions/v1":
        raise SystemExit(f"{path} 不是 chat_probe_questions/v1（实际 {data.get('schema')!r}）")
    conversations = data.get("conversations") or []
    for index, conv in enumerate(conversations, 1):
        missing = [key for key in ("name", "textbook", "questions") if not conv.get(key)]
        if missing:
            raise SystemExit(f"{path} 第 {index} 段会话缺字段：{missing}")
    if not conversations:
        raise SystemExit(f"{path} 里没有任何会话")
    return conversations


def select_conversations(conversations: list[dict[str, Any]], only: str | None) -> list[tuple[int, dict[str, Any]]]:
    """按 ``--only``（1 起的序号或名字子串，逗号分隔）挑会话；缺省全跑。

    Returns:
        ``(1 起序号, 会话)`` 列表——序号进 session_id，报告里对得上问题集。
    """
    numbered = list(enumerate(conversations, 1))
    if not only:
        return numbered
    picked: list[tuple[int, dict[str, Any]]] = []
    for token in (part.strip() for part in only.split(",")):
        if not token:
            continue
        if token.isdigit():
            hit = [item for item in numbered if item[0] == int(token)]
        else:
            hit = [item for item in numbered if token in item[1]["name"]]
        if not hit:
            raise SystemExit(f"--only {token!r} 没匹配到任何会话（可用序号或名字子串）")
        picked.extend(item for item in hit if item not in picked)
    return picked


# ── 与服务的交互（SSE）────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按问题集跑多轮对话，汇总缓存命中与上下文治理读数")
    parser.add_argument("--url", default="http://127.0.0.1:8000/study/chat")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="用于 /list 解析教材全名")
    parser.add_argument("--user", default="chatprobe", help="X-User-Id（隔离用）")
    parser.add_argument("--questions", type=Path, default=_DEFAULT_QUESTIONS, help="问题集 JSON 路径")
    parser.add_argument("--only", default=None, help="只跑这几段：序号或名字子串，逗号分隔（如 1,3 或 图）")
    parser.add_argument("--turns", type=int, default=0, help="每段只跑前 N 轮；0 表示跑满")
    parser.add_argument("--deep-every", type=int, default=3, help="每 N 轮走一次深度检索；0 表示全不 deep")
    parser.add_argument("--sleep", type=float, default=1.0, help="轮间隔秒数")
    parser.add_argument("--timeout", type=float, default=300.0, help="单轮 SSE 读流超时秒数")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--timeline", action="store_true", help="额外打印每轮的 steps 时间线（很长）")
    parser.add_argument("--report", type=Path, default=None, help="把报告另存一份到该文件（UTF-8）")
    return parser.parse_args(argv)


def build_client(timeout: float) -> httpx.AsyncClient:
    """探针用的 HTTP 客户端：**不走代理**。

    httpx 默认 ``trust_env=True``，在 Windows 上会读注册表里的系统代理（不是环境变量），
    于是发往 127.0.0.1 的请求被代理截走、返回 502（真机踩过：curl 200、httpx 502，同一台服务）。
    探针只跟本机服务说话，关掉代理信任；若哪天要探远端服务再另加参数。
    """
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


async def _resolve_textbook(client: httpx.AsyncClient, base: str, match: str) -> str:
    """从 /list 里按子串找教材全名（教材名带出版社/来源后缀，手敲不现实）。

    匹配不到抛 ``LookupError``（由调用方决定跳过还是退出）：一轮探针跑二十来分钟，
    一段会话的教材名写错不该把其余几段的报告也带走；子串撞上多本时取 /list 的第一本并告警。
    """
    # 服务刚起时 Milvus 连接还没就绪，/list 会 502（真机踩过：启动日志里 Milvus 连接比
    # "Application startup complete" 晚 1 秒）：5xx 重试几次再放弃，别让一次抢跑废掉整轮
    for _ in range(3):
        resp = await client.get(f"{base}/list", params={"page": 1, "page_size": 100})
        if resp.status_code < 500:
            break
        await asyncio.sleep(2)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    hits = [item["textbook_name"] for item in items if match in item["textbook_name"]]
    if not hits:
        names = "\n".join(f"  - {item['textbook_name']}" for item in items)
        raise LookupError(f"教材库里没有匹配 {match!r} 的教材，现有：\n{names}")
    if len(hits) > 1:
        logger.warning(f"教材子串 {match!r} 匹配到 {len(hits)} 本，取第一本：{hits[0]}（其余：{hits[1:]}）")
    return hits[0]


async def _send_turn(client: httpx.AsyncClient, args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """发一轮并读完 SSE：只收下面要用到的字段（状态、request_id、工具数、首字延迟、澄清问题）。"""
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
            print("        -> 撞上写盘确认，自动放行续跑")
            out = await _send_turn(client, args, {"textbook_name": payload["textbook_name"],
                                                  "query": "", "session_id": payload["session_id"],
                                                  "decision": "approve"})
        elif out["status"] == "ask_clarify":
            print(f"        -> 撞上澄清反问（{len(out['questions'])} 问），按首个建议选项作答")
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


# ── 记录读取与统计 ────────────────────────────────────────


def read_records(logs_dir: Path) -> list[dict[str, Any]]:
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


def session_records(records: list[dict[str, Any]], session_id: str, cutoff: str) -> list[dict[str, Any]]:
    """切出某会话本轮的记录（会话 + 开跑时刻双条件），按时序返回。

    不按 request_id 对：被写盘确认拦下的请求不发 done 事件（拿不到 id），而那正是要看的那条。
    """
    picked = [r for r in records if r.get("session_id") == session_id and str(r.get("ts", "")) >= cutoff]
    picked.sort(key=lambda r: str(r.get("ts", "")))
    return picked


def _tokens(record: dict[str, Any]) -> tuple[int, int, int]:
    """一条记录的（输入, 命中, 输出）token。"""
    return record.get("prompt_tokens", 0), record.get("cached_tokens", 0), record.get("completion_tokens", 0)


def _ratio(cached: int, prompt: int) -> float | None:
    return cached / prompt if prompt else None


def session_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """一段会话的汇总量：轮数、token、命中率（整体/首轮/稳态）、峰值输入、成本、压缩事件。

    「轮」只数带问题的记录；写盘确认等续跑（query 为空）单独计，但 token 与成本**照算**——
    钱是实打实花掉的，只把轮数口径分开。
    """
    question_records = [r for r in records if r.get("query")]
    totals = {"prompt": 0, "cached": 0, "out": 0, "cost": 0.0}
    peak = 0
    for record in records:
        prompt, cached, out = _tokens(record)
        totals["prompt"] += prompt
        totals["cached"] += cached
        totals["out"] += out
        totals["cost"] += record.get("turn_cost", 0.0)
        peak = max(peak, prompt)
    first = question_records[0] if question_records else {}
    steady = question_records[1:]
    first_prompt, first_cached, _ = _tokens(first) if first else (0, 0, 0)
    steady_prompt = sum(_tokens(r)[0] for r in steady)
    steady_cached = sum(_tokens(r)[1] for r in steady)
    compactions = [r["compacted"] for r in records if r.get("compacted")]
    return {
        "turns": len(question_records),
        "resumes": len(records) - len(question_records),
        "records": records,
        "total_prompt": totals["prompt"],
        "total_cached": totals["cached"],
        "total_out": totals["out"],
        "cost": totals["cost"],
        "hit": _ratio(totals["cached"], totals["prompt"]),
        "first_hit": _ratio(first_cached, first_prompt),
        "steady_hit": _ratio(steady_cached, steady_prompt),
        "peak_prompt": peak,
        "compactions": compactions,
    }


def turn_hit_curve(stats_by_index: dict[int, dict[str, Any]]) -> list[tuple[int, float, int]]:
    """跨会话的「第 N 轮命中率」曲线：(轮位, 平均命中率, 参与会话数)。

    按轮位而不是按时间对齐：第 1 轮永远没有跨轮缓存，第 2 轮起才是治理效果说话的地方。
    """
    curve: list[tuple[int, float, int]] = []
    turns = {index: [r for r in stat["records"] if r.get("query")] for index, stat in stats_by_index.items()}
    for position in range(1, max((len(v) for v in turns.values()), default=0) + 1):
        ratios = []
        for records in turns.values():
            if len(records) < position:
                continue
            prompt, cached, _ = _tokens(records[position - 1])
            ratio = _ratio(cached, prompt)
            if ratio is not None:
                ratios.append(ratio)
        if ratios:
            curve.append((position, sum(ratios) / len(ratios), len(ratios)))
    return curve


# ── 报告 ────────────────────────────────────────────────


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


def _pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "-"


def _turn_rows(records: list[dict[str, Any]]) -> list[str]:
    """逐轮明细表（含续跑记录，query 为空的那种标出来免得读成新问题）。"""
    rows = []
    for i, record in enumerate(records, 1):
        prompt, cached, out = _tokens(record)
        llm_steps = [s for s in record.get("steps") or [] if s.get("kind") == "llm"]
        tool_steps = [s for s in record.get("steps") or [] if s.get("kind") == "tool"]
        total_ms = (record.get("stage_ms") or {}).get("total_ms") or 0.0
        query = (record.get("query") or "（续跑·无新问题）")[:30]
        error = f"  [error={record['error_type']}]" if record.get("error_type") else ""
        compact = ""
        if record.get("compacted"):
            info = record["compacted"]
            compact = f"  [压缩] {info.get('dropped_msgs')}条/{info.get('dropped_tokens')}tok({info.get('trigger_source')})"
        rows.append(
            f"{i:>3}  {len(llm_steps):>3} {len(tool_steps):>4} {prompt:>8} {cached:>8} {out:>7} "
            f"{_pct(_ratio(cached, prompt)):>7} {record.get('turn_cost', 0):>8.0f} {total_ms:>7.0f}ms  {query}{error}{compact}"
        )
    return rows


def build_report(named_stats: list[tuple[int, str, dict[str, Any]]], timeline: bool, total: int | None = None) -> str:
    """生成三张表、压缩事件与跨会话命中率曲线。"""
    if not named_stats:
        return "没有可汇总的会话：全部被跳过或没跑成（控制台里有各段的 [!] 原因）。"
    out: list[str] = []
    out.append("=" * 100)
    if total is not None and len(named_stats) < total:
        out.append(f"[!] 计划 {total} 段会话，实际完成 {len(named_stats)} 段（缺的段见控制台 [!] 行）")
    header = f"{'轮':>3}  {'LLM':>3} {'工具':>4} {'输入':>8} {'命中':>8} {'输出':>7} {'命中率':>7} {'成本':>8} {'总耗时':>8}  问题"
    for index, name, stat in named_stats:
        out.append("")
        out.append(f"── 会话 {index:>2} · {name} ──")
        out.append(header)
        out.extend(_turn_rows(stat["records"]))

    out.append("")
    out.append("=" * 100)
    out.append("会话汇总（命中率 = 命中 token / 输入 token；稳态 = 第 2 轮起）")
    out.append(f"{'会话':>4}  {'轮':>3} {'续跑':>4} {'输入':>9} {'命中':>9} {'整体':>7} "
               f"{'首轮':>7} {'稳态':>7} {'峰值输入':>9} {'成本':>9}  名字")
    grand = {"prompt": 0, "cached": 0, "cost": 0.0}
    for index, name, stat in named_stats:
        grand["prompt"] += stat["total_prompt"]
        grand["cached"] += stat["total_cached"]
        grand["cost"] += stat["cost"]
        out.append(
            f"{index:>4}  {stat['turns']:>3} {stat['resumes']:>4} {stat['total_prompt']:>9} {stat['total_cached']:>9} "
            f"{_pct(stat['hit']):>10} {_pct(stat['first_hit']):>7} {_pct(stat['steady_hit']):>7} "
            f"{stat['peak_prompt']:>9} {stat['cost']:>9.0f}  {name}"
        )
    out.append(
        f"{'合计':>4}  {'':>3} {'':>4} {grand['prompt']:>9} {grand['cached']:>9} "
        f"{_pct(_ratio(grand['cached'], grand['prompt'])):>10} {'':>7} {'':>7} {'':>9} {grand['cost']:>9.0f}"
    )

    stats_by_index = {index: stat for index, _, stat in named_stats}
    curve = turn_hit_curve(stats_by_index)
    if curve:
        out.append("")
        out.append("跨会话命中率曲线（按轮位平均；第 1 轮没有跨轮缓存，从第 2 轮起才看得到治理效果）")
        bars = " ".join(f"{position}:{ratio:.0%}" for position, ratio, _ in curve)
        out.append("  " + bars)
        out.append("  轮位 → 命中率（括号内为参与平均的会话数）：")
        for position, ratio, n in curve:
            out.append(f"  {position:>3}  {'█' * int(ratio * 40):<40} {ratio:>5.0%}  ({n})")

    compactions = [(index, name, info) for index, name, stat in named_stats for info in stat["compactions"]]
    out.append("")
    if compactions:
        out.append(f"会话压缩事件（{len(compactions)} 次）：触发即一次跨轮前缀作废 + 一次摘要调用，故要看出现在哪几轮")
        for index, name, info in compactions:
            out.append(
                f"  会话 {index} {name}：触发读数 {info.get('trigger_tokens')}（{info.get('trigger_source')}）"
                f" → 压掉 {info.get('dropped_msgs')} 条 / ≈{info.get('dropped_tokens')} token，保留 {info.get('kept_msgs')} 条"
            )
    else:
        out.append(
            "本轮没有触发会话压缩。触发线 = 模型窗口 × 50%（服务端启动时的 MODEL/DEEPSEEK_CONTEXT_WINDOW），"
            "默认 50w 窗口 → 25w 触发，十来轮的会话够不着；要看压缩，用小窗口起服务："
            "DEEPSEEK_CONTEXT_WINDOW=20000 uv run python -m app.main"
        )

    if timeline:
        out.append("")
        out.append("=" * 100)
        out.append("逐轮 steps 时间线")
        for index, name, stat in named_stats:
            for i, record in enumerate(stat["records"], 1):
                out.append(f"[会话 {index} {name} · 轮 {i}] {record.get('query') or '（续跑）'}")
                out.extend(_steps_lines(record) or ["  （无 steps —— 观测没挂上？）"])
                out.append("")
    return "\n".join(out)


# ── 主流程 ──────────────────────────────────────────────


async def main() -> None:
    # 控制台编码兜底：GBK 终端打不出装饰字符（箭头、警告符之类），print 会直接抛
    # UnicodeEncodeError 把整轮探针打死（真机踩过）。打不出的字符降级成 ?，不中断。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = _parse_args()
    conversations = select_conversations(load_conversations(args.questions), args.only)
    total_turns = sum(min(len(c["questions"]), args.turns or len(c["questions"])) for _, c in conversations)
    print(f"问题集：{args.questions}\n会话 {len(conversations)} 段 · 共 {total_turns} 轮（每 {args.deep_every} 轮一次 deep）\n")

    named_stats: list[tuple[int, str, dict[str, Any]]] = []
    try:
        async with build_client(args.timeout) as client:
            for index, conv in conversations:
                try:
                    textbook = await _resolve_textbook(client, args.base, conv["textbook"])
                except LookupError as exc:
                    # 一段会话的教材名写错不拖垮整轮：跳过它，剩下的照跑，报告里点明缺了哪段
                    print(f"   [!] 跳过会话 {index}·{conv['name']}：{exc}\n")
                    continue
                session_id = f"chatprobe-{index:02d}-{uuid.uuid4().hex[:6]}"
                questions = conv["questions"][: args.turns or len(conv["questions"])]
                print(f"── 会话 {index} · {conv['name']} ──\n   教材：{textbook}\n   会话：{session_id}（{len(questions)} 轮）")
                cutoff = datetime.now().astimezone().isoformat(timespec="milliseconds")
                for turn, question in enumerate(questions, 1):
                    deep = args.deep_every > 0 and turn % args.deep_every == 0
                    out = await _ask_with_resume(client, args, {
                        "textbook_name": textbook, "query": question, "deep": deep, "session_id": session_id,
                    })
                    first = f"{out['first_token_ms'] / 1000:.1f}s" if out["first_token_ms"] else "-"
                    print(f"   轮 {turn:>2} [{'deep' if deep else 'fast'}] {out['status']:<12} "
                          f"工具×{out['tools']}  首字 {first}  总 {out['ms'] / 1000:.1f}s  {question[:22]}")
                    if out["status"] == "error":
                        print(f"        [!] {out['error']}")
                    await asyncio.sleep(args.sleep)
                named_stats.append((index, conv["name"], session_stats(session_records(read_records(args.logs_dir), session_id, cutoff))))
                print()
    finally:
        # 报告放 finally：跑二十分钟的探针不能因为一段会话出错就一份数据都不留
        report = build_report(named_stats, args.timeline, total=len(conversations))
        print(report)
        if args.report:
            args.report.write_text(report, encoding="utf-8")
            print(f"\n报告已另存：{args.report}")


if __name__ == "__main__":
    asyncio.run(main())
