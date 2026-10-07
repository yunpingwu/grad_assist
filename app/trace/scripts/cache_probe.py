"""提示词前缀缓存探针：单 session 串行跑完固定 10 轮对话，再从 metrics.log 取回逐次 LLM 调用的 (role, input, cached)。

服务需先起来（默认 ``http://127.0.0.1:8000/study/chat``），对话内容固定在
``data/trace/prefix_cache_dialogue.json``，两次跑只改 ``--tag``::

    uv run python -m app.trace.scripts.cache_probe --tag before
    # 当前树（system prompt 只留全局静态壳；教材/意图块/意图注/任务目标拼成「本轮指令」
    # SystemMessage 在轮首落进 state 一次）跑一遍，重启服务后
    uv run python -m app.trace.scripts.cache_probe --tag after
    uv run python -m app.trace.scripts.cache_probe --compare \\
        data/trace/runs/before_*.jsonl data/trace/runs/after_*.jsonl

盯的数只有一个：**每轮首次 agent 调用的 cached**。它按 ``llm_calls_detail`` 里 role 为空的那条取——
一轮里排在最前的往往是意图兜底层那次 267~371 token 的小调用（2026-10-06 实见），拿它当首轮就是把
重排效果读成噪声；意图/hyde/summary 三种调用点各自打了 ``trace_role:`` 标签，剩下的才是 agent 的。

百炼隐式前缀缓存的命中量实测是 **256 token 的整数倍**（39 个采样全为 256 倍数，其中 10752、12544
不是 1024 的倍数，故此前记的「1024 一块」作废）。命中判据是「从第一条 token 起完全相同的公共前缀」，
所以判读看的是**跨轮命中量是否随历史增长**：重排前它被钉死在 2048（每轮 agent 首轮 input 从 3618
涨到 22928，命中纹丝不动，占比从 57% 掉到 9%）——说明静态壳之后仍有每轮都变的字段把前缀截断。
轮内第 2 次起的调用本来就能命中（实测最高 33792/34211=98.7%），与重排无关，单独留着不进判读。

失败即停、不重试：重发同一轮会让对话历史多出一份，探针自己就污染了要测的分母。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# 项目根：app/trace/scripts/cache_probe.py → scripts → trace → app → 根
ROOT = Path(__file__).resolve().parents[3]
CACHE_BLOCK_TOKENS = 256  # 百炼隐式前缀缓存的匹配块大小（实测），命中量只能是它的整数倍


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单 session 多轮对话跑一遍，量化每轮首次 LLM 调用的前缀缓存命中")
    parser.add_argument("--fixture", type=Path, default=ROOT / "data" / "trace" / "prefix_cache_dialogue.json")
    parser.add_argument("--url", default="http://127.0.0.1:8000/study/chat", help="/study/chat 的 SSE 地址")
    parser.add_argument("--tag", default="run", help="本轮标记（before/after），写进输出文件名")
    parser.add_argument("--turns", type=int, default=0, help="只跑前 N 轮，0 表示全跑；先用 2 轮验管道再放全量")
    parser.add_argument("--timeout", type=float, default=300.0, help="单轮 SSE 读流超时秒数")
    parser.add_argument("--sleep", type=float, default=1.0, help="轮间隔秒数；隐式缓存无固定 TTL（实测约 6 分钟仍命中），别把间隔拉过长")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs", help="服务端 metrics.log 所在目录")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "trace" / "runs", help="探针结果输出目录")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None, help="只比对两份已有结果，不发请求")
    args = parser.parse_args(argv)
    if args.turns < 0:
        parser.error("--turns 不能为负数")
    if args.timeout <= 0 or args.sleep < 0:
        parser.error("--timeout 必须大于 0，--sleep 不能为负")
    return args


async def _send_turn(client: httpx.AsyncClient, url: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """发一轮对话并读完 SSE，只取探针需要的三件：done 的 request_id、状态、首 token 时延。"""
    out: dict[str, Any] = {"status": "no_done", "request_id": None, "answer_chars": 0, "first_token_ms": None}
    t_send = time.perf_counter()
    async with client.stream("POST", url, json=payload, headers={"X-User-Id": user_id}) as response:
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
                if out["first_token_ms"] is None:
                    out["first_token_ms"] = round((time.perf_counter() - t_send) * 1000, 2)
                out["answer_chars"] += len(str(event.get("content", "")))
            elif etype == "done":
                out["status"] = "done"
                out["request_id"] = event.get("request_id")
            elif etype in ("ask_clarify", "ask_confirm"):
                out["status"] = etype
            elif etype == "error":
                out["status"] = "error"
                out["error"] = event.get("message")
    return out


def _load_records(logs_dir: Path) -> list[dict[str, Any]]:
    """读回全部观测记录：兼容历史 ``logs/metrics.log`` 与按天隔离的 ``logs/<日期>/metrics.log``。

    metrics.log 是「缩进 JSON 块 + 空行」拼接写入的，不能 json.load，只能流式 raw_decode。
    """
    paths = [logs_dir / "metrics.log", *sorted(logs_dir.glob("*/metrics.log"))]
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    for path in paths:
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
            obj, index = decoder.raw_decode(text, index)
            records.append(obj)
    return records


def _match_records(records: list[dict[str, Any]], session_id: str, sent: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """把本 session 的记录按轮序对上：优先 request_id，缺 done 的轮按时间顺序兜位。"""
    mine = sorted((r for r in records if r.get("session_id") == session_id), key=lambda r: str(r.get("ts", "")))
    by_id = {r.get("request_id"): r for r in mine}
    used: set[str] = set()
    rows: list[dict[str, Any] | None] = []
    for turn in sent:
        record = by_id.get(turn["request_id"]) if turn.get("request_id") else None
        if record is not None:
            used.add(str(record.get("request_id")))
        rows.append(record)
    spare = [r for r in mine if str(r.get("request_id")) not in used]
    for i, row in enumerate(rows):
        if row is None and spare:
            rows[i] = spare.pop(0)
    return rows


def _split_by_role(detail: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按角色归堆：role 为空串即 agent 主体调用（intent/hyde/summary 由调用点打标）。

    整批都没有 role 字段 = 角色标注上线之前跑的历史结果，此时退回「按序号猜首轮」的老读法，
    并由调用方标注 ``labeled=False``——那份数据读到的首轮其实是意图小调用，不能用来判读重排。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in detail:
        grouped.setdefault(str(entry.get("role", "")), []).append(entry)
    return grouped


def _turn_row(tag: str, index: int, query: str, sent: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    detail = list((record or {}).get("llm_calls_detail") or [])
    labeled = any("role" in entry for entry in detail)
    grouped = _split_by_role(detail)
    agent_calls = grouped.get("", [])
    intent_calls = grouped.get("intent", [])
    summary_calls = grouped.get("summary", [])
    # 未标注的历史结果只能沿旧口径取序号 0（那是意图小调用），并在行里留 labeled=False 说明局限
    first_agent = agent_calls[0] if labeled and agent_calls else (detail[0] if detail else {})
    inputs = [int(d.get("input", 0) or 0) for d in detail]
    cacheds = [int(d.get("cached", 0) or 0) for d in detail]
    agent_cacheds = [int(d.get("cached", 0) or 0) for d in agent_calls]
    return {
        "kind": "turn",
        "tag": tag,
        "turn": index,
        "query": query,
        "status": sent["status"],
        "request_id": sent.get("request_id"),
        "error": (record or {}).get("error_type") or sent.get("error"),
        "intent": (record or {}).get("intent"),
        "intent_confidence": (record or {}).get("intent_confidence"),
        "labeled": labeled,
        "llm_calls": len(detail),
        "agent_calls": len(agent_calls) if labeled else None,
        "agent_first_input": int(first_agent.get("input", 0) or 0) or None,
        "agent_first_cached": int(first_agent.get("cached", 0) or 0) if first_agent else None,
        # 意图兜底层那次调用单独看：它每轮都排在全轮最前，重排动不到它
        "intent_input": int(intent_calls[0].get("input", 0) or 0) if intent_calls else None,
        "intent_cached": int(intent_calls[0].get("cached", 0) or 0) if intent_calls else None,
        # 摘要触发的轮次要挑出来：它会把历史形态整段换掉，前后两轮的分母不再同构
        "summary_calls": len(summary_calls) if labeled else None,
        # 轮内 agent 后续调用的命中是 agent loop 天然能拿到的，与重排无关，留着免得混进判读
        "agent_later_cached": agent_cacheds[1:] if labeled else cacheds[1:],
        "prompt_tokens": (record or {}).get("prompt_tokens"),
        "cached_tokens": (record or {}).get("cached_tokens"),
        "completion_tokens": (record or {}).get("completion_tokens"),
        "hit_ratio": round(sum(cacheds) / sum(inputs), 4) if inputs and sum(inputs) else None,
        "total_ms": (record or {}).get("stage_ms", {}).get("total_ms"),
        "answer_chars": sent.get("answer_chars"),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """打表：ag_* 是 agent 主体调用（判读只看它），intent_c 是意图兜底层那次，sum 是本轮触发的摘要调用数。"""
    widths = (4, 9, 5, 8, 10, 8, 9, 4, 9, 9, 6, 9)
    header = ("turn", "status", "calls", "ag_in", "ag_cached", "ag_hit%", "intent_c", "sum", "cached", "input", "hit%", "total_ms")
    print(" ".join(name.rjust(width) for name, width in zip(header, widths, strict=True)))
    for row in rows:
        ag_in = row["agent_first_input"] or 0
        ag_cached = row["agent_first_cached"] or 0
        hit = row["hit_ratio"]
        cells = (
            str(row["turn"]),
            row["status"],
            str(row["llm_calls"]),
            str(ag_in),
            str(ag_cached),
            "-" if not ag_in else f"{ag_cached / ag_in * 100:.0f}",
            str(row["intent_cached"]),
            "-" if row["summary_calls"] is None else str(row["summary_calls"]),
            str(row["cached_tokens"]),
            str(row["prompt_tokens"]),
            "-" if hit is None else f"{hit * 100:.1f}",
            str(row["total_ms"]),
        )
        print(" ".join(cell.rjust(width) for cell, width in zip(cells, widths, strict=True)))
    _print_judgement(rows)


def _print_judgement(rows: list[dict[str, Any]]) -> None:
    """判读首轮 agent 命中是否随历史增长——重排生效的唯一直接信号。"""
    if any(not row.get("labeled", True) for row in rows):
        print("\n警告：至少一轮的记录没有 role 字段（角色标注上线前跑的结果），ag_cached 退回按序号取首轮，那个位置是意图小调用，不能用它判读重排。")
    seq = [(row["turn"], row["agent_first_input"] or 0, row["agent_first_cached"] or 0) for row in rows if row["agent_first_input"]]
    if not seq:
        return
    print("\n首轮 agent (input -> cached): " + "  ".join(f"t{t}:{i}->{c}" for t, i, c in seq))
    summary_turns = [row["turn"] for row in rows if (row["summary_calls"] or 0) > 0]
    if summary_turns:
        print(f"摘要触发轮次: {summary_turns}——这些轮之后历史形态换成「摘要+最近 5 条」，与之前的轮不同构，对比时要分段看。")
    # 首轮冷启动必然没缓存可命中（t1 的 0 不是重排能改的），判"钉死/增长"只看第二轮起
    warm = [c for _, _, c in seq[1:]]
    if not warm:
        print("判读：不足两轮，看不出增长趋势——至少跑到第 2 轮再判。")
        return
    plateau_value, plateau_count = Counter(warm).most_common(1)[0]
    if plateau_count == len(warm) and plateau_value > 0:
        print(
            f"判读：命中钉死在 {plateau_value} token（{plateau_value // CACHE_BLOCK_TOKENS} 个 {CACHE_BLOCK_TOKENS} 块）不随历史增长——"
            f"静态壳之后仍有每轮都变的字段在截断前缀，或写入侧根本没攒到壳以外。"
        )
    elif warm[-1] > warm[0]:
        print(f"判读：命中随轮上升（{warm[0]} → {warm[-1]}），静态壳之外的历史也在被复用。")
    else:
        print(f"判读：命中量散落在 {min(warm)}~{max(warm)}，没有稳定增长——先看动态字段是否还在壳前，再怀疑 TTL 与写入时机。")


def _display(path: Path) -> str:
    """项目内的路径显示为相对路径，项目外（临时 --out-dir/--logs-dir）保留绝对路径。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_run(args: argparse.Namespace, fixture: dict[str, Any], session_id: str, rows: list[dict[str, Any]]) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out_dir / f"{args.tag}_{stamp}.jsonl"
    run_meta = {
        "kind": "run",
        "tag": args.tag,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "url": args.url,
        "fixture": str(args.fixture),
        "textbook_name": fixture["textbook_name"],
        "user_id": fixture["user_id"],
        "turn_count": len(rows),
        "cache_block_tokens": CACHE_BLOCK_TOKENS,
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(run_meta, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


async def _drive(args: argparse.Namespace) -> int:
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    turns = fixture["turns"] if not args.turns else fixture["turns"][: args.turns]
    session_id = f"pcprobe-{args.tag}-{uuid.uuid4().hex[:8]}"
    print(f"session={session_id} 轮数={len(turns)} url={args.url}")

    sent: list[dict[str, Any]] = []
    # trust_env=False：探针打的是本机服务，httpx 默认会读系统（注册表）代理把 127.0.0.1 也甩给代理，实见 502
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        for position, item in enumerate(turns):
            payload = {
                "textbook_name": fixture["textbook_name"],
                "query": item["query"],
                "session_id": session_id,
            }
            try:
                out = await _send_turn(client, args.url, fixture["user_id"], payload)
            except Exception as exc:
                print(f"第 {item['index']} 轮发送/读流失败，探针停止（重试会重复历史，污染要测的分母）：{type(exc).__name__}: {exc}")
                sent.append(
                    {
                        "status": "client_error",
                        "request_id": None,
                        "answer_chars": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            sent.append(out)
            print(f"  turn {item['index']:>2} {out['status']:>9} request_id={out['request_id']}")
            if args.sleep and position < len(turns) - 1:
                await asyncio.sleep(args.sleep)

    records = _load_records(args.logs_dir)
    matched = _match_records(records, session_id, sent)
    missing = sum(1 for r in matched if r is None)
    if missing:
        print(f"警告：{missing}/{len(turns)} 轮在 {args.logs_dir} 里没找到对应记录（服务端未开 metrics 落盘，或写进了别的目录）")
    rows = [_turn_row(args.tag, item["index"], item["query"], turn, record) for item, turn, record in zip(turns, sent, matched, strict=False)]
    path = _write_run(args, fixture, session_id, rows)
    print(f"\n结果已写入 {_display(path)}\n")
    _print_table(rows)
    return 0 if len(rows) == len(turns) else 1


def _load_run(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    meta: dict[str, Any] = {}
    turns: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("kind") == "run":
            meta = obj
        else:
            turns[int(obj["turn"])] = obj
    return meta, turns


def _pick(row: dict[str, Any] | None, field: str) -> int:
    """缺记录/缺字段一律按 0 计：不因某一轮没落库就让整列对比作废。"""
    return 0 if row is None else int(row.get(field) or 0)


def _compare(before_path: str, after_path: str) -> int:
    """按轮并排两次结果，只看首轮 agent 命中与总量差；其余指标不动判读，免得把重排的账记到轮内命中上。"""
    before_meta, before = _load_run(Path(before_path))
    after_meta, after = _load_run(Path(after_path))
    fields = ("agent_first_cached", "cached_tokens", "prompt_tokens")
    slots = ("first", "cached", "input")
    for name, run in (("before", before), ("after", after)):
        if any(not row.get("labeled", True) for row in run.values()):
            print(f"警告：{name} 里有未标注角色的轮次（角色标注上线前跑的结果），首轮列读的是意图小调用，对比作废。")
    print(f"{'turn':>4} {'first_cached b->a':>20} {'cached_sum b->a':>18} {'input_sum b->a':>16}")
    sums = {slot: [0, 0] for slot in slots}
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        pairs = tuple((_pick(b, field), _pick(a, field)) for field in fields)
        for slot, (bv, av) in zip(slots, pairs, strict=True):
            sums[slot][0] += bv
            sums[slot][1] += av
        cells = [str(key)] + [f"{bv} -> {av}" for bv, av in pairs]
        print(f"{cells[0]:>4} {cells[1]:>20} {cells[2]:>18} {cells[3]:>16}")
    n = len(set(before) | set(after)) or 1
    b_seq = [_pick(before.get(key), "agent_first_cached") for key in sorted(before)]
    a_seq = [_pick(after.get(key), "agent_first_cached") for key in sorted(after)]
    mean_b = sums["first"][0] / n
    mean_a = sums["first"][1] / n
    gained = sums["cached"][1] - sums["cached"][0]
    print(
        f"\n首轮 agent cached 均值: before={mean_b:.0f} after={mean_a:.0f}"
        f"（块粒度 {CACHE_BLOCK_TOKENS} token）\n"
        f"首轮 agent cached 序列: before={b_seq}\n"
        f"                        after ={a_seq}\n"
        f"整轮 cached 合计: before={sums['cached'][0]} after={sums['cached'][1]} Δ={gained}\n"
        f"输入 token 合计: before={sums['input'][0]} after={sums['input'][1]}"
        f"（两轮分母应当接近，差得多说明对话内容或轮序变了，对比作废）\n"
        f"session: before={before_meta.get('session_id')} after={after_meta.get('session_id')}"
    )
    flat_b = len(set(b_seq)) <= 1
    flat_a = len(set(a_seq)) <= 1
    if gained <= 0:
        print("判读：after 的命中总量没超过 before——重排没换来命中，先查 system prompt 顶部还留着什么每轮都变的字段。")
    elif not flat_a and flat_b:
        base = b_seq[0] if b_seq else 0
        print(f"判读：重排生效——首轮命中从钉死 {base} 变成随历史增长（最高 {max(a_seq)}），多命中约 {gained} token（命中部分按约 20% 计价）。")
    elif mean_a > mean_b:
        print(
            f"判读：首轮命中均值从 {mean_b:.0f} 涨到 {mean_a:.0f}，多命中约 {gained} token；"
            "但序列仍不发散，历史部分大概没进公共前缀，查摘要与存根化是否在改早期消息。"
        )
    else:
        print("判读：两轮首轮命中相当，重排没有可见差异。")
    return 0


def _reconfigure_stdout() -> None:
    """Windows 控制台默认 cp936，中文问答会花屏；探针自己的输出固定 UTF-8。"""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdout()
    args = _parse_args(argv)
    if args.compare:
        return _compare(*args.compare)
    if not args.fixture.is_file():
        print(f"找不到对话内容：{args.fixture}")
        return 1
    return asyncio.run(_drive(args))


if __name__ == "__main__":
    raise SystemExit(main())
