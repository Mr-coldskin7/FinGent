"""
FinGent 可靠性评测（K 次运行成功率 + 链路追踪）

回答三个问题：
  1. 稳定吗？        → 每个用例跑 K 次，报告成功率 + Wilson 95% 置信区间
  2. 延迟如何？      → 每次运行耗时，给出 p50 / p95
  3. 出错在哪一步？  → 每次运行写一份 trace.jsonl（每个进度事件带时间戳），可逐步回放

注意边界：这里衡量的是「可靠性」（跑得通、输出结构合法、路由符合预期），
不是「决策准确率」——后者需要真值与 point-in-time 回测（见 Agent.md P1/P2）。

用法：
    uv run python -m Eval.reliability --k 5                 # 全部用例
    uv run python -m Eval.reliability --k 2 --only chat     # 只跑 chat 用例（冒烟）
    uv run python -m Eval.reliability --k 5 --cases Eval/cases.json --out Eval/runs

产物（每次运行一个目录 Eval/runs/<时间戳>/）：
    report.json   汇总指标（机器可读）
    report.md     汇总表（人读）
    trace.jsonl   每次运行的事件流 + 结束记录
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT_ROOT / "Eval" / "cases.json"
DEFAULT_OUT = PROJECT_ROOT / "Eval" / "runs"

VALID_VOTES = {"BUY", "HOLD", "SELL", "STRONG_BUY", "STRONG_SELL", "REDUCE"}
# 图正常结束的状态：completed = 有结论；waiting_for_clarification = 合法的澄清分支
PIPELINE_OK_STATUSES = {"completed", "waiting_for_clarification"}


# ───────────────────────── 统计工具（纯函数，可单测） ─────────────────────────
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 得分区间。小样本（K=5）下比正态近似可靠，且 k=n 时上界不会超过 1。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """最近秩法分位数。空列表返回 None。"""
    if not values:
        return None
    xs = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(xs)))
    return xs[min(rank, len(xs)) - 1]


# ───────────────────────── 路由识别与结果判定（纯函数，可单测） ─────────────────────────
def classify_route(events: List[Dict[str, Any]]) -> str:
    """从进度事件推断本次走的路由：chat / clarify / single / multi / none。"""
    stages = [e.get("stage") for e in events]
    if "chat" in stages:
        return "chat"
    if "clarify" in stages:
        return "clarify"
    if "sector" in stages:
        return "sector"
    started = sum(1 for e in events if e.get("stage") == "agent_start")
    if started == 1:
        return "single"
    if started > 1:
        return "multi"
    return "none"


def evaluate_run(
    result: Dict[str, Any],
    events: List[Dict[str, Any]],
    expect_route: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """判定单次运行是否成功。

    两个层级（分开报告，避免过度声称）：
      pipeline_ok：没有抛异常，状态合法，有可展示的结果文本（"能跑通"）
      strict_ok  ：pipeline_ok 且路由符合预期、无降级 agent、投票值合法（"跑得对"）
    """
    reasons: List[str] = []
    status = (result or {}).get("status")
    text = (result or {}).get("result") or ""

    if error:
        reasons.append(f"exception: {error[:120]}")
    if status not in PIPELINE_OK_STATUSES:
        reasons.append(f"bad status: {status}")
    if not str(text).strip():
        reasons.append("empty result text")
    pipeline_ok = (not error) and status in PIPELINE_OK_STATUSES and bool(str(text).strip())

    route = classify_route(events)
    route_ok = route == expect_route
    if not route_ok:
        reasons.append(f"route {route} != expected {expect_route}")

    agent_results: Dict[str, Any] = (result or {}).get("agent_results") or {}
    # 降级 = 异常兜底（error）或 JSON 解析失败兜底（parse_fallback）；两者都会被兜底成 HOLD，
    # 若只看 error，解析失败会被误计为"正常观望"
    degraded = [
        name for name, r in agent_results.items() if r.get("error") or r.get("parse_fallback")
    ]
    if degraded:
        reasons.append(f"degraded agents: {degraded}")

    bad_votes = [
        name for name, r in agent_results.items() if r.get("vote") not in VALID_VOTES
    ]
    if bad_votes:
        reasons.append(f"invalid votes: {bad_votes}")
    # chat/clarify 没有 agent 投票，投票合法性视为满足
    votes_valid = not bad_votes

    strict_ok = pipeline_ok and route_ok and not degraded and votes_valid

    final_vote = ((result or {}).get("final_decision") or {}).get("final_vote")
    if final_vote is None and agent_results:
        # 单 agent：取唯一一票
        first = next(iter(agent_results.values()))
        final_vote = first.get("vote")

    return {
        "pipeline_ok": pipeline_ok,
        "strict_ok": strict_ok,
        "route": route,
        "route_ok": route_ok,
        "degraded_agents": degraded,
        "votes_valid": votes_valid,
        "final_vote": final_vote,
        "reasons": reasons,
    }


# ───────────────────────── 汇总（纯函数，可单测） ─────────────────────────
def summarize_case(case_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    pipe_k = sum(1 for r in records if r["eval"]["pipeline_ok"])
    strict_k = sum(1 for r in records if r["eval"]["strict_ok"])
    latencies = [r["latency_s"] for r in records]
    with_agents = [r for r in records if r["eval"]["route"] in ("single", "multi")]
    degraded_runs = sum(1 for r in with_agents if r["eval"]["degraded_agents"])
    votes: Dict[str, int] = {}
    for r in records:
        v = r["eval"].get("final_vote")
        if v:
            votes[v] = votes.get(v, 0) + 1
    p_lo, p_hi = wilson_interval(pipe_k, n)
    s_lo, s_hi = wilson_interval(strict_k, n)
    return {
        "case_id": case_id,
        "runs": n,
        "pipeline_ok": pipe_k,
        "pipeline_rate": round(pipe_k / n, 3) if n else 0.0,
        "pipeline_ci95": [round(p_lo, 3), round(p_hi, 3)],
        "strict_ok": strict_k,
        "strict_rate": round(strict_k / n, 3) if n else 0.0,
        "strict_ci95": [round(s_lo, 3), round(s_hi, 3)],
        "latency_p50_s": _r(percentile(latencies, 50)),
        "latency_p95_s": _r(percentile(latencies, 95)),
        "latency_mean_s": _r(sum(latencies) / n if n else None),
        "degraded_run_rate": round(degraded_runs / len(with_agents), 3) if with_agents else None,
        "routes": _count([r["eval"]["route"] for r in records]),
        "final_votes": votes,
    }


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 2)


def _count(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# FinGent 可靠性报告 · {report['generated_at']}",
        "",
        f"K = {report['k']}，用例数 = {len(report['cases'])}，总运行 = {report['total_runs']}",
        "",
        "> 「能跑通」= 无异常 + 状态合法 + 有结果。「跑得对」= 能跑通 + 路由符合预期 + 无降级 agent + 投票合法。",
        "> 本报告衡量可靠性，不衡量决策准确率。",
        "",
        "| 用例 | 运行数 | 能跑通 | 95% CI | 跑得对 | 95% CI | p50 延迟 | p95 延迟 | 降级率 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        deg = "—" if c["degraded_run_rate"] is None else f"{c['degraded_run_rate']:.0%}"
        lines.append(
            f"| {c['case_id']} | {c['runs']} | {c['pipeline_ok']}/{c['runs']} "
            f"| [{c['pipeline_ci95'][0]:.2f}, {c['pipeline_ci95'][1]:.2f}] "
            f"| {c['strict_ok']}/{c['runs']} "
            f"| [{c['strict_ci95'][0]:.2f}, {c['strict_ci95'][1]:.2f}] "
            f"| {c['latency_p50_s']}s | {c['latency_p95_s']}s | {deg} |"
        )
    lines += ["", "## 失败原因（去重计数）", ""]
    reason_counts: Dict[str, int] = {}
    for run in report["runs"]:
        for reason in run["eval"]["reasons"]:
            key = reason.split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    if reason_counts:
        for k_, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {k_}：{v}")
    else:
        lines.append("- 无")
    lines += ["", "明细与逐步事件见同目录 `trace.jsonl`。"]
    return "\n".join(lines) + "\n"


# ───────────────────────── 执行 ─────────────────────────
async def run_once(
    graph,
    case: Dict[str, Any],
    run_idx: int,
    trace_fp,
) -> Dict[str, Any]:
    """执行一次用例运行，记录事件流并写入 trace。"""
    run_id = f"{case['id']}#{run_idx}-{uuid.uuid4().hex[:6]}"
    events: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    async def progress_cb(evt: Dict[str, Any]) -> None:
        events.append({"t": round(time.perf_counter() - t0, 3), **evt})

    thread_id = f"eval_{run_id}"
    error: Optional[str] = None
    result: Dict[str, Any] = {}
    try:
        result = await asyncio.wait_for(
            graph.run(
                case["query"],
                thread_id,
                user_id="eval_harness",
                agents=case.get("agents"),
                progress_cb=progress_cb,
            ),
            timeout=case.get("timeout_s", 300),
        )
    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as e:  # 评测要记录异常而不是中断整批
        error = f"{type(e).__name__}: {e}"

    # 让尾部的进度任务（create_task 派发）有机会落地
    for _ in range(3):
        await asyncio.sleep(0)
    latency = time.perf_counter() - t0

    ev = evaluate_run(result, events, case["expect_route"], error=error)
    record = {
        "run_id": run_id,
        "case_id": case["id"],
        "run_idx": run_idx,
        "query": case["query"],
        "latency_s": round(latency, 2),
        "status": (result or {}).get("status"),
        "result_preview": str((result or {}).get("result") or "")[:200],
        "agents": _agent_summary((result or {}).get("agent_results") or {}),
        "eval": ev,
    }

    for e in events:
        trace_fp.write(json.dumps({"run_id": run_id, "type": "event", **e}, ensure_ascii=False) + "\n")
    trace_fp.write(json.dumps({"run_id": run_id, "type": "end", **record}, ensure_ascii=False, default=str) + "\n")
    trace_fp.flush()
    return record


def _agent_summary(agent_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """每个 Agent 的可复核摘要：投票 + 是否异常兜底 + 是否解析兜底。"""
    return {
        name: {
            "vote": r.get("vote"),
            "error": bool(r.get("error")),
            "parse_fallback": bool(r.get("parse_fallback")),
        }
        for name, r in agent_results.items()
    }


def _log_entry(fp, event: str, level: str = "INFO", **fields: Any) -> None:
    """写一行结构化日志（JSON Lines）。字段固定：ts / level / event，其余按事件变化。"""
    row = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "level": level,
        "event": event,
        **fields,
    }
    fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    fp.flush()


async def run_suite(
    graph,
    cases: List[Dict[str, Any]],
    k: int,
    trace_path: Path,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    log_path = log_path or trace_path.parent / "run_log.jsonl"
    with trace_path.open("w", encoding="utf-8") as fp, log_path.open("w", encoding="utf-8") as lg:
        _log_entry(lg, "suite_start", cases=[c["id"] for c in cases], k=k)
        for case in cases:
            for i in range(1, k + 1):
                print(f"▶ [{case['id']}] 第 {i}/{k} 次 ...", flush=True)
                _log_entry(lg, "run_start", case_id=case["id"], run_idx=i)
                rec = await run_once(graph, case, i, fp)
                ev = rec["eval"]
                mark = "✅" if ev["strict_ok"] else ("🟡" if ev["pipeline_ok"] else "❌")
                print(
                    f"  {mark} route={ev['route']} latency={rec['latency_s']}s "
                    f"vote={ev['final_vote']} reasons={ev['reasons']}",
                    flush=True,
                )
                level = "ERROR" if not ev["pipeline_ok"] else ("WARN" if not ev["strict_ok"] else "INFO")
                _log_entry(
                    lg,
                    "run_end",
                    level,
                    run_id=rec["run_id"],
                    case_id=rec["case_id"],
                    run_idx=i,
                    latency_s=rec["latency_s"],
                    status=rec["status"],
                    route=ev["route"],
                    pipeline_ok=ev["pipeline_ok"],
                    strict_ok=ev["strict_ok"],
                    final_vote=ev["final_vote"],
                    reasons=ev["reasons"],
                    agents=rec["agents"],
                )
                records.append(rec)
        _log_entry(lg, "suite_end", total_runs=len(records))

    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_case.setdefault(r["case_id"], []).append(r)
    summaries = [summarize_case(cid, rs) for cid, rs in by_case.items()]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "total_runs": len(records),
        "cases": summaries,
        "runs": records,
    }


def load_cases(path: Path, only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]
    if not cases:
        raise SystemExit(f"没有可运行的用例（cases={path}, only={only}）")
    return cases


def build_graph():
    """复用 main.create_fin_graph，但关闭 L3 记忆写入，避免评测数据污染用户记忆。"""
    from main import create_fin_graph  # 延迟导入：仅 CLI 需要真实模型

    graph = create_fin_graph()
    graph.memory_manager = None
    return graph


def parse_trace(trace_path: Path) -> List[Dict[str, Any]]:
    """把 trace.jsonl 按 run_id 分组，返回结构化列表（API 与 --show 共用）。

    每个元素：{run_id, case_id, latency_s, status, route, strict_ok, pipeline_ok,
              final_vote, reasons, query, events:[{t, stage, agent?, vote?, error?}], complete}
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        rid = it["run_id"]
        g = groups.setdefault(
            rid,
            {
                "run_id": rid,
                "case_id": rid.split("#")[0],
                "run_idx": None,
                "events": [],
                "complete": False,
                "latency_s": None,
                "status": None,
                "query": None,
                "route": None,
                "strict_ok": None,
                "pipeline_ok": None,
                "final_vote": None,
                "reasons": [],
            },
        )
        if it["type"] == "event":
            ev = {
                k: it[k]
                for k in (
                    "t",
                    "stage",
                    "agent",
                    "vote",
                    "error",
                    "mode",
                    "reason",
                    "source",
                )
                if k in it
            }
            g["events"].append(ev)
        elif it["type"] == "end":
            g["complete"] = True
            g["latency_s"] = it.get("latency_s")
            g["run_idx"] = it.get("run_idx")
            g["status"] = it.get("status")
            g["query"] = it.get("query")
            ev = it.get("eval", {})
            g["route"] = ev.get("route")
            g["strict_ok"] = ev.get("strict_ok")
            g["pipeline_ok"] = ev.get("pipeline_ok")
            g["final_vote"] = ev.get("final_vote")
            g["reasons"] = ev.get("reasons", [])
    return list(groups.values())


def format_trace(trace_path: Path) -> str:
    """把 trace.jsonl 渲染成按运行分组的时间线（命令行用）。"""
    out: List[str] = []
    for g in parse_trace(trace_path):
        if g["complete"]:
            mark = "✅" if g["strict_ok"] else ("🟡" if g["pipeline_ok"] else "❌")
            head = f"■ {g['run_id']}  {g['route']} · {g['latency_s']}s · {mark}"
        else:
            head = f"■ {g['run_id']}  (无结束记录，运行可能中断)"
        out.append(head)
        for ev in g["events"]:
            detail = " ".join(f"{k}={ev[k]}" for k in ("agent", "vote", "error") if k in ev)
            out.append(f"    {ev['t']:>7.2f}s  {ev['stage']:<12} {detail}".rstrip())
        for reason in g["reasons"]:
            out.append(f"    ! {reason}")
        out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FinGent K 次运行可靠性评测")
    parser.add_argument("--k", type=int, default=5, help="每个用例运行次数")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", type=str, default="", help="逗号分隔的用例 id，如 chat,single_morefit")
    parser.add_argument("--show", type=Path, default=None, help="只渲染已有的 trace.jsonl 时间线，不运行")
    args = parser.parse_args(argv)

    if args.show:
        print(format_trace(args.show))
        return

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    cases = load_cases(args.cases, only)
    total = len(cases) * args.k
    print(f"用例 {len(cases)} 个 × K={args.k} = {total} 次运行（会消耗 LLM 配额，多 agent 用例约 1–2 分钟/次）")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph()
    report = asyncio.run(
        run_suite(graph, cases, args.k, out_dir / "trace.jsonl", out_dir / "run_log.jsonl")
    )
    report["cases_file"] = str(args.cases)

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")

    print("\n" + render_markdown(report))
    print(f"产物目录: {out_dir}")


if __name__ == "__main__":
    main()
