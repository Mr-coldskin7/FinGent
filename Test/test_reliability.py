"""
Eval/reliability.py 的单元测试（全部使用假图，不消耗 LLM 配额）。
"""

import asyncio
import json
from typing import Optional

import pytest

from Eval.reliability import (
    classify_route,
    evaluate_run,
    percentile,
    render_markdown,
    run_suite,
    summarize_case,
    wilson_interval,
)


# ── 统计 ────────────────────────────────────────────────────────────
def test_wilson_all_success_upper_bound_capped_at_one():
    lo, hi = wilson_interval(5, 5)
    assert hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.566, abs=0.001)


def test_wilson_zero_success_lower_bound_zero():
    lo, hi = wilson_interval(0, 5)
    assert lo == pytest.approx(0.0)
    assert 0 < hi < 0.5


def test_wilson_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_percentile_nearest_rank():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == 5
    assert percentile(xs, 95) == 10
    assert percentile([], 50) is None


# ── 路由识别 ────────────────────────────────────────────────────────
def test_classify_route_chat():
    assert classify_route([{"stage": "preprocess"}, {"stage": "chat"}]) == "chat"


def test_classify_route_clarify():
    assert classify_route([{"stage": "preprocess"}, {"stage": "clarify"}]) == "clarify"


def test_classify_route_single_and_multi():
    single = [{"stage": "agent_start", "agent": "morefit"}]
    multi = [
        {"stage": "agent_start", "agent": "morefit"},
        {"stage": "agent_start", "agent": "risk-manager"},
    ]
    assert classify_route(single) == "single"
    assert classify_route(multi) == "multi"
    assert classify_route([]) == "none"


# ── 单次判定 ────────────────────────────────────────────────────────
GOOD_MULTI = {
    "status": "completed",
    "result": "综合结论",
    "agent_results": {
        "morefit": {"vote": "BUY", "error": False},
        "risk-manager": {"vote": "HOLD", "error": False},
    },
    "final_decision": {"final_vote": "BUY"},
}
MULTI_EVENTS = [
    {"stage": "agent_start", "agent": "morefit"},
    {"stage": "agent_start", "agent": "risk-manager"},
]


def test_evaluate_run_strict_ok():
    ev = evaluate_run(GOOD_MULTI, MULTI_EVENTS, expect_route="multi")
    assert ev["pipeline_ok"] and ev["strict_ok"]
    assert ev["final_vote"] == "BUY"
    assert ev["reasons"] == []


def test_evaluate_run_parse_fallback_counts_as_degraded():
    """JSON 解析失败被兜底为 HOLD 时，必须计为降级，不能伪装成正常观望。"""
    result = {
        **GOOD_MULTI,
        "agent_results": {
            "morefit": {"vote": "HOLD", "error": False, "parse_fallback": True},
            "risk-manager": {"vote": "BUY", "error": False},
        },
    }
    ev = evaluate_run(result, MULTI_EVENTS, expect_route="multi")
    assert ev["degraded_agents"] == ["morefit"]
    assert ev["strict_ok"] is False


def test_parse_agent_output_json_failure_sets_parse_fallback():
    from LLM.graph import FinGraph

    # 非 JSON 输出 → HOLD 兜底，且必须打标记
    bad = FinGraph._parse_agent_output(None, "这不是 json", "json", "NVDA")
    assert bad["vote"] == "HOLD" and bad["parse_fallback"] is True
    # 合法 JSON → 正常解析，不打标记
    good_json = '{"decisions":[{"symbol":"NVDA","vote":"BUY","reason":"x"}]}'
    good = FinGraph._parse_agent_output(None, good_json, "json", "NVDA")
    assert good["vote"] == "BUY" and not good.get("parse_fallback")
    # 纯文本解析器不属于"解析失败"，不打标记
    text = FinGraph._parse_agent_output(None, "自由文本", "text", "NVDA")
    assert not text.get("parse_fallback")


def test_evaluate_run_degraded_agent_is_pipeline_ok_but_not_strict():
    result = {
        **GOOD_MULTI,
        "agent_results": {
            "morefit": {"vote": "HOLD", "error": True},
            "risk-manager": {"vote": "HOLD", "error": False},
        },
    }
    ev = evaluate_run(result, MULTI_EVENTS, expect_route="multi")
    assert ev["pipeline_ok"] is True
    assert ev["strict_ok"] is False
    assert ev["degraded_agents"] == ["morefit"]


def test_evaluate_run_route_mismatch_not_strict():
    ev = evaluate_run(GOOD_MULTI, MULTI_EVENTS, expect_route="single")
    assert ev["pipeline_ok"] is True
    assert ev["strict_ok"] is False
    assert ev["route_ok"] is False


def test_evaluate_run_invalid_vote_not_strict():
    result = {**GOOD_MULTI, "agent_results": {"morefit": {"vote": "MAYBE", "error": False}}}
    ev = evaluate_run(result, MULTI_EVENTS[:1], expect_route="single")
    assert ev["votes_valid"] is False
    assert ev["strict_ok"] is False


def test_evaluate_run_exception_is_not_pipeline_ok():
    ev = evaluate_run({}, [], expect_route="chat", error="RuntimeError: boom")
    assert ev["pipeline_ok"] is False
    assert ev["strict_ok"] is False
    assert any("exception" in r for r in ev["reasons"])


def test_evaluate_run_empty_result_is_not_pipeline_ok():
    ev = evaluate_run({"status": "completed", "result": "  "}, [], expect_route="chat")
    assert ev["pipeline_ok"] is False


def test_evaluate_run_clarification_is_pipeline_ok():
    result = {"status": "waiting_for_clarification", "result": "请提供股票名称"}
    ev = evaluate_run(result, [{"stage": "clarify"}], expect_route="clarify")
    assert ev["pipeline_ok"] is True
    assert ev["strict_ok"] is True


# ── 汇总 ────────────────────────────────────────────────────────────
def _rec(pipe, strict, latency, route="multi", degraded=None, vote: Optional[str] = "BUY"):
    return {
        "latency_s": latency,
        "eval": {
            "pipeline_ok": pipe,
            "strict_ok": strict,
            "route": route,
            "degraded_agents": degraded or [],
            "final_vote": vote,
            "reasons": [],
        },
    }


def test_summarize_case_rates_and_degraded_rate():
    records = [
        _rec(True, True, 10.0),
        _rec(True, False, 20.0, degraded=["morefit"]),
        _rec(False, False, 30.0, vote=None),
        _rec(True, True, 40.0, route="chat", vote=None),
    ]
    s = summarize_case("c", records)
    assert s["runs"] == 4
    assert s["pipeline_ok"] == 3
    assert s["strict_ok"] == 2
    assert s["pipeline_rate"] == 0.75
    # 有 agent 的运行只有 3 次（chat 不计入），其中 1 次降级
    assert s["degraded_run_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert s["final_votes"] == {"BUY": 2}
    assert s["routes"] == {"multi": 3, "chat": 1}


def test_render_markdown_contains_case_row():
    report = {
        "generated_at": "2026-10-16T00:00:00",
        "k": 2,
        "total_runs": 2,
        "cases": [summarize_case("chat", [_rec(True, True, 1.0, route="chat", vote=None)] * 2)],
        "runs": [{"eval": {"reasons": []}}] * 2,
    }
    md = render_markdown(report)
    assert "| chat | 2 |" in md
    assert "能跑通" in md


# ── 端到端（假图）：trace 落盘 + 汇总 ──────────────────────────────
class FakeGraph:
    """模拟 FinGraph.run：按用例 query 决定事件与结果，支持注入异常。"""

    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()
        self.calls = 0

    async def run(self, user_input, thread_id, user_id=None, agents=None, progress_cb=None):
        self.calls += 1
        if user_input in self.fail_on:
            raise RuntimeError("injected failure")
        if user_input == "你好":
            await progress_cb({"stage": "preprocess"})
            await progress_cb({"stage": "chat"})
            return {"status": "completed", "result": "你好！我是 FinGent"}
        # 多 agent
        names = agents or ["morefit", "risk-manager"]
        for n in names:
            await progress_cb({"stage": "agent_start", "agent": n})
        for n in names:
            await progress_cb({"stage": "agent_done", "agent": n, "vote": "BUY"})
        await progress_cb({"stage": "voting"})
        return {
            "status": "completed",
            "result": "综合",
            "agent_results": {n: {"vote": "BUY", "error": False} for n in names},
            "final_decision": {"final_vote": "BUY"},
        }


CASES = [
    {"id": "chat", "query": "你好", "expect_route": "chat"},
    {"id": "multi", "query": "英伟达", "agents": ["morefit", "risk-manager"], "expect_route": "multi"},
]


async def test_run_suite_end_to_end_writes_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    report = await run_suite(FakeGraph(), CASES, k=2, trace_path=trace_path)

    assert report["total_runs"] == 4
    by_id = {c["case_id"]: c for c in report["cases"]}
    assert by_id["chat"]["strict_ok"] == 2
    assert by_id["multi"]["strict_ok"] == 2
    assert by_id["multi"]["routes"] == {"multi": 2}

    lines = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    ends = [l for l in lines if l["type"] == "end"]
    events = [l for l in lines if l["type"] == "event"]
    assert len(ends) == 4
    assert any(e["stage"] == "agent_done" and e["vote"] == "BUY" for e in events)
    # 每个事件都带时间戳，便于回放
    assert all("t" in e for e in events)


async def test_run_suite_records_exception_without_aborting(tmp_path):
    graph = FakeGraph(fail_on={"英伟达"})
    trace_path = tmp_path / "trace.jsonl"
    report = await run_suite(graph, CASES, k=2, trace_path=trace_path)

    multi = next(c for c in report["cases"] if c["case_id"] == "multi")
    assert multi["runs"] == 2
    assert multi["pipeline_ok"] == 0
    assert multi["strict_ok"] == 0
    # 异常后仍然跑完了 chat 用例
    chat = next(c for c in report["cases"] if c["case_id"] == "chat")
    assert chat["strict_ok"] == 2
    assert graph.calls == 4


def test_format_trace_renders_timeline_and_reasons(tmp_path):
    from Eval.reliability import format_trace

    path = tmp_path / "trace.jsonl"
    rows = [
        {"run_id": "r1", "type": "event", "t": 0.0, "stage": "preprocess"},
        {"run_id": "r1", "type": "event", "t": 1.5, "stage": "agent_done", "agent": "morefit", "vote": "HOLD"},
        {
            "run_id": "r1",
            "type": "end",
            "latency_s": 2.0,
            "eval": {"strict_ok": False, "pipeline_ok": True, "route": "single", "reasons": ["degraded agents: ['morefit']"]},
        },
        {"run_id": "r2", "type": "event", "t": 0.0, "stage": "preprocess"},  # \u65e0 end\uff1a\u6a21\u62df\u4e2d\u65ad
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    text = format_trace(path)
    assert "r1" in text and "single" in text
    assert "agent=morefit vote=HOLD" in text
    assert "degraded agents" in text
    assert "\u65e0\u7ed3\u675f\u8bb0\u5f55" in text


async def test_run_suite_writes_structured_run_log(tmp_path):
    """run_log.jsonl：每行合法 JSON，固定字段 ts/level/event，run_end 带每个 Agent 摘要。"""
    trace_path = tmp_path / "trace.jsonl"
    await run_suite(FakeGraph(), CASES, k=2, trace_path=trace_path)
    log_path = tmp_path / "run_log.jsonl"
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["event"] == "suite_start" and rows[0]["k"] == 2
    assert rows[-1]["event"] == "suite_end" and rows[-1]["total_runs"] == 4
    for r in rows:
        assert {"ts", "level", "event"} <= r.keys()
    ends = [r for r in rows if r["event"] == "run_end"]
    assert len(ends) == 4
    multi_end = next(r for r in ends if r["case_id"] == "multi")
    assert set(multi_end["agents"]) == {"morefit", "risk-manager"}
    assert {"vote", "error", "parse_fallback"} <= multi_end["agents"]["morefit"].keys()


async def test_run_suite_log_level_reflects_failure(tmp_path):
    graph = FakeGraph(fail_on={"英伟达"})
    await run_suite(graph, CASES, k=1, trace_path=tmp_path / "trace.jsonl")
    rows = [json.loads(l) for l in (tmp_path / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    ends = {r["case_id"]: r for r in rows if r["event"] == "run_end"}
    assert ends["multi"]["level"] == "ERROR"
    assert ends["chat"]["level"] == "INFO"
