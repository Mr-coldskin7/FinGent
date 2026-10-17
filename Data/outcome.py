"""
结果验证（前向检验）：分析之后，用真实行情判断各 Agent 的投票对不对。

设计要点
- 入场价 = 决策日之后第一个共同交易日的开盘价。决策发生在哪个时刻都不会用到其后的价格（无前视偏差）。
  代价是偏保守：决策日盘中的那段走势不计入。
- 出场价 = 入场日起第 horizon 个交易日的收盘价。
- 用超额收益判定（个股收益 - 基准收益），避免大牛市里"乱买都对"。
  美股基准 ^GSPC；A 股基准 沪深300（000300.SS / 000300.SZ）。
- 判定规则：BUY/STRONG_BUY 超额 > +阈值 为对；SELL/STRONG_SELL/REDUCE 超额 < -阈值 为对；
  HOLD 在 ±阈值内为对。阈值 OUTCOME_THRESHOLD（默认 2%）。
- 数据不足（还没到期、或数据源失败）记为"待验证"，下次再试，不会误判。

用法：
    uv run python -m Data.outcome            # 验证所有到期记录并更新客观权重
    uv run python -m Data.outcome --no-weights
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from Data.memory import MIN_VERIFIED_SAMPLES, PRIMARY_HORIZON, MemoryManager
from Data.providers.yahoo_us import fetch_daily_bars
from Data.symbol_alias import alias_symbol

OUTCOME_THRESHOLD = 0.02
VERIFY_HORIZONS: tuple[int, ...] = (5, 20)

BULLISH = {"BUY", "STRONG_BUY"}
BEARISH = {"SELL", "STRONG_SELL", "REDUCE"}

Fetcher = Callable[[str, date, date], Awaitable[List[Dict[str, Any]]]]


def _is_a_share_code(symbol: str) -> bool:
    s = symbol.strip()
    return s.isdigit() and len(s) == 6


def yahoo_symbol(symbol: str) -> str:
    aliased = alias_symbol(symbol)
    if aliased:
        return aliased
    s = symbol.strip().upper()
    if _is_a_share_code(s):
        return f"{s}.SS" if s.startswith("6") else f"{s}.SZ"
    return s


def benchmark_for(symbol: str) -> str:
    if alias_symbol(symbol):  # 别名一律是美股代码，基准用标普500
        return "^GSPC"
    s = symbol.strip().upper()
    if _is_a_share_code(s):
        return "000300.SS" if s.startswith("6") else "000300.SZ"
    return "^GSPC"


def judge_vote(vote: str, excess_return: float, threshold: float = OUTCOME_THRESHOLD) -> Optional[int]:
    """判定某个投票是否正确：1 对 / 0 错 / None 无法判定（未知投票类型）。"""
    if vote in BULLISH:
        return int(excess_return > threshold)
    if vote in BEARISH:
        return int(excess_return < -threshold)
    if vote == "HOLD":
        return int(abs(excess_return) <= threshold)
    return None


def compute_outcome(
    stock_bars: Sequence[Dict[str, Any]],
    bench_bars: Sequence[Dict[str, Any]],
    decision_day: date,
    horizon: int,
) -> Optional[Dict[str, Any]]:
    """计算超额收益。bars 为 [{date, open, close}] 升序；数据不足返回 None（待验证）。"""
    sm = {b["date"]: b for b in stock_bars}
    bm = {b["date"]: b for b in bench_bars}
    common = sorted(d for d in sm if d in bm)

    entry_idx = next((i for i, d in enumerate(common) if d > decision_day), None)
    if entry_idx is None:
        return None
    exit_idx = entry_idx + horizon - 1
    if exit_idx >= len(common):
        return None

    e_day, x_day = common[entry_idx], common[exit_idx]
    s_entry, s_exit = sm[e_day]["open"], sm[x_day]["close"]
    b_entry, b_exit = bm[e_day]["open"], bm[x_day]["close"]
    if min(s_entry, s_exit, b_entry, b_exit) <= 0:
        return None

    stock_return = s_exit / s_entry - 1
    bench_return = b_exit / b_entry - 1
    return {
        "entry_date": e_day.isoformat(),
        "exit_date": x_day.isoformat(),
        "entry_price": s_entry,
        "exit_price": s_exit,
        "stock_return": stock_return,
        "bench_return": bench_return,
        "excess_return": stock_return - bench_return,
    }


def _decision_day(created_at: str) -> date:
    """SQLite CURRENT_TIMESTAMP 是 UTC 字符串。"""
    return datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).date()


async def verify_pending(
    mm: MemoryManager,
    horizons: Sequence[int] = VERIFY_HORIZONS,
    limit: int = 200,
    fetch: Fetcher = fetch_daily_bars,
) -> Dict[str, Any]:
    """验证所有已到期的记录，写入 agent_outcomes。返回汇总（可用于日志/评测）。"""
    summary: Dict[str, Any] = {"checked": 0, "verified": 0, "pending": 0, "errors": 0, "by_horizon": {}}
    cache: Dict[tuple, List[Dict[str, Any]]] = {}

    async def bars(sym: str, start: date, end: date) -> List[Dict[str, Any]]:
        key = (sym, start, end)
        if key not in cache:
            cache[key] = await fetch(sym, start, end)
        return cache[key]

    for h in horizons:
        # 交易日 h 个 ≈ 日历日 h*1.4；加几天余量覆盖节假日
        min_age_days = int(h * 1.4) + 4
        pending = await mm.pending_verifications(h, min_age_days, limit)
        h_stats = {"checked": 0, "verified": 0, "pending": 0}
        for item in pending:
            summary["checked"] += 1
            h_stats["checked"] += 1
            try:
                d0 = _decision_day(item["created_at"])
                start, end = d0 - timedelta(days=7), d0 + timedelta(days=int(h * 1.4) + 10)
                stock_bars = await bars(yahoo_symbol(item["stock_symbol"]), start, end)
                bench_bars = await bars(benchmark_for(item["stock_symbol"]), start, end)
                out = compute_outcome(stock_bars, bench_bars, d0, h)
            except Exception:
                out = None
                summary["errors"] += 1

            if out is None:
                summary["pending"] += 1
                h_stats["pending"] += 1
                continue

            rows = []
            for agent, vote in item["votes"].items():
                correct = judge_vote(vote, out["excess_return"])
                rows.append((
                    agent, vote, out["entry_date"], out["exit_date"], out["entry_price"],
                    out["exit_price"], out["stock_return"], out["bench_return"],
                    out["excess_return"], correct,
                ))
            await mm.save_outcome_rows(item["id"], h, rows, mark_verified=(h == PRIMARY_HORIZON))
            summary["verified"] += 1
            h_stats["verified"] += 1
        summary["by_horizon"][h] = h_stats
    return summary


async def run(mm: MemoryManager, horizons: Sequence[int], limit: int, update_weights: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {"verify": await verify_pending(mm, horizons, limit)}
    if update_weights:
        result["objective_weights"] = await mm.refresh_objective_weights(PRIMARY_HORIZON, MIN_VERIFIED_SAMPLES)
    # 自挖掘规则：从验证失败中提炼候选纠正规则（默认禁用，需在规则面板人工启用）
    try:
        result["self_mined_rules"] = await mm.mine_outcome_rules(PRIMARY_HORIZON)
    except Exception as e:  # 挖掘失败不影响验证/权重主流程
        result["self_mined_rules"] = [{"error": str(e)}]
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="结果验证（前向检验）与客观权重更新")
    parser.add_argument("--horizons", default=",".join(str(h) for h in VERIFY_HORIZONS))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--no-weights", action="store_true", help="只验证，不更新客观权重")
    parser.add_argument("--db", default=None, help="记忆库路径（默认 FINGENT_MEMORY_DB / memory/l3_memory.db）")
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    mm = MemoryManager(db_path=args.db)
    result = asyncio.run(run(mm, horizons, args.limit, update_weights=not args.no_weights))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _main()
