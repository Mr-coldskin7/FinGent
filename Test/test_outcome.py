"""结果验证（前向检验）：判定规则、无前视偏差、待验证处理、客观权重。全部用合成数据，不走网络。"""
import asyncio
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from Data import outcome as O
from Data.memory import MemoryManager
from Data.providers.yahoo_us import parse_history


def _bars(start: date, n: int, base: float = 100.0, step: float = 0.0, spike_day: date = None, spike_close: float = None):
    out = []
    d = start
    i = 0
    while len(out) < n:
        if d.weekday() < 5:  # 跳过周末，模拟交易日
            price = base + step * i
            close = spike_close if (spike_day and d == spike_day) else price
            out.append({"date": d, "open": price, "close": close})
            i += 1
        d += timedelta(days=1)
    return out


# ---------- 判定规则 ----------
@pytest.mark.parametrize("vote,excess,expected", [
    ("BUY", 0.05, 1), ("STRONG_BUY", 0.021, 1), ("BUY", 0.01, 0), ("BUY", -0.05, 0),
    ("SELL", -0.05, 1), ("REDUCE", -0.03, 1), ("SELL", 0.0, 0),
    ("HOLD", 0.01, 1), ("HOLD", -0.019, 1), ("HOLD", 0.05, 0),
    ("WEIRD", 0.5, None),
])
def test_judge_vote(vote, excess, expected):
    assert O.judge_vote(vote, excess) == expected


# ---------- 无前视偏差 ----------
def test_entry_uses_next_day_open_not_decision_day_close():
    d0 = date(2026, 10, 5)  # 周一
    # 决策日收盘出现大涨（若误用会得到超高收益）
    stock = _bars(d0, 40, base=100, spike_day=d0, spike_close=500)
    bench = _bars(d0, 40, base=100)
    out = O.compute_outcome(stock, bench, d0, horizon=5)
    entry = [b for b in stock if b["date"] > d0][0]
    assert out["entry_date"] == entry["date"].isoformat()
    assert out["entry_price"] == entry["open"]  # 决策日的 500 不参与入场


def test_excess_return_is_stock_minus_benchmark():
    d0 = date(2026, 10, 5)
    stock = _bars(d0, 40, base=100, step=1.0)   # 个股上涨
    bench = _bars(d0, 40, base=100, step=0.2)   # 基准涨得少
    out = O.compute_outcome(stock, bench, d0, horizon=5)
    assert out["excess_return"] == pytest.approx(out["stock_return"] - out["bench_return"])
    assert out["excess_return"] > 0


def test_not_enough_bars_is_pending_not_wrong():
    d0 = date(2026, 10, 5)
    stock = _bars(d0, 3)
    bench = _bars(d0, 3)
    assert O.compute_outcome(stock, bench, d0, horizon=5) is None


def test_missing_benchmark_is_pending():
    d0 = date(2026, 10, 5)
    assert O.compute_outcome(_bars(d0, 40), [], d0, horizon=5) is None


# ---------- 符号与基准 ----------
def test_symbol_and_benchmark_mapping():
    assert O.yahoo_symbol("nvda") == "NVDA"
    assert O.yahoo_symbol("600519") == "600519.SS"
    assert O.yahoo_symbol("000001") == "000001.SZ"
    assert O.benchmark_for("NVDA") == "^GSPC"
    assert O.benchmark_for("600519") == "000300.SS"
    assert O.benchmark_for("300750") == "000300.SZ"


# ---------- Yahoo 历史解析 ----------
def test_parse_history_uses_exchange_local_dates_and_skips_nulls():
    ts = [int(datetime(2026, 10, 14, 13, 30, tzinfo=timezone.utc).timestamp()),
          int(datetime(2026, 10, 15, 13, 30, tzinfo=timezone.utc).timestamp()),
          int(datetime(2026, 10, 16, 13, 30, tzinfo=timezone.utc).timestamp())]
    payload = {"chart": {"result": [{
        "meta": {"gmtoffset": -4 * 3600},
        "timestamp": ts,
        "indicators": {"quote": [{"open": [1.0, None, 3.0], "close": [1.5, 2.0, 3.5]}]},
    }]}}
    bars = parse_history(payload)
    assert [b["date"] for b in bars] == [date(2026, 10, 14), date(2026, 10, 16)]
    assert bars[0]["open"] == 1.0 and bars[1]["close"] == 3.5


def test_parse_history_bad_payload_is_empty():
    assert parse_history({}) == []
    assert parse_history({"chart": {"result": None}}) == []


# ---------- 端到端（临时库 + 假数据源） ----------
def _insert_old_analysis(db, created_days_ago: int, votes: dict, symbol="NVDA") -> int:
    created = (datetime.now(timezone.utc) - timedelta(days=created_days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db) as c:
        cur = c.execute(
            "INSERT INTO analysis_history (user_id, session_id, stock_symbol, agent_votes, final_decision, created_at) "
            "VALUES ('u', 's', ?, ?, 'BUY', ?)",
            (symbol, str(votes).replace("'", '"'), created),
        )
        rid = cur.lastrowid
        for a, v in votes.items():
            c.execute("INSERT INTO agent_votes (analysis_id, agent_name, vote) VALUES (?, ?, ?)", (rid, a, v))
        c.commit()
        return rid


def test_verify_pending_writes_outcomes_and_marks_primary(tmp_path):
    db = str(tmp_path / "v.db")
    mm = MemoryManager(db_path=db)
    rid = _insert_old_analysis(db, created_days_ago=60, votes={"A": "BUY", "B": "SELL"})

    async def fake_fetch(symbol, start, end):
        d0 = date.today() - timedelta(days=60)
        if symbol == "^GSPC":
            return _bars(d0 - timedelta(days=7), 90, base=100, step=0.1)
        return _bars(d0 - timedelta(days=7), 90, base=100, step=1.0)  # 个股强势

    summary = asyncio.run(O.verify_pending(mm, horizons=(5, 20), fetch=fake_fetch))
    assert summary["verified"] >= 2
    rows = {r["agent"]: r for r in asyncio.run(mm.analysis_outcomes(rid)) if r["horizon_days"] == 20}
    assert rows["A"]["correct"] == 1   # BUY 在强势股上判对
    assert rows["B"]["correct"] == 0   # SELL 在强势股上判错
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT market_outcome FROM analysis_history WHERE id=?", (rid,)).fetchone()[0] == "verified"


def test_verify_pending_is_idempotent_and_skips_unavailable_data(tmp_path):
    db = str(tmp_path / "v2.db")
    mm = MemoryManager(db_path=db)
    _insert_old_analysis(db, created_days_ago=60, votes={"A": "BUY"})

    async def no_data(symbol, start, end):
        return []

    first = asyncio.run(O.verify_pending(mm, horizons=(20,), fetch=no_data))
    assert first["verified"] == 0 and first["pending"] >= 1
    assert with_rows(db) == 0  # 数据缺失不写入任何结果


def with_rows(db) -> int:
    with sqlite3.connect(db) as c:
        return c.execute("SELECT COUNT(*) FROM agent_outcomes").fetchone()[0]


def test_recent_analysis_not_yet_due_is_not_checked(tmp_path):
    db = str(tmp_path / "v3.db")
    mm = MemoryManager(db_path=db)
    _insert_old_analysis(db, created_days_ago=1, votes={"A": "BUY"})

    async def fake(symbol, start, end):
        raise AssertionError("未到期的记录不应拉取数据")

    summary = asyncio.run(O.verify_pending(mm, horizons=(5, 20), fetch=fake))
    assert summary["checked"] == 0


# ---------- P1-2 客观权重（样本门槛） ----------
def test_objective_weight_requires_min_samples(tmp_path):
    db = str(tmp_path / "w.db")
    mm = MemoryManager(db_path=db)
    rows = [(f"A", "BUY", "2026-01-01", "2026-01-31", 1, 1, 0, 0, 0.05, 1)]
    # 只有 5 个样本：不更新
    for i in range(5):
        asyncio.run(mm.save_outcome_rows(1000 + i, 20, rows, mark_verified=False))
    rep = asyncio.run(mm.refresh_objective_weights(20, min_samples=30))
    assert rep["A"]["n"] == 5 and rep["A"]["weight"] is None
    assert asyncio.run(mm.resolve_weights("u", ["A"]))["A"] == pytest.approx(1.0)


def test_objective_weight_follows_hit_rate(tmp_path):
    db = str(tmp_path / "w2.db")
    mm = MemoryManager(db_path=db)
    # 30 个样本，全部判对 -> 1 + 2*(1 - 1/3) = 1.6 -> 截断到 1.5
    for i in range(30):
        asyncio.run(mm.save_outcome_rows(2000 + i, 20,
                    [("GOOD", "BUY", "2026-01-01", "2026-01-31", 1, 1, 0, 0, 0.05, 1)], mark_verified=False))
    # 30 个样本，全部判错 -> 1 + 2*(0 - 1/3) = 0.333
    for i in range(30):
        asyncio.run(mm.save_outcome_rows(3000 + i, 20,
                    [("BAD", "BUY", "2026-01-01", "2026-01-31", 1, 1, 0, 0, -0.05, 0)], mark_verified=False))
    rep = asyncio.run(mm.refresh_objective_weights(20, min_samples=30))
    assert rep["GOOD"]["weight"] == pytest.approx(1.5)
    assert rep["BAD"]["weight"] == pytest.approx(round(1 + 2 * (0 - 1 / 3), 4), abs=1e-3)
    assert asyncio.run(mm.resolve_weights("u", ["BAD"]))["BAD"] < 0.5
