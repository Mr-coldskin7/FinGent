"""Yahoo Finance 美股兜底解析测试（纯函数，不走网络）。"""
from datetime import datetime, timezone

from Data.providers.yahoo_us import parse_chart

EXPECTED_KEYS = {"symbol", "name", "price", "change", "changePercent", "volume", "dayHigh", "dayLow"}


def _ts(y, m, d, hh=13, mm=30):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp())


def _payload(timestamps, closes, **meta_overrides):
    meta = {
        "symbol": "AAPL",
        "shortName": "Apple Inc.",
        "gmtoffset": -4 * 3600,
        "regularMarketPrice": 342.0,
        "regularMarketTime": _ts(2026, 10, 15, 20, 0),
        "regularMarketVolume": 39605715,
        "regularMarketDayHigh": 347.85,
        "regularMarketDayLow": 341.6,
    }
    meta.update(meta_overrides)
    return {"chart": {"result": [{"meta": meta, "timestamp": timestamps, "indicators": {"quote": [{"close": closes}]}}]}}


def test_parse_normal_uses_previous_bar_when_last_bar_is_today():
    ts = [_ts(2026, 10, 14), _ts(2026, 10, 15)]
    out = parse_chart("AAPL", _payload(ts, [340.0, 342.0]))
    assert set(out) == EXPECTED_KEYS
    assert out["name"] == "Apple Inc."
    assert out["price"] == 342.0
    assert out["change"] == 2.0  # 342 - 340
    assert out["changePercent"] == round(2 / 340 * 100, 2)
    assert out["volume"] == 39605715
    assert out["dayHigh"] == 347.85 and out["dayLow"] == 341.6


def test_parse_when_today_bar_missing_last_bar_is_previous_close():
    # 今日尚无 K 线：最后一根就是昨收，不能再往前退一根
    ts = [_ts(2026, 10, 13), _ts(2026, 10, 14), _ts(2026, 10, 15)]
    # 今日现价 345，昨收 = 最后一根 342（不是倒数第二根 340）
    out = parse_chart("AAPL", _payload(ts, [336.0, 340.0, 342.0], regularMarketPrice=345.0, regularMarketTime=_ts(2026, 10, 16, 13, 0)))
    assert out["change"] == 3.0


def test_parse_skips_null_closes():
    ts = [_ts(2026, 10, 13), _ts(2026, 10, 14), _ts(2026, 10, 15)]
    out = parse_chart("AAPL", _payload(ts, [336.0, None, 342.0]))
    assert out["change"] == 6.0  # 342 vs 336（跳过 None）


def test_parse_empty_or_malformed_returns_none():
    assert parse_chart("X", {}) is None
    assert parse_chart("X", {"chart": {"result": None}}) is None
    assert parse_chart("X", _payload([], [])) is None
    assert parse_chart("X", _payload([_ts(2026, 10, 15)], [None])) is None


def test_parse_zero_previous_close_returns_none():
    ts = [_ts(2026, 10, 14), _ts(2026, 10, 15)]
    assert parse_chart("X", _payload(ts, [0.0, 342.0])) is None


def test_parse_falls_back_to_symbol_when_no_name():
    ts = [_ts(2026, 10, 14), _ts(2026, 10, 15)]
    out = parse_chart("ZZZ", _payload(ts, [10.0, 11.0], shortName=None, longName=None))
    assert out["name"] == "ZZZ"
