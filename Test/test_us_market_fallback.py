"""美股行情主/兜底源：Yahoo 直连为主源，失败/不完整时切 Tiingo（monkeypatch，不走网络）。

注（2026-10-17 主源互换）：Tiingo 免费额度按小时限流，持续访问会超额（原 Agent.md §7.14 估算
120s TTL 下约 600 次/小时）；Yahoo 直连无 key、无配额，故改为 Yahoo 主源、Tiingo 兜底。
"""
import asyncio

import api_server as a

_YAHOO_ROW = {"symbol": "AAPL", "name": "Apple Inc.", "price": 1.0, "change": 0.0,
              "changePercent": 0.0, "volume": 1, "dayHigh": 1.0, "dayLow": 1.0}


def _run(yahoo_result, tiingo_result):
    """替换两个源函数：yahoo_result 可传 dict / Exception；tiingo_result 同理。"""
    orig_y, orig_t = a._get_us_market_yahoo, a._get_us_market_tiingo

    async def fake_yahoo():
        if isinstance(yahoo_result, Exception):
            raise yahoo_result
        return yahoo_result

    async def fake_tiingo():
        if isinstance(tiingo_result, Exception):
            raise tiingo_result
        return tiingo_result

    a._get_us_market_yahoo, a._get_us_market_tiingo = fake_yahoo, fake_tiingo
    try:
        return asyncio.run(a._get_us_market())
    finally:
        a._get_us_market_yahoo, a._get_us_market_tiingo = orig_y, orig_t


def _full_yahoo():
    return {"success": True, "market": "us", "source": "yahoo", "count": len(a.US_HOT_STOCKS),
            "stocks": [dict(_YAHOO_ROW, symbol=s) for s in a.US_HOT_STOCKS]}


def _full_tiingo():
    return {"success": True, "market": "us", "source": "tiingo", "count": len(a.US_HOT_STOCKS),
            "stocks": [dict(_YAHOO_ROW, symbol=s) for s in a.US_HOT_STOCKS]}


def test_yahoo_complete_is_used():
    r = _run(_full_yahoo(), None)
    assert r["success"] and r["source"] == "yahoo" and r["count"] == len(a.US_HOT_STOCKS)


def test_yahoo_failure_falls_back_to_tiingo():
    r = _run({"success": False, "error": "网络错误"}, _full_tiingo())
    assert r["success"] and r["source"] == "tiingo" and r["count"] == len(a.US_HOT_STOCKS)


def test_yahoo_partial_switches_whole_set_to_tiingo():
    partial = {"success": True, "market": "us", "source": "yahoo", "count": 10,
               "stocks": [dict(_YAHOO_ROW, symbol=s) for s in a.US_HOT_STOCKS[:10]]}
    r = _run(partial, _full_tiingo())
    assert r["source"] == "tiingo" and r["count"] == len(a.US_HOT_STOCKS)


def test_yahoo_exception_falls_back():
    r = _run(RuntimeError("boom"), _full_tiingo())
    assert r["source"] == "tiingo"


def test_both_sources_down_returns_chinese_error_not_raise():
    r = _run(RuntimeError("断网"), {"success": False, "error": "TIINGO_API_KEY 未配置"})
    assert r["success"] is False and "Yahoo" in r["error"] and "Tiingo" in r["error"]


def test_tiingo_partial_too_returns_chinese_error():
    partial = {"success": True, "market": "us", "source": "tiingo", "count": 8,
               "stocks": [dict(_YAHOO_ROW, symbol=s) for s in a.US_HOT_STOCKS[:8]]}
    r = _run({"success": False, "error": "Yahoo 无返回"}, partial)
    assert r["success"] is False and "Tiingo" in r["error"] and "仅返回" in r["error"]
