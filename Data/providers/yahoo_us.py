"""
Yahoo Finance 美股行情（直连，作为 Tiingo 的兜底数据源）。

使用 v8 chart 接口：无需 API key / crumb，返回 OHLCV 与 meta。
v7 quote 接口需要 crumb，直连会 401，因此不用。

输出字段与 api_server 中 Tiingo 路径保持一致：
symbol / name / price / change / changePercent / volume / dayHigh / dayLow
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

import httpx

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
_CONCURRENCY = 5  # 对 Yahoo 保持礼貌，避免瞬时并发过高


def _local_date(ts: int, gmtoffset: int) -> date:
    return datetime.fromtimestamp(ts + gmtoffset, tz=timezone.utc).date()


def parse_chart(symbol: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """把 Yahoo chart 返回解析成统一行情结构。解析失败或数据不足返回 None（纯函数，便于测试）。"""
    try:
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return None
        r = result[0]
        meta = r.get("meta") or {}
        quote = ((r.get("indicators") or {}).get("quote") or [{}])[0]
        timestamps = r.get("timestamp") or []
        closes = quote.get("close") or []

        # 只保留有收盘价的 K 线
        bars = [(t, c) for t, c in zip(timestamps, closes) if c is not None]
        if not bars:
            return None

        price = meta.get("regularMarketPrice")
        if price is None:
            price = bars[-1][1]
        price = float(price)

        # 昨收：若最后一根 K 线是"今天"（与 regularMarketTime 同日），则取倒数第二根；
        # 否则最后一根已是昨收（例如今日尚未开盘/数据未更新）。
        gmtoffset = int(meta.get("gmtoffset") or 0)
        market_time = meta.get("regularMarketTime")
        last_ts = bars[-1][0]
        if market_time is not None and _local_date(last_ts, gmtoffset) == _local_date(int(market_time), gmtoffset):
            prev_candidates = bars[:-1]
        else:
            prev_candidates = bars
        if not prev_candidates:
            return None
        prev_close = float(prev_candidates[-1][1])
        if prev_close == 0:
            return None

        change = price - prev_close
        change_percent = change / prev_close * 100

        return {
            "symbol": symbol,
            "name": meta.get("shortName") or meta.get("longName") or symbol,
            "price": round(price, 2),
            "change": round(change, 2),
            "changePercent": round(change_percent, 2),
            "volume": int(meta.get("regularMarketVolume") or 0),
            "dayHigh": float(meta.get("regularMarketDayHigh") or 0),
            "dayLow": float(meta.get("regularMarketDayLow") or 0),
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return None


async def _fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, symbol: str, days: int) -> Optional[dict]:
    async with sem:
        try:
            r = await client.get(
                CHART_URL.format(symbol=symbol),
                params={"range": f"{days}d", "interval": "1d"},
                headers=_HEADERS,
            )
            if r.status_code != 200:
                return None
            return parse_chart(symbol, r.json())
        except (httpx.HTTPError, ValueError):
            return None


async def fetch_us_quotes(symbols: Iterable[str], days: int = 7, timeout: float = 10.0) -> list[dict]:
    """并发拉取多只美股，返回解析成功的行情列表（按原顺序，失败的跳过）。"""
    sem = asyncio.Semaphore(_CONCURRENCY)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        results = await asyncio.gather(*(_fetch_one(client, sem, s, days) for s in symbols))
    return [r for r in results if r]


# ───────────────────────── 历史日线（结果验证用） ─────────────────────────
def parse_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """把 chart 返回解析成按交易所本地日期的日线 [{date, open, close}]，升序。纯函数，便于测试。"""
    try:
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return []
        r = result[0]
        meta = r.get("meta") or {}
        gmtoffset = int(meta.get("gmtoffset") or 0)
        quote = ((r.get("indicators") or {}).get("quote") or [{}])[0]
        timestamps = r.get("timestamp") or []
        opens = quote.get("open") or []
        closes = quote.get("close") or []
        bars = []
        for ts, o, c in zip(timestamps, opens, closes):
            if o is None or c is None:
                continue
            bars.append({"date": _local_date(int(ts), gmtoffset), "open": float(o), "close": float(c)})
        return sorted(bars, key=lambda b: b["date"])
    except (TypeError, ValueError, KeyError, IndexError):
        return []


async def fetch_daily_bars(symbol: str, start: date, end: date, timeout: float = 15.0) -> list[dict[str, Any]]:
    """拉取 [start, end] 的日线（Yahoo 直连）。失败返回空列表，由上层判定为"待验证"。"""
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp())
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            r = await client.get(
                CHART_URL.format(symbol=symbol),
                params={"period1": p1, "period2": p2, "interval": "1d"},
                headers=_HEADERS,
            )
            if r.status_code != 200:
                return []
            return parse_history(r.json())
        except (httpx.HTTPError, ValueError):
            return []
